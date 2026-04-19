#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FD003 clean reviewer-table script using the tuned GF-CoLLM training schedule.

Rows produced:
- small_only
- corrected_large_side
- threshold_routing
- mlp_gate
- type1_fuzzy_routing
- it2_fuzzy_routing
- gf_collm

Columns produced:
- rmse
- mae
- coverage
- interval_width
- latency_ms_per_sample

Required files/folder layout
----------------------------
./FD003/train_FD003.txt
./FD003/test_FD003.txt
./FD003/RUL_FD003.txt

For the model file, any one of these names is accepted:
./paper8_fd003.py
./paper8_fd003 (4).py
./paper8_fd003 (3).py

Optional:
./fd003_split.json
If absent, the script creates a deterministic 80/20 unit-level split.
"""

import os
import json
import math
import copy
import time
import random
import importlib
import importlib.util
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


FD = "FD003"
DATA_DIR = "FD003"
SPLIT_JSON = "fd003_split.json"

SEED = 42
MAX_EPOCHS = 50
PATIENCE = 7
CALIBRATION_ALPHA = 0.10


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_json(split_path):
    with open(split_path, "r") as f:
        return json.load(f)


def maybe_make_split_json(train_df: pd.DataFrame, split_path: str, seed: int = 42):
    if os.path.exists(split_path):
        return

    unit_col = "unit" if "unit" in train_df.columns else "unit_id"
    unit_ids = sorted(train_df[unit_col].unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(unit_ids)

    n_val = max(1, int(round(0.20 * len(unit_ids))))
    val_ids = unit_ids[:n_val]
    train_ids = unit_ids[n_val:]

    split = {
        "train_engine_ids": train_ids,
        "val_engine_ids": val_ids,
    }
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)


def get_engine_id_col(df):
    for c in ["unit_id", "unit", "engine_id", "id"]:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find engine id column. Available columns: {df.columns.tolist()}")


def import_module_from_file(module_name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {filepath}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def infer_p8_module():
    try:
        return importlib.import_module("paper8_fd003")
    except Exception:
        pass

    candidates = [
        "paper8_fd003.py",
        "paper8_fd003 (4).py",
        "paper8_fd003 (3).py",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            return import_module_from_file("paper8_fd003_dynamic", fp)

    raise FileNotFoundError(
        "Could not find paper8_fd003 model file. Expected one of: "
        "paper8_fd003.py, paper8_fd003 (4).py, paper8_fd003 (3).py"
    )


def load_raw_data_with_p8(p8, data_dir, fd):
    fd = fd.upper()
    train_file = os.path.join(data_dir, f"train_{fd}.txt")
    test_file = os.path.join(data_dir, f"test_{fd}.txt")
    rul_file = os.path.join(data_dir, f"RUL_{fd}.txt")

    load_fn = None
    for name in [f"load_{fd.lower()}", "load_fd003", "load_fd001", "load_cmapss"]:
        if hasattr(p8, name):
            load_fn = getattr(p8, name)
            break
    if load_fn is None:
        raise AttributeError("Could not find a dataset loader in p8 module.")

    train_df = load_fn(train_file)
    test_df = load_fn(test_file)
    train_df = p8.add_rul_labels(train_df)
    test_rul_true = p8.load_rul_file(rul_file)
    return train_df, test_df, test_rul_true


def get_feature_cols(train_df, fd=None):
    # FD003 follows the sensor-only choice from the FD003 tuner
    useless_sensors = [1, 5, 6, 10, 16, 18, 19]
    sensor_cols = [f"s{i}" for i in range(1, 22) if f"s{i}" in train_df.columns and i not in useless_sensors]
    if sensor_cols:
        return sensor_cols
    sensor_cols = [f"sensor_{i}" for i in range(1, 22) if f"sensor_{i}" in train_df.columns and i not in useless_sensors]
    if sensor_cols:
        return sensor_cols
    raise ValueError(f"Could not infer feature columns from columns: {train_df.columns.tolist()}")


def split_train_val_by_engine(train_df, split_json):
    train_ids = set(split_json["train_engine_ids"])
    val_ids = set(split_json["val_engine_ids"])
    engine_col = get_engine_id_col(train_df)
    train_part = train_df[train_df[engine_col].isin(train_ids)].copy()
    val_part = train_df[train_df[engine_col].isin(val_ids)].copy()
    return train_part, val_part


def normalize_with_train_only(train_part, val_part, test_df, feature_cols):
    train_part = train_part.copy()
    val_part = val_part.copy()
    test_df = test_df.copy()

    train_part[feature_cols] = train_part[feature_cols].astype(np.float32)
    val_part[feature_cols] = val_part[feature_cols].astype(np.float32)
    test_df[feature_cols] = test_df[feature_cols].astype(np.float32)

    mean = train_part[feature_cols].mean(axis=0)
    std = train_part[feature_cols].std(axis=0).replace(0, 1.0)

    train_norm = train_part.copy()
    val_norm = val_part.copy()
    test_norm = test_df.copy()

    train_norm[feature_cols] = ((train_part[feature_cols] - mean) / std).astype(np.float32)
    val_norm[feature_cols] = ((val_part[feature_cols] - mean) / std).astype(np.float32)
    test_norm[feature_cols] = ((test_df[feature_cols] - mean) / std).astype(np.float32)
    return train_norm, val_norm, test_norm


def create_train_windows(df, feature_cols, window_size, unit_ids=None):
    X_list, y_list = [], []
    if unit_ids is None:
        grouped = df.groupby("unit")
    else:
        grouped = [(uid, df[df["unit"] == uid]) for uid in unit_ids]

    for _, group in grouped:
        group = group.sort_values("time")
        data = group[feature_cols].values
        rul = group["RUL"].values
        if len(group) < window_size:
            continue
        for start in range(0, len(group) - window_size + 1):
            end = start + window_size
            X_list.append(data[start:end, :])
            y_list.append(rul[end - 1])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def create_test_last_windows(test_df, feature_cols, window_size):
    X_list, unit_ids = [], []
    for uid, group in test_df.groupby("unit"):
        group = group.sort_values("time")
        data = group[feature_cols].values
        if len(group) >= window_size:
            seq = data[-window_size:, :]
        else:
            pad_len = window_size - len(group)
            pad_block = np.repeat(data[0:1, :], pad_len, axis=0)
            seq = np.vstack([pad_block, data])
        X_list.append(seq)
        unit_ids.append(uid)
    return np.array(X_list, dtype=np.float32), np.array(unit_ids, dtype=np.int32)


def create_windows_trainvaltest(p8, train_norm, val_norm, test_norm, test_rul_true, feature_cols, window_size):
    X_train, y_train = create_train_windows(train_norm, feature_cols, window_size)
    X_val, y_val = create_train_windows(val_norm, feature_cols, window_size)
    X_test, test_units = create_test_last_windows(test_norm, feature_cols, window_size)

    n_common = min(len(test_rul_true), len(test_units))
    X_test = X_test[:n_common]
    y_test = test_rul_true[:n_common]
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def evaluate_components(model, loader, device):
    model.eval()
    ys_s, ys_l, ys_c, ys_f, ys_true = [], [], [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            ys_s.append(out["y_s"].cpu().numpy())
            ys_l.append(out["y_l"].cpu().numpy())
            ys_c.append(out["y_c"].cpu().numpy())
            ys_f.append(out["y_fused"].cpu().numpy())
            ys_true.append(yb.numpy())

    y_true = np.concatenate(ys_true, axis=0)
    y_s = np.concatenate(ys_s, axis=0)
    y_l = np.concatenate(ys_l, axis=0)
    y_c = np.concatenate(ys_c, axis=0)
    y_f = np.concatenate(ys_f, axis=0)

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _mae(a, b):
        return float(np.mean(np.abs(a - b)))

    return {
        "s_rmse": _rmse(y_true, y_s),
        "l_rmse": _rmse(y_true, y_l),
        "c_rmse": _rmse(y_true, y_c),
        "f_rmse": _rmse(y_true, y_f),
        "f_mae": _mae(y_true, y_f),
    }


def branch_error_correlation_loss(y_s, y_l, y_true):
    err_s = y_s - y_true
    err_l = y_l - y_true
    err_s = err_s - torch.mean(err_s)
    err_l = err_l - torch.mean(err_l)
    numerator = torch.mean(err_s * err_l)
    denominator = torch.std(err_s) * torch.std(err_l) + 1e-6
    corr = numerator / denominator
    return torch.abs(corr)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}


def train_gfcollm_with_schedule(
    p8,
    input_dim,
    train_loader,
    val_loader,
    device,
    max_epochs=50,
    patience=7,
    lambda_res=0.2,
    lambda_route=0.1,
    lambda_conf=0.1,
    lr=2e-4,
    lambda_no_harm_strong=40.0,
):
    model = p8.GFCoLLM(input_dim=input_dim, window_size=p8.WINDOW_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    ema = EMA(model, decay=0.999)

    best_state = None
    best_val = float("inf")
    patience_ctr = 0

    print("FD003 tuner v4 ACTIVE: safe-switch gating + disagreement-aware gate + refine cap ±1 + powered no-harm")

    for epoch in range(1, max_epochs + 1):
        fusion_trainable = epoch > 10
        for name, param in model.named_parameters():
            if any(k in name for k in ["fusion", "route", "trust_net", "anchor_gate_net", "base_anchor_net", "final_gate_net"]):
                param.requires_grad = fusion_trainable

        model.train()
        total = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            out = model(xb)
            y_s = out["y_s"]
            y_l = out["y_l"]
            y_c = out["y_c"]
            y_fused = out["y_fused"]

            loss_s = mse(y_s, yb)
            loss_l = mse(y_l, yb)
            loss_pred = mse(y_fused, yb)
            loss_res = mse(y_c, yb)

            residual_hat = out.get("residual_hat", torch.zeros_like(yb))
            route_logits = out.get("route_logits", None)
            conf_score = out.get("conf_score", None)
            gate_weights = out.get("gate_weights", None)
            gate_logits = out.get("gate_logits", None)
            feat_s = out.get("feat_s", None)
            feat_l_proj = out.get("feat_l_proj", None)

            with torch.no_grad():
                per_sample_err_s = (y_s - yb) ** 2
                per_sample_err_l = (y_l - yb) ** 2
                per_sample_err_c = (y_c - yb) ** 2
                branch_errs = torch.stack([per_sample_err_s, per_sample_err_l, per_sample_err_c], dim=1)
                best_idx = torch.argmin(branch_errs, dim=1)
                best_branch_err = torch.min(branch_errs, dim=1)[0]

            per_sample_err_fused = (y_fused - yb) ** 2
            loss_no_harm = torch.mean(torch.pow(torch.relu(per_sample_err_fused - best_branch_err), 1.5))

            loss_route = torch.tensor(0.0, device=device)
            if route_logits is not None:
                loss_route = ce(route_logits, best_idx)

            loss_conf = torch.tensor(0.0, device=device)
            if conf_score is not None:
                loss_conf = torch.mean((conf_score - torch.abs(y_fused - yb)) ** 2)

            loss_gate_supervision = torch.tensor(0.0, device=device)
            if gate_logits is not None:
                loss_gate_supervision = ce(gate_logits, best_idx)
            elif gate_weights is not None:
                loss_gate_supervision = ce(torch.log(gate_weights + 1e-8), best_idx)

            gate_entropy = torch.tensor(0.0, device=device)
            if gate_weights is not None:
                gate_entropy = -torch.mean(torch.sum(gate_weights * torch.log(gate_weights + 1e-6), dim=-1))

            loss_resid_refine = torch.mean(residual_hat ** 2)

            loss_ortho = torch.tensor(0.0, device=device)
            if feat_s is not None and feat_l_proj is not None and feat_s.shape == feat_l_proj.shape:
                loss_ortho = torch.mean(torch.abs(torch.sum(feat_s * feat_l_proj, dim=-1)))

            loss_diversity = torch.mean(torch.exp(-torch.abs(y_s - y_l)))

            if epoch <= 10:
                loss = (
                    1.5 * loss_s
                    + 1.0 * loss_l
                    + 0.1 * loss_ortho
                    + 0.05 * loss_diversity
                )
            elif epoch <= 15:
                loss = (
                    loss_pred
                    + 0.5 * loss_res
                    + 1.0 * (loss_s + loss_l)
                    + 5.0 * loss_no_harm
                    + lambda_res * loss_resid_refine
                    + 0.35 * loss_gate_supervision
                    + 0.05 * gate_entropy
                    + 0.1 * loss_ortho
                    + 0.05 * loss_diversity
                )
            else:
                lambda_no_harm = lambda_no_harm_strong
                loss = (
                    loss_pred
                    + 0.5 * loss_res
                    + lambda_no_harm * loss_no_harm
                    + lambda_res * loss_resid_refine
                    + lambda_route * loss_route
                    + lambda_conf * loss_conf
                    + 0.75 * loss_gate_supervision
                    + 0.05 * gate_entropy
                    + 0.1 * loss_ortho
                    + 0.1 * loss_diversity
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ema.update(model)
            total += loss.item() * xb.size(0)

        ema.apply_shadow(model)
        val_metrics = evaluate_components(model, val_loader, device)
        ema.restore(model)

        train_obj = math.sqrt(total / len(train_loader.dataset))
        print(
            f"[Epoch {epoch:02d}] train_obj={train_obj:.4f} "
            f"val_fused_RMSE={val_metrics['f_rmse']:.4f} "
            f"(s={val_metrics['s_rmse']:.4f}, l={val_metrics['l_rmse']:.4f}, c={val_metrics['c_rmse']:.4f})"
        )

        if val_metrics["f_rmse"] < best_val:
            best_val = val_metrics["f_rmse"]
            ema.apply_shadow(model)
            best_state = copy.deepcopy(model.state_dict())
            ema.restore(model)
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val


def infer_outputs(model, loader, device):
    model.eval()
    ys_true = []
    merged = {
        "y_s": [], "y_l": [], "y_c": [], "y_fused": [],
        "route_logits": [], "gate_weights": [], "conf_s": [], "conf_l": [], "conf_score": []
    }
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            for k in merged:
                merged[k].append(out[k].detach().cpu().numpy())
            ys_true.append(yb.numpy())

    y_true = np.concatenate(ys_true, axis=0)
    merged = {k: np.concatenate(v, axis=0) for k, v in merged.items()}
    return y_true, merged


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def conformal_width(cal_y, cal_pred, alpha=0.10):
    abs_res = np.abs(cal_y - cal_pred)
    n = len(abs_res)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    try:
        q = float(np.quantile(abs_res, q_level, method="higher"))
    except TypeError:
        q = float(np.quantile(abs_res, q_level, interpolation="higher"))
    return q


def large_side_conf(conf_l, conf_score):
    return 0.7 * conf_l + 0.3 * (1.0 / (1.0 + conf_score))


def fuzzy_score_type1(conf_s, conf_l, conf_score, y_s, y_c):
    disagreement = np.abs(y_s - y_c)
    lside = large_side_conf(conf_l, conf_score)
    score_l = 0.70 * lside + 0.30 * (1.0 / (1.0 + disagreement))
    score_s = 0.70 * conf_s + 0.30 * (1.0 / (1.0 + disagreement))
    return score_s, score_l


def fuzzy_score_it2(conf_s, conf_l, conf_score, y_s, y_c):
    disagreement = np.abs(y_s - y_c)
    low = 1.0 / (1.0 + conf_score + 0.15)
    high = 1.0 / np.maximum(1.0, 1.0 + conf_score - 0.15)
    lside_center = 0.5 * (0.7 * conf_l + 0.3 * low + 0.7 * conf_l + 0.3 * high)
    score_l = 0.65 * lside_center + 0.35 * (1.0 / (1.0 + disagreement))
    score_s = 0.70 * conf_s + 0.30 * (1.0 / (1.0 + disagreement))
    return score_s, score_l


def predict_by_mode(mode: str, outputs: Dict[str, np.ndarray]):
    y_s = outputs["y_s"]
    y_l = outputs["y_l"]
    y_c = outputs["y_c"]
    y_f = outputs["y_fused"]
    conf_s = outputs["conf_s"]
    conf_l = outputs["conf_l"]
    conf_score = outputs["conf_score"]
    route_logits = outputs["route_logits"]
    gate_weights = outputs["gate_weights"]

    if mode == "small_only":
        pred = y_s
    elif mode == "corrected_large_side":
        pred = y_c
    elif mode == "threshold_routing":
        lside = large_side_conf(conf_l, conf_score)
        choose_large = lside >= (conf_s + 0.01)
        pred = np.where(choose_large, y_c, y_s)
    elif mode == "mlp_gate":
        route_idx = np.argmax(route_logits, axis=1)
        pred = np.where(route_idx == 0, y_s, np.where(route_idx == 1, y_l, y_c))
    elif mode == "type1_fuzzy_routing":
        score_s, score_l = fuzzy_score_type1(conf_s, conf_l, conf_score, y_s, y_c)
        choose_large = score_l >= score_s
        pred = np.where(choose_large, y_c, y_s)
    elif mode == "it2_fuzzy_routing":
        score_s, score_l = fuzzy_score_it2(conf_s, conf_l, conf_score, y_s, y_c)
        choose_large = score_l >= score_s
        pred = np.where(choose_large, y_c, y_s)
    elif mode == "gf_collm":
        pred = y_f
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return pred.astype(np.float32)


def measure_mode_latency(model, loader, device, mode: str):
    model.eval()
    total_time = 0.0
    total_samples = 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = model(xb)
            out_np = {
                "y_s": out["y_s"].detach().cpu().numpy(),
                "y_l": out["y_l"].detach().cpu().numpy(),
                "y_c": out["y_c"].detach().cpu().numpy(),
                "y_fused": out["y_fused"].detach().cpu().numpy(),
                "route_logits": out["route_logits"].detach().cpu().numpy(),
                "gate_weights": out["gate_weights"].detach().cpu().numpy(),
                "conf_s": out["conf_s"].detach().cpu().numpy(),
                "conf_l": out["conf_l"].detach().cpu().numpy(),
                "conf_score": out["conf_score"].detach().cpu().numpy(),
            }
            _pred = predict_by_mode(mode, out_np)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_time += elapsed
            total_samples += xb.size(0)
    return 1000.0 * total_time / max(1, total_samples)


def main():
    set_seed(SEED)
    p8 = infer_p8_module()
    device = getattr(p8, "DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    train_df, test_df, test_rul_true = load_raw_data_with_p8(p8, DATA_DIR, FD)
    maybe_make_split_json(train_df, SPLIT_JSON, seed=SEED)
    split_json = load_split_json(SPLIT_JSON)

    feature_cols = get_feature_cols(train_df, fd=FD)
    train_part, val_part = split_train_val_by_engine(train_df, split_json)
    train_norm, val_norm, test_norm = normalize_with_train_only(train_part, val_part, test_df, feature_cols)

    window_size = getattr(p8, "WINDOW_SIZE", 50)
    batch_size = getattr(p8, "BATCH_SIZE", 64)
    input_dim = len(feature_cols)

    X_train, y_train, X_val, y_val, X_test, y_test = create_windows_trainvaltest(
        p8, train_norm, val_norm, test_norm, test_rul_true, feature_cols, window_size
    )
    train_loader, val_loader, test_loader = make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)

    cfg = {
        "lr": 2e-4,
        "lambda_res": 0.3,
        "lambda_route": 0.1,
        "lambda_conf": 0.1,
        "lambda_no_harm_strong": 40.0,
    }
    print(f"Training tuned FD003 GF-CoLLM with config: {cfg}")

    model, best_val_rmse = train_gfcollm_with_schedule(
        p8=p8,
        input_dim=input_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        lambda_res=cfg["lambda_res"],
        lambda_route=cfg["lambda_route"],
        lambda_conf=cfg["lambda_conf"],
        lr=cfg["lr"],
        lambda_no_harm_strong=cfg["lambda_no_harm_strong"],
    )

    print(f"Best validation fused RMSE: {best_val_rmse:.4f}")

    val_y, val_outputs = infer_outputs(model, val_loader, device)
    test_y, test_outputs = infer_outputs(model, test_loader, device)

    modes = [
        "small_only",
        "corrected_large_side",
        "threshold_routing",
        "mlp_gate",
        "type1_fuzzy_routing",
        "it2_fuzzy_routing",
        "gf_collm",
    ]

    rows = []
    for mode in modes:
        cal_pred = predict_by_mode(mode, val_outputs)
        test_pred = predict_by_mode(mode, test_outputs)

        width = conformal_width(val_y, cal_pred, alpha=CALIBRATION_ALPHA)
        lower = test_pred - width
        upper = test_pred + width
        coverage = float(np.mean((test_y >= lower) & (test_y <= upper)))

        base_rmse = rmse(test_y, test_pred)
        base_mae = mae(test_y, test_pred)
        latency_ms = measure_mode_latency(model, test_loader, device, mode)

        rows.append({
            "model_routing_mode": mode,
            "rmse": round(base_rmse, 4),
            "mae": round(base_mae, 4),
            "coverage": round(coverage, 4),
            "interval_width": round(2.0 * width, 4),
            "latency_ms_per_sample": round(latency_ms, 4),
        })

    out_df = pd.DataFrame(rows)
    os.makedirs("fd003_branch_value_outputs", exist_ok=True)
    out_csv = "fd003_branch_value_outputs/fd003_branch_value_table_clean.csv"
    out_df.to_csv(out_csv, index=False)

    print("\n=== Clean FD003 Branch-Value Table ===")
    print(out_df.to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
