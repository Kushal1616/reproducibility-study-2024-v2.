#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FD003 multi-horizon CMAPSS baselines with Split Conformal Prediction (Split CP).

Methods
-------
- GRU + Split CP
- Transformer + Split CP
- CoLLM-C + Split CP
- GF-CoLLM + Split CP

FD003 tuning choices
--------------------
- sensor-only informative features
- direct multi-horizon training (one scalar model per horizon)
- train/val engine split used for early stopping and Split CP calibration
- GF-CoLLM schedule adapted from the FD003 tuning runner:
  * lr = 2e-4
  * EMA validation
  * fusion/gating frozen until epoch 10
  * stronger no-harm penalty in later stages
  * gate supervision + gate entropy
"""

from __future__ import annotations

import os
import json
import math
import copy
import glob
import random
import argparse
import importlib
import importlib.util
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


CMAPSS_COLS = ["unit", "time", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
USELESS_SENSORS = [1, 5, 6, 10, 16, 18, 19]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cmapss(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] == 27:
        df = df.iloc[:, :-1]
    df.columns = CMAPSS_COLS
    return df


def add_train_rul(df: pd.DataFrame, max_rul_cap: int = 125) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby("unit")["time"].max().rename("max_time")
    df = df.merge(max_cycle, on="unit", how="left")
    df["RUL"] = (df["max_time"] - df["time"]).clip(upper=max_rul_cap)
    return df.drop(columns=["max_time"])


def load_rul_file(path: str) -> np.ndarray:
    rul_df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    return rul_df.iloc[:, 0].values.astype(np.float32)


def add_test_rul(df: pd.DataFrame, rul_offsets: np.ndarray, max_rul_cap: int = 125) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby("unit")["time"].max().rename("max_time")
    df = df.merge(max_cycle, on="unit", how="left")
    offset_map = {i + 1: float(rul_offsets[i]) for i in range(len(rul_offsets))}
    df["rul_offset"] = df["unit"].map(offset_map)
    df["full_failure_cycle"] = df["max_time"] + df["rul_offset"]
    df["RUL"] = (df["full_failure_cycle"] - df["time"]).clip(upper=max_rul_cap)
    return df.drop(columns=["max_time", "rul_offset", "full_failure_cycle"])


def load_split_json(path: str) -> Dict[str, List[int]]:
    with open(path, "r") as f:
        split = json.load(f)
    train_ids = split.get("train_engine_ids", split.get("train_units"))
    val_ids = split.get("val_engine_ids", split.get("val_units"))
    if train_ids is None or val_ids is None:
        raise KeyError(f"Split JSON missing train/val engine ids. Found keys: {list(split.keys())}")
    return {"train_engine_ids": list(train_ids), "val_engine_ids": list(val_ids)}


def get_feature_cols(train_df: pd.DataFrame) -> List[str]:
    sensor_cols = [f"s{i}" for i in range(1, 22) if i not in USELESS_SENSORS and f"s{i}" in train_df.columns]
    if not sensor_cols:
        sensor_cols = [f"sensor_{i}" for i in range(1, 22) if i not in USELESS_SENSORS and f"sensor_{i}" in train_df.columns]
    if not sensor_cols:
        raise ValueError(f"Could not infer feature columns from columns: {train_df.columns.tolist()}")
    return sensor_cols


def fit_standardizer(train_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    mean = train_df[feature_cols].mean(axis=0)
    std = train_df[feature_cols].std(axis=0).replace(0, 1.0)
    return mean, std


def apply_standardizer(df: pd.DataFrame, feature_cols: List[str], mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = (out[feature_cols] - mean) / std
    return out


def build_multihorizon_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    horizon: int,
    unit_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list, Y_list = [], []
    groups = df.groupby("unit") if unit_ids is None else [(uid, df[df["unit"] == uid]) for uid in unit_ids]
    for _, g in groups:
        g = g.sort_values("time").reset_index(drop=True)
        feats = g[feature_cols].values.astype(np.float32)
        rul = g["RUL"].values.astype(np.float32)
        for end_idx in range(window_size - 1, len(g) - horizon):
            start_idx = end_idx - window_size + 1
            X_list.append(feats[start_idx:end_idx + 1])
            Y_list.append(rul[end_idx + 1:end_idx + 1 + horizon])
    if not X_list:
        raise ValueError("No windows created.")
    return np.asarray(X_list, dtype=np.float32), np.asarray(Y_list, dtype=np.float32)


def make_scalar_loaders(X_train, y_train_h, X_val, y_val_h, X_test, y_test_h, batch_size):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_h.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_h.astype(np.float32)))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test_h.astype(np.float32)))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def import_module_from_file(module_name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {filepath}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def maybe_import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        candidates = sorted(glob.glob(f"{name}*.py"))
        if not candidates:
            raise
        return import_module_from_file(f"{name}_attached", candidates[0])


class GRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, proj_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim)
        self.gru = nn.GRU(proj_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=0.1 if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        x = self.proj(x)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, ff_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
                                               dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
    def forward(self, x):
        z = self.proj(x)
        z = self.encoder(z)
        return self.head(z[:, -1, :]).squeeze(-1)


def evaluate_scalar_model(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    rmse = math.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    return rmse, mae, preds, trues


def train_simple_with_early_stopping(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                                     device: str, lr: float, max_epochs: int, patience: int):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_state, best_val_rmse, patience_ctr = None, float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_rmse = math.sqrt(total_loss / len(train_loader.dataset))
        val_rmse, val_mae, _, _ = evaluate_scalar_model(model, val_loader, device)
        print(f"[Epoch {epoch:02d}] train_RMSE={train_rmse:.4f} val_RMSE={val_rmse:.4f} val_MAE={val_mae:.4f}")
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_rmse


def split_cp_from_val(y_val_true: np.ndarray, y_val_pred: np.ndarray, alpha: float = 0.1) -> float:
    scores = np.abs(y_val_true - y_val_pred).astype(np.float64)
    n = len(scores)
    q_level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def predict_collmc_point(model_s, model_l, agent_f, reflect_r, loader: DataLoader, device: str):
    model_s.eval(); model_l.eval(); agent_f.eval(); reflect_r.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            y_s, feat_s = model_s(xb)
            y_l, feat_l = model_l(xb)
            mu = agent_f(feat_s)
            refl = reflect_r(feat_l)
            y_collab = mu * y_s + (1.0 - mu) * (y_l + refl)
            preds.append(y_collab.detach().cpu().numpy())
            trues.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def train_p7_collm_with_early_stopping(p7, input_dim: int, train_loader: DataLoader, val_loader: DataLoader,
                                       device: str, max_epochs: int = 50, patience: int = 7):
    model_s = p7.SmallModelS(input_dim=input_dim, embed_dim=32, hidden_dim=64, feature_dim=32).to(device)
    model_l = p7.PatchTransformerL(input_dim=input_dim, patch_size=4, patch_stride=4,
                                   embed_dim=768, num_heads=8, ff_dim=1024, num_layers=2).to(device)
    agent_f = p7.FuzzyDecisionAgent(feature_dim=32, num_memberships=64).to(device)
    reflect_r = p7.SelfReflection(input_dim=768).to(device)
    e1 = max(3, max_epochs // 5); e2 = max(3, max_epochs // 5); e3 = max(5, max_epochs - e1 - e2)
    mse = nn.MSELoss()

    def eval_small():
        model_s.eval(); preds=[]; trues=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); y_hat, _ = model_s(xb)
                preds.append(y_hat.cpu().numpy()); trues.append(yb.numpy())
        p = np.concatenate(preds); t = np.concatenate(trues)
        return math.sqrt(mean_squared_error(t, p))

    def eval_large():
        model_l.eval(); preds=[]; trues=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); y_hat, _ = model_l(xb)
                preds.append(y_hat.cpu().numpy()); trues.append(yb.numpy())
        p = np.concatenate(preds); t = np.concatenate(trues)
        return math.sqrt(mean_squared_error(t, p))

    best_s=None; best_val=float("inf"); patience_ctr=0
    opt = torch.optim.Adam(model_s.parameters(), lr=getattr(p7, "LR_STAGE1", 1e-3))
    for epoch in range(1, e1 + 1):
        model_s.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); y_hat, _ = model_s(xb); loss = mse(y_hat, yb); loss.backward(); opt.step()
        val_rmse = eval_small()
        print(f"[CoLLM-C Stage1 Epoch {epoch:02d}] val_RMSE={val_rmse:.4f}")
        if val_rmse < best_val:
            best_val = val_rmse; best_s = copy.deepcopy(model_s.state_dict()); patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience: break
    if best_s is not None: model_s.load_state_dict(best_s)
    for p in model_s.parameters(): p.requires_grad = False

    best_l=None; best_val=float("inf"); patience_ctr=0
    opt = torch.optim.Adam(model_l.parameters(), lr=getattr(p7, "LR_STAGE2", 1e-3))
    for epoch in range(1, e2 + 1):
        model_l.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); y_hat, _ = model_l(xb); loss = mse(y_hat, yb); loss.backward(); opt.step()
        val_rmse = eval_large()
        print(f"[CoLLM-C Stage2 Epoch {epoch:02d}] val_RMSE={val_rmse:.4f}")
        if val_rmse < best_val:
            best_val = val_rmse; best_l = copy.deepcopy(model_l.state_dict()); patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience: break
    if best_l is not None: model_l.load_state_dict(best_l)
    for p in model_l.parameters(): p.requires_grad = False

    params = list(agent_f.parameters()) + list(reflect_r.parameters())
    opt = torch.optim.Adam(params, lr=getattr(p7, "LR_STAGE3", 1e-3))
    best_pack=None; best_val=float("inf"); patience_ctr=0
    for epoch in range(1, e3 + 1):
        model_s.eval(); model_l.eval(); agent_f.train(); reflect_r.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                y_s, feat_s = model_s(xb); y_l, feat_l = model_l(xb)
            opt.zero_grad()
            mu = agent_f(feat_s); refl = reflect_r(feat_l.detach())
            y_collab = mu * y_s + (1.0 - mu) * (y_l + refl)
            loss = mse(y_collab, yb); loss.backward(); opt.step()
        val_pred, val_true = predict_collmc_point(model_s, model_l, agent_f, reflect_r, val_loader, device)
        val_rmse = math.sqrt(mean_squared_error(val_true, val_pred))
        print(f"[CoLLM-C Stage3 Epoch {epoch:02d}] val_RMSE={val_rmse:.4f}")
        if val_rmse < best_val:
            best_val = val_rmse
            best_pack = {"s": copy.deepcopy(model_s.state_dict()), "l": copy.deepcopy(model_l.state_dict()),
                         "f": copy.deepcopy(agent_f.state_dict()), "r": copy.deepcopy(reflect_r.state_dict())}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience: break
    if best_pack is not None:
        model_s.load_state_dict(best_pack["s"]); model_l.load_state_dict(best_pack["l"])
        agent_f.load_state_dict(best_pack["f"]); reflect_r.load_state_dict(best_pack["r"])
    return model_s, model_l, agent_f, reflect_r, best_val


def evaluate_components(model, loader, device):
    model.eval()
    ys_s, ys_l, ys_c, ys_f, ys_true = [], [], [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); out = model(xb)
            ys_s.append(out["y_s"].cpu().numpy()); ys_l.append(out["y_l"].cpu().numpy())
            ys_c.append(out["y_c"].cpu().numpy()); ys_f.append(out["y_fused"].cpu().numpy())
            ys_true.append(yb.numpy())
    y_true = np.concatenate(ys_true, axis=0); y_s = np.concatenate(ys_s, axis=0)
    y_l = np.concatenate(ys_l, axis=0); y_c = np.concatenate(ys_c, axis=0); y_f = np.concatenate(ys_f, axis=0)
    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    return {"s_rmse": rmse(y_true, y_s), "l_rmse": rmse(y_true, y_l), "c_rmse": rmse(y_true, y_c), "f_rmse": rmse(y_true, y_f)}


def branch_error_correlation_loss(y_s, y_l, y_true):
    err_s = y_s - y_true; err_l = y_l - y_true
    err_s = err_s - torch.mean(err_s); err_l = err_l - torch.mean(err_l)
    numerator = torch.mean(err_s * err_l)
    denominator = torch.std(err_s) * torch.std(err_l) + 1e-6
    return torch.abs(numerator / denominator)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay; self.shadow = {}; self.backup = {}
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


def train_p8_gfcollm_fd003_with_early_stopping(p8, input_dim: int, train_loader: DataLoader, val_loader: DataLoader,
                                               device: str, max_epochs: int = 50, patience: int = 7):
    model = p8.GFCoLLM(input_dim=input_dim, window_size=p8.WINDOW_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    mse = nn.MSELoss(); ce = nn.CrossEntropyLoss(); ema = EMA(model, decay=0.999)
    best_state=None; best_val=float("inf"); patience_ctr=0

    print("FD003 tuner v4: safe-switch gating + disagreement-aware gate + refine cap ±1 + powered no-harm")
    for epoch in range(1, max_epochs + 1):
        fusion_trainable = epoch > 10
        for name, param in model.named_parameters():
            if any(k in name for k in ["fusion", "route", "trust_net", "anchor_gate_net", "base_anchor_net", "final_gate_net"]):
                param.requires_grad = fusion_trainable

        model.train(); total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            y_s = out["y_s"]; y_l = out["y_l"]; y_c = out["y_c"]; y_fused = out["y_fused"]
            loss_s = mse(y_s, yb); loss_l = mse(y_l, yb); loss_pred = mse(y_fused, yb); loss_res = mse(y_c, yb)

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

            loss_no_harm = torch.mean(torch.relu(per_sample_err_fused - best_branch_err) ** 2)
            loss_route = ce(route_logits, best_idx) if route_logits is not None else torch.tensor(0.0, device=device)
            loss_conf = torch.mean((conf_score - torch.abs(y_fused - yb)) ** 2) if conf_score is not None else torch.tensor(0.0, device=device)
            loss_gate_supervision = ce(gate_logits, best_idx) if gate_logits is not None else torch.tensor(0.0, device=device)
            gate_entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=1).mean() if gate_weights is not None else torch.tensor(0.0, device=device)
            loss_resid_refine = mse(residual_hat, (yb - y_l))

            loss_ortho = torch.tensor(0.0, device=device)
            if feat_s is not None and feat_l_proj is not None and feat_s.shape == feat_l_proj.shape:
                loss_ortho = torch.mean(torch.abs(F.cosine_similarity(feat_s, feat_l_proj, dim=-1)))

            loss_diversity = branch_error_correlation_loss(y_s, y_l, yb)

            if epoch <= 10:
                loss = 1.0 * loss_s + 1.0 * loss_l + 0.1 * loss_ortho + 0.05 * loss_diversity
            elif epoch <= 25:
                loss = (loss_pred + 0.5 * loss_res + 0.8 * (loss_s + loss_l) + 8.0 * loss_no_harm
                        + 0.2 * loss_resid_refine + 0.25 * loss_gate_supervision + 0.01 * gate_entropy
                        + 0.1 * loss_ortho + 0.05 * loss_diversity)
            else:
                loss = (loss_pred + 0.5 * loss_res + 40.0 * loss_no_harm + 0.2 * loss_resid_refine
                        + 0.1 * loss_route + 0.1 * loss_conf + 0.5 * loss_gate_supervision
                        + 0.01 * gate_entropy + 0.1 * loss_ortho + 0.1 * loss_diversity)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step(); ema.update(model); total += loss.item() * xb.size(0)

        ema.apply_shadow(model)
        val_metrics = evaluate_components(model, val_loader, device)
        print(f"[Epoch {epoch:02d}] train_obj={math.sqrt(total/len(train_loader.dataset)):.4f} val_fused_RMSE={val_metrics['f_rmse']:.4f} (s={val_metrics['s_rmse']:.4f}, l={val_metrics['l_rmse']:.4f}, c={val_metrics['c_rmse']:.4f})")
        if val_metrics["f_rmse"] < best_val:
            best_val = val_metrics["f_rmse"]; best_state = copy.deepcopy(model.state_dict()); patience_ctr = 0
        else:
            patience_ctr += 1
        ema.restore(model)
        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch}"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict_gfcollm_point(model, loader: DataLoader, device: str):
    model.eval(); preds=[]; trues=[]
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); out = model(xb)
            preds.append(out["y_fused"].detach().cpu().numpy()); trues.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def tune_and_fit_gru(input_dim, train_loader, val_loader, device, max_epochs, patience):
    search = [
        {"hidden_dim": 64,  "num_layers": 2, "lr": 2e-4},
        {"hidden_dim": 128, "num_layers": 2, "lr": 2e-4},
        {"hidden_dim": 128, "num_layers": 3, "lr": 2e-4},
        {"hidden_dim": 128, "num_layers": 2, "lr": 5e-4},
    ]
    best_model=None; best_val=float("inf"); best_cfg=None
    for cfg in search:
        print(f"[GRU tuning] trying {cfg}")
        model = GRURegressor(input_dim=input_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"])
        model, val_rmse = train_simple_with_early_stopping(model, train_loader, val_loader, device,
                                                           lr=cfg["lr"], max_epochs=max_epochs, patience=patience)
        if val_rmse < best_val:
            best_model, best_val, best_cfg = model, val_rmse, cfg
    print(f"[GRU tuning] best_cfg={best_cfg} best_val={best_val:.4f}")
    return best_model, best_val


def tune_and_fit_transformer(input_dim, train_loader, val_loader, device, max_epochs, patience):
    search = [
        {"d_model": 64,  "nhead": 4, "num_layers": 2, "ff_dim": 128, "lr": 2e-4},
        {"d_model": 128, "nhead": 4, "num_layers": 2, "ff_dim": 256, "lr": 2e-4},
        {"d_model": 64,  "nhead": 4, "num_layers": 3, "ff_dim": 128, "lr": 2e-4},
        {"d_model": 64,  "nhead": 4, "num_layers": 2, "ff_dim": 128, "lr": 5e-4},
    ]
    best_model=None; best_val=float("inf"); best_cfg=None
    for cfg in search:
        print(f"[Transformer tuning] trying {cfg}")
        model = TimeSeriesTransformer(input_dim=input_dim, d_model=cfg["d_model"], nhead=cfg["nhead"],
                                      num_layers=cfg["num_layers"], ff_dim=cfg["ff_dim"])
        model, val_rmse = train_simple_with_early_stopping(model, train_loader, val_loader, device,
                                                           lr=cfg["lr"], max_epochs=max_epochs, patience=patience)
        if val_rmse < best_val:
            best_model, best_val, best_cfg = model, val_rmse, cfg
    print(f"[Transformer tuning] best_cfg={best_cfg} best_val={best_val:.4f}")
    return best_model, best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=str, default="FD003")
    parser.add_argument("--data_dir", type=str, default="FD003")
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--methods", nargs="*", default=["GRU", "Transformer", "CoLLM-C", "GF-CoLLM"])
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rul_cap", type=int, default=125)
    parser.add_argument("--out_csv", type=str, default="fd003_multihorizon_splitcp_summary.csv")
    parser.add_argument("--detail_csv", type=str, default=None)
    parser.add_argument("--save_npz_dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.fd.upper() != "FD003":
        raise ValueError("This script is for FD003 only.")

    train_path = os.path.join(args.data_dir, "train_FD003.txt")
    test_path = os.path.join(args.data_dir, "test_FD003.txt")
    rul_path = os.path.join(args.data_dir, "RUL_FD003.txt")

    train_df = add_train_rul(load_cmapss(train_path), max_rul_cap=args.max_rul_cap)
    test_df = add_test_rul(load_cmapss(test_path), load_rul_file(rul_path), max_rul_cap=args.max_rul_cap)
    split = load_split_json(args.split_json)
    feature_cols = get_feature_cols(train_df)

    train_part = train_df[train_df["unit"].isin(split["train_engine_ids"])].copy()
    val_part = train_df[train_df["unit"].isin(split["val_engine_ids"])].copy()

    mean, std = fit_standardizer(train_part, feature_cols)
    train_norm = apply_standardizer(train_part, feature_cols, mean, std)
    val_norm = apply_standardizer(val_part, feature_cols, mean, std)
    test_norm = apply_standardizer(test_df, feature_cols, mean, std)

    X_train, Y_train = build_multihorizon_windows(train_norm, feature_cols, args.window_size, args.horizon)
    X_val, Y_val = build_multihorizon_windows(val_norm, feature_cols, args.window_size, args.horizon)
    X_test, Y_test = build_multihorizon_windows(test_norm, feature_cols, args.window_size, args.horizon)
    input_dim = X_train.shape[-1]

    if args.save_npz_dir:
        os.makedirs(args.save_npz_dir, exist_ok=True)

    p7 = maybe_import_module("paper7_fd003") if "CoLLM-C" in args.methods else None
    p8 = maybe_import_module("paper8_fd003") if "GF-CoLLM" in args.methods else None
    if p8 is not None and hasattr(p8, "WINDOW_SIZE"):
        p8.WINDOW_SIZE = args.window_size

    summary_rows = []
    detail_rows = []

    for method in args.methods:
        print(f"\n===== Method: {method} | Dataset: FD003 | H={args.horizon} =====")
        all_test_preds, all_test_true, all_q = [], [], []

        for h in range(args.horizon):
            y_train_h = Y_train[:, h]; y_val_h = Y_val[:, h]; y_test_h = Y_test[:, h]
            train_loader, val_loader, test_loader = make_scalar_loaders(
                X_train, y_train_h, X_val, y_val_h, X_test, y_test_h, args.batch_size
            )

            if method == "GRU":
                model, best_val = tune_and_fit_gru(input_dim, train_loader, val_loader, device, args.max_epochs, args.patience)
                _, _, val_pred, val_true = evaluate_scalar_model(model, val_loader, device)
                _, _, test_pred, test_true = evaluate_scalar_model(model, test_loader, device)
            elif method == "Transformer":
                model, best_val = tune_and_fit_transformer(input_dim, train_loader, val_loader, device, args.max_epochs, args.patience)
                _, _, val_pred, val_true = evaluate_scalar_model(model, val_loader, device)
                _, _, test_pred, test_true = evaluate_scalar_model(model, test_loader, device)
            elif method == "CoLLM-C":
                model_s, model_l, agent_f, reflect_r, best_val = train_p7_collm_with_early_stopping(
                    p7, input_dim, train_loader, val_loader, device, args.max_epochs, args.patience
                )
                val_pred, val_true = predict_collmc_point(model_s, model_l, agent_f, reflect_r, val_loader, device)
                test_pred, test_true = predict_collmc_point(model_s, model_l, agent_f, reflect_r, test_loader, device)
            elif method == "GF-CoLLM":
                model, best_val = train_p8_gfcollm_fd003_with_early_stopping(
                    p8, input_dim, train_loader, val_loader, device, args.max_epochs, args.patience
                )
                val_pred, val_true = predict_gfcollm_point(model, val_loader, device)
                test_pred, test_true = predict_gfcollm_point(model, test_loader, device)
            else:
                raise ValueError(f"Unsupported method: {method}")

            q_hat = split_cp_from_val(val_true, val_pred, alpha=args.alpha)
            lower = test_pred - q_hat; upper = test_pred + q_hat
            coverage = float(np.mean((test_true >= lower) & (test_true <= upper)))
            width = float(np.mean(upper - lower))
            rmse_h = float(np.sqrt(mean_squared_error(test_true, test_pred)))
            mae_h = float(mean_absolute_error(test_true, test_pred))

            detail_rows.append({
                "Method": method, "Dataset": "FD003", "Horizon": args.horizon, "Step": h + 1,
                "Val_RMSE": best_val, "Test_RMSE": rmse_h, "Test_MAE": mae_h,
                "Coverage": coverage, "Avg_Interval_Width": width, "q_hat": q_hat,
            })

            all_test_preds.append(test_pred.reshape(-1, 1)); all_test_true.append(test_true.reshape(-1, 1))
            all_q.append(np.full_like(test_pred.reshape(-1, 1), q_hat))

            if args.save_npz_dir:
                fname = f"FD003_{method.replace(' ', '_').replace('+', 'plus').replace('-', '_')}_h{h+1}.npz"
                np.savez_compressed(os.path.join(args.save_npz_dir, fname),
                                    val_pred=val_pred, val_true=val_true,
                                    test_pred=test_pred, test_true=test_true,
                                    q_hat=np.array([q_hat], dtype=np.float32),
                                    lower=lower, upper=upper)

        test_pred_mat = np.concatenate(all_test_preds, axis=1)
        test_true_mat = np.concatenate(all_test_true, axis=1)
        q_mat = np.concatenate(all_q, axis=1)

        avg_rmse = float(np.mean([np.sqrt(mean_squared_error(test_true_mat[:, i], test_pred_mat[:, i]))
                                  for i in range(test_true_mat.shape[1])]))
        final_h_rmse = float(np.sqrt(mean_squared_error(test_true_mat[:, -1], test_pred_mat[:, -1])))
        coverage_all = float(np.mean((test_true_mat >= (test_pred_mat - q_mat)) & (test_true_mat <= (test_pred_mat + q_mat))))
        avg_width_all = float(np.mean((test_pred_mat + q_mat) - (test_pred_mat - q_mat)))

        summary_rows.append({
            "Method": method, "Dataset": "FD003", "Horizon": args.horizon,
            "Avg_RMSE": avg_rmse, "Final_Horizon_RMSE": final_h_rmse,
            "Coverage": coverage_all, "Avg_Interval_Width": avg_width_all,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_csv, index=False)
    print("\nSummary:"); print(summary_df)

    if args.detail_csv:
        detail_df = pd.DataFrame(detail_rows)
        detail_df.to_csv(args.detail_csv, index=False)
        print(f"\nSaved detail CSV to: {args.detail_csv}")
    print(f"\nSaved summary CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
