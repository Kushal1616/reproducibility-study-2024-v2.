#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tune GF-CoLLM for CMAPSS FD004.

Design choice:
- FD004 is treated as a multi-condition, multi-fault dataset.
- Start from the FD002-style condition-aware recipe with stronger guardrails:
  * locked low learning-rate region
  * paper8_fd004.py provides condition-aware FIG + safe-switch gating
  * EMA validation
  * gate supervision + gate entropy
  * stronger no-harm loss and longer warm-up
- Run a small, sensible sweep around the stability-first region.
"""

import os
import math
import json
import copy
import time
import random
import argparse
import importlib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cmapss(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] == 27:
        df = df.iloc[:, :-1]
    cols = ["unit", "time", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    df.columns = cols
    return df


def add_rul_labels(train_df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = train_df.groupby("unit")["time"].max().reset_index()
    max_cycle.columns = ["unit", "max_time"]
    df = train_df.merge(max_cycle, on="unit", how="left")
    df["RUL"] = df["max_time"] - df["time"]
    df.drop(columns=["max_time"], inplace=True)
    return df


def load_rul_file(path: str) -> np.ndarray:
    rul_df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    return rul_df.iloc[:, 0].values.astype(np.float32)


def get_feature_cols(train_df, fd=None):
    # FD004: include operating conditions first so the condition-aware FIG layer can use them explicitly
    useless_sensors = [1, 5, 6, 10, 16, 18, 19]
    sensor_cols = [f"s{i}" for i in range(1, 22) if f"s{i}" in train_df.columns and i not in useless_sensors]
    if not sensor_cols:
        sensor_cols = [f"sensor_{i}" for i in range(1, 22) if f"sensor_{i}" in train_df.columns and i not in useless_sensors]
    if not sensor_cols:
        raise ValueError(f"Could not infer feature columns from columns: {train_df.columns.tolist()}")
    op_cols = [c for c in ["os1", "os2", "os3"] if c in train_df.columns]
    return op_cols + sensor_cols


def fit_standardizer(train_df, feature_cols):
    mean = train_df[feature_cols].mean(axis=0)
    std = train_df[feature_cols].std(axis=0).replace(0, 1.0)
    return mean, std


def apply_standardizer(df, feature_cols, mean, std):
    out = df.copy()
    out[feature_cols] = (out[feature_cols] - mean) / std
    return out


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

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


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
    X = np.array(X_list, dtype=np.float32)
    unit_ids = np.array(unit_ids, dtype=np.int32)
    return X, unit_ids


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


def evaluate_split(model, loader, device):
    model.eval()
    y_true, ys, yl, yc, yf = [], [], [], [], []
    gate_weights_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            ys.append(out["y_s"].cpu().numpy())
            yl.append(out["y_l"].cpu().numpy())
            yc.append(out["y_c"].cpu().numpy())
            yf.append(out["y_fused"].cpu().numpy())
            y_true.append(yb.numpy())
            if "gate_weights" in out:
                gate_weights_all.append(out["gate_weights"].detach().cpu())

    y_true = np.concatenate(y_true)
    ys = np.concatenate(ys)
    yl = np.concatenate(yl)
    yc = np.concatenate(yc)
    yf = np.concatenate(yf)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    metrics = {
        "s": rmse(ys, y_true),
        "l": rmse(yl, y_true),
        "c": rmse(yc, y_true),
        "fused": rmse(yf, y_true),
    }

    if gate_weights_all:
        avg_weights = torch.cat(gate_weights_all, dim=0).mean(dim=0).cpu().numpy()
        print(f"\n>>> DIAGNOSTIC: Test Avg Gate Weights [L, C] = {avg_weights.tolist()}")

    return metrics


def train_gfcollm_with_schedule(p8, input_dim, train_loader, val_loader, device,
                                max_epochs=50, patience=7,
                                lambda_res=0.2, lambda_route=0.1, lambda_conf=0.1, lr=2e-4,
                                lambda_no_harm_strong=40.0):
    model = p8.GFCoLLM(input_dim=input_dim, window_size=p8.WINDOW_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    ema = EMA(model, decay=0.999)

    best_state = None
    best_val = float("inf")
    patience_ctr = 0

    print("FD004 tuner v12: dual-expert fusion (Large vs Corrected) + gate-weight logging")

    for epoch in range(1, max_epochs + 1):
        # Freeze fusion/gating early so branches stabilize first
        fusion_trainable = epoch > 20
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

                # 1) Best across ALL branches (for route supervision and no-harm reference)
                branch_errs_all = torch.stack([per_sample_err_s, per_sample_err_l, per_sample_err_c], dim=1)
                best_idx_all = torch.argmin(branch_errs_all, dim=1)   # 0, 1, or 2
                best_branch_err = torch.min(branch_errs_all, dim=1)[0]

                # 2) Best across GATE-ONLY branches (Large vs Corrected)
                branch_errs_gate = torch.stack([per_sample_err_l, per_sample_err_c], dim=1)
                best_idx_gate = torch.argmin(branch_errs_gate, dim=1)  # 0 or 1

            per_sample_err_fused = (y_fused - yb) ** 2
            loss_no_harm = torch.mean(torch.pow(torch.relu(per_sample_err_fused - best_branch_err), 2.0))

            loss_route = torch.tensor(0.0, device=device)
            if route_logits is not None:
                loss_route = ce(route_logits, best_idx_all)

            loss_conf = torch.tensor(0.0, device=device)
            if conf_score is not None:
                loss_conf = torch.mean((conf_score - torch.abs(y_fused - yb)) ** 2)

            loss_gate_supervision = torch.tensor(0.0, device=device)
            if gate_logits is not None:
                loss_gate_supervision = ce(gate_logits, best_idx_gate)
            elif gate_weights is not None:
                loss_gate_supervision = ce(torch.log(gate_weights + 1e-8), best_idx_gate)

            gate_entropy = torch.tensor(0.0, device=device)
            if gate_weights is not None:
                gate_entropy = -torch.mean(torch.sum(gate_weights * torch.log(gate_weights + 1e-6), dim=-1))

            loss_resid_refine = torch.mean(residual_hat ** 2)

            loss_ortho = torch.tensor(0.0, device=device)
            if feat_s is not None and feat_l_proj is not None and feat_s.shape == feat_l_proj.shape:
                loss_ortho = torch.mean(torch.abs(torch.sum(feat_s * feat_l_proj, dim=-1)))

            loss_diversity = torch.mean(torch.exp(-torch.abs(y_s - y_l)))

            # FD004 V8 schedule: earlier iron guardrail for multi-condition safety.
            if epoch <= 20:
                loss = (
                    1.0 * loss_s
                    + 1.0 * loss_l
                    + 0.1 * loss_ortho
                    + 0.05 * loss_diversity
                )
            elif epoch <= 30:
                loss = (
                    loss_pred
                    + 0.5 * loss_res
                    + 0.8 * (loss_s + loss_l)
                    + 2.0 * loss_no_harm
                    + lambda_res * loss_resid_refine
                    + 0.1 * loss_ortho
                    + 0.05 * loss_diversity
                )
            else:
                lambda_no_harm = lambda_no_harm_strong
                loss = (
                    loss_pred
                    + 0.5 * loss_res
                    + 0.5 * (loss_s + loss_l)
                    + lambda_no_harm * loss_no_harm
                    + lambda_res * loss_resid_refine
                    + lambda_route * loss_route
                    + lambda_conf * loss_conf
                    + 0.5 * loss_gate_supervision
                    + 0.05 * gate_entropy
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ema.update(model)
            total += loss.item() * xb.size(0)

        ema.apply_shadow(model)
        val_metrics = evaluate_split(model, val_loader, device)
        ema.restore(model)

        train_obj = math.sqrt(total / len(train_loader.dataset))
        print(
            f"[Epoch {epoch:02d}] train_obj={train_obj:.4f} "
            f"val_fused_RMSE={val_metrics['fused']:.4f} "
            f"(s={val_metrics['s']:.4f}, l={val_metrics['l']:.4f}, c={val_metrics['c']:.4f})"
        )

        if val_metrics["fused"] < best_val:
            best_val = val_metrics["fused"]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=str, default="FD004")
    parser.add_argument("--data_dir", type=str, default="FD004")
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--out_csv", type=str, default="fd004_gfcollm_tuning.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p8 = importlib.import_module("paper8_fd004")

    train_file = os.path.join(args.data_dir, f"train_{args.fd}.txt")
    test_file = os.path.join(args.data_dir, f"test_{args.fd}.txt")
    rul_file = os.path.join(args.data_dir, f"RUL_{args.fd}.txt")

    train_df = load_cmapss(train_file)
    test_df = load_cmapss(test_file)
    train_df = add_rul_labels(train_df)
    test_rul_true = load_rul_file(rul_file)

    with open(args.split_json, "r") as f:
        split = json.load(f)

    # Support both naming styles from different split generators
    train_units = split.get("train_units", split.get("train_engine_ids"))
    val_units = split.get("val_units", split.get("val_engine_ids"))
    if train_units is None or val_units is None:
        raise KeyError(f"Split JSON must contain either ('train_units','val_units') or ('train_engine_ids','val_engine_ids'). Found keys: {list(split.keys())}")

    feature_cols = get_feature_cols(train_df, fd=args.fd)
    mean, std = fit_standardizer(train_df[train_df["unit"].isin(train_units)], feature_cols)
    train_df_norm = apply_standardizer(train_df, feature_cols, mean, std)
    test_df_norm = apply_standardizer(test_df, feature_cols, mean, std)

    X_train, y_train = create_train_windows(train_df_norm, feature_cols, p8.WINDOW_SIZE, unit_ids=train_units)
    X_val, y_val = create_train_windows(train_df_norm, feature_cols, p8.WINDOW_SIZE, unit_ids=val_units)
    X_test, test_units = create_test_last_windows(test_df_norm, feature_cols, p8.WINDOW_SIZE)
    n_common = min(len(test_rul_true), len(test_units))
    X_test, y_test = X_test[:n_common], test_rul_true[:n_common]

    print(f"Dataset: {args.fd}")
    print(f"Input dim: {len(feature_cols)}")
    print(f"Window size: {p8.WINDOW_SIZE}")
    print(f"Train windows: {len(X_train)} | Val windows: {len(X_val)} | Test engines: {len(X_test)}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=p8.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=p8.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                             batch_size=p8.BATCH_SIZE, shuffle=False)

    # Small, sensible sweep around the FD004 stability-first region
    configs = [
        {"lr": 2e-4, "lambda_res": 0.3, "lambda_route": 0.1, "lambda_conf": 0.1, "lambda_no_harm_strong": 60.0},  # FD004 V12 target
    ]

    rows = []
    best_cfg = None
    best_val = float("inf")

    for i, cfg in enumerate(configs, 1):
        print("\n" + "=" * 90)
        print(f"Config {i}/{len(configs)}: {cfg}")

        # write selected values into imported module globals for consistency
        p8.LR = cfg["lr"]
        p8.LAMBDA_RESIDUAL = cfg["lambda_res"]
        p8.LAMBDA_ROUTE = cfg["lambda_route"]
        p8.LAMBDA_CONF = cfg["lambda_conf"]

        model, val_best = train_gfcollm_with_schedule(
            p8=p8,
            input_dim=len(feature_cols),
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lambda_res=cfg["lambda_res"],
            lambda_route=cfg["lambda_route"],
            lambda_conf=cfg["lambda_conf"],
            lr=cfg["lr"],
            lambda_no_harm_strong=cfg["lambda_no_harm_strong"],
        )

        test_metrics = evaluate_split(model, test_loader, device)

        row = {
            "FD": args.fd,
            "Seed": args.seed,
            "lr": cfg["lr"],
            "lambda_res": cfg["lambda_res"],
            "lambda_route": cfg["lambda_route"],
            "lambda_conf": cfg["lambda_conf"],
            "lambda_no_harm_strong": cfg["lambda_no_harm_strong"],
            "Val_fused_RMSE": val_best,
            "Test_s_RMSE": test_metrics["s"],
            "Test_l_RMSE": test_metrics["l"],
            "Test_c_RMSE": test_metrics["c"],
            "Test_fused_RMSE": test_metrics["fused"],
        }
        rows.append(row)

        if val_best < best_val:
            best_val = val_best
            best_cfg = cfg

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)

    print("\nBest config by validation fused RMSE:")
    print(best_cfg)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
