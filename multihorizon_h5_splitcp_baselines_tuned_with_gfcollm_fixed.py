#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H=5 multi-horizon CMAPSS baselines with Split Conformal Prediction (Split CP)
for:
  - GRU + Split CP
  - Transformer + Split CP
  - CoLLM-C + Split CP
  - GF-CoLLM + Split CP

This version includes small validation-based tuning for all four methods.

Design notes
------------
1) Direct multi-horizon strategy: one scalar predictor per horizon h=1..H.
2) Uses train/val engine split JSON for both model selection and Split CP calibration.
3) Uses reconstructed per-cycle test RUL labels, so metrics are computed over all valid
   multi-horizon windows.
4) CoLLM-C is trained with the staged collaborative recipe from paper7_fd*.py.
5) GF-CoLLM is trained using the provided paper8_fd*.py module and selected by fused
   validation RMSE.
"""
from __future__ import annotations

import os
import json
import math
import time
import copy
import random
import argparse
import importlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CMAPSS_COLS = ["unit", "time", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
USELESS_SENSORS = [1, 5, 6, 10, 16, 18, 19]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data utilities
# ============================================================

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
        raise KeyError(
            "Split JSON must contain either train_engine_ids/val_engine_ids or "
            "train_units/val_units."
        )
    return {"train_engine_ids": list(train_ids), "val_engine_ids": list(val_ids)}


def get_feature_cols(train_df: pd.DataFrame, fd: str) -> List[str]:
    fd = fd.upper()
    sensor_cols = [f"s{i}" for i in range(1, 22) if i not in USELESS_SENSORS and f"s{i}" in train_df.columns]
    if fd in {"FD002", "FD004"}:
        op_cols = [c for c in ["os1", "os2", "os3"] if c in train_df.columns]
        return op_cols + sensor_cols
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
        raise ValueError("No windows were created. Check window_size, horizon, and split.")

    return np.asarray(X_list, dtype=np.float32), np.asarray(Y_list, dtype=np.float32)


def make_scalar_loaders(
    X_train: np.ndarray,
    y_train_h: np.ndarray,
    X_val: np.ndarray,
    y_val_h: np.ndarray,
    X_test: np.ndarray,
    y_test_h: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_h.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_h.astype(np.float32)))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test_h.astype(np.float32)))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


# ============================================================
# Models: GRU / Transformer
# ============================================================

class GRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, proj_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim)
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        out, _ = self.gru(x)
        z = out[:, -1, :]
        return self.head(z).squeeze(-1)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, ff_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.encoder(z)
        z = z[:, -1, :]
        return self.head(z).squeeze(-1)


# ============================================================
# Generic training / inference
# ============================================================

def evaluate_scalar_model(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    rmse = math.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    return rmse, mae, preds, trues


def train_simple_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    lr: float = 1e-3,
    max_epochs: int = 50,
    patience: int = 7,
    weight_decay: float = 0.0,
) -> Tuple[nn.Module, float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_state = None
    best_val_rmse = float("inf")
    patience_ctr = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        val_rmse, _, _, _ = evaluate_scalar_model(model, val_loader, device)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_rmse


# ============================================================
# CoLLM-C / GF-CoLLM helpers
# ============================================================

def predict_collmc_point(model_s, model_l, agent_f, reflect_r, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
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


def evaluate_collmc_point(model_s, model_l, agent_f, reflect_r, loader: DataLoader, device: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    preds, trues = predict_collmc_point(model_s, model_l, agent_f, reflect_r, loader, device)
    return math.sqrt(mean_squared_error(trues, preds)), mean_absolute_error(trues, preds), preds, trues


def train_p7_collm_with_early_stopping(p7, input_dim: int, train_loader: DataLoader, val_loader: DataLoader,
                                       device: str, max_epochs: int = 50, patience: int = 7,
                                       lr_stage1: Optional[float] = None, lr_stage2: Optional[float] = None,
                                       lr_stage3: Optional[float] = None):
    model_s = p7.SmallModelS(input_dim=input_dim, embed_dim=32, hidden_dim=64, feature_dim=32).to(device)
    model_l = p7.PatchTransformerL(
        input_dim=input_dim,
        patch_size=4,
        patch_stride=4,
        embed_dim=768,
        num_heads=8,
        ff_dim=1024,
        num_layers=2,
    ).to(device)
    agent_f = p7.FuzzyDecisionAgent(feature_dim=32, num_memberships=64).to(device)
    reflect_r = p7.SelfReflection(input_dim=768).to(device)

    e1 = max(3, max_epochs // 5)
    e2 = max(3, max_epochs // 5)
    e3 = max(5, max_epochs - e1 - e2)
    mse = nn.MSELoss()

    # Stage 1
    best_state, best_val, patience_ctr = None, float("inf"), 0
    opt = torch.optim.Adam(model_s.parameters(), lr=lr_stage1 if lr_stage1 is not None else getattr(p7, "LR_STAGE1", 1e-3))
    for _ in range(1, e1 + 1):
        model_s.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            y_hat, _ = model_s(xb)
            loss = mse(y_hat, yb)
            loss.backward()
            opt.step()
        val_rmse, _, _, _ = evaluate_p7_small(model_s, val_loader, device)
        if val_rmse < best_val:
            best_val = val_rmse; best_state = copy.deepcopy(model_s.state_dict()); patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            break
    if best_state is not None:
        model_s.load_state_dict(best_state)
    for p in model_s.parameters():
        p.requires_grad = False

    # Stage 2
    best_state, best_val, patience_ctr = None, float("inf"), 0
    opt = torch.optim.Adam(model_l.parameters(), lr=lr_stage2 if lr_stage2 is not None else getattr(p7, "LR_STAGE2", 1e-3))
    for _ in range(1, e2 + 1):
        model_l.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            y_hat, _ = model_l(xb)
            loss = mse(y_hat, yb)
            loss.backward()
            opt.step()
        val_rmse, _, _, _ = evaluate_p7_large(model_l, val_loader, device)
        if val_rmse < best_val:
            best_val = val_rmse; best_state = copy.deepcopy(model_l.state_dict()); patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            break
    if best_state is not None:
        model_l.load_state_dict(best_state)

    # Stage 3
    for p in model_s.parameters():
        p.requires_grad = False
    for p in model_l.parameters():
        p.requires_grad = False

    params = list(agent_f.parameters()) + list(reflect_r.parameters())
    opt = torch.optim.Adam(params, lr=lr_stage3 if lr_stage3 is not None else getattr(p7, "LR_STAGE3", 1e-3))
    best_pack, best_val, patience_ctr = None, float("inf"), 0
    for _ in range(1, e3 + 1):
        agent_f.train(); reflect_r.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                y_s, feat_s = model_s(xb)
                y_l, feat_l = model_l(xb)
            opt.zero_grad()
            mu = agent_f(feat_s)
            refl = reflect_r(feat_l)
            y_collab = mu * y_s + (1.0 - mu) * (y_l + refl)
            loss = mse(y_collab, yb)
            loss.backward()
            opt.step()
        val_rmse, _, _, _ = evaluate_collmc_point(model_s, model_l, agent_f, reflect_r, val_loader, device)
        if val_rmse < best_val:
            best_val = val_rmse
            best_pack = {
                "agent_f": copy.deepcopy(agent_f.state_dict()),
                "reflect_r": copy.deepcopy(reflect_r.state_dict()),
            }
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            break
    if best_pack is not None:
        agent_f.load_state_dict(best_pack["agent_f"])
        reflect_r.load_state_dict(best_pack["reflect_r"])
    return model_s, model_l, agent_f, reflect_r, best_val


def evaluate_p7_small(model_s, loader: DataLoader, device: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model_s.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            y_hat, _ = model_s(xb)
            preds.append(y_hat.detach().cpu().numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return math.sqrt(mean_squared_error(trues, preds)), mean_absolute_error(trues, preds), preds, trues


def evaluate_p7_large(model_l, loader: DataLoader, device: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model_l.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            y_hat, _ = model_l(xb)
            preds.append(y_hat.detach().cpu().numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return math.sqrt(mean_squared_error(trues, preds)), mean_absolute_error(trues, preds), preds, trues


def evaluate_gf_fused(model, loader: DataLoader, device: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.append(out["y_fused"].detach().cpu().numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return math.sqrt(mean_squared_error(trues, preds)), mean_absolute_error(trues, preds), preds, trues


# ============================================================
# Split conformal prediction
# ============================================================

def split_cp_from_val(y_val: np.ndarray, y_val_pred: np.ndarray, alpha: float = 0.1) -> float:
    scores = np.abs(y_val - y_val_pred)
    n = len(scores)
    q_level = math.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = min(max(q_level, 0.0), 1.0)
    return float(np.quantile(scores, q_level, method="higher" if hasattr(np, 'quantile') else 'linear'))


def coverage_and_width(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Tuple[float, float]:
    covered = ((y_true >= lower) & (y_true <= upper)).astype(np.float32)
    width = upper - lower
    return float(covered.mean()), float(width.mean())


# ============================================================
# Tuning spaces
# ============================================================

def gru_search_space():
    return [
        {"hidden_dim": 64, "num_layers": 2, "lr": 2e-4, "weight_decay": 0.0},
        {"hidden_dim": 128, "num_layers": 2, "lr": 2e-4, "weight_decay": 0.0},
        {"hidden_dim": 128, "num_layers": 2, "lr": 5e-4, "weight_decay": 0.0},
    ]


def transformer_search_space():
    return [
        {"d_model": 64, "nhead": 4, "num_layers": 2, "ff_dim": 128, "lr": 2e-4, "weight_decay": 0.0},
        {"d_model": 128, "nhead": 4, "num_layers": 2, "ff_dim": 256, "lr": 2e-4, "weight_decay": 0.0},
        {"d_model": 64, "nhead": 4, "num_layers": 3, "ff_dim": 128, "lr": 5e-4, "weight_decay": 0.0},
    ]


def collmc_search_space(p7):
    base1 = getattr(p7, "LR_STAGE1", 1e-3)
    base2 = getattr(p7, "LR_STAGE2", 1e-3)
    base3 = getattr(p7, "LR_STAGE3", 1e-3)
    return [
        {"lr_stage1": base1, "lr_stage2": base2, "lr_stage3": base3},
        {"lr_stage1": base1 * 0.5, "lr_stage2": base2 * 0.5, "lr_stage3": base3 * 0.5},
        {"lr_stage1": base1, "lr_stage2": base2, "lr_stage3": base3 * 0.5},
    ]


def gfcollm_search_space():
    return [
        {"lr": 2e-4, "epochs": 40},
        {"lr": 2e-4, "epochs": 50},
        {"lr": 5e-4, "epochs": 40},
    ]


# ============================================================
# Horizon-wise tuned training wrappers
# ============================================================

def fit_best_gru(train_loader, val_loader, input_dim, device, max_epochs, patience):
    best = None
    for cfg in gru_search_space():
        model = GRURegressor(input_dim=input_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"])
        model, val_rmse = train_simple_with_early_stopping(
            model, train_loader, val_loader, device,
            lr=cfg["lr"], max_epochs=max_epochs, patience=patience, weight_decay=cfg["weight_decay"]
        )
        if best is None or val_rmse < best[0]:
            best = (val_rmse, copy.deepcopy(model.state_dict()), cfg)
    cfg = best[2]
    model = GRURegressor(input_dim=input_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"]).to(device)
    model.load_state_dict(best[1])
    return model, best[0], cfg


def fit_best_transformer(train_loader, val_loader, input_dim, device, max_epochs, patience):
    best = None
    for cfg in transformer_search_space():
        model = TimeSeriesTransformer(
            input_dim=input_dim, d_model=cfg["d_model"], nhead=cfg["nhead"],
            num_layers=cfg["num_layers"], ff_dim=cfg["ff_dim"]
        )
        model, val_rmse = train_simple_with_early_stopping(
            model, train_loader, val_loader, device,
            lr=cfg["lr"], max_epochs=max_epochs, patience=patience, weight_decay=cfg["weight_decay"]
        )
        if best is None or val_rmse < best[0]:
            best = (val_rmse, copy.deepcopy(model.state_dict()), cfg)
    cfg = best[2]
    model = TimeSeriesTransformer(
        input_dim=input_dim, d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], ff_dim=cfg["ff_dim"]
    ).to(device)
    model.load_state_dict(best[1])
    return model, best[0], cfg


def fit_best_collmc(p7, train_loader, val_loader, input_dim, device, max_epochs, patience):
    best = None
    for cfg in collmc_search_space(p7):
        model_s, model_l, agent_f, reflect_r, val_rmse = train_p7_collm_with_early_stopping(
            p7, input_dim, train_loader, val_loader, device,
            max_epochs=max_epochs, patience=patience,
            lr_stage1=cfg["lr_stage1"], lr_stage2=cfg["lr_stage2"], lr_stage3=cfg["lr_stage3"]
        )
        pack = {
            "model_s": copy.deepcopy(model_s.state_dict()),
            "model_l": copy.deepcopy(model_l.state_dict()),
            "agent_f": copy.deepcopy(agent_f.state_dict()),
            "reflect_r": copy.deepcopy(reflect_r.state_dict()),
            "cfg": cfg,
        }
        if best is None or val_rmse < best[0]:
            best = (val_rmse, pack)
    pack = best[1]
    model_s = p7.SmallModelS(input_dim=input_dim, embed_dim=32, hidden_dim=64, feature_dim=32).to(device)
    model_l = p7.PatchTransformerL(input_dim=input_dim, patch_size=4, patch_stride=4, embed_dim=768, num_heads=8, ff_dim=1024, num_layers=2).to(device)
    agent_f = p7.FuzzyDecisionAgent(feature_dim=32, num_memberships=64).to(device)
    reflect_r = p7.SelfReflection(input_dim=768).to(device)
    model_s.load_state_dict(pack["model_s"])
    model_l.load_state_dict(pack["model_l"])
    agent_f.load_state_dict(pack["agent_f"])
    reflect_r.load_state_dict(pack["reflect_r"])
    return model_s, model_l, agent_f, reflect_r, best[0], pack["cfg"]


def fit_best_gfcollm(p8, train_loader, val_loader, input_dim, window_size, device):
    """
    Robust GF-CoLLM tuner.

    Some paper8 modules train in-place and return None from train_gf_collm(...).
    Others may return the trained model. This wrapper supports both behaviors.
    """
    best = None
    for cfg in gfcollm_search_space():
        model = p8.GFCoLLM(input_dim=input_dim, window_size=window_size).to(device)

        train_ret = p8.train_gf_collm(model, train_loader, device, epochs=cfg["epochs"], lr=cfg["lr"])
        if train_ret is not None:
            model = train_ret

        val_result = p8.evaluate_gf_collm(model, val_loader, device)
        val_rmse = float(val_result["fused"]["rmse"])

        if best is None or val_rmse < best[0]:
            best = (val_rmse, copy.deepcopy(model.state_dict()), cfg)

    model = p8.GFCoLLM(input_dim=input_dim, window_size=window_size).to(device)
    model.load_state_dict(best[1])
    return model, best[0], best[2]


def import_dataset_module(base_name: str):
    try:
        return importlib.import_module(base_name)
    except Exception:
        # Fallback for files renamed like "paper8_fd001 (4).py" if user did not rename them.
        import importlib.util
        from pathlib import Path
        candidates = sorted(Path('.').glob(f"{base_name}*.py"))
        if not candidates:
            raise
        path = str(candidates[0])
        spec = importlib.util.spec_from_file_location(base_name, path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod


# ============================================================
# Main evaluation routine
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=str, required=True, choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--methods", nargs="*", default=["GRU", "Transformer", "CoLLM-C", "GF-CoLLM"])
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max_rul_cap", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default="mh_splitcp_summary.csv")
    parser.add_argument("--detail_csv", type=str, default="mh_splitcp_detail.csv")
    parser.add_argument("--save_npz_dir", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fd = args.fd.upper()
    train_path = os.path.join(args.data_dir, f"train_{fd}.txt")
    test_path = os.path.join(args.data_dir, f"test_{fd}.txt")
    rul_path = os.path.join(args.data_dir, f"RUL_{fd}.txt")

    train_df = add_train_rul(load_cmapss(train_path), max_rul_cap=args.max_rul_cap)
    test_df = add_test_rul(load_cmapss(test_path), load_rul_file(rul_path), max_rul_cap=args.max_rul_cap)
    split = load_split_json(args.split_json)
    feature_cols = get_feature_cols(train_df, fd)

    train_part = train_df[train_df["unit"].isin(split["train_engine_ids"])].copy()
    val_part = train_df[train_df["unit"].isin(split["val_engine_ids"])].copy()

    mean, std = fit_standardizer(train_part, feature_cols)
    train_part = apply_standardizer(train_part, feature_cols, mean, std)
    val_part = apply_standardizer(val_part, feature_cols, mean, std)
    test_df = apply_standardizer(test_df, feature_cols, mean, std)

    X_train, Y_train = build_multihorizon_windows(train_part, feature_cols, args.window_size, args.horizon)
    X_val, Y_val = build_multihorizon_windows(val_part, feature_cols, args.window_size, args.horizon)
    X_test, Y_test = build_multihorizon_windows(test_df, feature_cols, args.window_size, args.horizon)
    input_dim = X_train.shape[-1]

    p7 = None
    p8 = None
    if "CoLLM-C" in args.methods:
        p7 = import_dataset_module(f"paper7_{fd.lower()}")
    if "GF-CoLLM" in args.methods:
        p8 = import_dataset_module(f"paper8_{fd.lower()}")

    summary_rows = []
    detail_rows = []
    if args.save_npz_dir:
        os.makedirs(args.save_npz_dir, exist_ok=True)

    for method in args.methods:
        print(f"\n===== Method: {method} | Dataset: {fd} | H={args.horizon} =====")
        per_h_rmse, per_h_width, per_h_cov = [], [], []
        final_h_rmse = None
        chosen_cfgs = []

        for h_idx in range(args.horizon):
            h = h_idx + 1
            y_train_h = Y_train[:, h_idx]
            y_val_h = Y_val[:, h_idx]
            y_test_h = Y_test[:, h_idx]

            train_loader, val_loader, test_loader = make_scalar_loaders(
                X_train, y_train_h, X_val, y_val_h, X_test, y_test_h, args.batch_size
            )

            if method == "GRU":
                model, best_val_rmse, cfg = fit_best_gru(train_loader, val_loader, input_dim, device, args.max_epochs, args.patience)
                test_rmse, test_mae, test_pred, y_true = evaluate_scalar_model(model, test_loader, device)
                _, _, val_pred, _ = evaluate_scalar_model(model, val_loader, device)
            elif method == "Transformer":
                model, best_val_rmse, cfg = fit_best_transformer(train_loader, val_loader, input_dim, device, args.max_epochs, args.patience)
                test_rmse, test_mae, test_pred, y_true = evaluate_scalar_model(model, test_loader, device)
                _, _, val_pred, _ = evaluate_scalar_model(model, val_loader, device)
            elif method == "CoLLM-C":
                model_s, model_l, agent_f, reflect_r, best_val_rmse, cfg = fit_best_collmc(
                    p7, train_loader, val_loader, input_dim, device, args.max_epochs, args.patience
                )
                test_rmse, test_mae, test_pred, y_true = evaluate_collmc_point(model_s, model_l, agent_f, reflect_r, test_loader, device)
                _, _, val_pred, _ = evaluate_collmc_point(model_s, model_l, agent_f, reflect_r, val_loader, device)
            elif method == "GF-CoLLM":
                model, best_val_rmse, cfg = fit_best_gfcollm(p8, train_loader, val_loader, input_dim, args.window_size, device)
                test_rmse, test_mae, test_pred, y_true = evaluate_gf_fused(model, test_loader, device)
                _, _, val_pred, _ = evaluate_gf_fused(model, val_loader, device)
            else:
                raise ValueError(f"Unsupported method: {method}")

            qhat = split_cp_from_val(y_val_h, val_pred, alpha=args.alpha)
            lower = test_pred - qhat
            upper = test_pred + qhat
            coverage, avg_width = coverage_and_width(y_true, lower, upper)

            per_h_rmse.append(test_rmse)
            per_h_cov.append(coverage)
            per_h_width.append(avg_width)
            chosen_cfgs.append(str(cfg))
            if h == args.horizon:
                final_h_rmse = test_rmse

            detail_rows.append({
                "Method": method,
                "Dataset": fd,
                "Horizon": h,
                "RMSE": test_rmse,
                "MAE": test_mae,
                "Coverage": coverage,
                "Avg_Interval_Width": avg_width,
                "Val_RMSE_Best": best_val_rmse,
                "qhat": qhat,
                "Config": str(cfg),
            })

            if args.save_npz_dir:
                np.savez_compressed(
                    os.path.join(args.save_npz_dir, f"{fd}_{method.replace('-', '').replace(' ', '_')}_h{h}.npz"),
                    y_true=y_true, y_pred=test_pred, lower=lower, upper=upper,
                )

        summary_rows.append({
            "Method": method,
            "Dataset": fd,
            "Horizon": args.horizon,
            "Avg_RMSE": float(np.mean(per_h_rmse)),
            "Final_Horizon_RMSE": float(final_h_rmse),
            "Coverage": float(np.mean(per_h_cov)),
            "Avg_Interval_Width": float(np.mean(per_h_width)),
            "Chosen_Configs": " | ".join(chosen_cfgs),
        })

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    summary_df.to_csv(args.out_csv, index=False)
    detail_df.to_csv(args.detail_csv, index=False)

    print("\nSummary:")
    print(summary_df[["Method", "Dataset", "Horizon", "Avg_RMSE", "Final_Horizon_RMSE", "Coverage", "Avg_Interval_Width"]])
    print(f"\nSaved summary to: {args.out_csv}")
    print(f"Saved details to : {args.detail_csv}")


if __name__ == "__main__":
    main()
