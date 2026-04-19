# unified_earlystop_comparison_runner.py

import os
import json
import math
import time
import copy
import random
import importlib
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Generic non-LLM baselines
# ============================================================

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, 32)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = self.proj(x)
        out, _ = self.lstm(x)
        z = out[:, -1, :]
        return self.head(z).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, 32)
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = self.proj(x)
        out, _ = self.gru(x)
        z = out[:, -1, :]
        return self.head(z).squeeze(-1)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        res = res[:, :, :out.size(2)]
        return out + res


class TCNRegressor(nn.Module):
    def __init__(self, input_dim, channels=64):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.block1 = TemporalBlock(channels, channels, dilation=1)
        self.block2 = TemporalBlock(channels, channels, dilation=2)
        self.block3 = TemporalBlock(channels, channels, dilation=4)
        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.input_proj(x)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        z = z[:, :, -1]
        return self.head(z).squeeze(-1)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, ff_dim=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        z = self.proj(x)
        z = self.encoder(z)
        z = z[:, -1, :]
        return self.head(z).squeeze(-1)


# ============================================================
# Split / module / data helpers
# ============================================================

def load_split_json(split_path):
    with open(split_path, "r") as f:
        return json.load(f)


def infer_fd_modules(fd):
    """
    Explicit module routing so the runner picks the intended dataset-specific files.
    """
    fd = fd.upper()

    module_map = {
        "FD001": ("paper7_fd001", "paper8_fd001"),
        "FD002": ("paper7_fd002", "paper8_fd002"),
        "FD003": ("paper7_fd003", "paper8_fd003"),
        "FD004": ("paper7_fd004", "paper8_fd004"),
    }

    if fd not in module_map:
        raise ValueError(f"Unsupported FD subset: {fd}")

    p7_name, p8_name = module_map[fd]
    print(f"[Module Loader] Using {p7_name}.py and {p8_name}.py")

    p7 = importlib.import_module(p7_name)
    p8 = importlib.import_module(p8_name)
    return p7, p8


def get_engine_id_col(df):
    candidates = ["unit_id", "unit", "engine_id", "id"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find engine id column. Available columns: {df.columns.tolist()}")


def load_raw_data_with_p8(p8, data_dir, fd):
    fd = fd.upper()
    train_file = os.path.join(data_dir, f"train_{fd}.txt")
    test_file = os.path.join(data_dir, f"test_{fd}.txt")
    rul_file = os.path.join(data_dir, f"RUL_{fd}.txt")

    load_fn = None
    for name in [f"load_{fd.lower()}", "load_fd001", "load_cmapss"]:
        if hasattr(p8, name):
            load_fn = getattr(p8, name)
            break
    if load_fn is None:
        raise AttributeError("Could not find a dataset loader in p8 module.")

    train_df = load_fn(train_file)
    test_df = load_fn(test_file)

    print("Loaded train columns:", train_df.columns.tolist())
    print("Loaded test columns :", test_df.columns.tolist())

    if hasattr(p8, "add_rul_labels"):
        train_df = p8.add_rul_labels(train_df)
    else:
        raise AttributeError("p8 module must expose add_rul_labels(train_df).")

    if hasattr(p8, "load_rul_file"):
        test_rul_true = p8.load_rul_file(rul_file)
    else:
        raise AttributeError("p8 module must expose load_rul_file(rul_file).")

    return train_df, test_df, test_rul_true


def get_feature_cols(train_df, fd=None):
    """
    FD-aware feature selection.

    FD002 / FD004:
        include operating conditions first so the GF-CoLLM FIG layer can use them explicitly.
    FD001 / FD003:
        keep the original sensor-only behavior.
    """
    useless_sensors = [1, 5, 6, 10, 16, 18, 19]

    sensor_cols = [f"s{i}" for i in range(1, 22) if f"s{i}" in train_df.columns and i not in useless_sensors]
    if len(sensor_cols) == 0:
        sensor_cols = [f"sensor_{i}" for i in range(1, 22) if f"sensor_{i}" in train_df.columns and i not in useless_sensors]
    if len(sensor_cols) == 0:
        possible = [c for c in train_df.columns if c.startswith("s") or c.startswith("sensor_")]
        if possible:
            sensor_cols = possible

    if len(sensor_cols) == 0:
        raise ValueError(f"Could not infer feature columns from columns: {train_df.columns.tolist()}")

    fd = (fd or "").upper()
    if fd in {"FD002", "FD004"}:
        op_cols = [c for c in ["os1", "os2", "os3"] if c in train_df.columns]
        return op_cols + sensor_cols

    return sensor_cols


def split_train_val_by_engine(train_df, split_json):
    train_ids = set(split_json["train_engine_ids"])
    val_ids = set(split_json["val_engine_ids"])

    engine_col = get_engine_id_col(train_df)

    train_part = train_df[train_df[engine_col].isin(train_ids)].copy()
    val_part = train_df[train_df[engine_col].isin(val_ids)].copy()
    return train_part, val_part


def normalize_with_train_only(p8, train_part, val_part, test_df, feature_cols):
    if hasattr(p8, "normalize_features"):
        train_norm, val_norm, scaler = p8.normalize_features(train_part, val_part, feature_cols)
        _, test_norm, _ = p8.normalize_features(train_part, test_df, feature_cols)
        return train_norm, val_norm, test_norm, scaler
    raise AttributeError("p8 module must expose normalize_features(train_df, test_df, feature_cols).")


def create_windows_trainvaltest(p8, train_norm, val_norm, test_norm, test_rul_true, feature_cols, window_size):
    if not hasattr(p8, "create_train_windows"):
        raise AttributeError("p8 module must expose create_train_windows(df, feature_cols, window_size).")
    if not hasattr(p8, "create_test_last_windows"):
        raise AttributeError("p8 module must expose create_test_last_windows(df, feature_cols, window_size).")

    X_train, y_train = p8.create_train_windows(train_norm, feature_cols, window_size)
    X_val, y_val = p8.create_train_windows(val_norm, feature_cols, window_size)
    X_test, test_units = p8.create_test_last_windows(test_norm, feature_cols, window_size)

    n_common = min(len(test_rul_true), len(test_units))
    X_test = X_test[:n_common]
    y_test = test_rul_true[:n_common]

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_scalar_model(model, loader, device):
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
    return rmse, mae


def evaluate_p7_small(model_s, loader, device):
    model_s.eval()
    preds, trues = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            y_hat, _ = model_s(xb)
            preds.append(y_hat.cpu().numpy())
            trues.append(yb.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return math.sqrt(mean_squared_error(trues, preds)), mean_absolute_error(trues, preds)


def evaluate_p7_large(model_l, loader, device):
    model_l.eval()
    preds, trues = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            y_hat, _ = model_l(xb)
            preds.append(y_hat.cpu().numpy())
            trues.append(yb.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return math.sqrt(mean_squared_error(trues, preds)), mean_absolute_error(trues, preds)




def parse_gfcollm_eval_result(eval_result):
    """
    Accept either:
    1) tuple/list like (rmse, mae, ...)
    2) dict from p8.evaluate_gf_collm(...) with nested branch metrics

    Returns:
        (fused_rmse, fused_mae)
    """
    if isinstance(eval_result, dict):
        if "fused" in eval_result and isinstance(eval_result["fused"], dict):
            fused = eval_result["fused"]
            return float(fused["rmse"]), float(fused["mae"])
        raise ValueError(f"Unexpected GF-CoLLM eval dict format: {list(eval_result.keys())}")
    if isinstance(eval_result, (tuple, list)):
        if len(eval_result) >= 2:
            return float(eval_result[0]), float(eval_result[1])
        raise ValueError("GF-CoLLM eval tuple/list must have at least 2 items.")
    raise TypeError(f"Unsupported GF-CoLLM eval return type: {type(eval_result)}")
# ============================================================
# Early stopping for simple models
# ============================================================

def train_simple_with_early_stopping(model, train_loader, val_loader, device, lr=1e-3, max_epochs=50, patience=7):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_state = None
    best_val_rmse = float("inf")
    patience_ctr = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_rmse = math.sqrt(total_loss / len(train_loader.dataset))
        val_rmse, val_mae = evaluate_scalar_model(model, val_loader, device)
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


# ============================================================
# Early stopping wrappers for paper7/paper8 families
# ============================================================

def train_p7_collm_with_early_stopping(p7, input_dim, train_loader, val_loader, device, max_epochs=50, patience=7):
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

    best_s = None
    best_val = float("inf")
    patience_ctr = 0
    opt = torch.optim.Adam(model_s.parameters(), lr=getattr(p7, "LR_STAGE1", 1e-3))
    mse = nn.MSELoss()

    for epoch in range(1, e1 + 1):
        model_s.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            y_hat, _ = model_s(xb)
            loss = mse(y_hat, yb)
            loss.backward()
            opt.step()

        val_rmse, _ = evaluate_p7_small(model_s, val_loader, device)
        print(f"[CoLLM-C Stage1 Epoch {epoch:02d}] val_RMSE={val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_s = copy.deepcopy(model_s.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    if best_s is not None:
        model_s.load_state_dict(best_s)

    for p in model_s.parameters():
        p.requires_grad = False

    best_l = None
    best_val = float("inf")
    patience_ctr = 0
    opt = torch.optim.Adam(model_l.parameters(), lr=getattr(p7, "LR_STAGE2", 1e-3))

    for epoch in range(1, e2 + 1):
        model_l.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            y_hat, _ = model_l(xb)
            loss = mse(y_hat, yb)
            loss.backward()
            opt.step()

        val_rmse, _ = evaluate_p7_large(model_l, val_loader, device)
        print(f"[CoLLM-C Stage2 Epoch {epoch:02d}] val_RMSE={val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_l = copy.deepcopy(model_l.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    if best_l is not None:
        model_l.load_state_dict(best_l)

    for p in model_l.parameters():
        p.requires_grad = False

    params = list(agent_f.parameters()) + list(reflect_r.parameters())
    opt = torch.optim.Adam(params, lr=getattr(p7, "LR_STAGE3", 1e-3))

    best_pack = None
    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, e3 + 1):
        model_s.eval()
        model_l.eval()
        agent_f.train()
        reflect_r.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                y_s, feat_s = model_s(xb)
                y_l, feat_l = model_l(xb)

            opt.zero_grad()

            mu = agent_f(feat_s)
            refl = reflect_r(feat_l.detach())
            y_collab = mu * y_s + (1.0 - mu) * (y_l + refl)

            loss = mse(y_collab, yb)
            loss.backward()
            opt.step()

        val_rmse, val_mae, *_ = p7.evaluate_collm(
            model_s, model_l, agent_f, reflect_r, val_loader, device,
            tau1=getattr(p7, "TAU1", 0.10),
            tau2=getattr(p7, "TAU2", 0.20),
        )
        print(f"[CoLLM-C Stage3 Epoch {epoch:02d}] val_RMSE={val_rmse:.4f} val_MAE={val_mae:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_pack = {
                "s": copy.deepcopy(model_s.state_dict()),
                "l": copy.deepcopy(model_l.state_dict()),
                "f": copy.deepcopy(agent_f.state_dict()),
                "r": copy.deepcopy(reflect_r.state_dict()),
            }
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    if best_pack is not None:
        model_s.load_state_dict(best_pack["s"])
        model_l.load_state_dict(best_pack["l"])
        agent_f.load_state_dict(best_pack["f"])
        reflect_r.load_state_dict(best_pack["r"])

    return model_s, model_l, agent_f, reflect_r, best_val


def train_p8_gfcollm_with_early_stopping(p8, input_dim, train_loader, val_loader, device, max_epochs=50, patience=7, fd=None):
    """
    FD-aware early stopping for GF-CoLLM.

    FD003 uses the stronger V3 "safe-switch" recipe that matched the successful
    tuning runs:
    - earlier heavy guardrail
    - powered no-harm loss
    - stronger gate entropy
    - stronger no-harm weight
    """
    model = p8.GFCoLLM(input_dim=input_dim, window_size=p8.WINDOW_SIZE).to(device)

    lr = getattr(p8, "LR", 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    fd = (fd or "").upper()
    lambda_res = getattr(p8, "LAMBDA_RESIDUAL", 0.2)
    lambda_route = getattr(p8, "LAMBDA_ROUTE", 0.1)
    lambda_conf = getattr(p8, "LAMBDA_CONF", 0.1)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    ema = None
    if hasattr(p8, "EMA"):
        try:
            ema = p8.EMA(model, decay=0.999)
        except TypeError:
            try:
                ema = p8.EMA(model)
            except Exception:
                ema = None
        except Exception:
            ema = None

    def eval_with_optional_ema(loader):
        if ema is None:
            return p8.evaluate_gf_collm(model, loader, device)
        ema.apply_shadow(model)
        out = p8.evaluate_gf_collm(model, loader, device)
        ema.restore(model)
        return out

    def set_trainable_by_name(epoch_now):
        if fd in {"FD002", "FD004"}:
            fusion_trainable = epoch_now > 25
        else:
            fusion_trainable = epoch_now > 10

        for name, param in model.named_parameters():
            if any(k in name for k in ["fusion", "trust_net", "base_anchor_net", "final_gate_net", "route_head", "anchor_gate_net"]):
                param.requires_grad = fusion_trainable

    best_state = None
    best_val_rmse = float("inf")
    patience_ctr = 0

    for epoch in range(1, max_epochs + 1):
        set_trainable_by_name(epoch)
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

            route_logits = out.get("route_logits", None)
            conf_score = out.get("conf_score", None)
            gate_weights = out.get("gate_weights", None)
            gate_logits = out.get("gate_logits", None)
            residual_hat = out.get("residual_hat", torch.zeros_like(yb))

            with torch.no_grad():
                per_sample_err_s = (y_s - yb) ** 2
                per_sample_err_l = (y_l - yb) ** 2
                per_sample_err_c = (y_c - yb) ** 2
                branch_errs = torch.stack([per_sample_err_s, per_sample_err_l, per_sample_err_c], dim=1)
                best_idx = torch.argmin(branch_errs, dim=1)
                best_branch_err = torch.min(branch_errs, dim=1)[0]

            per_sample_err_fused = (y_fused - yb) ** 2

            # FD003 uses the stronger powered hinge no-harm penalty from the working V3 tuner.
            if fd == "FD003":
                loss_no_harm = torch.mean(torch.pow(torch.relu(per_sample_err_fused - best_branch_err), 1.5))
            else:
                loss_no_harm = torch.mean(torch.relu(per_sample_err_fused - best_branch_err))

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
            if "feat_s" in out and "feat_l" in out:
                fs = out["feat_s"]
                fl = out["feat_l"]
                if (
                    fs.dim() == 2
                    and fl.dim() == 2
                    and fs.size(0) == fl.size(0)
                    and fs.size(1) == fl.size(1)
                ):
                    loss_ortho = torch.mean((fs * fl).sum(dim=-1).abs())

            loss_diversity = torch.tensor(0.0, device=device)
            if "y_s" in out and "y_l" in out:
                loss_diversity = torch.mean(torch.exp(-torch.abs(y_s - y_l)))

            # FD-aware schedule
            if fd in {"FD002", "FD004"}:
                if epoch <= 25:
                    loss = (
                        1.0 * loss_s
                        + 1.0 * loss_l
                        + 0.1 * loss_ortho
                        + 0.05 * loss_diversity
                    )
                elif epoch <= 35:
                    loss = (
                        loss_pred
                        + 0.5 * loss_res
                        + 0.8 * (loss_s + loss_l)
                        + 2.0 * loss_no_harm
                        + lambda_res * loss_resid_refine
                        + 0.25 * loss_gate_supervision
                        + 0.01 * gate_entropy
                        + 0.1 * loss_ortho
                        + 0.05 * loss_diversity
                    )
                else:
                    lambda_no_harm = 20.0
                    loss = (
                        loss_pred
                        + 0.5 * loss_res
                        + lambda_no_harm * loss_no_harm
                        + lambda_res * loss_resid_refine
                        + lambda_route * loss_route
                        + lambda_conf * loss_conf
                        + 0.5 * loss_gate_supervision
                        + 0.01 * gate_entropy
                        + 0.1 * loss_ortho
                        + 0.1 * loss_diversity
                    )
            elif fd == "FD003":
                # Match the successful FD003 V3 tuning recipe
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
                    lambda_no_harm = 50.0
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
            else:
                if epoch <= 10:
                    loss = (
                        1.0 * loss_s
                        + 1.0 * loss_l
                        + 0.1 * loss_ortho
                        + 0.05 * loss_diversity
                    )
                elif epoch <= 20:
                    loss = (
                        loss_pred
                        + 0.5 * loss_res
                        + 0.8 * (loss_s + loss_l)
                        + 1.0 * loss_no_harm
                        + lambda_res * loss_resid_refine
                        + 0.25 * loss_gate_supervision
                        + 0.01 * gate_entropy
                        + 0.1 * loss_ortho
                        + 0.05 * loss_diversity
                    )
                else:
                    lambda_no_harm = 15.0
                    loss = (
                        loss_pred
                        + 0.5 * loss_res
                        + lambda_no_harm * loss_no_harm
                        + lambda_res * loss_resid_refine
                        + lambda_route * loss_route
                        + lambda_conf * loss_conf
                        + 0.5 * loss_gate_supervision
                        + 0.01 * gate_entropy
                        + 0.1 * loss_ortho
                        + 0.1 * loss_diversity
                    )

            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)

            total += loss.item() * xb.size(0)

        val_eval = eval_with_optional_ema(val_loader)
        val_rmse, val_mae = parse_gfcollm_eval_result(val_eval)
        train_obj = math.sqrt(total / len(train_loader.dataset))
        print(f"[GF-CoLLM Epoch {epoch:02d}] train_obj={train_obj:.4f} val_RMSE={val_rmse:.4f} val_MAE={val_mae:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            if ema is not None:
                ema.apply_shadow(model)
                best_state = copy.deepcopy(model.state_dict())
                ema.restore(model)
            else:
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


# ============================================================
# Main per-model runners
# ============================================================

def run_simple_model(model_name, input_dim, train_loader, val_loader, test_loader, device, max_epochs, patience):
    if model_name == "LSTM":
        model = LSTMRegressor(input_dim=input_dim)
    elif model_name == "GRU":
        model = GRURegressor(input_dim=input_dim)
    elif model_name == "TCN":
        model = TCNRegressor(input_dim=input_dim)
    elif model_name == "Transformer":
        model = TimeSeriesTransformer(input_dim=input_dim)
    else:
        raise ValueError(model_name)

    start = time.time()
    model, best_val_rmse = train_simple_with_early_stopping(
        model, train_loader, val_loader, device,
        lr=1e-3, max_epochs=max_epochs, patience=patience
    )
    rmse, mae = evaluate_scalar_model(model, test_loader, device)
    elapsed = time.time() - start
    return rmse, mae, best_val_rmse, elapsed


def run_smallonly(p7, input_dim, train_loader, val_loader, test_loader, device, max_epochs, patience):
    model = p7.SmallModelS(input_dim=input_dim, embed_dim=32, hidden_dim=64, feature_dim=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=getattr(p7, "LR_STAGE1", 1e-3))
    mse = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    patience_ctr = 0
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            y_hat, _ = model(xb)
            loss = mse(y_hat, yb)
            loss.backward()
            opt.step()

        val_rmse, _ = evaluate_p7_small(model, val_loader, device)
        print(f"[SmallOnly Epoch {epoch:02d}] val_RMSE={val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    rmse, mae = evaluate_p7_small(model, test_loader, device)
    elapsed = time.time() - start
    return rmse, mae, best_val, elapsed


def run_largeonly(p7, input_dim, train_loader, val_loader, test_loader, device, max_epochs, patience):
    start = time.time()

    model_s = p7.SmallModelS(input_dim=input_dim, embed_dim=32, hidden_dim=64, feature_dim=32).to(device)
    opt_s = torch.optim.Adam(model_s.parameters(), lr=getattr(p7, "LR_STAGE1", 1e-3))
    mse = nn.MSELoss()

    e1 = max(3, max_epochs // 4)
    e2 = max(3, max_epochs - e1)

    best_s = None
    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, e1 + 1):
        model_s.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_s.zero_grad()
            y_hat, _ = model_s(xb)
            loss = mse(y_hat, yb)
            loss.backward()
            opt_s.step()

        val_rmse, _ = evaluate_p7_small(model_s, val_loader, device)
        print(f"[LargeOnly Stage1 Epoch {epoch:02d}] val_RMSE(S)={val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_s = copy.deepcopy(model_s.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    if best_s is not None:
        model_s.load_state_dict(best_s)

    for p in model_s.parameters():
        p.requires_grad = False

    model_l = p7.PatchTransformerL(
        input_dim=input_dim,
        patch_size=4,
        patch_stride=4,
        embed_dim=768,
        num_heads=8,
        ff_dim=1024,
        num_layers=2,
    ).to(device)
    opt_l = torch.optim.Adam(model_l.parameters(), lr=getattr(p7, "LR_STAGE2", 1e-3))

    best_l = None
    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, e2 + 1):
        model_l.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_l.zero_grad()
            y_hat, _ = model_l(xb)
            loss = mse(y_hat, yb)
            loss.backward()
            opt_l.step()

        val_rmse, _ = evaluate_p7_large(model_l, val_loader, device)
        print(f"[LargeOnly Stage2 Epoch {epoch:02d}] val_RMSE(L)={val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_l = copy.deepcopy(model_l.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    if best_l is not None:
        model_l.load_state_dict(best_l)

    rmse, mae = evaluate_p7_large(model_l, test_loader, device)
    elapsed = time.time() - start
    return rmse, mae, best_val, elapsed


def run_collmc(p7, input_dim, train_loader, val_loader, test_loader, device, max_epochs, patience):
    start = time.time()
    model_s, model_l, agent_f, reflect_r, best_val = train_p7_collm_with_early_stopping(
        p7, input_dim, train_loader, val_loader, device,
        max_epochs=max_epochs, patience=patience
    )
    rmse, mae, *_ = p7.evaluate_collm(
        model_s, model_l, agent_f, reflect_r, test_loader, device,
        tau1=getattr(p7, "TAU1", 0.10),
        tau2=getattr(p7, "TAU2", 0.20),
    )
    elapsed = time.time() - start
    return rmse, mae, best_val, elapsed


def run_gfcollm(p8, input_dim, train_loader, val_loader, test_loader, device, max_epochs, patience, fd=None):
    start = time.time()
    model, best_val = train_p8_gfcollm_with_early_stopping(
        p8, input_dim, train_loader, val_loader, device,
        max_epochs=max_epochs, patience=patience, fd=fd
    )
    test_eval = p8.evaluate_gf_collm(model, test_loader, device)
    rmse, mae = parse_gfcollm_eval_result(test_eval)
    elapsed = time.time() - start
    return rmse, mae, best_val, elapsed


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=str, required=True, help="FD001 / FD002 / FD003 / FD004")
    parser.add_argument("--data_dir", type=str, required=True, help="folder containing train_FD00X.txt etc.")
    parser.add_argument("--split_json", type=str, required=True, help="path to split json file")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    fd = args.fd.upper()
    set_seed(args.seed)

    p7, p8 = infer_fd_modules(fd)
    device = getattr(p8, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    split_json = load_split_json(args.split_json)
    train_df, test_df, test_rul_true = load_raw_data_with_p8(p8, args.data_dir, fd)
    feature_cols = get_feature_cols(train_df, fd=fd)

    train_part, val_part = split_train_val_by_engine(train_df, split_json)
    train_norm, val_norm, test_norm, _ = normalize_with_train_only(p8, train_part, val_part, test_df, feature_cols)

    window_size = getattr(p8, "WINDOW_SIZE", 50)
    batch_size = getattr(p8, "BATCH_SIZE", 64)

    X_train, y_train, X_val, y_val, X_test, y_test = create_windows_trainvaltest(
        p8, train_norm, val_norm, test_norm, test_rul_true, feature_cols, window_size
    )

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size
    )

    input_dim = len(feature_cols)

    print(f"Device: {device}")
    print(f"Dataset: {fd}")
    print(f"Input dim: {input_dim}")
    print(f"Window size: {window_size}")
    print(f"Train windows: {len(X_train)} | Val windows: {len(X_val)} | Test engines: {len(X_test)}")

    experiments = [
        ("LSTM", "Non-LLM"),
        ("GRU", "Non-LLM"),
        ("TCN", "Non-LLM"),
        ("Transformer", "Non-LLM"),
        ("SmallOnly", "Collaborative/Large-model"),
        ("LargeOnly", "Collaborative/Large-model"),
        ("CoLLM-C", "Collaborative/Large-model"),
        ("GF-CoLLM", "Proposed"),
    ]

    rows = []

    for name, group in experiments:
        print("\n" + "=" * 80)
        print(f"Running: {name}")

        if name in ["LSTM", "GRU", "TCN", "Transformer"]:
            rmse, mae, best_val, elapsed = run_simple_model(
                name, input_dim, train_loader, val_loader, test_loader,
                device, args.max_epochs, args.patience
            )
        elif name == "SmallOnly":
            rmse, mae, best_val, elapsed = run_smallonly(
                p7, input_dim, train_loader, val_loader, test_loader,
                device, args.max_epochs, args.patience
            )
        elif name == "LargeOnly":
            rmse, mae, best_val, elapsed = run_largeonly(
                p7, input_dim, train_loader, val_loader, test_loader,
                device, args.max_epochs, args.patience
            )
        elif name == "CoLLM-C":
            rmse, mae, best_val, elapsed = run_collmc(
                p7, input_dim, train_loader, val_loader, test_loader,
                device, args.max_epochs, args.patience
            )
        elif name == "GF-CoLLM":
            rmse, mae, best_val, elapsed = run_gfcollm(
                p8, input_dim, train_loader, val_loader, test_loader,
                device, args.max_epochs, args.patience, fd=fd
            )
        else:
            raise ValueError(name)

        print(f"[{name}] test_RMSE={rmse:.4f} test_MAE={mae:.4f} best_val_RMSE={best_val:.4f} time={elapsed:.1f}s")

        rows.append({
            "Dataset": fd,
            "Seed": args.seed,
            "Group": group,
            "Model": name,
            "Val_RMSE_best": best_val,
            "Test_RMSE": rmse,
            "Test_MAE": mae,
            "Runtime_sec": elapsed,
        })

    df = pd.DataFrame(rows).sort_values(["Test_RMSE", "Test_MAE"]).reset_index(drop=True)
    print("\nFinal comparison table:\n")
    print(df)

    out_csv = args.out_csv if args.out_csv is not None else f"{fd.lower()}_seed{args.seed}_earlystop_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()