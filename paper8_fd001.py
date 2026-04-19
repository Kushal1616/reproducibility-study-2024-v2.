#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GF-CoLLM on CMAPSS FD001 with:
- expert-gating fusion over small / large / corrected branches (V7)
- bounded zero-initialized residual corrector
- small bounded fusion refinement path
- gated fuzzy adapters
- improved small branch
- orthogonality regularization
- branch-first warm-up schedule
"""

import os
import math
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# -----------------------------------------------------------
# Global config
# -----------------------------------------------------------

DATA_DIR = "."
TRAIN_FILE = os.path.join(DATA_DIR, "train_FD001.txt")
TEST_FILE = os.path.join(DATA_DIR, "test_FD001.txt")
RUL_FILE = os.path.join(DATA_DIR, "RUL_FD001.txt")

WINDOW_SIZE = 50
BATCH_SIZE = 256

EPOCHS = 40
LR = 2e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Auxiliary loss weights
LAMBDA_RESIDUAL = 0.2
LAMBDA_ROUTE = 0.1
LAMBDA_CONF = 0.1
LAMBDA_ORTHO = 0.05

# Branch supervision maxima
ALPHA_SMALL_MAX = 1.0
ALPHA_LARGE_MAX = 0.7


# -----------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# -----------------------------------------------------------
# Data loading utilities for FD001
# -----------------------------------------------------------

def load_fd001(path: str) -> pd.DataFrame:
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


def normalize_features(train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       feature_cols):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])

    train_norm = train_df.copy()
    test_norm = test_df.copy()
    train_norm[feature_cols] = train_scaled
    test_norm[feature_cols] = test_scaled
    return train_norm, test_norm, scaler


def create_train_windows(df: pd.DataFrame,
                         feature_cols,
                         window_size: int):
    X_list = []
    y_list = []

    for _, group in df.groupby("unit"):
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


def create_test_last_windows(test_df: pd.DataFrame,
                             feature_cols,
                             window_size: int):
    X_list = []
    unit_ids = []

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


# -----------------------------------------------------------
# FIG Layer
# -----------------------------------------------------------

class FIGLayer(nn.Module):
    def __init__(self, input_dim: int, window_size: int, granule_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(3 * input_dim, granule_dim)

    def forward(self, x):
        mean_t = x.mean(dim=1)
        std_t = x.std(dim=1)
        trend = x[:, -1, :] - x[:, 0, :]
        granule_raw = torch.cat([mean_t, std_t, trend], dim=-1)
        return torch.relu(self.proj(granule_raw))


# -----------------------------------------------------------
# Gated Fuzzy Adapters
# -----------------------------------------------------------

class GatedFuzzyAdapter(nn.Module):
    def __init__(self, input_dim: int, granule_dim: int, out_dim: int):
        super().__init__()
        self.gate = nn.Linear(granule_dim, out_dim)
        self.proj = nn.Linear(input_dim, out_dim)

    def forward(self, x, g):
        gate_weights = torch.sigmoid(self.gate(g)).unsqueeze(1)
        return self.proj(x) * gate_weights


# -----------------------------------------------------------
# Improved Small model S
# -----------------------------------------------------------

class ImprovedSmallModelS(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 feature_dim: int = 64):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.feat_proj = nn.Linear(hidden_dim, feature_dim)
        self.reg_head = nn.Linear(feature_dim, 1)
        self.unc_head = nn.Linear(feature_dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv(x))
        x = x.transpose(1, 2)

        out, _ = self.gru(x)
        h_last = out[:, -1, :]
        feat = torch.relu(self.feat_proj(h_last))
        y = self.reg_head(feat).squeeze(-1)
        u_raw = self.unc_head(feat).squeeze(-1)
        u = F.softplus(u_raw) + 1e-4
        return y, feat, u


# -----------------------------------------------------------
# Large model L
# -----------------------------------------------------------

class PatchTransformerL(nn.Module):
    def __init__(self,
                 input_dim: int,
                 patch_size: int = 4,
                 patch_stride: int = 4,
                 embed_dim: int = 256,
                 num_heads: int = 4,
                 ff_dim: int = 512,
                 num_layers: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        patch_dim = patch_size * input_dim
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.reg_head = nn.Linear(embed_dim, 1)

    def _patchify(self, x):
        B, T, F = x.shape
        ps = self.patch_size
        stride = self.patch_stride
        patches = []
        for start in range(0, T - ps + 1, stride):
            end = start + ps
            patches.append(x[:, start:end, :].reshape(B, -1))
        return torch.stack(patches, dim=1)

    def forward(self, x):
        patches = self._patchify(x)
        tokens = self.patch_embed(patches)
        tokens = tokens.transpose(0, 1)
        out = self.transformer(tokens)
        last_token = out[-1, :, :]
        y = self.reg_head(last_token).squeeze(-1)
        feat = last_token
        return y, feat


# -----------------------------------------------------------
# Routing Agent
# -----------------------------------------------------------

class RoutingAgent(nn.Module):
    def __init__(self, granule_dim: int, extra_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(granule_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, g, u_s, conf_score):
        x = torch.cat([g, u_s.unsqueeze(-1), conf_score.unsqueeze(-1)], dim=-1)
        return self.net(x)


# -----------------------------------------------------------
# Fuzzy–Conformal-style Calibration Head
# -----------------------------------------------------------

class FuzzyConformalHead(nn.Module):
    def __init__(self, granule_dim: int, feat_s_dim: int, feat_l_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(granule_dim + feat_s_dim + feat_l_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, g, feat_s, feat_l):
        x = torch.cat([g, feat_s, feat_l], dim=-1)
        return self.net(x).squeeze(-1)


# -----------------------------------------------------------
# Residual Corrector with dropout
# -----------------------------------------------------------

class ResidualCorrector(nn.Module):
    def __init__(self, granule_dim: int, feat_l_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(granule_dim + feat_l_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevent overfitting to training noise
            nn.Linear(hidden_dim, 1)
        )
        # CRITICAL: initialize to zero so it starts by doing nothing
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, g, feat_l):
        x = torch.cat([g, feat_l], dim=-1)
        # Bounded correction: cannot move RUL by more than +/- 20 cycles
        return torch.tanh(self.net(x).squeeze(-1)) * 20.0


# -----------------------------------------------------------
# Contextual Expert Gating Fusion
# -----------------------------------------------------------

class SharpenedExpertGating(nn.Module):
    def __init__(self, granule_dim=64, temperature=0.5, refine_scale=2.0):
        super().__init__()
        self.temp = temperature
        self.refine_scale = refine_scale
        # Input: granules + confidence/uncertainty cues from the branches
        self.gate_net = nn.Sequential(
            nn.Linear(granule_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.refine_net = nn.Sequential(
            nn.Linear(granule_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        nn.init.zeros_(self.refine_net[-1].weight)
        nn.init.zeros_(self.refine_net[-1].bias)

    def forward(self, y_s, y_l, y_c, g, conf_s, conf_l):
        gate_in = torch.cat([g, conf_s.unsqueeze(-1), conf_l.unsqueeze(-1)], dim=-1)
        logits = self.gate_net(gate_in)
        # Sharpened softmax: encourages confident selection rather than mushy averaging.
        weights = torch.softmax(logits / self.temp, dim=-1)

        y_weighted = (
            weights[:, 0] * y_s
            + weights[:, 1] * y_l
            + weights[:, 2] * y_c
        )

                # Tighter bounded refinement to avoid rewriting a strong expert prediction.
        refine = torch.tanh(self.refine_net(g).squeeze(-1)) * self.refine_scale
        return y_weighted + refine, weights, logits

# -----------------------------------------------------------
# GF-CoLLM wrapper
# -----------------------------------------------------------

class GFCoLLM(nn.Module):
    def __init__(self, input_dim: int, window_size: int):
        super().__init__()
        granule_dim = 64

        self.fig = FIGLayer(input_dim, window_size, granule_dim=granule_dim)

        self.adapter_s = GatedFuzzyAdapter(input_dim, granule_dim, out_dim=32)
        self.adapter_l = GatedFuzzyAdapter(input_dim, granule_dim, out_dim=64)

        self.small = ImprovedSmallModelS(
            input_dim=32,
            hidden_dim=128,
            feature_dim=64
        )

        self.large = PatchTransformerL(
            input_dim=64,
            patch_size=4,
            patch_stride=4,
            embed_dim=256,
            num_heads=4,
            ff_dim=512,
            num_layers=2
        )

        self.conf_head = FuzzyConformalHead(
            granule_dim=granule_dim,
            feat_s_dim=64,
            feat_l_dim=256,
            hidden_dim=64
        )

        self.corrector = ResidualCorrector(
            granule_dim=granule_dim,
            feat_l_dim=256,
            hidden_dim=64
        )

        self.routing = RoutingAgent(
            granule_dim=granule_dim,
            extra_dim=2,
            hidden_dim=64
        )

        self.fusion = SharpenedExpertGating(granule_dim=granule_dim, temperature=0.5, refine_scale=2.0)

        self.feat_l_to_s = nn.Linear(256, 64)

    def forward(self, x, y_true=None):
        g = self.fig(x)

        x_s = self.adapter_s(x, g)
        x_l = self.adapter_l(x, g)

        y_s, feat_s, u_s = self.small(x_s)
        y_l, feat_l = self.large(x_l)

        conf_score = self.conf_head(g, feat_s, feat_l)

        residual_hat = self.corrector(g, feat_l)
        y_c = y_l + residual_hat

        # Confidence cues for gating: lower uncertainty/nonconformity becomes higher trust.
        conf_s = 1.0 / (u_s + 1.0)
        conf_l = 1.0 / (conf_score + 1.0)
        y_fused, gate_weights, gate_logits = self.fusion(y_s, y_l, y_c, g, conf_s, conf_l)

        route_logits = self.routing(g, u_s, conf_score)
        feat_l_proj = self.feat_l_to_s(feat_l)

        return {
            "y_s": y_s,
            "y_l": y_l,
            "y_c": y_c,
            "y_fused": y_fused,
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "conf_s": conf_s,
            "conf_l": conf_l,
            "residual_hat": residual_hat,
            "route_logits": route_logits,
            "conf_score": conf_score,
            "feat_s": feat_s,
            "feat_l": feat_l,
            "feat_l_proj": feat_l_proj,
            "u_s": u_s,
        }


# -----------------------------------------------------------
# Training & evaluation
# -----------------------------------------------------------

def cosine_orthogonality_loss(feat_s, feat_l_proj):
    sim = F.cosine_similarity(feat_s, feat_l_proj, dim=-1)
    return torch.mean(torch.abs(sim))


def branch_weights(epoch, total_epochs):
    if epoch <= 20:
        return ALPHA_SMALL_MAX, ALPHA_LARGE_MAX
    decay = max(0.3, 1.0 - (epoch - 20) / max(1, total_epochs - 20))
    return ALPHA_SMALL_MAX * decay, ALPHA_LARGE_MAX * decay


def train_gf_collm(model, train_loader, device, epochs=EPOCHS, lr=LR):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    lambda_res = LAMBDA_RESIDUAL
    lambda_route = LAMBDA_ROUTE
    lambda_conf = LAMBDA_CONF
    lambda_ortho = LAMBDA_ORTHO

    print("ENHANCED GF-CoLLM TRAINING ACTIVE")

    for epoch in range(1, epochs + 1):
        model.train()

        alpha_small, alpha_large = branch_weights(epoch, epochs)

        total_loss = 0.0
        total_fused = 0.0
        total_small = 0.0
        total_large = 0.0
        total_corr = 0.0
        total_ortho = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            out = model(X_batch, y_true=y_batch)

            y_s = out["y_s"]
            y_l = out["y_l"]
            y_c = out["y_c"]
            y_hat = out["y_fused"]
            residual_hat = out["residual_hat"]
            route_logits = out["route_logits"]
            conf_score = out["conf_score"]
            feat_s = out["feat_s"]
            feat_l_proj = out["feat_l_proj"]

            L_small = mse_loss(y_s, y_batch)
            L_large = mse_loss(y_l, y_batch)
            L_corr = mse_loss(y_c, y_batch)
            L_fused = mse_loss(y_hat, y_batch)
            L_resid = mse_loss(residual_hat, (y_batch - y_l))
            L_ortho = cosine_orthogonality_loss(feat_s, feat_l_proj)

            with torch.no_grad():
                sq_err_s = (y_s - y_batch) ** 2
                sq_err_l = (y_l - y_batch) ** 2
                sq_err_c = (y_c - y_batch) ** 2
                errs = torch.stack([sq_err_s, sq_err_l, sq_err_c], dim=1)
                best_idx = torch.argmin(errs, dim=1)

            L_route = ce_loss(route_logits, best_idx)
            L_conf = mse_loss(conf_score, torch.abs(y_hat - y_batch))

            if epoch <= 10:
                L_total = (
                    alpha_small * L_small
                    + alpha_large * L_large
                    + lambda_ortho * L_ortho
                )
            elif epoch <= 20:
                L_total = (
                    L_fused
                    + alpha_small * L_small
                    + alpha_large * L_large
                    + lambda_res * L_resid
                    + 0.5 * L_corr
                    + lambda_ortho * L_ortho
                )
            else:
                L_total = (
                    L_fused
                    + alpha_small * L_small
                    + alpha_large * L_large
                    + lambda_res * L_resid
                    + 0.5 * L_corr
                    + lambda_route * L_route
                    + lambda_conf * L_conf
                    + lambda_ortho * L_ortho
                )

            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += L_total.item() * X_batch.size(0)
            total_fused += L_fused.item() * X_batch.size(0)
            total_small += L_small.item() * X_batch.size(0)
            total_large += L_large.item() * X_batch.size(0)
            total_corr += L_corr.item() * X_batch.size(0)
            total_ortho += L_ortho.item() * X_batch.size(0)

        n = len(train_loader.dataset)

        rmse_fused = math.sqrt(total_fused / n)
        rmse_small = math.sqrt(total_small / n)
        rmse_large = math.sqrt(total_large / n)
        rmse_corr = math.sqrt(total_corr / n)
        avg_ortho = total_ortho / n
        avg_total = total_loss / n

        print(
            f"[Epoch {epoch:02d}/{epochs}] "
            f"Total={avg_total:.4f} | "
            f"alpha_s={alpha_small:.3f} alpha_l={alpha_large:.3f} | "
            f"RMSE_fused={rmse_fused:.4f} | "
            f"RMSE_small={rmse_small:.4f} | "
            f"RMSE_large={rmse_large:.4f} | "
            f"RMSE_corr={rmse_corr:.4f} | "
            f"Ortho={avg_ortho:.4f}"
        )


def evaluate_gf_collm(model, test_loader, device):
    model.to(device)
    model.eval()

    preds_s, preds_l, preds_c, preds_f = [], [], [], []
    trues = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)

            preds_s.append(out["y_s"].cpu().numpy())
            preds_l.append(out["y_l"].cpu().numpy())
            preds_c.append(out["y_c"].cpu().numpy())
            preds_f.append(out["y_fused"].cpu().numpy())
            trues.append(y_batch.numpy())

    y_true = np.concatenate(trues, axis=0)
    y_s = np.concatenate(preds_s, axis=0)
    y_l = np.concatenate(preds_l, axis=0)
    y_c = np.concatenate(preds_c, axis=0)
    y_f = np.concatenate(preds_f, axis=0)

    def get_metrics(y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        return rmse, mae

    rmse_s, mae_s = get_metrics(y_s)
    rmse_l, mae_l = get_metrics(y_l)
    rmse_c, mae_c = get_metrics(y_c)
    rmse_f, mae_f = get_metrics(y_f)

    print(f"[Small ] RMSE: {rmse_s:.4f} | MAE: {mae_s:.4f}")
    print(f"[Large ] RMSE: {rmse_l:.4f} | MAE: {mae_l:.4f}")
    print(f"[Corr  ] RMSE: {rmse_c:.4f} | MAE: {mae_c:.4f}")
    print(f"[Fused ] RMSE: {rmse_f:.4f} | MAE: {mae_f:.4f}")

    return {
        "small": {"rmse": rmse_s, "mae": mae_s},
        "large": {"rmse": rmse_l, "mae": mae_l},
        "corr": {"rmse": rmse_c, "mae": mae_c},
        "fused": {"rmse": rmse_f, "mae": mae_f},
    }


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    print("Loading FD001 data...")
    train_df = load_fd001(TRAIN_FILE)
    test_df = load_fd001(TEST_FILE)
    train_df = add_rul_labels(train_df)
    test_rul_true = load_rul_file(RUL_FILE)

    useless_sensors = [1, 5, 6, 10, 16, 18, 19]
    feature_cols = [f"s{i}" for i in range(1, 22) if i not in useless_sensors]
    input_dim = len(feature_cols)
    print(f"Using {input_dim} sensor features: {feature_cols}")

    train_df_norm, test_df_norm, _ = normalize_features(train_df, test_df, feature_cols)

    print("Creating training windows...")
    X_train, y_train = create_train_windows(train_df_norm, feature_cols, WINDOW_SIZE)
    print(f"Train windows: X={X_train.shape}, y={y_train.shape}")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Creating test engine-level windows...")
    X_test, test_units = create_test_last_windows(test_df_norm, feature_cols, WINDOW_SIZE)
    n_common = min(len(test_rul_true), len(test_units))
    X_test = X_test[:n_common]
    y_test = test_rul_true[:n_common]

    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing enhanced GF-CoLLM model...")
    model = GFCoLLM(input_dim=input_dim, window_size=WINDOW_SIZE)

    print(f"Training enhanced GF-CoLLM ({EPOCHS} epochs)...")
    train_gf_collm(model, train_loader, DEVICE, epochs=EPOCHS, lr=LR)

    print("Evaluating enhanced GF-CoLLM on FD001 test set...")
    metrics = evaluate_gf_collm(model, test_loader, DEVICE)

    print(f"[TEST-FUSED] RMSE: {metrics['fused']['rmse']:.4f}")
    print(f"[TEST-FUSED] MAE:  {metrics['fused']['mae']:.4f}")


if __name__ == "__main__":
    main()