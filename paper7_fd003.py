#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoLLM-C (approximate implementation) on CMAPSS FD003.

Implements:
- Small model S (lightweight GRU-based regressor)
- Large model L (patch-embedding + Transformer encoder)
- Fuzzy decision agent F (Gaussian membership + MLP)
- Self-reflection model R (FCN)
- CoLLM-C thresholds (you can tune TAU1, TAU2 if needed)

Training:
- Stage 1: train S only (1 epoch)
- Stage 2: train L only (1 epoch, S frozen)
- Stage 3: train F and R only (1 epoch, S & L frozen)

Output:
- Prints RMSE and MAE on FD003 test set (engine-level evaluation)
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

# -----------------------------------------------------------
# Global config
# -----------------------------------------------------------

DATA_DIR = "."
TRAIN_FILE = os.path.join(DATA_DIR, "train_FD003.txt")
TEST_FILE  = os.path.join(DATA_DIR, "test_FD003.txt")
RUL_FILE   = os.path.join(DATA_DIR, "RUL_FD003.txt")

WINDOW_SIZE = 50        # t in the paper
BATCH_SIZE = 256
LR_STAGE1 = 2e-3
LR_STAGE2 = 2e-3
LR_STAGE3 = 2e-3

EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 5
EPOCHS_STAGE3 = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# CoLLM-C thresholds (you may tune based on FD003 behavior)
TAU1 = 0.6
TAU2 = 0.05

ALPHA_CONF = 10.0       # scaling factor for confidence labels


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
# FD003 loading & preprocessing
# -----------------------------------------------------------

def load_fd003(path: str) -> pd.DataFrame:
    """
    Load CMAPSS FD003 file.

    Format:
      col 1: unit (engine id)
      col 2: time (cycle)
      col 3-5: operating settings 1-3
      col 6-26: sensors 1-21
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Sometimes an extra empty column exists
    if df.shape[1] == 27:
        df = df.iloc[:, :-1]

    cols = ["unit", "time", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    df.columns = cols
    return df


def add_rul_labels(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard RUL: RUL = max_time(unit) - current_time.
    """
    max_cycle = train_df.groupby("unit")["time"].max().reset_index()
    max_cycle.columns = ["unit", "max_time"]
    df = train_df.merge(max_cycle, on="unit", how="left")
    df["RUL"] = df["max_time"] - df["time"]
    df.drop(columns=["max_time"], inplace=True)
    return df


def load_rul_file(path: str) -> np.ndarray:
    """
    RUL_FD003.txt -> array of size (#test_units,)
    """
    rul_df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    return rul_df.iloc[:, 0].values.astype(np.float32)


def normalize_features(train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       feature_cols):
    """
    z-score normalization over training data, apply to test.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled  = scaler.transform(test_df[feature_cols])

    train_df_norm = train_df.copy()
    test_df_norm  = test_df.copy()
    train_df_norm[feature_cols] = train_scaled
    test_df_norm[feature_cols]  = test_scaled
    return train_df_norm, test_df_norm, scaler


def create_train_windows(df: pd.DataFrame,
                         feature_cols,
                         window_size: int):
    """
    Sliding-window sequences from training data.

    Returns:
      X: (num_seq, window_size, num_features)
      y: (num_seq,)
    """
    X_list = []
    y_list = []

    for uid, group in df.groupby("unit"):
        group = group.sort_values("time")
        data = group[feature_cols].values
        rul  = group["RUL"].values

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
    """
    For each test unit, take the last window_size cycles (pad with first row if shorter).

    Returns:
      X_test: (num_units, window_size, num_features)
      unit_ids: (num_units,)
    """
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
# Models: S, L, F, R
# -----------------------------------------------------------

class SmallModelS(nn.Module):
    """
    Lightweight small model S:
      - Linear projection (input_dim -> embed_dim)
      - GRU (embed_dim -> hidden_dim)
      - Feature projection (hidden_dim -> feature_dim = ds)
      - Regression head (feature_dim -> 1)
    """

    def __init__(self,
                 input_dim: int,
                 embed_dim: int = 32,
                 hidden_dim: int = 64,
                 feature_dim: int = 32):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.feat_proj = nn.Linear(hidden_dim, feature_dim)
        self.reg_head = nn.Linear(feature_dim, 1)

    def forward(self, x):
        """
        x: (B, T, F)
        Returns:
          y: (B,)
          feat: (B, feature_dim)
        """
        e = torch.relu(self.embed(x))       # (B, T, embed_dim)
        out, _ = self.gru(e)                # (B, T, hidden_dim)
        h_last = out[:, -1, :]              # (B, hidden_dim)
        feat = torch.relu(self.feat_proj(h_last))  # (B, feature_dim)
        y = self.reg_head(feat).squeeze(-1)        # (B,)
        return y, feat


class PatchTransformerL(nn.Module):
    """
    Large model L (approximation):
      - Patch embedding over time dimension.
      - TransformerEncoder.
      - Regression head.
    """

    def __init__(self,
                 input_dim: int,
                 patch_size: int = 4,
                 patch_stride: int = 4,
                 embed_dim: int = 768,
                 num_heads: int = 8,
                 ff_dim: int = 1024,
                 num_layers: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.input_dim = input_dim

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
        """
        x: (B, T, F)
        Returns:
          patches: (B, S, patch_dim)
        """
        B, T, F = x.shape
        ps = self.patch_size
        stride = self.patch_stride

        patches = []
        for start in range(0, T - ps + 1, stride):
            end = start + ps
            patches.append(x[:, start:end, :].reshape(B, -1))  # (B, ps*F)

        patches = torch.stack(patches, dim=1)  # (B, S, patch_dim)
        return patches

    def forward(self, x):
        """
        x: (B, T, F)
        Returns:
          y: (B,)
          feat: (B, embed_dim)  (pooled last token)
        """
        patches = self._patchify(x)               # (B, S, patch_dim)
        tokens = self.patch_embed(patches)        # (B, S, E)
        tokens = tokens.transpose(0, 1)           # (S, B, E)

        out = self.transformer(tokens)            # (S, B, E)
        last_token = out[-1, :, :]                # (B, E)

        y = self.reg_head(last_token).squeeze(-1)
        feat = last_token
        return y, feat


class FuzzyDecisionAgent(nn.Module):
    """
    Fuzzy decision-making agent F:
      - Gaussian membership functions over feature vector.
      - MLP to output confidence score Q_s in [0, 1].
    """

    def __init__(self,
                 feature_dim: int,
                 num_memberships: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_memberships = num_memberships

        self.mu = nn.Parameter(torch.zeros(num_memberships, feature_dim))
        self.sigma = nn.Parameter(torch.ones(num_memberships, feature_dim))

        self.mlp = nn.Sequential(
            nn.Linear(num_memberships * feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        """
        feat: (B, feature_dim)
        Returns:
          Q_s: (B,)
        """
        B, D = feat.shape
        x = feat.unsqueeze(1)             # (B, 1, D)
        mu = self.mu.unsqueeze(0)         # (1, M, D)
        sigma = self.sigma.unsqueeze(0)   # (1, M, D)

        membership = torch.exp(- (x - mu) ** 2 / (sigma ** 2 + 1e-6))  # (B, M, D)
        membership_flat = membership.view(B, -1)                        # (B, M*D)
        Q_s = self.mlp(membership_flat).squeeze(-1)                     # (B,)
        return Q_s


class SelfReflection(nn.Module):
    """
    Self-reflection model R: FCN mapping large-model features -> confidence Q_l.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        """
        feat: (B, dl)
        """
        return self.net(feat).squeeze(-1)


# -----------------------------------------------------------
# Training helpers
# -----------------------------------------------------------

def train_stage1_small_model(model_s, train_loader, device, lr, epochs):
    print("=== Stage 1: Train Small Model S ===")
    model_s.to(device)
    optimizer = torch.optim.Adam(model_s.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model_s.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred, _ = model_s(X_batch)
            loss = mse_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        rmse = math.sqrt(avg_loss)
        print(f"Stage1 Epoch {epoch}/{epochs} - Train RMSE (S): {rmse:.4f}")


def train_stage2_large_model(model_s, model_l, train_loader, device, lr, epochs):
    print("=== Stage 2: Train Large Model L (S frozen) ===")
    model_s.to(device)
    model_l.to(device)

    for p in model_s.parameters():
        p.requires_grad = False
    model_s.eval()

    optimizer = torch.optim.Adam(model_l.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model_l.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred, _ = model_l(X_batch)
            loss = mse_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        rmse = math.sqrt(avg_loss)
        print(f"Stage2 Epoch {epoch}/{epochs} - Train RMSE (L): {rmse:.4f}")


def train_stage3_fuzzy_and_reflection(model_s, model_l,
                                      agent_f, reflect_r,
                                      train_loader, device, lr, epochs):
    print("=== Stage 3: Train Fuzzy Agent F and Self-Reflection R (S & L frozen) ===")

    model_s.to(device)
    model_l.to(device)
    agent_f.to(device)
    reflect_r.to(device)

    for p in model_s.parameters():
        p.requires_grad = False
    for p in model_l.parameters():
        p.requires_grad = False

    model_s.eval()
    model_l.eval()

    optimizer = torch.optim.Adam(
        list(agent_f.parameters()) + list(reflect_r.parameters()),
        lr=lr
    )
    mse_loss = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        agent_f.train()
        reflect_r.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            with torch.no_grad():
                y_s, feat_s = model_s(X_batch)
                y_l, feat_l = model_l(X_batch)

            Qs_star = 1.0 - torch.tanh(torch.abs(y_s - y_batch) / ALPHA_CONF)
            Ql_star = 1.0 - torch.tanh(torch.abs(y_l - y_batch) / ALPHA_CONF)

            Qs_pred = agent_f(feat_s.detach())
            Ql_pred = reflect_r(feat_l.detach())

            loss_f = mse_loss(Qs_pred, Qs_star)
            loss_r = mse_loss(Ql_pred, Ql_star)
            loss = loss_f + loss_r

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Stage3 Epoch {epoch}/{epochs} - Train loss (F+R): {avg_loss:.6f}")


# -----------------------------------------------------------
# CoLLM-C inference & evaluation
# -----------------------------------------------------------

def collm_forward_batch(model_s, model_l, agent_f, reflect_r,
                        X_batch, device, tau1, tau2):
    """
    Full CoLLM-C inference for a batch of sequences.

    Returns:
      y_final: (B,)
    """
    model_s.eval()
    model_l.eval()
    agent_f.eval()
    reflect_r.eval()

    with torch.no_grad():
        X_batch = X_batch.to(device)

        y_s, feat_s = model_s(X_batch)
        Q_s = agent_f(feat_s)

        need_large = Q_s < tau1
        y_final = y_s.clone()

        if need_large.any():
            y_l, feat_l = model_l(X_batch)
            Q_l = reflect_r(feat_l)

            delta = Q_s - Q_l

            use_large = need_large & (delta <= tau2)
            use_fuse  = need_large & (delta > tau2)

            y_final[use_large] = y_l[use_large]
            y_final[use_fuse]  = 0.5 * (y_s[use_fuse] + y_l[use_fuse])

        return y_final.cpu()


def evaluate_collm(model_s, model_l, agent_f, reflect_r,
                   data_loader, device, tau1, tau2):
    preds = []
    trues = []

    for X_batch, y_batch in data_loader:
        y_hat = collm_forward_batch(model_s, model_l, agent_f, reflect_r,
                                    X_batch, device, tau1, tau2)
        preds.append(y_hat.numpy())
        trues.append(y_batch.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = math.sqrt(mse)
    return rmse, mae, preds, trues


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print("Loading FD003 data...")

    train_df = load_fd003(TRAIN_FILE)
    test_df  = load_fd003(TEST_FILE)
    train_df = add_rul_labels(train_df)
    test_rul_true = load_rul_file(RUL_FILE)

    # Use 14 informative sensors (same subset as in FD001/FD002 version)
    useless_sensors = [1, 5, 6, 10, 16, 18, 19]
    sensor_cols = [f"s{i}" for i in range(1, 22) if i not in useless_sensors]
    feature_cols = sensor_cols
    input_dim = len(feature_cols)
    print(f"Using {input_dim} sensor features: {feature_cols}")

    train_df, test_df, _ = normalize_features(train_df, test_df, feature_cols)

    print("Creating training windows...")
    X_train, y_train = create_train_windows(train_df, feature_cols, WINDOW_SIZE)
    print(f"Train windows: X={X_train.shape}, y={y_train.shape}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Creating test engine-level windows...")
    X_test, test_units = create_test_last_windows(test_df, feature_cols, WINDOW_SIZE)
    print(f"Test engines: {X_test.shape[0]}")

    if len(test_rul_true) != len(test_units):
        print(f"WARNING: #test engines ({len(test_units)}) != #RUL entries ({len(test_rul_true)}). "
              f"Using minimum length.")
    n_common = min(len(test_rul_true), len(test_units))
    y_test = test_rul_true[:n_common]
    X_test = X_test[:n_common]

    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing models (S, L, F, R)...")

    model_s = SmallModelS(input_dim=input_dim,
                          embed_dim=32,
                          hidden_dim=64,
                          feature_dim=32)

    model_l = PatchTransformerL(input_dim=input_dim,
                                patch_size=4,
                                patch_stride=4,
                                embed_dim=768,
                                num_heads=8,
                                ff_dim=1024,
                                num_layers=2)

    agent_f   = FuzzyDecisionAgent(feature_dim=32, num_memberships=64)
    reflect_r = SelfReflection(input_dim=768)

    # Stage 1
    train_stage1_small_model(model_s, train_loader, DEVICE,
                             lr=LR_STAGE1, epochs=EPOCHS_STAGE1)

    # Stage 2
    train_stage2_large_model(model_s, model_l, train_loader, DEVICE,
                             lr=LR_STAGE2, epochs=EPOCHS_STAGE2)

    # Stage 3
    train_stage3_fuzzy_and_reflection(model_s, model_l,
                                      agent_f, reflect_r,
                                      train_loader, DEVICE,
                                      lr=LR_STAGE3, epochs=EPOCHS_STAGE3)

    print("=== Final CoLLM-C Evaluation on FD003 Test ===")
    rmse_test, mae_test, _, _ = evaluate_collm(
        model_s, model_l, agent_f, reflect_r,
        test_loader, DEVICE,
        tau1=TAU1, tau2=TAU2
    )

    print(f"[TEST] RMSE: {rmse_test:.4f}")
    print(f"[TEST] MAE:  {mae_test:.4f}")


if __name__ == "__main__":
    main()
