#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-2 fine-tuning for RUL prediction on NASA CMAPSS FD002.

Assumes the following files are in the SAME folder as this script:
    train_FD002.txt
    test_FD002.txt
    RUL_FD002.txt

Pipeline:
- Load FD002 train/test and build RUL labels for train.
- Standardize operating conditions + sensors.
- Build sliding window sequences (SEQ_LEN cycles).
- Fine-tune GPT-2 for 1 epoch for sequence regression.
- Report RMSE and MAE on validation set and test set.
- Estimate FLOPs and number of parameters for one forward pass (if thop is installed).
- Save test predictions to: gpt2_predictions_FD002.csv
"""

import os
import math
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import GPT2Config, GPT2Model

# Optional FLOPs computation
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
DATA_DIR = "."  # same folder as this script and data files
DATASET_NAME = "FD002"

TRAIN_FILE = os.path.join(DATA_DIR, f"train_{DATASET_NAME}.txt")
TEST_FILE = os.path.join(DATA_DIR, f"test_{DATASET_NAME}.txt")
RUL_FILE = os.path.join(DATA_DIR, f"RUL_{DATASET_NAME}.txt")

SEQ_LEN = 30          # time window length (in cycles)
BATCH_SIZE = 16       # smaller batch size because GPT-2 is heavy
EPOCHS = 1            # as requested: 1 epoch of fine-tuning
LR = 1e-4
VAL_SPLIT = 0.2       # fraction of units for validation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# --------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# --------------------------------------------------------------------
# Data loading and preprocessing
# --------------------------------------------------------------------
def load_cmapss_file(path):
    """
    Load CMAPSS FD00X file (train or test) as DataFrame with proper column names.

    Format (NASA CMAPSS):
        1) unit
        2) time (cycle)
        3-5) operating settings 1-3
        6-26) sensor 1-21
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Some versions have an extra empty column at the end
    if df.shape[1] == 27:
        df = df.iloc[:, :-1]

    col_names = ["unit", "time", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    df.columns = col_names
    return df


def add_rul_labels(train_df):
    """
    Add Remaining Useful Life (RUL) label for each row in FD002 training data:
        RUL = max_time(unit) - current_time
    """
    max_cycle = train_df.groupby("unit")["time"].max().reset_index()
    max_cycle.columns = ["unit", "max_time"]
    train_df = train_df.merge(max_cycle, on="unit", how="left")
    train_df["RUL"] = train_df["max_time"] - train_df["time"]
    train_df.drop(columns=["max_time"], inplace=True)
    return train_df


def load_rul_file(path):
    """
    Load RUL_FD00X.txt: each row k contains the true RUL at the last cycle
    of the k-th test engine.
    """
    rul_df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Remove possible extra columns
    if rul_df.shape[1] > 1:
        rul_df = rul_df.iloc[:, 0]
    else:
        rul_df = rul_df.iloc[:, 0]
    return rul_df.values.astype(np.float32)


def scale_features(train_df, test_df, feature_cols):
    """
    Standardize features using statistics from training data.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])

    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[feature_cols] = train_scaled
    test_df_scaled[feature_cols] = test_scaled
    return train_df_scaled, test_df_scaled, scaler


def make_sequences(df, feature_cols, seq_len, use_rul=True):
    """
    Convert unit-wise time series into sequences.

    If use_rul=True (train/val):
        returns X (num_seq, seq_len, num_features),
                y (num_seq,)

    If use_rul=False (test-time):
        returns X (num_units, seq_len, num_features),
                unit_ids (num_units,)
    """
    X_list = []
    y_list = []
    unit_ids = []

    for unit_id, group in df.groupby("unit"):
        group = group.sort_values("time")
        data = group[feature_cols].values

        if use_rul:
            rul = group["RUL"].values
            if len(group) >= seq_len:
                for i in range(len(group) - seq_len + 1):
                    X_list.append(data[i : i + seq_len])
                    y_list.append(rul[i + seq_len - 1])
        else:
            # last seq_len cycles (with left padding if needed)
            if len(group) >= seq_len:
                seq = data[-seq_len:]
            else:
                pad_len = seq_len - len(group)
                pad_block = np.repeat(data[0][None, :], pad_len, axis=0)
                seq = np.vstack([pad_block, data])
            X_list.append(seq)
            unit_ids.append(unit_id)

    X = np.array(X_list, dtype=np.float32)

    if use_rul:
        y = np.array(y_list, dtype=np.float32)
        return X, y
    else:
        unit_ids = np.array(unit_ids, dtype=np.int32)
        return X, unit_ids


# --------------------------------------------------------------------
# GPT-2 regression model for time series
# --------------------------------------------------------------------
class TimeSeriesGPT2(nn.Module):
    """
    Wraps a pre-trained GPT-2 transformer for time-series regression.

    - A linear layer maps time-series features -> GPT-2 embedding dimension.
    - The GPT-2 transformer processes the sequence.
    - A regression head maps the last token's hidden state -> scalar RUL.
    """

    def __init__(self, input_dim, seq_len=SEQ_LEN, gpt2_name="gpt2"):
        super(TimeSeriesGPT2, self).__init__()

        # Load GPT-2 config and model (pre-trained)
        self.config = GPT2Config.from_pretrained(gpt2_name)
        self.config.n_positions = max(self.config.n_positions, seq_len)
        self.config.n_ctx = max(self.config.n_ctx, seq_len)

        self.gpt2 = GPT2Model.from_pretrained(gpt2_name, config=self.config)

        # Linear projection from input_dim to GPT-2 embedding size
        self.feature_proj = nn.Linear(input_dim, self.config.n_embd)

        # Regression head
        self.reg_head = nn.Linear(self.config.n_embd, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        # Project features to embedding dimension
        emb = self.feature_proj(x)  # (batch, seq_len, n_embd)

        # Pass as inputs_embeds to GPT-2
        outputs = self.gpt2(inputs_embeds=emb)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, n_embd)
        pooled = last_hidden[:, -1, :]          # last token
        y_hat = self.reg_head(pooled).squeeze(-1)
        return y_hat


# --------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    mse_loss = nn.MSELoss()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = mse_loss(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    avg_mse = total_loss / len(loader.dataset)
    rmse = math.sqrt(avg_mse)
    return avg_mse, rmse


def evaluate(model, loader, device):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)

            preds.append(y_pred.cpu().numpy())
            trues.append(y_batch.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = math.sqrt(mse)

    return rmse, mae, preds, trues


def estimate_flops_and_params(model, seq_len, input_dim, device):
    """
    Estimate FLOPs and parameter count using thop, if available.
    """
    if not HAS_THOP:
        print("thop is not installed. Skipping FLOPs computation.")
        return None, None

    model.eval()
    dummy = torch.randn(1, seq_len, input_dim).to(device)

    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy,), verbose=False)

    return flops, params


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    print(f"Using device: {DEVICE}")
    print(f"Dataset: {DATASET_NAME}")

    # 1) Load data
    print("Loading FD002 train/test data...")
    train_df = load_cmapss_file(TRAIN_FILE)
    test_df = load_cmapss_file(TEST_FILE)

    # 2) Add RUL labels to training data
    print("Adding RUL labels to training data...")
    train_df = add_rul_labels(train_df)

    # 3) Load ground-truth RUL for test set
    print("Loading ground-truth RUL for test set...")
    test_rul_true = load_rul_file(RUL_FILE)  # one value per test unit

    # 4) Define feature columns (operating settings + sensors)
    feature_cols = ["os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    input_dim = len(feature_cols)

    # 5) Scale features
    print("Scaling features...")
    train_df, test_df, scaler = scale_features(train_df, test_df, feature_cols)

    # 6) Train/validation split by unit ID
    all_units = train_df["unit"].unique()
    train_units, val_units = train_test_split(
        all_units, test_size=VAL_SPLIT, random_state=SEED
    )

    train_df_split = train_df[train_df["unit"].isin(train_units)].copy()
    val_df_split = train_df[train_df["unit"].isin(val_units)].copy()

    # 7) Create sequences
    print("Creating sequences...")
    X_train, y_train = make_sequences(
        train_df_split, feature_cols, SEQ_LEN, use_rul=True
    )
    X_val, y_val = make_sequences(
        val_df_split, feature_cols, SEQ_LEN, use_rul=True
    )

    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    # 8) Build DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # 9) Initialize GPT-2 model for time-series regression
    print("Initializing GPT-2 model (fine-tuning)...")
    model = TimeSeriesGPT2(input_dim=input_dim, seq_len=SEQ_LEN, gpt2_name="gpt2")
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 10) Train for 1 epoch
    print("Starting training (1 epoch)...")
    for epoch in range(1, EPOCHS + 1):
        train_mse, train_rmse = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_rmse, val_mae, _, _ = evaluate(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"- Train RMSE: {train_rmse:.4f} "
            f"| Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}"
        )

    # 11) Final evaluation on validation set (RMSE & MAE)
    print("Final evaluation on validation set...")
    val_rmse, val_mae, _, _ = evaluate(model, val_loader, DEVICE)
    print(f"[VAL RESULT] RMSE: {val_rmse:.4f}")
    print(f"[VAL RESULT] MAE:  {val_mae:.4f}")

    # 12) FLOPs and parameter count
    print("Estimating FLOPs and parameter count for one forward pass...")
    flops, params = estimate_flops_and_params(model, SEQ_LEN, input_dim, DEVICE)
    if flops is not None:
        print(f"[FLOPs] Approximate FLOPs per forward pass: {flops:.3e}")
        print(f"[PARAMS] Number of parameters: {params:.3e}")

    # 13) Prepare test sequences and evaluate on test set using RUL_FD002
    print("Preparing test sequences and computing test RMSE/MAE...")
    X_test, test_units = make_sequences(test_df, feature_cols, SEQ_LEN, use_rul=False)
    X_test_tensor = torch.from_numpy(X_test).to(DEVICE)

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor).cpu().numpy()

    # Align test ground-truth RUL with test units:
    # FD002 RUL file is assumed to be in ascending unit_id order (1..num_units).
    # groupby in make_sequences also yields units in ascending order by default.
    if len(test_rul_true) != len(test_units):
        print(
            f"WARNING: Number of test units in data ({len(test_units)}) "
            f"!= number of rows in RUL file ({len(test_rul_true)}). "
            f"Will align by min length."
        )
    n_common = min(len(test_rul_true), len(test_units), len(test_preds))
    test_rul_true_aligned = test_rul_true[:n_common]
    test_preds_aligned = test_preds[:n_common]

    test_rmse = math.sqrt(mean_squared_error(test_rul_true_aligned, test_preds_aligned))
    test_mae = mean_absolute_error(test_rul_true_aligned, test_preds_aligned)

    print(f"[TEST RESULT] RMSE: {test_rmse:.4f}")
    print(f"[TEST RESULT] MAE:  {test_mae:.4f}")

    # 14) Save test predictions to CSV
    pred_df = pd.DataFrame(
        {
            "unit": test_units[:n_common],
            "predicted_RUL": test_preds_aligned,
            "true_RUL": test_rul_true_aligned,
        }
    )
    pred_df = pred_df.sort_values("unit")
    out_file = os.path.join(DATA_DIR, f"gpt2_predictions_{DATASET_NAME}.csv")
    pred_df.to_csv(out_file, index=False)

    print(f"Saved GPT-2 RUL predictions for test set to: {out_file}")
    print("Done.")


if __name__ == "__main__":
    main()
