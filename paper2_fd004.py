#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Llama-2-7B fine-tuning for RUL prediction on NASA CMAPSS FD004.

Files expected in the SAME folder:
    train_FD004.txt
    test_FD004.txt
    RUL_FD004.txt

What this script does:
- Load FD004 train/test, build RUL labels for train.
- Standardize operating settings + sensors.
- Build sliding-window time-series sequences (SEQ_LEN cycles).
- Fine-tune Llama-2-7B (as a time-series transformer) for 1 epoch.
- Report RMSE & MAE on validation and test sets.

NOTE:
- This is full fine-tuning of a 7B-parameter model. You will need a large GPU
  or adapt this to LoRA / 8-bit loading if you hit OOM.
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

from transformers import AutoConfig, AutoModel

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
DATA_DIR = "."   # same folder as data files and this script
DATASET_NAME = "FD004"

TRAIN_FILE = os.path.join(DATA_DIR, f"train_{DATASET_NAME}.txt")
TEST_FILE = os.path.join(DATA_DIR, f"test_{DATASET_NAME}.txt")
RUL_FILE = os.path.join(DATA_DIR, f"RUL_{DATASET_NAME}.txt")

# Llama-2-7B HF id (you must have access to this repo on Hugging Face)
LLAMA_MODEL_NAME = "meta-llama/Llama-2-7b-hf"

SEQ_LEN = 30
BATCH_SIZE = 8          # small, because model is huge
EPOCHS = 1              # as requested
LR = 1e-5               # conservative LR for big model
VAL_SPLIT = 0.2

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
def load_cmapss(path):
    """
    Load CMAPSS FD00X train/test file with standard column names.

    Format:
        1) unit
        2) time (cycle)
        3-5) operating settings 1-3
        6-26) sensor 1-21
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Some versions have an extra blank column at end
    if df.shape[1] == 27:
        df = df.iloc[:, :-1]

    cols = ["unit", "time", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    df.columns = cols
    return df


def add_rul_labels(train_df):
    """
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
    Load RUL_FD00X.txt -> np.array of RUL values, one per test engine.
    """
    rul_df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if rul_df.shape[1] > 1:
        rul_df = rul_df.iloc[:, 0]
    else:
        rul_df = rul_df.iloc[:, 0]
    return rul_df.values.astype(np.float32)


def scale_features(train_df, test_df, feature_cols):
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
    If use_rul=True (train/val):
        returns X (num_seq, seq_len, num_features), y (num_seq,)

    If use_rul=False (test):
        returns X (num_units, seq_len, num_features), unit_ids
    """
    X_list = []
    y_list = []
    unit_ids = []

    for uid, group in df.groupby("unit"):
        group = group.sort_values("time")
        data = group[feature_cols].values

        if use_rul:
            rul = group["RUL"].values
            if len(group) >= seq_len:
                for i in range(len(group) - seq_len + 1):
                    X_list.append(data[i : i + seq_len])
                    y_list.append(rul[i + seq_len - 1])
        else:
            # last seq_len cycles (pad at start if too short)
            if len(group) >= seq_len:
                seq = data[-seq_len:]
            else:
                pad_len = seq_len - len(group)
                pad_block = np.repeat(data[0][None, :], pad_len, axis=0)
                seq = np.vstack([pad_block, data])
            X_list.append(seq)
            unit_ids.append(uid)

    X = np.array(X_list, dtype=np.float32)

    if use_rul:
        y = np.array(y_list, dtype=np.float32)
        return X, y
    else:
        unit_ids = np.array(unit_ids, dtype=np.int32)
        return X, unit_ids

# -------------------------------------------------------------------
# Llama-2 time-series regression model
# --------------------------------------------------------------------
class TimeSeriesLlama2(nn.Module):
    """
    Use Llama-2-7B as a backbone transformer for time-series regression:

    - Project numeric feature vectors -> Llama hidden size.
    - Feed as inputs_embeds to LlamaModel (no tokenizer).
    - Pool last token and map to scalar RUL.
    """

    def __init__(self, input_dim, llama_name=LLAMA_MODEL_NAME):
        super().__init__()

        # Load config and model
        config = AutoConfig.from_pretrained(llama_name)
        self.backbone = AutoModel.from_pretrained(
            llama_name,
            config=config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        hidden_size = config.hidden_size

        self.feature_proj = nn.Linear(input_dim, hidden_size)
        self.reg_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        # For large models, it's often beneficial to keep backbone in float16
        orig_dtype = x.dtype
        x = x.to(self.feature_proj.weight.dtype)

        emb = self.feature_proj(x)  # (B, T, H)

        outputs = self.backbone(inputs_embeds=emb)
        last_hidden = outputs.last_hidden_state  # (B, T, H)
        pooled = last_hidden[:, -1, :]           # last time step

        y_hat = self.reg_head(pooled).squeeze(-1)

        return y_hat.to(orig_dtype)

# --------------------------------------------------------------------
# Training / evaluation helpers
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    print(f"Using device: {DEVICE}")
    print(f"Dataset: {DATASET_NAME}")
    print("Loading FD004 train/test data...")
    train_df = load_cmapss(TRAIN_FILE)
    test_df = load_cmapss(TEST_FILE)

    print("Adding RUL labels to training data...")
    train_df = add_rul_labels(train_df)

    print("Loading ground-truth RUL for test set...")
    test_rul_true = load_rul_file(RUL_FILE)

    feature_cols = ["os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    input_dim = len(feature_cols)

    print("Scaling features...")
    train_df, test_df, _ = scale_features(train_df, test_df, feature_cols)

    # Split by unit into train/val
    all_units = train_df["unit"].unique()
    train_units, val_units = train_test_split(
        all_units, test_size=VAL_SPLIT, random_state=SEED
    )

    train_df_split = train_df[train_df["unit"].isin(train_units)].copy()
    val_df_split = train_df[train_df["unit"].isin(val_units)].copy()

    print("Creating sequences...")
    X_train, y_train = make_sequences(train_df_split, feature_cols, SEQ_LEN, use_rul=True)
    X_val, y_val = make_sequences(val_df_split, feature_cols, SEQ_LEN, use_rul=True)

    print(f"Train sequences: {X_train.shape}")
    print(f"Val sequences:   {X_val.shape}")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing Llama-2-7B time-series model...")
    model = TimeSeriesLlama2(input_dim=input_dim)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    print("Starting training (1 epoch)...")
    for epoch in range(1, EPOCHS + 1):
        train_mse, train_rmse = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_rmse, val_mae, _, _ = evaluate(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"- Train RMSE: {train_rmse:.4f} "
            f"| Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}"
        )

    print("Final validation evaluation...")
    val_rmse, val_mae, _, _ = evaluate(model, val_loader, DEVICE)
    print(f"[VAL] RMSE: {val_rmse:.4f}")
    print(f"[VAL] MAE:  {val_mae:.4f}")

    # ------------- Test set evaluation -------------
    print("Preparing test sequences...")
    X_test, test_units = make_sequences(test_df, feature_cols, SEQ_LEN, use_rul=False)
    X_test_tensor = torch.from_numpy(X_test).to(DEVICE)

    print("Predicting on test set...")
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor).cpu().numpy()

    # Align with RUL_FD004 ordering (unit id ascending)
    if len(test_rul_true) != len(test_units):
        print(
            f"WARNING: test units ({len(test_units)}) != RUL entries ({len(test_rul_true)}). "
            "Aligning by min length."
        )
    n_common = min(len(test_rul_true), len(test_units), len(test_preds))
    test_rul_true_aligned = test_rul_true[:n_common]
    test_preds_aligned = test_preds[:n_common]

    test_rmse = math.sqrt(mean_squared_error(test_rul_true_aligned, test_preds_aligned))
    test_mae = mean_absolute_error(test_rul_true_aligned, test_preds_aligned)

    print(f"[TEST] RMSE: {test_rmse:.4f}")
    print(f"[TEST] MAE:  {test_mae:.4f}")

    # Save predictions
    out_df = pd.DataFrame(
        {
            "unit": test_units[:n_common],
            "predicted_RUL": test_preds_aligned,
            "true_RUL": test_rul_true_aligned,
        }
    ).sort_values("unit")

    out_path = os.path.join(DATA_DIR, f"llama2_7b_predictions_{DATASET_NAME}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved test predictions to: {out_path}")


if __name__ == "__main__":
    main()
