#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AutoTimes-style model for RUL prediction on NASA CMAPSS FD002.

This script adapts the core ideas of AutoTimes:
- Segment-based "tokens" along the time axis
- SegmentEmbedding MLP that maps each segment into the LLM latent space
- Frozen GPT-2 decoder-only backbone
- Trainable segment embedding, position embeddings, and regression head

Here we adapt the architecture to regress RUL (scalar) instead of forecasting.

Files expected in the SAME folder as this script:
    train_FD002.txt
    test_FD002.txt
    RUL_FD002.txt

Outputs:
- Prints RMSE & MAE on validation and test sets.
- Saves test predictions to `autotimes_fd002_predictions.csv`.
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

from transformers import GPT2Model

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
DATA_DIR = "."   # same folder as data files and this script
DATASET_NAME = "FD002"

TRAIN_FILE = os.path.join(DATA_DIR, f"train_{DATASET_NAME}.txt")
TEST_FILE  = os.path.join(DATA_DIR, f"test_{DATASET_NAME}.txt")
RUL_FILE   = os.path.join(DATA_DIR, f"RUL_{DATASET_NAME}.txt")

# GPT-2 backbone
GPT2_BACKBONE = "gpt2"   # small GPT-2

# Time-series specifics
SEQ_LEN_RAW  = 60         # history window (time steps)
SEGMENT_LEN  = 10         # segment length along time axis (token length)
BATCH_SIZE   = 16
EPOCHS       = 1          # as requested
LR           = 1e-4
VAL_SPLIT    = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42

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
def load_cmapss_fd002(path: str) -> pd.DataFrame:
    """
    Load CMAPSS FD002 train/test file.

    Format:
        col 1: unit (engine id)
        col 2: time (cycle)
        col 3-5: operating settings 1-3
        col 6-26: sensor 1-21
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Some distributions have a trailing empty column
    if df.shape[1] == 27:
        df = df.iloc[:, :-1]

    cols = ["unit", "time", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    df.columns = cols
    return df


def add_rul_labels(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    RUL = max_time(unit) - current_time
    """
    max_cycle = train_df.groupby("unit")["time"].max().reset_index()
    max_cycle.columns = ["unit", "max_time"]
    df = train_df.merge(max_cycle, on="unit", how="left")
    df["RUL"] = df["max_time"] - df["time"]
    df.drop(columns=["max_time"], inplace=True)
    return df


def load_rul_file(path: str) -> np.ndarray:
    """
    Load RUL_FD002.txt -> array of RUL values, one per test engine.
    """
    rul_df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    rul = rul_df.iloc[:, 0].values.astype(np.float32)
    return rul


def scale_features(train_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   feature_cols):
    """
    Standardize features with StandardScaler.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled  = scaler.transform(test_df[feature_cols])

    train_df_scaled = train_df.copy()
    test_df_scaled  = test_df.copy()
    train_df_scaled[feature_cols] = train_scaled
    test_df_scaled[feature_cols]  = test_scaled
    return train_df_scaled, test_df_scaled, scaler


def make_sequences_fd002(df: pd.DataFrame,
                         feature_cols,
                         seq_len_raw: int,
                         use_rul: bool = True):
    """
    Build sequences of fixed length for FD002.

    If use_rul=True (train/val):
        returns X (num_seq, seq_len_raw, num_features), y (num_seq,)
        Sliding windows; label is RUL at last time step.

    If use_rul=False (test):
        returns X (num_units, seq_len_raw, num_features), unit_ids
        Use last seq_len_raw cycles (pad at start if too short).
    """
    X_list = []
    y_list = []
    unit_ids = []

    for uid, group in df.groupby("unit"):
        group = group.sort_values("time")
        data = group[feature_cols].values

        if use_rul:
            rul = group["RUL"].values
            if len(group) >= seq_len_raw:
                for i in range(len(group) - seq_len_raw + 1):
                    X_list.append(data[i: i + seq_len_raw])
                    y_list.append(rul[i + seq_len_raw - 1])
        else:
            # take last seq_len_raw cycles; pad at start if needed
            if len(group) >= seq_len_raw:
                seq = data[-seq_len_raw:]
            else:
                pad_len   = seq_len_raw - len(group)
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

# --------------------------------------------------------------------
# AutoTimes-style model
# --------------------------------------------------------------------
class AutoTimesRUL(nn.Module):
    """
    AutoTimes-style model adapted for scalar RUL regression.

    - Segment-based tokens along time axis:
        Each token is a contiguous segment of length SEGMENT_LEN over all features.
    - SegmentEmbedding (MLP) maps each segment to GPT-2 hidden dimension.
    - Learnable position embeddings per token index.
    - Frozen GPT-2 backbone processes the token sequence.
    - Last token representation -> regression head -> RUL.
    """

    def __init__(self,
                 input_dim: int,
                 seq_len_raw: int,
                 segment_len: int,
                 gpt2_name: str = GPT2_BACKBONE):
        super().__init__()
        assert seq_len_raw % segment_len == 0, \
            "SEQ_LEN_RAW must be divisible by SEGMENT_LEN"

        self.seq_len_raw  = seq_len_raw
        self.segment_len  = segment_len
        self.num_tokens   = seq_len_raw // segment_len
        self.input_dim    = input_dim

        # Load GPT-2 and freeze it
        self.gpt2 = GPT2Model.from_pretrained(gpt2_name)
        hidden_size = self.gpt2.config.n_embd

        for p in self.gpt2.parameters():
            p.requires_grad = False

        # SegmentEmbedding: flatten segment (segment_len * features) to hidden_size
        seg_input_dim = segment_len * input_dim
        self.segment_embed = nn.Sequential(
            nn.Linear(seg_input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Learnable position embeddings for tokens
        self.pos_emb = nn.Embedding(self.num_tokens, hidden_size)

        # Regression head from last token representation to scalar RUL
        self.reg_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T_raw, F) where T_raw = seq_len_raw, F = input_dim
        """
        B, T, F = x.shape
        assert T == self.seq_len_raw

        # Reshape into tokens: (B, num_tokens, segment_len, F)
        x = x.view(B, self.num_tokens, self.segment_len, F)
        # Flatten segment: (B, num_tokens, segment_len * F)
        x = x.reshape(B, self.num_tokens, self.segment_len * F)

        # Segment embeddings
        seg_emb = self.segment_embed(x)  # (B, num_tokens, H)

        # Position indices [0..num_tokens-1]
        token_ids = torch.arange(self.num_tokens, device=x.device)\
                       .unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_emb(token_ids)  # (B, num_tokens, H)

        # Token embeddings = segment + position
        token_embeds = seg_emb + pos_emb  # (B, num_tokens, H)

        # Feed into GPT-2 via inputs_embeds (backbone is frozen)
        outputs = self.gpt2(inputs_embeds=token_embeds)
        last_hidden = outputs.last_hidden_state  # (B, num_tokens, H)

        # Take last token representation
        pooled = last_hidden[:, -1, :]           # (B, H)

        # Regression
        y_hat = self.reg_head(pooled).squeeze(-1)  # (B,)
        return y_hat

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
    print("Loading FD002 train/test data...")
    train_df = load_cmapss_fd002(TRAIN_FILE)
    test_df  = load_cmapss_fd002(TEST_FILE)

    print("Adding RUL labels to training data...")
    train_df = add_rul_labels(train_df)

    print("Loading ground-truth RUL for test set...")
    test_rul_true = load_rul_file(RUL_FILE)

    # Features: operating settings + 21 sensors
    feature_cols = ["os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    input_dim = len(feature_cols)

    print("Scaling features...")
    train_df, test_df, _ = scale_features(train_df, test_df, feature_cols)

    # Split engines into train/val
    all_units = train_df["unit"].unique()
    train_units, val_units = train_test_split(
        all_units, test_size=VAL_SPLIT, random_state=SEED
    )

    train_df_split = train_df[train_df["unit"].isin(train_units)].copy()
    val_df_split   = train_df[train_df["unit"].isin(val_units)].copy()

    print("Creating sequences...")
    X_train, y_train = make_sequences_fd002(
        train_df_split, feature_cols, SEQ_LEN_RAW, use_rul=True
    )
    X_val, y_val = make_sequences_fd002(
        val_df_split, feature_cols, SEQ_LEN_RAW, use_rul=True
    )

    print(f"Train sequences: {X_train.shape}")
    print(f"Val sequences:   {X_val.shape}")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing AutoTimes-style model with GPT-2 backbone...")
    model = AutoTimesRUL(
        input_dim=input_dim,
        seq_len_raw=SEQ_LEN_RAW,
        segment_len=SEGMENT_LEN,
        gpt2_name=GPT2_BACKBONE,
    )
    model.to(DEVICE)

    # Only segment embedding, position embedding, and regression head are trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: "
          f"{sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

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

    # --------- Test evaluation (per engine) ----------
    print("Preparing test sequences...")
    X_test, test_units = make_sequences_fd002(
        test_df, feature_cols, SEQ_LEN_RAW, use_rul=False
    )
    X_test_tensor = torch.from_numpy(X_test).to(DEVICE)

    print("Predicting RUL on test set...")
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor).cpu().numpy()

    # Align with RUL_FD002 ordering (unit id ascending)
    if len(test_rul_true) != len(test_units):
        print(
            f"WARNING: test units ({len(test_units)}) != RUL entries ({len(test_rul_true)}). "
            "Aligning by min length."
        )
    n_common = min(len(test_rul_true), len(test_units), len(test_preds))
    test_rul_true_aligned = test_rul_true[:n_common]
    test_preds_aligned     = test_preds[:n_common]

    test_rmse = math.sqrt(mean_squared_error(test_rul_true_aligned, test_preds_aligned))
    test_mae  = mean_absolute_error(test_rul_true_aligned, test_preds_aligned)

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

    out_path = os.path.join(DATA_DIR, "autotimes_fd002_predictions.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved test predictions to: {out_path}")


if __name__ == "__main__":
    main()
