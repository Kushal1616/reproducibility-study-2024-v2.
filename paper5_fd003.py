#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5-0.5B-style model for RUL prediction on NASA CMAPSS FD003.

Architecture:
- Tokenization of time series via fixed segments (AutoTimes-style).
- Segment embedding → Qwen hidden dim.
- Learnable token-wise position embeddings.
- Frozen Qwen2.5-0.5B Transformer used as a feature extractor.
- Regression head predicts RUL.

Training: 1 epoch, AdamW, MSE.
Evaluation: RMSE and MAE for val + test.

Expected files in SAME directory:
    train_FD003.txt
    test_FD003.txt
    RUL_FD003.txt

Output:
    qwen25_fd003_predictions.csv
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

from transformers import AutoModel

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
DATA_DIR = "."
DATASET_NAME = "FD003"

TRAIN_FILE = os.path.join(DATA_DIR, f"train_{DATASET_NAME}.txt")
TEST_FILE  = os.path.join(DATA_DIR, f"test_{DATASET_NAME}.txt")
RUL_FILE   = os.path.join(DATA_DIR, f"RUL_{DATASET_NAME}.txt")

QWEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B" 

SEQ_LEN_RAW  = 60
SEGMENT_LEN  = 10
BATCH_SIZE   = 16
EPOCHS       = 1
LR           = 1e-4
VAL_SPLIT    = 0.2

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
# Data loading
# --------------------------------------------------------------------
def load_cmapss_fd003(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] == 27:
        df = df.iloc[:, :-1]

    cols = ["unit", "time", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    df.columns = cols
    return df

def add_rul_labels(train_df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = train_df.groupby("unit")["time"].max().reset_index()
    max_cycle.columns = ["unit", "max_time"]
    df = train_df.merge(max_cycle, on="unit")
    df["RUL"] = df["max_time"] - df["time"]
    return df.drop(columns=["max_time"])

def load_rul_file(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    return df.iloc[:, 0].values.astype(np.float32)

def scale_features(train_df, test_df, feature_cols):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled  = scaler.transform(test_df[feature_cols])
    train_df_scaled = train_df.copy()
    test_df_scaled  = test_df.copy()
    train_df_scaled[feature_cols] = train_scaled
    test_df_scaled[feature_cols]  = test_scaled
    return train_df_scaled, test_df_scaled

# --------------------------------------------------------------------
# Sequence builder (same logic as FD001/FD002)
# --------------------------------------------------------------------
def make_sequences_fd003(df, feature_cols, seq_len_raw, use_rul=True):
    X_list, y_list, unit_ids = [], [], []

    for uid, group in df.groupby("unit"):
        group = group.sort_values("time")
        data = group[feature_cols].values

        if use_rul:
            rul = group["RUL"].values
            if len(group) >= seq_len_raw:
                for i in range(len(group) - seq_len_raw + 1):
                    X_list.append(data[i:i+seq_len_raw])
                    y_list.append(rul[i + seq_len_raw - 1])
        else:
            if len(group) >= seq_len_raw:
                seq = data[-seq_len_raw:]
            else:
                pad = seq_len_raw - len(group)
                seq = np.vstack([np.repeat(data[0][None, :], pad, axis=0), data])
            X_list.append(seq)
            unit_ids.append(uid)

    X = np.array(X_list, dtype=np.float32)
    if use_rul:
        return X, np.array(y_list, dtype=np.float32)
    return X, np.array(unit_ids)

# --------------------------------------------------------------------
# Qwen2.5 AutoTimes model
# --------------------------------------------------------------------
class QwenAutoTimesRUL(nn.Module):
    def __init__(self, input_dim, seq_len_raw, segment_len, model_name=QWEN_MODEL_NAME):
        super().__init__()
        assert seq_len_raw % segment_len == 0
        self.seq_len_raw  = seq_len_raw
        self.segment_len  = segment_len
        self.num_tokens   = seq_len_raw // segment_len

        self.qwen = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden = self.qwen.config.hidden_size

        for p in self.qwen.parameters():
            p.requires_grad = False

        self.segment_embed = nn.Sequential(
            nn.Linear(segment_len * input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden)
        )
        self.pos_emb = nn.Embedding(self.num_tokens, hidden)
        self.reg_head = nn.Linear(hidden, 1)

    def forward(self, x):
        B, T, F = x.shape
        x = x.view(B, self.num_tokens, self.segment_len, F)
        x = x.view(B, self.num_tokens, -1)

        seg = self.segment_embed(x)

        pos = self.pos_emb(
            torch.arange(self.num_tokens, device=x.device)
            .unsqueeze(0).expand(B, -1)
        )

        embeds = seg + pos
        out = self.qwen(inputs_embeds=embeds).last_hidden_state
        last = out[:, -1, :]
        return self.reg_head(last).squeeze(-1)

# --------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------
def train_one_epoch(model, loader, opt, device):
    model.train()
    mse_loss = nn.MSELoss()
    total = 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(Xb)
        loss = mse_loss(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * Xb.size(0)

    mse = total / len(loader.dataset)
    return mse, math.sqrt(mse)

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mse = mean_squared_error(trues, preds)
    return math.sqrt(mse), mean_absolute_error(trues, preds), preds, trues

# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print("Loading FD003...")

    train_df = load_cmapss_fd003(TRAIN_FILE)
    test_df  = load_cmapss_fd003(TEST_FILE)

    train_df = add_rul_labels(train_df)
    test_rul = load_rul_file(RUL_FILE)

    feature_cols = ["os1","os2","os3"] + [f"s{i}" for i in range(1,22)]
    input_dim = len(feature_cols)

    train_df, test_df = scale_features(train_df, test_df, feature_cols)

    units = train_df["unit"].unique()
    train_units, val_units = train_test_split(units, test_size=VAL_SPLIT, random_state=SEED)

    X_train, y_train = make_sequences_fd003(train_df[train_df.unit.isin(train_units)], feature_cols, SEQ_LEN_RAW)
    X_val, y_val     = make_sequences_fd003(train_df[train_df.unit.isin(val_units)],   feature_cols, SEQ_LEN_RAW)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                              batch_size=BATCH_SIZE)

    model = QwenAutoTimesRUL(input_dim, SEQ_LEN_RAW, SEGMENT_LEN).to(DEVICE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    print("Training...")
    for e in range(1, EPOCHS+1):
        mse, rmse = train_one_epoch(model, train_loader, opt, DEVICE)
        val_rmse, val_mae, _, _ = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {e}: Train RMSE={rmse:.4f} | Val RMSE={val_rmse:.4f} | Val MAE={val_mae:.4f}")

    print("Preparing test sequences...")
    X_test, test_units = make_sequences_fd003(test_df, feature_cols, SEQ_LEN_RAW, use_rul=False)
    X_test_t = torch.from_numpy(X_test).to(DEVICE)

    print("Testing...")
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()

    n = min(len(preds), len(test_rul))
    rmse = math.sqrt(mean_squared_error(test_rul[:n], preds[:n]))
    mae  = mean_absolute_error(test_rul[:n], preds[:n])

    print(f"[TEST] RMSE: {rmse:.4f}")
    print(f"[TEST] MAE : {mae:.4f}")

    out = pd.DataFrame({
        "unit": test_units[:n],
        "predicted_RUL": preds[:n],
        "true_RUL": test_rul[:n]
    })
    out.to_csv("qwen25_fd003_predictions.csv", index=False)
    print("Saved predictions to qwen25_fd003_predictions.csv")

if __name__ == "__main__":
    main()
