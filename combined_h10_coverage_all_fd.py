import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================================
# Combined H=10 multi-horizon coverage script for FD001-004
# Coverage is computed using horizon-wise split-conformal
# intervals calibrated on a validation split.
# =========================================================

WINDOW_SIZE = 30
HORIZON = 10
BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 2
MAX_RUL_CAP = 125
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Nominal coverage = 1 - ALPHA
ALPHA = 0.10
VAL_RATIO = 0.20

DATASETS = {
    "FD001": {
        "train_path": "FD001/train_FD001.txt",
        "test_path": "FD001/test_FD001.txt",
        "rul_path": "FD001/RUL_FD001.txt",
    },
    "FD002": {
        "train_path": "FD002/train_FD002.txt",
        "test_path": "FD002/test_FD002.txt",
        "rul_path": "FD002/RUL_FD002.txt",
    },
    "FD003": {
        "train_path": "FD003/train_FD003.txt",
        "test_path": "FD003/test_FD003.txt",
        "rul_path": "FD003/RUL_FD003.txt",
    },
    "FD004": {
        "train_path": "FD004/train_FD004.txt",
        "test_path": "FD004/test_FD004.txt",
        "rul_path": "FD004/RUL_FD004.txt",
    },
}

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

cols = ["unit_id", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]


def load_cmapss_file(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = cols
    return df


def add_train_rul(df, cap=125):
    df = df.copy()
    max_cycle = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycle, on="unit_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df.drop(columns=["max_cycle"])


def add_test_rul(df, rul_offsets, cap=125):
    df = df.copy()
    max_cycle = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycle, on="unit_id", how="left")
    offset_map = {i + 1: int(rul_offsets.iloc[i, 0]) for i in range(len(rul_offsets))}
    df["rul_offset"] = df["unit_id"].map(offset_map)
    df["full_failure_cycle"] = df["max_cycle"] + df["rul_offset"]
    df["RUL"] = df["full_failure_cycle"] - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df.drop(columns=["max_cycle", "rul_offset", "full_failure_cycle"])


def split_train_val_by_unit(df, val_ratio=0.20, seed=42):
    unit_ids = sorted(df["unit_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = np.array(unit_ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    val_units = set(shuffled[:n_val].tolist())
    train_units = set(shuffled[n_val:].tolist())
    train_df = df[df["unit_id"].isin(train_units)].copy().reset_index(drop=True)
    val_df = df[df["unit_id"].isin(val_units)].copy().reset_index(drop=True)
    return train_df, val_df


def build_multihorizon_windows(df, feature_cols, window_size=30, horizon=10):
    X_list, Y_list = [], []
    for _, g in df.groupby("unit_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        feats = g[feature_cols].values.astype(np.float32)
        rul = g["RUL"].values.astype(np.float32)
        for end_idx in range(window_size - 1, len(g) - horizon):
            start_idx = end_idx - window_size + 1
            x = feats[start_idx:end_idx + 1]
            y = rul[end_idx + 1:end_idx + 1 + horizon]
            X_list.append(x)
            Y_list.append(y)
    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class LSTMMultiHorizon(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


def predict_model(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())
    return np.vstack(preds), np.vstack(trues)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def horizonwise_metrics(trues, preds):
    mae_per_h = []
    rmse_per_h = []
    for h in range(trues.shape[1]):
        mae_per_h.append(float(mean_absolute_error(trues[:, h], preds[:, h])))
        rmse_per_h.append(rmse(trues[:, h], preds[:, h]))
    return mae_per_h, rmse_per_h


def conformal_widths(val_trues, val_preds, alpha=0.10):
    widths = []
    n = val_trues.shape[0]
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    for h in range(val_trues.shape[1]):
        abs_res = np.abs(val_trues[:, h] - val_preds[:, h])
        try:
            q = float(np.quantile(abs_res, q_level, method="higher"))
        except TypeError:
            q = float(np.quantile(abs_res, q_level, interpolation="higher"))
        widths.append(q)
    return np.array(widths, dtype=np.float32)


def coverage_from_widths(test_trues, test_preds, widths):
    lower = test_preds - widths.reshape(1, -1)
    upper = test_preds + widths.reshape(1, -1)
    covered = ((test_trues >= lower) & (test_trues <= upper)).astype(np.float32)
    coverage_per_h = covered.mean(axis=0).tolist()
    avg_coverage = float(np.mean(coverage_per_h))
    return coverage_per_h, avg_coverage, lower, upper


def run_one_dataset(name, cfg):
    print("\n" + "=" * 80)
    print(f"Running {name}")
    print("=" * 80)

    train_df = load_cmapss_file(cfg["train_path"])
    test_df = load_cmapss_file(cfg["test_path"])
    rul_df = pd.read_csv(cfg["rul_path"], sep=r"\s+", header=None, names=["rul_offset"])

    train_df = add_train_rul(train_df, MAX_RUL_CAP)
    test_df = add_test_rul(test_df, rul_df, MAX_RUL_CAP)

    train_sub_df, val_df = split_train_val_by_unit(train_df, val_ratio=VAL_RATIO, seed=SEED)

    feature_cols = [c for c in train_sub_df.columns if c not in ["unit_id", "cycle", "RUL"]]
    stds = train_sub_df[feature_cols].std()
    feature_cols = [c for c in feature_cols if stds[c] > 1e-8]

    train_sub_df[feature_cols] = train_sub_df[feature_cols].astype(np.float32)
    val_df[feature_cols] = val_df[feature_cols].astype(np.float32)
    test_df[feature_cols] = test_df[feature_cols].astype(np.float32)

    scaler = StandardScaler()
    train_sub_df[feature_cols] = scaler.fit_transform(train_sub_df[feature_cols]).astype(np.float32)
    val_df[feature_cols] = scaler.transform(val_df[feature_cols]).astype(np.float32)
    test_df[feature_cols] = scaler.transform(test_df[feature_cols]).astype(np.float32)

    X_train, Y_train = build_multihorizon_windows(train_sub_df, feature_cols, WINDOW_SIZE, HORIZON)
    X_val, Y_val = build_multihorizon_windows(val_df, feature_cols, WINDOW_SIZE, HORIZON)
    X_test, Y_test = build_multihorizon_windows(test_df, feature_cols, WINDOW_SIZE, HORIZON)

    print(f"Train windows: {X_train.shape}, {Y_train.shape}")
    print(f"Val   windows: {X_val.shape}, {Y_val.shape}")
    print(f"Test  windows: {X_test.shape}, {Y_test.shape}")

    train_loader = DataLoader(SeqDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SeqDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMMultiHorizon(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        horizon=HORIZON,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"{name} | Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {avg_loss:.6f}")

    val_preds, val_trues = predict_model(model, val_loader, DEVICE)
    test_preds, test_trues = predict_model(model, test_loader, DEVICE)

    mae_per_h, rmse_per_h = horizonwise_metrics(test_trues, test_preds)
    avg_mae = float(np.mean(mae_per_h))
    avg_rmse = float(np.mean(rmse_per_h))
    final_h_rmse = float(rmse_per_h[-1])

    widths = conformal_widths(val_trues, val_preds, alpha=ALPHA)
    coverage_per_h, avg_coverage, lower, upper = coverage_from_widths(test_trues, test_preds, widths)
    final_h_coverage = float(coverage_per_h[-1])

    print(f"\n{name} | Multi-Horizon Results (H={HORIZON})")
    for h in range(HORIZON):
        print(
            f"Horizon {h + 1:02d} | "
            f"MAE: {mae_per_h[h]:.4f} | "
            f"RMSE: {rmse_per_h[h]:.4f} | "
            f"Coverage: {coverage_per_h[h]:.4f} | "
            f"Width: {widths[h]:.4f}"
        )

    print(f"\n{name} | Average MAE        : {avg_mae:.4f}")
    print(f"{name} | Average RMSE       : {avg_rmse:.4f}")
    print(f"{name} | Final-Horizon RMSE : {final_h_rmse:.4f}")
    print(f"{name} | Average Coverage   : {avg_coverage:.4f}")
    print(f"{name} | Final-H Coverage   : {final_h_coverage:.4f}")

    results = {
        "dataset": name,
        "nominal_coverage": 1.0 - ALPHA,
        "avg_mae": avg_mae,
        "avg_rmse": avg_rmse,
        "final_h_rmse": final_h_rmse,
        "avg_coverage": avg_coverage,
        "final_h_coverage": final_h_coverage,
    }

    for h in range(HORIZON):
        results[f"h{h + 1}_mae"] = mae_per_h[h]
        results[f"h{h + 1}_rmse"] = rmse_per_h[h]
        results[f"h{h + 1}_coverage"] = coverage_per_h[h]
        results[f"h{h + 1}_width"] = float(widths[h])

    os.makedirs("coverage_outputs_h10", exist_ok=True)
    np.save(f"coverage_outputs_h10/{name}_test_preds.npy", test_preds)
    np.save(f"coverage_outputs_h10/{name}_test_trues.npy", test_trues)
    np.save(f"coverage_outputs_h10/{name}_lower.npy", lower)
    np.save(f"coverage_outputs_h10/{name}_upper.npy", upper)
    np.save(f"coverage_outputs_h10/{name}_widths.npy", widths)

    return results


def main():
    all_results = []
    for name, cfg in DATASETS.items():
        result = run_one_dataset(name, cfg)
        all_results.append(result)

    results_df = pd.DataFrame(all_results)

    ordered_cols = [
        "dataset",
        "nominal_coverage",
        "avg_mae",
        "avg_rmse",
        "final_h_rmse",
        "avg_coverage",
        "final_h_coverage",
    ]
    for h in range(1, HORIZON + 1):
        ordered_cols += [f"h{h}_rmse", f"h{h}_coverage", f"h{h}_width"]

    results_df = results_df[ordered_cols]

    os.makedirs("coverage_outputs_h10", exist_ok=True)
    out_csv = "coverage_outputs_h10/coverage_summary_h10_all_datasets.csv"
    results_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\nSaved summary to {out_csv}")


if __name__ == "__main__":
    main()
