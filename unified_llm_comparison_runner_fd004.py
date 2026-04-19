#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Uniform seeded early-stop benchmark runner for FD004.

What this does
--------------
Runs a common split / preprocessing / early-stopping protocol for:

  1. GPT-2               (Paper1 FD001.py)
  2. Llama-2-7B          (paper2_fd001.py)
  3. OneFitsAll / FPT    (paper3_fd001.py)
  4. AutoTimes           (paper4_fd001.py)
  5. Qwen2.5-0.5B        (paper5_fd001.py)
  6. Qwen3-0.6B          (paper6_fd001.py)
  7. CoLLM-C             (paper7_fd001.py, if available)
  8. GF-CoLLM            (paper8_fd001.py, if available)

Outputs one CSV with:
  Dataset, Seed, Group, Model, Val_RMSE_best, Test_RMSE, Test_MAE, Runtime_sec, Status

Notes
-----
- This runner is designed for FD004 only.
- The six attached paper files are imported directly by filename.
- CoLLM-C / GF-CoLLM are treated as optional. If their modules are unavailable or do
  not expose the expected classes/functions, the runner writes a SKIPPED/ERROR row.
- For the six attached methods, early stopping is implemented uniformly in this file.
- Very large backbones (e.g., Llama / Qwen) may be slow or exceed memory limits.
"""

import os
import sys
import math
import json
import time
import copy
import random
import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------
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
    return df.drop(columns=["max_time"])


def load_rul_file(path: str) -> np.ndarray:
    rul_df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    return rul_df.iloc[:, 0].values.astype(np.float32)


def load_split_json(path: str) -> Tuple[List[int], List[int]]:
    with open(path, "r") as f:
        split = json.load(f)
    train_units = split.get("train_units", split.get("train_engine_ids"))
    val_units = split.get("val_units", split.get("val_engine_ids"))
    if train_units is None or val_units is None:
        raise KeyError(
            f"Split JSON must contain either "
            f"('train_units','val_units') or ('train_engine_ids','val_engine_ids'). "
            f"Found keys: {list(split.keys())}"
        )
    return train_units, val_units


def feature_cols_fd004(train_df: pd.DataFrame) -> List[str]:
    # Uniform feature set for FD004: operating settings + sensors
    return ["os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]


def fit_standardizer(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    mean = df[feature_cols].mean(axis=0)
    std = df[feature_cols].std(axis=0).replace(0, 1.0)
    return mean, std


def apply_standardizer(df: pd.DataFrame, feature_cols: List[str], mean, std) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = (out[feature_cols] - mean) / std
    return out


def create_train_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    unit_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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


def create_test_last_windows(
    test_df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    return np.array(X_list, dtype=np.float32), np.array(unit_ids, dtype=np.int32)


# ---------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------
def import_module_from_file(module_name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {filepath}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def maybe_import_by_name_or_file(module_name: str, filename: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        if os.path.exists(filename):
            return import_module_from_file(module_name, filename)
        raise


# ---------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------
def build_method_registry(root_dir: str) -> List[Dict]:
    return [
        {
            "name": "GPT-2",
            "group": "LLM",
            "module_name": "paper1_fd004_attached",
            "filepath": os.path.join(root_dir, "paper1_fd004.py"),
            "class_name": "TimeSeriesGPT2",
            "seq_len_attr": "SEQ_LEN",
            "default_seq_len": 30,
            "batch_attr": "BATCH_SIZE",
            "default_batch": 16,
            "lr_attr": "LR",
            "default_lr": 1e-4,
            "builder": lambda mod, input_dim, seq_len: mod.TimeSeriesGPT2(input_dim=input_dim, seq_len=seq_len),
        },
        {
            "name": "Llama-2-7B",
            "group": "LLM",
            "module_name": "paper2_fd004_attached",
            "filepath": os.path.join(root_dir, "paper2_fd004.py"),
            "class_name": "TimeSeriesLlama2",
            "seq_len_attr": "SEQ_LEN",
            "default_seq_len": 30,
            "batch_attr": "BATCH_SIZE",
            "default_batch": 8,
            "lr_attr": "LR",
            "default_lr": 1e-5,
            "builder": lambda mod, input_dim, seq_len: mod.TimeSeriesLlama2(input_dim=input_dim),
        },
        {
            "name": "OneFitsAll-FPT",
            "group": "LLM",
            "module_name": "paper3_fd004_attached",
            "filepath": os.path.join(root_dir, "paper3_fd004.py"),
            "class_name": "OneFitsAllFPT",
            "seq_len_attr": "SEQ_LEN_RAW",
            "default_seq_len": 60,
            "batch_attr": "BATCH_SIZE",
            "default_batch": 16,
            "lr_attr": "LR",
            "default_lr": 1e-4,
            "builder": lambda mod, input_dim, seq_len: mod.OneFitsAllFPT(
                input_dim=input_dim,
                seq_len_raw=seq_len,
                patch_len=getattr(mod, "PATCH_LEN", 10),
            ),
        },
        {
            "name": "AutoTimes",
            "group": "LLM",
            "module_name": "paper4_fd004_attached",
            "filepath": os.path.join(root_dir, "paper4_fd004.py"),
            "class_name": "AutoTimesRUL",
            "seq_len_attr": "SEQ_LEN_RAW",
            "default_seq_len": 60,
            "batch_attr": "BATCH_SIZE",
            "default_batch": 16,
            "lr_attr": "LR",
            "default_lr": 1e-4,
            "builder": lambda mod, input_dim, seq_len: mod.AutoTimesRUL(
                input_dim=input_dim,
                seq_len_raw=seq_len,
                segment_len=getattr(mod, "SEGMENT_LEN", 10),
            ),
        },
        {
            "name": "Qwen2.5-0.5B",
            "group": "LLM",
            "module_name": "paper5_fd004_attached",
            "filepath": os.path.join(root_dir, "paper5_fd004.py"),
            "class_name": "QwenAutoTimesRUL",
            "seq_len_attr": "SEQ_LEN_RAW",
            "default_seq_len": 60,
            "batch_attr": "BATCH_SIZE",
            "default_batch": 16,
            "lr_attr": "LR",
            "default_lr": 1e-4,
            "builder": lambda mod, input_dim, seq_len: mod.QwenAutoTimesRUL(
                input_dim=input_dim,
                seq_len_raw=seq_len,
                segment_len=getattr(mod, "SEGMENT_LEN", 10),
            ),
        },
        {
            "name": "Qwen3-0.6B",
            "group": "LLM",
            "module_name": "paper6_fd004_attached",
            "filepath": os.path.join(root_dir, "paper6_fd004.py"),
            "class_name": "Qwen3AutoTimesRUL",
            "seq_len_attr": "SEQ_LEN_RAW",
            "default_seq_len": 60,
            "batch_attr": "BATCH_SIZE",
            "default_batch": 16,
            "lr_attr": "LR",
            "default_lr": 1e-4,
            "builder": lambda mod, input_dim, seq_len: mod.Qwen3AutoTimesRUL(
                input_dim=input_dim,
                seq_len_raw=seq_len,
                segment_len=getattr(mod, "SEGMENT_LEN", 10),
            ),
        },
        {
            "name": "CoLLM-C",
            "group": "Collaborative/Large-model",
            "module_name": "paper7_fd004",
            "filepath": os.path.join(root_dir, "paper7_fd004.py"),
            "optional": True,
        },
        {
            "name": "GF-CoLLM",
            "group": "Proposed",
            "module_name": "paper8_fd004",
            "filepath": os.path.join(root_dir, "paper8_fd004.py"),
            "optional": True,
        },
    ]


# ---------------------------------------------------------------------
# Generic training / evaluation for papers 1-6
# ---------------------------------------------------------------------
def evaluate_regression(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            y_pred.append(pred)
            y_true.append(yb.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return rmse, mae


def train_generic_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    max_epochs: int,
    patience: int,
    grad_clip: float = 5.0,
) -> Tuple[nn.Module, float]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += loss.item() * xb.size(0)

        val_rmse, val_mae = evaluate_regression(model, val_loader, device)
        train_obj = math.sqrt(total / max(1, len(train_loader.dataset)))
        print(f"[Epoch {epoch:02d}] train_obj={train_obj:.4f} val_RMSE={val_rmse:.4f} val_MAE={val_mae:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# ---------------------------------------------------------------------
# CoLLM-C / GF-CoLLM helpers
# ---------------------------------------------------------------------
def run_collm_methods_with_existing_runner(
    root_dir: str,
    fd: str,
    data_dir: str,
    split_json: str,
    seed: int,
    max_epochs: int,
    patience: int,
) -> List[Dict]:
    """
    Try to leverage an existing comparison runner in the working folder, if available,
    to get CoLLM-C / GF-CoLLM rows under the same split.
    """
    candidates = [
        os.path.join(root_dir, "unified_earlystop_comparison_runner_fdaware_fixed.py"),
        os.path.join(root_dir, "unified_earlystop_comparison_runner_fdaware.py"),
        os.path.join(root_dir, "unified_earlystop_comparison_runner_fixed.py"),
        os.path.join(root_dir, "unified_earlystop_comparison_runner.py"),
        os.path.join(root_dir, "unified_earlystop_comparison_runner_explicit_modules.py"),
    ]
    found = None
    for c in candidates:
        if os.path.exists(c):
            found = c
            break

    if found is None:
        return []

    tmp_out = os.path.join(root_dir, f"_tmp_uniform_collm_{fd.lower()}_seed{seed}.csv")
    cmd = (
        f'"{sys.executable}" "{found}" --fd {fd} --data_dir "{data_dir}" '
        f'--split_json "{split_json}" --seed {seed} --max_epochs {max_epochs} '
        f'--patience {patience} --out_csv "{tmp_out}"'
    )
    print(f"[INFO] Running existing collaborative comparison runner:\n{cmd}")
    rc = os.system(cmd)
    if rc != 0 or not os.path.exists(tmp_out):
        print("[WARN] Collaborative runner failed or produced no CSV; skipping CoLLM-C / GF-CoLLM.")
        return []

    df = pd.read_csv(tmp_out)
    keep = df[df["Model"].isin(["CoLLM-C", "GF-CoLLM"])].copy()
    rows = keep.to_dict(orient="records")
    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=str, default="FD004")
    parser.add_argument("--data_dir", type=str, default="FD004")
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size_override", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="fd004_uniform_llm_comparison.csv")
    parser.add_argument("--root_dir", type=str, default=".")
    args = parser.parse_args()

    if args.fd.upper() != "FD004":
        raise ValueError("This uniform runner is intended for FD004 only.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_file = os.path.join(args.data_dir, "train_FD004.txt")
    test_file = os.path.join(args.data_dir, "test_FD004.txt")
    rul_file = os.path.join(args.data_dir, "RUL_FD004.txt")

    train_df = load_cmapss(train_file)
    test_df = load_cmapss(test_file)
    train_df = add_rul_labels(train_df)
    test_rul = load_rul_file(rul_file)

    train_units, val_units = load_split_json(args.split_json)

    feature_cols = feature_cols_fd004(train_df)
    mean, std = fit_standardizer(train_df[train_df["unit"].isin(train_units)], feature_cols)
    train_df = apply_standardizer(train_df, feature_cols, mean, std)
    test_df = apply_standardizer(test_df, feature_cols, mean, std)

    methods = build_method_registry(args.root_dir)
    rows: List[Dict] = []

    # First run attached methods 1-6 using common logic
    for m in methods[:6]:
        start = time.time()
        try:
            print("\n" + "=" * 90)
            print(f"Running: {m['name']}")

            mod = import_module_from_file(m["module_name"], m["filepath"])
            seq_len = int(getattr(mod, m["seq_len_attr"], m["default_seq_len"]))
            batch_size = int(getattr(mod, m["batch_attr"], m["default_batch"]))
            if args.batch_size_override > 0:
                batch_size = args.batch_size_override
            lr = float(getattr(mod, m["lr_attr"], m["default_lr"]))

            X_train, y_train = create_train_windows(train_df, feature_cols, seq_len, unit_ids=train_units)
            X_val, y_val = create_train_windows(train_df, feature_cols, seq_len, unit_ids=val_units)
            X_test, test_units = create_test_last_windows(test_df, feature_cols, seq_len)
            n_common = min(len(test_rul), len(test_units), len(X_test))
            X_test, y_test = X_test[:n_common], test_rul[:n_common]

            train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                                      batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                                    batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                                     batch_size=batch_size, shuffle=False)

            model = m["builder"](mod, len(feature_cols), seq_len)
            model, best_val = train_generic_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                lr=lr,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
            test_rmse, test_mae = evaluate_regression(model, test_loader, device)

            rows.append({
                "Dataset": "FD004",
                "Seed": args.seed,
                "Group": m["group"],
                "Model": m["name"],
                "Val_RMSE_best": best_val,
                "Test_RMSE": test_rmse,
                "Test_MAE": test_mae,
                "Runtime_sec": time.time() - start,
                "Status": "OK",
            })
        except Exception as e:
            rows.append({
                "Dataset": "FD004",
                "Seed": args.seed,
                "Group": m["group"],
                "Model": m["name"],
                "Val_RMSE_best": np.nan,
                "Test_RMSE": np.nan,
                "Test_MAE": np.nan,
                "Runtime_sec": time.time() - start,
                "Status": f"ERROR: {type(e).__name__}: {e}",
            })
            print(f"[ERROR] {m['name']} failed: {type(e).__name__}: {e}")

    # Then try to add CoLLM-C / GF-CoLLM using an existing collaborative runner
    collm_rows = run_collm_methods_with_existing_runner(
        root_dir=args.root_dir,
        fd="FD004",
        data_dir=args.data_dir,
        split_json=args.split_json,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    if collm_rows:
        for r in collm_rows:
            r["Status"] = "OK"
        rows.extend(collm_rows)
    else:
        for m in methods[6:]:
            rows.append({
                "Dataset": "FD004",
                "Seed": args.seed,
                "Group": m["group"],
                "Model": m["name"],
                "Val_RMSE_best": np.nan,
                "Test_RMSE": np.nan,
                "Test_MAE": np.nan,
                "Runtime_sec": 0.0,
                "Status": "SKIPPED: collaborative runner unavailable or failed",
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved comparison CSV to: {args.out_csv}")
    print("\nFinal table:")
    print(out_df[["Dataset", "Seed", "Group", "Model", "Val_RMSE_best", "Test_RMSE", "Test_MAE", "Runtime_sec", "Status"]])


if __name__ == "__main__":
    main()
