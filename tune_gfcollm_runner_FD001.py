import os
import json
import math
import copy
import random
import importlib
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_json(split_path):
    with open(split_path, "r") as f:
        return json.load(f)


def get_engine_id_col(df):
    for c in ["unit_id", "unit", "engine_id", "id"]:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find engine id column. Available columns: {df.columns.tolist()}")


def infer_p8_module(fd):
    return importlib.import_module(f"paper8_{fd.lower()}")


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
    train_df = p8.add_rul_labels(train_df)
    test_rul_true = p8.load_rul_file(rul_file)
    return train_df, test_df, test_rul_true


def get_feature_cols(train_df):
    useless_sensors = [1, 5, 6, 10, 16, 18, 19]
    sensor_cols = [f"s{i}" for i in range(1, 22) if f"s{i}" in train_df.columns and i not in useless_sensors]
    if sensor_cols:
        return sensor_cols
    sensor_cols = [f"sensor_{i}" for i in range(1, 22) if f"sensor_{i}" in train_df.columns and i not in useless_sensors]
    if sensor_cols:
        return sensor_cols
    raise ValueError(f"Could not infer feature columns from columns: {train_df.columns.tolist()}")


def split_train_val_by_engine(train_df, split_json):
    train_ids = set(split_json["train_engine_ids"])
    val_ids = set(split_json["val_engine_ids"])
    engine_col = get_engine_id_col(train_df)
    train_part = train_df[train_df[engine_col].isin(train_ids)].copy()
    val_part = train_df[train_df[engine_col].isin(val_ids)].copy()
    return train_part, val_part


def normalize_with_train_only(p8, train_part, val_part, test_df, feature_cols):
    train_norm, val_norm, _ = p8.normalize_features(train_part, val_part, feature_cols)
    _, test_norm, _ = p8.normalize_features(train_part, test_df, feature_cols)
    return train_norm, val_norm, test_norm


def create_windows_trainvaltest(p8, train_norm, val_norm, test_norm, test_rul_true, feature_cols, window_size):
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
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def evaluate_components(model, loader, device):
    model.eval()
    ys_s, ys_l, ys_c, ys_f, ys_true = [], [], [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            ys_s.append(out["y_s"].cpu().numpy())
            ys_l.append(out["y_l"].cpu().numpy())
            ys_c.append(out["y_c"].cpu().numpy())
            ys_f.append(out["y_fused"].cpu().numpy())
            ys_true.append(yb.numpy())

    y_true = np.concatenate(ys_true, axis=0)
    y_s = np.concatenate(ys_s, axis=0)
    y_l = np.concatenate(ys_l, axis=0)
    y_c = np.concatenate(ys_c, axis=0)
    y_f = np.concatenate(ys_f, axis=0)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    return {
        "s_rmse": rmse(y_true, y_s),
        "l_rmse": rmse(y_true, y_l),
        "c_rmse": rmse(y_true, y_c),
        "f_rmse": rmse(y_true, y_f),
        "f_mae": mae(y_true, y_f),
    }


def branch_error_correlation_loss(y_s, y_l, y_true):
    """
    Penalize correlation between branch errors.
    Lower is better; zero means more independent mistakes.
    """
    err_s = y_s - y_true
    err_l = y_l - y_true

    err_s = err_s - torch.mean(err_s)
    err_l = err_l - torch.mean(err_l)

    numerator = torch.mean(err_s * err_l)
    denominator = torch.std(err_s) * torch.std(err_l) + 1e-6
    corr = numerator / denominator
    return torch.abs(corr)




class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()
        self.backup = {}

    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup and param.requires_grad:
                param.data.copy_(self.backup[name].data)
        self.backup = {}
def train_gfcollm_with_schedule(
    p8,
    input_dim,
    train_loader,
    val_loader,
    device,
    lr,
    lambda_res,
    lambda_route,
    lambda_conf,
    max_epochs=50,
    patience=7,
):
    model = p8.GFCoLLM(input_dim=input_dim, window_size=p8.WINDOW_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ema = ModelEMA(model, decay=0.995)

    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()

    best_state = None
    best_val_rmse = float("inf")
    patience_ctr = 0

    print("TUNER V7: refine cap ±2 + EMA validation + locked lr=2e-4 + lambda_entropy=0.01 + lambda_no_harm=15")

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0

        # Warm-up: keep fusion head fixed while branches become trustworthy.
        fusion_trainable = epoch > 5
        for p in model.fusion.parameters():
            p.requires_grad = fusion_trainable

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            out = model(xb)

            y_s = out["y_s"]
            y_l = out["y_l"]
            y_c = out["y_c"]
            y_fused = out["y_fused"]
            feat_s = out["feat_s"]
            feat_l_proj = out["feat_l_proj"]
            residual_hat = out["residual_hat"]
            route_logits = out["route_logits"]
            conf_score = out["conf_score"]
            gate_weights = out["gate_weights"]
            gate_logits = out["gate_logits"]

            # Direct branch supervision
            loss_s = mse(y_s, yb)
            loss_l = mse(y_l, yb)
            loss_ortho = torch.mean(torch.abs(F.cosine_similarity(feat_s, feat_l_proj, dim=-1)))

            # Diversity penalty
            loss_diversity = branch_error_correlation_loss(y_s, y_l, yb)

            # Refinement losses
            loss_pred = mse(y_fused, yb)
            loss_res = mse(y_c, yb)
            loss_resid_refine = mse(residual_hat, (yb - y_l))

            with torch.no_grad():
                errs = torch.stack([
                    (y_s - yb) ** 2,
                    (y_l - yb) ** 2,
                    (y_c - yb) ** 2,
                ], dim=1)
                best_idx = torch.argmin(errs, dim=1)

            loss_route = ce(route_logits, best_idx)
            loss_conf = torch.mean((conf_score - torch.abs(y_fused - yb)) ** 2)

            branch_sq_errs = torch.stack([
                (y_s - yb) ** 2,
                (y_l - yb) ** 2,
                (y_c - yb) ** 2,
            ], dim=1)
            per_sample_err_fused = (y_fused - yb) ** 2
            per_sample_err_best_branch, best_branch_idx = torch.min(branch_sq_errs, dim=1)
            loss_no_harm = torch.mean(torch.relu(per_sample_err_fused - per_sample_err_best_branch))

            # Sharper selection over averaging.
            gate_entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=1).mean()
            loss_gate_supervision = ce(gate_logits, best_branch_idx)

            # Staged objective
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ema.update(model)
            total += loss.item() * xb.size(0)

        ema.apply_shadow(model)
        val_metrics = evaluate_components(model, val_loader, device)
        print(
            f"[Epoch {epoch:02d}] train_obj={math.sqrt(total/len(train_loader.dataset)):.4f} "
            f"val_fused_RMSE={val_metrics['f_rmse']:.4f} "
            f"(s={val_metrics['s_rmse']:.4f}, l={val_metrics['l_rmse']:.4f}, c={val_metrics['c_rmse']:.4f})"
        )

        if val_metrics["f_rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["f_rmse"]
            # Save the EMA-smoothed weights because validation is computed under EMA.
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
        ema.restore(model)

        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_val_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    fd = args.fd.upper()
    set_seed(args.seed)

    p8 = infer_p8_module(fd)
    device = getattr(p8, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    split_json = load_split_json(args.split_json)
    train_df, test_df, test_rul_true = load_raw_data_with_p8(p8, args.data_dir, fd)
    feature_cols = get_feature_cols(train_df)

    train_part, val_part = split_train_val_by_engine(train_df, split_json)
    train_norm, val_norm, test_norm = normalize_with_train_only(p8, train_part, val_part, test_df, feature_cols)

    window_size = getattr(p8, "WINDOW_SIZE", 50)
    batch_size = getattr(p8, "BATCH_SIZE", 64)

    X_train, y_train, X_val, y_val, X_test, y_test = create_windows_trainvaltest(
        p8, train_norm, val_norm, test_norm, test_rul_true, feature_cols, window_size
    )
    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size
    )

    input_dim = len(feature_cols)

    # Focus the search on the healthier region observed across seeds.
    configs = [
        {"lr": 2e-4, "lambda_res": 0.2, "lambda_route": 0.1, "lambda_conf": 0.1},
    ]

    rows = []
    best_val = float("inf")
    best_cfg = None

    for i, cfg in enumerate(configs, start=1):
        print("\n" + "=" * 90)
        print(f"Config {i}/{len(configs)}: {cfg}")

        model, best_val_rmse = train_gfcollm_with_schedule(
            p8,
            input_dim,
            train_loader,
            val_loader,
            device,
            lr=cfg["lr"],
            lambda_res=cfg["lambda_res"],
            lambda_route=cfg["lambda_route"],
            lambda_conf=cfg["lambda_conf"],
            max_epochs=args.max_epochs,
            patience=args.patience,
        )

        # The returned model already contains the best EMA-smoothed checkpoint.
        val_metrics = evaluate_components(model, val_loader, device)
        test_metrics = evaluate_components(model, test_loader, device)

        row = {
            "Dataset": fd,
            "Seed": args.seed,
            **cfg,
            "Val_s_RMSE": val_metrics["s_rmse"],
            "Val_l_RMSE": val_metrics["l_rmse"],
            "Val_c_RMSE": val_metrics["c_rmse"],
            "Val_fused_RMSE": val_metrics["f_rmse"],
            "Test_s_RMSE": test_metrics["s_rmse"],
            "Test_l_RMSE": test_metrics["l_rmse"],
            "Test_c_RMSE": test_metrics["c_rmse"],
            "Test_fused_RMSE": test_metrics["f_rmse"],
            "Test_fused_MAE": test_metrics["f_mae"],
        }
        rows.append(row)

        if val_metrics["f_rmse"] < best_val:
            best_val = val_metrics["f_rmse"]
            best_cfg = cfg.copy()

    df = pd.DataFrame(rows).sort_values("Val_fused_RMSE").reset_index(drop=True)
    out_csv = args.out_csv or f"{fd.lower()}_seed{args.seed}_gfcollm_tuning.csv"
    df.to_csv(out_csv, index=False)

    print("\nBest config by validation fused RMSE:")
    print(best_cfg)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()