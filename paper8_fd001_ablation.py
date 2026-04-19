
"""
GF-CoLLM (FD001) — Primary Ablation Study Script (Edge-only vs Cloud-only vs Hybrid Edge–Cloud)

What this script provides (as requested):
- Labels normalized with MAX_RUL=125 at training time.
- Evaluation + plots inverse-scaled back to cycles.
- Conformal confidence computed from a calibration split (NOT min-max proxy).
- Large Model uses a genuinely pretrained backbone: GPT2Model.from_pretrained("gpt2") (frozen backbone).
- 100 epochs training (configurable), and an ablation runner that reports:
    latency, memory footprint, large-model call rate, RMSE/MAE (in cycles).

Expected NASA C-MAPSS files in `--data_dir`:
- train_FD001.txt
- test_FD001.txt
- RUL_FD001.txt

Run:
  python paper8_fd001_ablation.py --data_dir /path/to/CMAPSS --epochs 100

Notes:
- `transformers` is required for pretrained GPT-2. If unavailable, the script aborts by default.
  (You can override with --allow_unpretrained_lm, but that will not satisfy "genuinely pretrained".)
"""

from __future__ import annotations

import os
import time
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Optional imports
try:
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    raise ImportError("scikit-learn is required (StandardScaler). Please install scikit-learn.") from e

try:
    from transformers import GPT2Model, GPT2Config
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


# ----------------------------
# Configuration
# ----------------------------

MAX_RUL_DEFAULT = 125  # training-time normalization constant (requested)


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data / windowing
    seq_len: int = 30
    stride: int = 1
    batch_size: int = 256
    num_workers: int = 0  # safer default across OS/notebooks
    calib_fraction: float = 0.10  # conformal calibration split (requested)

    # Training
    epochs: int = 100
    lr_sm: float = 1e-3
    lr_lm_head: float = 5e-4
    weight_decay: float = 1e-4

    # Conformal
    alpha: float = 0.10  # 90% coverage; used for q_hat (and confidence p-value)
    # Routing
    conf_tau: float = 0.55  # if SM confidence is high, prefer edge
    w_conf: float = 0.60
    w_latency: float = 0.25
    w_memory: float = 0.15


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_size_mb(model: nn.Module) -> float:
    """Parameter-only footprint estimate (MB)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total / (1024 ** 2)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


# ----------------------------
# Data loading (NASA C-MAPSS)
# ----------------------------

NASA_COLS = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

def read_cmapss_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] < len(NASA_COLS):
        # Some files may have trailing spaces producing empty columns
        df = df.iloc[:, :len(NASA_COLS)]
    df.columns = NASA_COLS
    return df


def add_train_rul(df_train: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    """Train RUL is (max_cycle(unit) - cycle), clipped at max_rul."""
    df = df_train.copy()
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(lower=0, upper=max_rul)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def add_test_rul(df_test: pd.DataFrame, rul_txt_path: str, max_rul: int) -> pd.DataFrame:
    """
    Correct FD001 test RUL (NASA):
    For each unit u, NASA provides RUL at end-of-test: RUL_end[u].
    Then for each cycle t in that unit:
        RUL(t) = min(max_rul, RUL_end[u] + (max_cycle_test[u] - t))
    """
    df = df_test.copy()
    rul_end = pd.read_csv(rul_txt_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=np.int64)
    # units are 1-indexed in files; rul_end is 0-indexed array
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit")
    df["RUL_end"] = df["unit"].apply(lambda u: int(rul_end[int(u) - 1]))
    df["RUL"] = (df["RUL_end"] + (df["max_cycle"] - df["cycle"])).clip(lower=0, upper=max_rul)
    df.drop(columns=["max_cycle", "RUL_end"], inplace=True)
    return df


def make_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows per unit. Label is RUL at the last timestep in window.
    Returns:
      X: [N, seq_len, D]
      y: [N]
    """
    xs = []
    ys = []
    for unit_id, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        feat = g[feature_cols].to_numpy(dtype=np.float32)
        rul = g["RUL"].to_numpy(dtype=np.float32)
        T = len(g)
        for start in range(0, T - seq_len + 1, stride):
            end = start + seq_len
            xs.append(feat[start:end])
            ys.append(rul[end - 1])
    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.float32)
    return X, y


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(-1)  # (N,1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ----------------------------
# Models
# ----------------------------

class SmallRULNet(nn.Module):
    """Edge/small model: GRU regressor."""
    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 2, dropout: float = 0.10):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        out, _ = self.gru(x)
        h = out[:, -1, :]  # last step
        return self.head(h)  # (B,1)


class LargeGPT2RULNet(nn.Module):
    """
    Cloud/large model: pretrained GPT-2 backbone (frozen), plus a learned feature projection and regression head.
    We treat each timestep as a "token" and feed inputs_embeds.
    """
    def __init__(self, in_dim: int, gpt_name: str = "gpt2", freeze_backbone: bool = True):
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for GPT2Model.from_pretrained. "
                "Install via: pip install transformers"
            )

        self.gpt = GPT2Model.from_pretrained(gpt_name)
        if freeze_backbone:
            for p in self.gpt.parameters():
                p.requires_grad = False

        n_embd = self.gpt.config.n_embd
        self.proj = nn.Linear(in_dim, n_embd)
        self.head = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, n_embd // 2),
            nn.GELU(),
            nn.Linear(n_embd // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        embeds = self.proj(x)  # (B,T,n_embd)
        out = self.gpt(inputs_embeds=embeds).last_hidden_state  # (B,T,n_embd)
        h = out[:, -1, :]
        return self.head(h)  # (B,1)


# ----------------------------
# Conformal calibration (exact p-value confidence)
# ----------------------------

class ConformalCalibrator:
    """
    Nonconformity score: s = |y_hat - y| (in cycles, AFTER inverse scaling if you calibrated in cycles)
    We compute:
      p_value(x) = (#{s_calib >= s(x)} + 1) / (n_calib + 1)
      confidence(x) = 1 - p_value(x)
    And q_hat (for alpha) using standard split conformal regression:
      q_hat = quantile_{ceil((n+1)(1-alpha))/n} of s_calib
    """
    def __init__(self, alpha: float = 0.10):
        self.alpha = float(alpha)
        self.s_calib: Optional[np.ndarray] = None
        self.q_hat: Optional[float] = None

    def fit(self, y_hat: np.ndarray, y_true: np.ndarray) -> None:
        s = np.abs(np.asarray(y_hat, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))
        s = np.maximum(s, 0.0)
        self.s_calib = s
        n = len(s)
        # split conformal: kth order statistic
        k = int(math.ceil((n + 1) * (1.0 - self.alpha)))
        k = min(max(k, 1), n)
        self.q_hat = float(np.partition(s, k - 1)[k - 1])

    def confidence(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        if self.s_calib is None:
            raise RuntimeError("Calibrator not fit. Call fit() first.")
        s = np.abs(np.asarray(y_hat, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))
        sc = self.s_calib
        # vectorized count of >= can be memory heavy; do chunked
        conf = np.zeros_like(s, dtype=np.float64)
        n = sc.shape[0]
        # chunk to keep RAM low
        chunk = 4096
        for i in range(0, len(s), chunk):
            ss = s[i:i+chunk]
            # broadcasting: (chunk,n) -> may be large; do loop per element for safety
            for j, val in enumerate(ss):
                ge = int(np.sum(sc >= val))
                p = (ge + 1.0) / (n + 1.0)
                conf[i + j] = 1.0 - p
        return conf

    def interval(self, y_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.q_hat is None:
            raise RuntimeError("Calibrator not fit. Call fit() first.")
        y_hat = np.asarray(y_hat, dtype=np.float64)
        lo = y_hat - self.q_hat
        hi = y_hat + self.q_hat
        return lo, hi


# ----------------------------
# Routing controller (fuzzy-MCDM style weighted utility)
# ----------------------------

@dataclass
class RouteDecision:
    use_lm: bool
    util_sm: float
    util_lm: float
    conf_sm: float
    conf_lm: float


class FuzzyMCDMRouter:
    """
    A simple (transparent) fuzzy-MCDM-like controller:
      utility = w_conf * conf  - w_latency * latency_norm - w_memory * mem_norm
    We estimate latency_norm and mem_norm from pre-measured averages and model param MB.

    Hybrid policy:
      if conf_sm >= conf_tau and util_sm >= util_lm: choose SM
      else: choose LM
    """
    def __init__(self, conf_tau: float, w_conf: float, w_latency: float, w_memory: float):
        self.conf_tau = float(conf_tau)
        self.w_conf = float(w_conf)
        self.w_latency = float(w_latency)
        self.w_memory = float(w_memory)

        # set later
        self.lat_sm_ms = 1.0
        self.lat_lm_ms = 5.0
        self.mem_sm_mb = 1.0
        self.mem_lm_mb = 10.0

    def set_costs(self, lat_sm_ms: float, lat_lm_ms: float, mem_sm_mb: float, mem_lm_mb: float) -> None:
        self.lat_sm_ms = max(float(lat_sm_ms), 1e-6)
        self.lat_lm_ms = max(float(lat_lm_ms), 1e-6)
        self.mem_sm_mb = max(float(mem_sm_mb), 1e-6)
        self.mem_lm_mb = max(float(mem_lm_mb), 1e-6)

    def _norm(self, x: float, a: float, b: float) -> float:
        # normalize x in [a,b] to [0,1]
        if b <= a:
            return 0.0
        return float((x - a) / (b - a))

    def decide(self, conf_sm: float, conf_lm: float) -> RouteDecision:
        # normalize costs between SM and LM (2-point scaling)
        lat_sm_n = self._norm(self.lat_sm_ms, self.lat_sm_ms, self.lat_lm_ms)
        lat_lm_n = self._norm(self.lat_lm_ms, self.lat_sm_ms, self.lat_lm_ms)
        mem_sm_n = self._norm(self.mem_sm_mb, self.mem_sm_mb, self.mem_lm_mb)
        mem_lm_n = self._norm(self.mem_lm_mb, self.mem_sm_mb, self.mem_lm_mb)

        util_sm = self.w_conf * conf_sm - self.w_latency * lat_sm_n - self.w_memory * mem_sm_n
        util_lm = self.w_conf * conf_lm - self.w_latency * lat_lm_n - self.w_memory * mem_lm_n

        use_lm = not (conf_sm >= self.conf_tau and util_sm >= util_lm)
        return RouteDecision(use_lm=use_lm, util_sm=float(util_sm), util_lm=float(util_lm),
                            conf_sm=float(conf_sm), conf_lm=float(conf_lm))


# ----------------------------
# Training / Evaluation
# ----------------------------

@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    yh, yt = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        yh.append(out.detach().cpu().numpy().reshape(-1))
        yt.append(y.detach().cpu().numpy().reshape(-1))
    return np.concatenate(yh), np.concatenate(yt)


def train_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    tag: str,
) -> None:
    model.to(device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        total = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu().item()) * x.size(0)
            n += x.size(0)
        dt = time.perf_counter() - t0
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            mse = total / max(n, 1)
            print(f"[{tag}] Epoch {ep:03d}/{epochs} | Train MSE={mse:.6f} | time={dt:.1f}s")


@torch.no_grad()
def measure_latency_ms(model: nn.Module, loader: DataLoader, device: str, n_batches: int = 20) -> float:
    model.eval()
    model.to(device)
    # warmup
    it = iter(loader)
    for _ in range(3):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device)
        _ = model(x)

    times = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    if not times:
        return 0.0
    return float(np.mean(times))


def evaluate_mode(
    mode: str,
    sm: nn.Module,
    lm: nn.Module,
    router: FuzzyMCDMRouter,
    calib_sm: ConformalCalibrator,
    calib_lm: ConformalCalibrator,
    test_loader: DataLoader,
    device: str,
    max_rul: int,
) -> Dict[str, float]:
    """
    mode in {"edge_only","cloud_only","hybrid"}.
    Returns metrics in *cycles*.
    """
    sm.eval().to(device)
    lm.eval().to(device)

    y_true_all = []
    y_pred_all = []
    conf_all = []
    use_lm_all = []
    lat_ms = []

    # for hybrid: we need per-sample confidence, but we compute in cycles
    for x, y_norm in test_loader:
        # y_norm is normalized [0,1]; inverse-scale for calibration/confidence
        y_true = (y_norm.squeeze(-1).cpu().numpy() * max_rul).astype(np.float64)

        x = x.to(device)
        y_norm = y_norm.to(device)

        # SM prediction
        t0 = time.perf_counter()
        y_sm_norm = sm(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # LM prediction
        t2 = time.perf_counter()
        y_lm_norm = lm(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        y_sm = (y_sm_norm.detach().cpu().numpy().reshape(-1) * max_rul).astype(np.float64)
        y_lm = (y_lm_norm.detach().cpu().numpy().reshape(-1) * max_rul).astype(np.float64)

        # Conformal confidence (exact p-value based)
        conf_sm = calib_sm.confidence(y_sm, y_true)
        conf_lm = calib_lm.confidence(y_lm, y_true)

        if mode == "edge_only":
            y_pred = y_sm
            conf = conf_sm
            use_lm = np.zeros_like(y_pred, dtype=np.int32)
            # latency uses SM only
            lat = (t1 - t0) * 1000.0
        elif mode == "cloud_only":
            y_pred = y_lm
            conf = conf_lm
            use_lm = np.ones_like(y_pred, dtype=np.int32)
            lat = (t3 - t2) * 1000.0
        elif mode == "hybrid":
            # decide per-sample
            y_pred = np.zeros_like(y_sm)
            conf = np.zeros_like(conf_sm)
            use_lm = np.zeros_like(y_sm, dtype=np.int32)
            # approximate batch latency: SM + LM computed here; but routed inference would call only one.
            # For reporting, we compute routed latency using pre-measured means (router.lat_*).
            # Here we still compute actual outputs for both to decide + to plot fairness.
            for i in range(len(y_sm)):
                d = router.decide(conf_sm[i], conf_lm[i])
                if d.use_lm:
                    y_pred[i] = y_lm[i]
                    conf[i] = conf_lm[i]
                    use_lm[i] = 1
                else:
                    y_pred[i] = y_sm[i]
                    conf[i] = conf_sm[i]
                    use_lm[i] = 0
            # routed latency estimate: mix of per-model means
            lat = float(np.mean(use_lm) * router.lat_lm_ms + (1.0 - np.mean(use_lm)) * router.lat_sm_ms)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        conf_all.append(conf)
        use_lm_all.append(use_lm)
        lat_ms.append(lat)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    conf_all = np.concatenate(conf_all)
    use_lm_all = np.concatenate(use_lm_all)

    out = {
        "mode": mode,
        "rmse_cycles": rmse(y_pred_all, y_true_all),
        "mae_cycles": mae(y_pred_all, y_true_all),
        "avg_conf": float(np.mean(conf_all)),
        "lm_call_rate": float(np.mean(use_lm_all)),
        "avg_latency_ms": float(np.mean(lat_ms)),
    }
    return out


# ----------------------------
# Main: FD001 ablation runner
# ----------------------------

def run_fd001_ablation(data_dir: str, cfg: TrainConfig, max_rul: int, allow_unpretrained_lm: bool = False) -> None:
    set_seed(cfg.seed)

    train_path = os.path.join(data_dir, "train_FD001.txt")
    test_path = os.path.join(data_dir, "test_FD001.txt")
    rul_path = os.path.join(data_dir, "RUL_FD001.txt")

    if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(rul_path)):
        raise FileNotFoundError(
            "Missing NASA files. Expected in data_dir:\n"
            f"  {train_path}\n  {test_path}\n  {rul_path}"
        )

    if not _HAS_TRANSFORMERS and not allow_unpretrained_lm:
        raise ImportError(
            "transformers is not available, but pretrained GPT-2 is REQUIRED for the Large Model.\n"
            "Install: pip install transformers\n"
            "Or run with --allow_unpretrained_lm (NOT paper-faithful)."
        )

    print(f"Device: {cfg.device}")
    print(f"Loading FD001 from: {data_dir}")

    df_train = read_cmapss_txt(train_path)
    df_test = read_cmapss_txt(test_path)

    df_train = add_train_rul(df_train, max_rul=max_rul)
    df_test = add_test_rul(df_test, rul_txt_path=rul_path, max_rul=max_rul)

    # Features: op settings + sensors (exclude unit/cycle/RUL)
    feature_cols = [c for c in df_train.columns if c not in ("unit", "cycle", "RUL")]
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].to_numpy(dtype=np.float32))

    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    df_train_scaled[feature_cols] = scaler.transform(df_train_scaled[feature_cols])
    df_test_scaled[feature_cols] = scaler.transform(df_test_scaled[feature_cols])

    # Windowize
    X_train, y_train_cycles = make_windows(df_train_scaled, feature_cols, cfg.seq_len, cfg.stride)
    X_test, y_test_cycles = make_windows(df_test_scaled, feature_cols, cfg.seq_len, cfg.stride)

    # Normalize labels at training-time
    y_train = (y_train_cycles / float(max_rul)).astype(np.float32)
    y_test = (y_test_cycles / float(max_rul)).astype(np.float32)

    # Train / calibration split
    n = X_train.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_cal = max(1, int(cfg.calib_fraction * n))
    cal_idx = idx[:n_cal]
    tr_idx = idx[n_cal:]

    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_cal, y_cal = X_train[cal_idx], y_train[cal_idx]

    train_loader = DataLoader(WindowDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.device.startswith("cuda"))
    calib_loader = DataLoader(WindowDataset(X_cal, y_cal), batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=cfg.device.startswith("cuda"))
    test_loader = DataLoader(WindowDataset(X_test, y_test), batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=cfg.device.startswith("cuda"))

    in_dim = X_train.shape[-1]

    # Models
    sm = SmallRULNet(in_dim=in_dim)
    lm = LargeGPT2RULNet(in_dim=in_dim) if _HAS_TRANSFORMERS else None
    if lm is None:
        raise RuntimeError("LM unavailable.")

    print(f"Small model params: {param_size_mb(sm):.2f} MB")
    print(f"Large model params: {param_size_mb(lm):.2f} MB (GPT-2 frozen backbone + trainable head/proj)")

    # Train SM + LM head
    train_regressor(sm, train_loader, cfg.device, cfg.epochs, cfg.lr_sm, cfg.weight_decay, tag="SM")
    train_regressor(lm, train_loader, cfg.device, cfg.epochs, cfg.lr_lm_head, cfg.weight_decay, tag="LM")

    # Calibration residuals in cycles (inverse-scaled)
    yhat_sm_cal_norm, ytrue_cal_norm = predict(sm, calib_loader, cfg.device)
    yhat_lm_cal_norm, _ = predict(lm, calib_loader, cfg.device)

    yhat_sm_cal = yhat_sm_cal_norm * max_rul
    yhat_lm_cal = yhat_lm_cal_norm * max_rul
    ytrue_cal = ytrue_cal_norm * max_rul

    calib_sm = ConformalCalibrator(alpha=cfg.alpha)
    calib_lm = ConformalCalibrator(alpha=cfg.alpha)
    calib_sm.fit(yhat_sm_cal, ytrue_cal)
    calib_lm.fit(yhat_lm_cal, ytrue_cal)

    print(f"Conformal q_hat (SM): {calib_sm.q_hat:.3f} cycles @ alpha={cfg.alpha}")
    print(f"Conformal q_hat (LM): {calib_lm.q_hat:.3f} cycles @ alpha={cfg.alpha}")

    # Measure latency means for routing (batch-level); use for hybrid routed latency estimate.
    lat_sm = measure_latency_ms(sm, test_loader, cfg.device, n_batches=20)
    lat_lm = measure_latency_ms(lm, test_loader, cfg.device, n_batches=20)
    mem_sm = param_size_mb(sm)
    mem_lm = param_size_mb(lm)

    router = FuzzyMCDMRouter(conf_tau=cfg.conf_tau, w_conf=cfg.w_conf, w_latency=cfg.w_latency, w_memory=cfg.w_memory)
    router.set_costs(lat_sm_ms=lat_sm, lat_lm_ms=lat_lm, mem_sm_mb=mem_sm, mem_lm_mb=mem_lm)

    # Evaluate ablation modes
    results = []
    for mode in ("edge_only", "cloud_only", "hybrid"):
        print(f"\nEvaluating mode: {mode}")
        r = evaluate_mode(mode, sm, lm, router, calib_sm, calib_lm, test_loader, cfg.device, max_rul=max_rul)
        results.append(r)
        print(
            f"  RMSE={r['rmse_cycles']:.3f} | MAE={r['mae_cycles']:.3f} | "
            f"LM_rate={r['lm_call_rate']:.3f} | latency={r['avg_latency_ms']:.2f}ms | conf={r['avg_conf']:.3f}"
        )

    out_df = pd.DataFrame(results)
    out_csv = os.path.join(os.getcwd(), "ablation_fd001_summary.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Save a simple resource summary
    resource = pd.DataFrame([{
        "lat_sm_ms": lat_sm,
        "lat_lm_ms": lat_lm,
        "mem_sm_mb": mem_sm,
        "mem_lm_mb": mem_lm,
        "conf_tau": cfg.conf_tau,
        "w_conf": cfg.w_conf,
        "w_latency": cfg.w_latency,
        "w_memory": cfg.w_memory,
        "alpha": cfg.alpha,
        "qhat_sm": calib_sm.q_hat,
        "qhat_lm": calib_lm.q_hat,
    }])
    res_csv = os.path.join(os.getcwd(), "ablation_fd001_resources.csv")
    resource.to_csv(res_csv, index=False)
    print(f"Saved: {res_csv}")

    # Plot: confidence vs error for each mode (optional if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        # Recompute per-sample results for hybrid to plot scatter
        sm.eval().to(cfg.device); lm.eval().to(cfg.device)
        y_true_all = []
        y_sm_all = []
        y_lm_all = []
        conf_sm_all = []
        conf_lm_all = []

        for x, y_norm in test_loader:
            y_true = (y_norm.squeeze(-1).cpu().numpy() * max_rul).astype(np.float64)
            x = x.to(cfg.device)
            y_sm = (sm(x).detach().cpu().numpy().reshape(-1) * max_rul).astype(np.float64)
            y_lm = (lm(x).detach().cpu().numpy().reshape(-1) * max_rul).astype(np.float64)
            conf_sm = calib_sm.confidence(y_sm, y_true)
            conf_lm = calib_lm.confidence(y_lm, y_true)

            y_true_all.append(y_true)
            y_sm_all.append(y_sm)
            y_lm_all.append(y_lm)
            conf_sm_all.append(conf_sm)
            conf_lm_all.append(conf_lm)

        y_true_all = np.concatenate(y_true_all)
        y_sm_all = np.concatenate(y_sm_all)
        y_lm_all = np.concatenate(y_lm_all)
        conf_sm_all = np.concatenate(conf_sm_all)
        conf_lm_all = np.concatenate(conf_lm_all)

        # Hybrid routed
        y_hyb = np.zeros_like(y_true_all)
        conf_hyb = np.zeros_like(conf_sm_all)
        use_lm = np.zeros_like(y_true_all, dtype=np.int32)
        for i in range(len(y_true_all)):
            d = router.decide(conf_sm_all[i], conf_lm_all[i])
            if d.use_lm:
                y_hyb[i] = y_lm_all[i]
                conf_hyb[i] = conf_lm_all[i]
                use_lm[i] = 1
            else:
                y_hyb[i] = y_sm_all[i]
                conf_hyb[i] = conf_sm_all[i]
                use_lm[i] = 0

        # Figure: confidence histogram + RMSE by bin for each mode
        def conf_bins(conf, yhat, ytrue, name):
            bins = np.linspace(0, 1, 6)
            idx = np.digitize(conf, bins) - 1
            xs = []
            ys = []
            ps = []
            for b in range(5):
                m = idx == b
                if np.any(m):
                    xs.append((bins[b] + bins[b+1]) / 2)
                    ys.append(rmse(yhat[m], ytrue[m]))
                    ps.append(100.0 * np.mean(m))
                else:
                    xs.append((bins[b] + bins[b+1]) / 2)
                    ys.append(np.nan)
                    ps.append(0.0)
            return np.array(xs), np.array(ys), np.array(ps)

        plt.figure(figsize=(8, 4))
        for conf, yhat, name in [(conf_sm_all, y_sm_all, "SM"), (conf_lm_all, y_lm_all, "LM"), (conf_hyb, y_hyb, "Hybrid")]:
            xs, ys, ps = conf_bins(conf, yhat, y_true_all, name)
            plt.plot(xs, ys, marker="o", label=name)
        plt.xlabel("Conformal confidence (binned)")
        plt.ylabel("RMSE (cycles)")
        plt.legend()
        fig_path = os.path.join(os.getcwd(), "ablation_fd001_conf_vs_rmse.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=900)
        plt.close()
        print(f"Saved: {fig_path}")

        # Figure: LM call rate as a function of confidence threshold
        taus = np.linspace(0, 1, 21)
        rates = []
        for tau in taus:
            router2 = FuzzyMCDMRouter(conf_tau=float(tau), w_conf=cfg.w_conf, w_latency=cfg.w_latency, w_memory=cfg.w_memory)
            router2.set_costs(lat_sm_ms=lat_sm, lat_lm_ms=lat_lm, mem_sm_mb=mem_sm, mem_lm_mb=mem_lm)
            use = []
            for i in range(len(y_true_all)):
                d = router2.decide(conf_sm_all[i], conf_lm_all[i])
                use.append(1 if d.use_lm else 0)
            rates.append(np.mean(use))
        plt.figure(figsize=(7, 4))
        plt.plot(taus, rates, marker="o")
        plt.xlabel("SM confidence threshold (tau)")
        plt.ylabel("LM call rate")
        plt.tight_layout()
        fig_path2 = os.path.join(os.getcwd(), "ablation_fd001_lm_call_rate_vs_tau.png")
        plt.savefig(fig_path2, dpi=900)
        plt.close()
        print(f"Saved: {fig_path2}")

    except Exception as e:
        print(f"[WARN] Skipping plots due to: {e}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Folder containing train_FD001.txt, test_FD001.txt, RUL_FD001.txt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_rul", type=int, default=MAX_RUL_DEFAULT)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--calib_fraction", type=float, default=0.10)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--conf_tau", type=float, default=0.55)
    p.add_argument("--allow_unpretrained_lm", action="store_true", help="NOT paper-faithful; for debugging only.")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = TrainConfig(
        epochs=int(args.epochs),
        seq_len=int(args.seq_len),
        batch_size=int(args.batch_size),
        calib_fraction=float(args.calib_fraction),
        alpha=float(args.alpha),
        conf_tau=float(args.conf_tau),
    )
    run_fd001_ablation(
        data_dir=args.data_dir,
        cfg=cfg,
        max_rul=int(args.max_rul),
        allow_unpretrained_lm=bool(args.allow_unpretrained_lm),
    )


if __name__ == "__main__":
    main()
