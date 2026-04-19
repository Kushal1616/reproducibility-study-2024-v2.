
"""
Convenience runner for FD001 ablation (100 epochs by default).

Usage (terminal):
  python fd_001big_ablation.py --data_dir /path/to/CMAPSS --epochs 100

Usage (Jupyter/Colab cell):
  !python fd_001big_ablation.py --data_dir "/content/CMAPSS" --epochs 100

This runner simply forwards arguments to paper8_fd001_ablation.py, keeping your earlier workflow.
"""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_rul", type=int, default=125)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--calib_fraction", type=float, default=0.10)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--conf_tau", type=float, default=0.55)
    p.add_argument("--allow_unpretrained_lm", action="store_true")
    args = p.parse_args()

    script = Path(__file__).with_name("paper8_fd001_ablation.py")
    cmd = [
        sys.executable, str(script),
        "--data_dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--max_rul", str(args.max_rul),
        "--seq_len", str(args.seq_len),
        "--batch_size", str(args.batch_size),
        "--calib_fraction", str(args.calib_fraction),
        "--alpha", str(args.alpha),
        "--conf_tau", str(args.conf_tau),
    ]
    if args.allow_unpretrained_lm:
        cmd.append("--allow_unpretrained_lm")

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
