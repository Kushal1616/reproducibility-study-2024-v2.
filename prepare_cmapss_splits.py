import os
import json
import argparse
import numpy as np
import pandas as pd


def load_cmapss(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    cols = ["unit_id", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
    df.columns = cols
    return df


def make_engine_split(train_path, val_ratio=0.2, seed=42):
    df = load_cmapss(train_path)
    engine_ids = sorted(df["unit_id"].unique().tolist())

    rng = np.random.default_rng(seed)
    engine_ids = np.array(engine_ids)
    rng.shuffle(engine_ids)

    n_val = max(1, int(round(len(engine_ids) * val_ratio)))
    val_ids = sorted(engine_ids[:n_val].tolist())
    train_ids = sorted(engine_ids[n_val:].tolist())

    return {
        "train_engine_ids": train_ids,
        "val_engine_ids": val_ids
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=str, required=True, help="FD001 / FD002 / FD003 / FD004")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="splits")
    args = parser.parse_args()

    fd = args.fd.upper()
    train_file = os.path.join(args.data_dir, f"train_{fd}.txt")

    split = make_engine_split(train_file, val_ratio=args.val_ratio, seed=args.seed)
    split["fd"] = fd
    split["seed"] = args.seed
    split["val_ratio"] = args.val_ratio

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{fd}_seed{args.seed}.json")

    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Saved split file: {out_path}")
    print(f"Train engines: {len(split['train_engine_ids'])}")
    print(f"Val engines: {len(split['val_engine_ids'])}")


if __name__ == "__main__":
    main()