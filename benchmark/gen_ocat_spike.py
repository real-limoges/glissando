#!/usr/bin/env python3
"""
Generate synthetic ordered-categorical (ocat) train/test parquets for the
Phase 0 spike (Guide 3).

DGP:
  latent f(x1, x2) = sin(x1) + 0.4 * x2^2
  cumulative thresholds θ = (-1.0, 0.0, 1.2)
  P(y ≤ k | x) = logistic(θ_k - f(x))
  y ∈ {1, 2, 3, 4}  drawn from the implied category probabilities

Train (n=1500): columns y, x1, x2, le1 (y≤1), le2 (y≤2), le3 (y≤3)
Test  (n=500):  same + p1..p4 (true population category probabilities per row)
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

THRESHOLDS = (-1.0, 0.0, 1.2)


def logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate(rng: np.random.Generator, n: int):
    x1 = rng.uniform(0.0, 2 * np.pi, size=n)
    x2 = rng.uniform(0.0, 1.0, size=n)
    f = np.sin(x1) + 0.4 * x2**2

    theta = np.array(THRESHOLDS)
    # cumulative probs P(y <= k) = logistic(theta_k - f), shape (n, 3)
    cum = logistic(theta[None, :] - f[:, None])  # (n, 3)

    # category probs (n, 4)
    p1 = cum[:, 0]
    p2 = np.clip(cum[:, 1] - cum[:, 0], 0.0, 1.0)
    p3 = np.clip(cum[:, 2] - cum[:, 1], 0.0, 1.0)
    p4 = np.clip(1.0 - cum[:, 2], 0.0, 1.0)

    # renormalize, just to mop up float rounding
    total = p1 + p2 + p3 + p4
    p1 /= total
    p2 /= total
    p3 /= total
    p4 /= total

    # draw y ∈ {0..3}, then bump to {1..4}
    probs_mat = np.stack([p1, p2, p3, p4], axis=1)  # (n, 4)
    y_idx = np.array(
        [rng.choice(4, p=probs_mat[i]) for i in range(n)], dtype=float
    )
    y = y_idx + 1.0  # now 1-indexed

    return {
        "y":   y,
        "x1":  x1,
        "x2":  x2,
        "le1": (y <= 1.0).astype(float),
        "le2": (y <= 2.0).astype(float),
        "le3": (y <= 3.0).astype(float),
        # true population probabilities, for scoring later
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
    }


def write_parquet(data: dict, path: Path) -> None:
    df = pl.DataFrame(
        {k: pl.Series(k, v.tolist(), dtype=pl.Float64) for k, v in data.items()}
    )
    df.write_parquet(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-test",  type=int, default=500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng   = np.random.default_rng(args.seed)
    train = generate(rng, args.n_train)
    test  = generate(rng, args.n_test)

    train_path = args.output_dir / "ocat_train.parquet"
    test_path  = args.output_dir / "ocat_test.parquet"
    write_parquet(train, train_path)
    write_parquet(test,  test_path)

    print(f"[gen] train n={args.n_train} → {train_path}")
    print(f"[gen] test  n={args.n_test}  → {test_path}")
    y = train["y"]
    for k in range(1, 5):
        print(f"  cat {k}: {int((y == k).sum()):4d}  ({(y == k).mean():.1%})")


if __name__ == "__main__":
    main()
