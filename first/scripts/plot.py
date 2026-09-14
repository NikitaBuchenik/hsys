"""
Plot benchmark results.

Expected CSV columns:
n,cuda_seconds,eigen_seconds

Example:
python scripts/plot.py results.csv
"""

import csv
import math
import sys
import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "n": int(row["n"]),
                "cuda": float(row["cuda_seconds"]),
                "eigen": float(row["eigen_seconds"]),
            })
    return rows


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/plot.py results.csv")
        raise SystemExit(1)

    rows = read_csv(sys.argv[1])
    rows.sort(key=lambda x: x["n"])

    n = [r["n"] for r in rows]
    cuda = [r["cuda"] for r in rows]
    eigen = [r["eigen"] for r in rows]
    speedup = [e / c for e, c in zip(eigen, cuda)]

    plt.figure()
    plt.loglog(n, cuda, "o-", label="CUDA Vector")
    plt.loglog(n, eigen, "o-", label="Eigen VectorXf")
    plt.xlabel("Vector size n")
    plt.ylabel("Time, seconds")
    plt.title("Vector addition: real complexity")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("complexity.png", dpi=160)

    plt.figure()
    plt.semilogx(n, speedup, "o-")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Vector size n")
    plt.ylabel("Speedup (Eigen / CUDA)")
    plt.title("CUDA Vector speedup over Eigen")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("speedup.png", dpi=160)

    print("Created complexity.png and speedup.png")


if __name__ == "__main__":
    main()
