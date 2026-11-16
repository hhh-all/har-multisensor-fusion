"""Plot robustness CSV files in a separate process from PyTorch on Windows."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(r"D:\研一\多传感器智能信息融合\作业3")
SHORT_LABELS = ("Walk", "Up", "Down", "Sit", "Stand", "Lie")
NOISE_LEVELS = (0.1, 0.3, 0.5)


def read_summary(path):
    summary = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            summary[(row["model"], row["condition"], float(row["noise_sigma"]))] = {
                "macro_f1_mean": float(row["macro_f1_mean"]),
                "macro_f1_std": float(row["macro_f1_std"]),
            }
    return summary


def plot_missing(summary, output):
    labels = ("Clean", "No acc", "No gyro")
    conditions = ("clean", "missing_acc", "missing_gyro")
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for name, shift, color in (("C", -0.18, "#607D8B"), ("D", 0.18, "#E07A3F")):
        values = [summary[(name, cond, 0.0)]["macro_f1_mean"] for cond in conditions]
        bars = ax.bar(x + shift, values, width=0.36, label=name, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Missing sensor evaluation (same test set)")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "missing_modality.png", dpi=180)
    plt.close(fig)


def plot_noise(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout="constrained")
    x = [0.0, *NOISE_LEVELS]
    for ax, sensor, title in zip(axes, ("acc", "gyro"),
                                 ("Noise on accelerometer", "Noise on gyroscope")):
        for name, color in (("C", "#607D8B"), ("D", "#E07A3F")):
            rows = [summary[(name, "clean", 0.0)]] + [
                summary[(name, f"noise_{sensor}", sigma)] for sigma in NOISE_LEVELS
            ]
            means = np.array([row["macro_f1_mean"] for row in rows])
            stds = np.array([row["macro_f1_std"] for row in rows])
            ax.plot(x, means, marker="o", color=color, label=name)
            ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.15)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xlabel("Gaussian noise std (normalized units)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Test Macro-F1")
    axes[1].legend(title="Model")
    fig.savefig(output / "noise_curve.png", dpi=180)
    plt.close(fig)


def plot_confusions(output, base):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    folders = {
        "A": "experiment_A_acc", "B": "experiment_B_gyro",
        "C": "experiment_C_concat", "D": "experiment_D_gated",
    }
    for ax, name in zip(axes.flat, ("A", "B", "C", "D")):
        path = base / folders[name] / "confusion_matrix.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.reader(file))
        cm = np.array([[int(cell) for cell in row[1:]] for row in rows[1:]], dtype=int)
        if cm.shape != (6, 6):
            raise ValueError(f"{path}: 混淆矩阵应为 6x6")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=550)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > 270 else "black")
        ax.set_title(f"Model {name}")
        ax.set_xticks(range(6), SHORT_LABELS, rotation=35, ha="right")
        ax.set_yticks(range(6), SHORT_LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.savefig(output / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    output = args.input.expanduser().resolve()
    summary = read_summary(output / "robustness_summary.csv")
    plot_missing(summary, output)
    plot_noise(summary, output)
    plot_confusions(output, args.base.expanduser().resolve())
    print("Created missing_modality.png, noise_curve.png, confusion_matrices.png")


if __name__ == "__main__":
    main()
