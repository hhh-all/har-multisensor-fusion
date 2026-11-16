"""Step 4: evaluate fixed A-D results and C/D robustness on the official test set.

Run with: python evaluate_robustness.py
Requires: numpy, torch, matplotlib, and the C/D training scripts in this folder.

Positive `macro_f1_drop_pp` means performance fell relative to clean input.
Noise standard deviations are measured after train-only standardization.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import numpy as np

from train_concat_fusion import ConcatFusionCNN
from train_gated_fusion import GatedFusionCNN


BASE = Path(r"D:\研一\多传感器智能信息融合\作业3")
NOISE_LEVELS = (0.1, 0.3, 0.5)


def model_dirs(base):
    return {
        "A": base / "experiment_A_acc",
        "B": base / "experiment_B_gyro",
        "C": base / "experiment_C_concat",
        "D": base / "experiment_D_gated",
    }


def read_test(data_dir):
    path = data_dir / "test.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        acc = data["X_acc"].astype(np.float32)
        gyro = data["X_gyro"].astype(np.float32)
        y = data["y"].astype(np.int64)
    if acc.shape != gyro.shape or acc.ndim != 3 or acc.shape[1:] != (3, 128):
        raise ValueError(f"测试集两路信号形状不正确：{acc.shape}, {gyro.shape}")
    if len(acc) != len(y):
        raise ValueError("测试集信号与标签数量不一致")
    return acc, gyro, y


def load_model(name, folder):
    if name == "C":
        model, checkpoint_name = ConcatFusionCNN(), "best_concat_cnn.pt"
    elif name == "D":
        model, checkpoint_name = GatedFusionCNN(), "best_gated_cnn.pt"
    else:
        raise ValueError(name)
    path = folder / checkpoint_name
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def metric(y_true, y_pred):
    cm = np.zeros((6, 6), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    f1 = []
    for i in range(6):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(y_true == y_pred)), float(np.mean(f1))


@torch.no_grad()
def predict(model, name, acc, gyro, missing=None, batch_size=128):
    model.eval()
    predictions = []
    for start in range(0, len(acc), batch_size):
        stop = min(start + batch_size, len(acc))
        a = torch.from_numpy(np.ascontiguousarray(acc[start:stop]))
        g = torch.from_numpy(np.ascontiguousarray(gyro[start:stop]))
        if name == "D":
            mask_acc = torch.ones((len(a), 1))
            mask_gyro = torch.ones((len(g), 1))
            if missing == "acc":
                mask_acc.zero_()
            elif missing == "gyro":
                mask_gyro.zero_()
            logits, _ = model(a, g, mask_acc, mask_gyro)
        else:
            logits = model(a, g)
        predictions.append(logits.argmax(dim=1).numpy())
    return np.concatenate(predictions)


def evaluate_condition(models, acc, gyro, y, condition, noise=None):
    if condition == "missing_acc":
        acc_input, gyro_input, missing = np.zeros_like(acc), gyro, "acc"
    elif condition == "missing_gyro":
        acc_input, gyro_input, missing = acc, np.zeros_like(gyro), "gyro"
    elif condition == "noise_acc":
        acc_input, gyro_input, missing = acc + noise, gyro, None
    elif condition == "noise_gyro":
        acc_input, gyro_input, missing = acc, gyro + noise, None
    elif condition == "clean":
        acc_input, gyro_input, missing = acc, gyro, None
    else:
        raise ValueError(condition)
    results = {}
    for name, model in models.items():
        pred = predict(model, name, acc_input, gyro_input, missing)
        results[name] = metric(y, pred)
    return results


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE,
                        help="包含 uci_har_ready 和四个实验目录的项目根目录")
    parser.add_argument("--data", type=Path, default=None,
                        help="已整理数据目录；默认 <base>/uci_har_ready")
    parser.add_argument("--output", type=Path, default=None,
                        help="结果目录；默认 <base>/experiment_4_robustness")
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.noise_repeats < 1:
        parser.error("noise-repeats 必须大于 0")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    base = args.base.expanduser().resolve()
    data_dir = (args.data or base / "uci_har_ready").expanduser().resolve()
    out = (args.output or base / "experiment_4_robustness").expanduser().resolve()
    dirs = model_dirs(base)
    out.mkdir(parents=True, exist_ok=True)
    acc, gyro, y = read_test(data_dir)
    protocol = {
        "test_samples": len(y),
        "models_frozen": True,
        "missing_sensor_simulation": "set all standardized values of that sensor to zero; D also receives an availability mask",
        "noise": "zero-mean unit Gaussian noise scaled by sigma, added after train-only standardization to one sensor",
        "noise_sigmas": NOISE_LEVELS,
        "noise_repeats": args.noise_repeats,
        "noise_seed_first_repeat": args.seed,
        "same_perturbation_for_C_and_D": True,
        "macro_f1_drop_pp": "100 * (clean_macro_f1 - perturbed_macro_f1); positive means worse",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    models = {name: load_model(name, dirs[name]) for name in ("C", "D")}

    baseline_rows = []
    saved_metrics = {}
    for name in ("A", "B", "C", "D"):
        path = dirs[name] / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        saved_metrics[name] = json.loads(path.read_text(encoding="utf-8"))
        m = saved_metrics[name]
        if m["test_samples"] != len(y):
            raise ValueError(f"模型 {name} 的测试样本数与当前测试集不同")
        baseline_rows.append((name, m["test_accuracy"], m["test_macro_f1"],
                              m["best_epoch_by_validation_macro_f1"]))
    write_csv(out / "baseline_results.csv",
              ("model", "test_accuracy", "test_macro_f1", "best_validation_epoch"), baseline_rows)

    raw = []

    def add(condition, sigma, repeat, values):
        for name, (accuracy, macro_f1) in values.items():
            raw.append((name, condition, sigma, repeat, accuracy, macro_f1))

    add("clean", 0.0, 0, evaluate_condition(models, acc, gyro, y, "clean"))
    for name in ("C", "D"):
        measured = next(row for row in raw if row[0] == name and row[1] == "clean")
        expected = saved_metrics[name]
        if (abs(measured[4] - expected["test_accuracy"]) > 1e-8
                or abs(measured[5] - expected["test_macro_f1"]) > 1e-8):
            raise ValueError(f"模型 {name} 的重新评估成绩与原 metrics.json 不一致")
    print("Clean C/D scores match the saved experiment results.", flush=True)

    for condition in ("missing_acc", "missing_gyro"):
        add(condition, 0.0, 0, evaluate_condition(models, acc, gyro, y, condition))
        print(f"Finished: {condition}", flush=True)

    for repeat in range(args.noise_repeats):
        rng = np.random.default_rng(args.seed + repeat)
        base_acc_noise = rng.standard_normal(acc.shape, dtype=np.float32)
        base_gyro_noise = rng.standard_normal(gyro.shape, dtype=np.float32)
        for sigma in NOISE_LEVELS:
            add("noise_acc", sigma, repeat,
                evaluate_condition(models, acc, gyro, y, "noise_acc", sigma * base_acc_noise))
            add("noise_gyro", sigma, repeat,
                evaluate_condition(models, acc, gyro, y, "noise_gyro", sigma * base_gyro_noise))
        print(f"Finished noise repeat {repeat + 1}/{args.noise_repeats}", flush=True)

    write_csv(out / "robustness_raw.csv",
              ("model", "condition", "noise_sigma", "repeat", "accuracy", "macro_f1"), raw)
    summary = {}
    for name in ("C", "D"):
        keys = sorted({(condition, sigma) for model, condition, sigma, *_ in raw if model == name})
        for condition, sigma in keys:
            selected = [row for row in raw if row[0] == name and row[1] == condition and row[2] == sigma]
            accs = np.array([row[4] for row in selected])
            f1s = np.array([row[5] for row in selected])
            summary[(name, condition, sigma)] = {
                "repeats": len(selected),
                "accuracy_mean": float(accs.mean()),
                "accuracy_std": float(accs.std(ddof=0)),
                "macro_f1_mean": float(f1s.mean()),
                "macro_f1_std": float(f1s.std(ddof=0)),
            }
    summary_rows = []
    for (name, condition, sigma), value in sorted(summary.items()):
        clean = summary[(name, "clean", 0.0)]
        summary_rows.append((name, condition, sigma, value["repeats"],
                             value["accuracy_mean"], value["accuracy_std"],
                             value["macro_f1_mean"], value["macro_f1_std"],
                             100 * (clean["accuracy_mean"] - value["accuracy_mean"]),
                             100 * (clean["macro_f1_mean"] - value["macro_f1_mean"])))
    write_csv(out / "robustness_summary.csv",
              ("model", "condition", "noise_sigma", "repeats", "accuracy_mean", "accuracy_std",
               "macro_f1_mean", "macro_f1_std", "accuracy_drop_pp", "macro_f1_drop_pp"), summary_rows)

    subprocess.run([sys.executable, str(Path(__file__).with_name("plot_robustness.py")),
                    "--input", str(out), "--base", str(base)], check=True)
    print("\nMissing-sensor Macro-F1 (clean -> missing acc / missing gyro):")
    for name in ("C", "D"):
        values = [summary[(name, cond, 0.0)]["macro_f1_mean"]
                  for cond in ("clean", "missing_acc", "missing_gyro")]
        print(f"{name}: {values[0]:.4f} -> {values[1]:.4f} / {values[2]:.4f}")
    print(f"Saved tables and figures to: {out}")


if __name__ == "__main__":
    main()
