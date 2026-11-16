"""Experiment C: two-branch feature concatenation on prepared UCI HAR data.

Run with: python train_concat_fusion.py
Requires: numpy, torch
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_DATA = Path(r"D:\研一\多传感器智能信息融合\作业3\uci_har_ready")
DEFAULT_OUTPUT = Path(r"D:\研一\多传感器智能信息融合\作业3\experiment_C_concat")
CLASS_NAMES = (
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
)


class SensorEncoder(nn.Module):
    """Convert one three-axis signal window (3,128) to a 64-D feature."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ConcatFusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.acc_encoder = SensorEncoder()
        self.gyro_encoder = SensorEncoder()
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 6))

    def forward(self, x_acc, x_gyro):
        acc_feature = self.acc_encoder(x_acc)
        gyro_feature = self.gyro_encoder(x_gyro)
        fused_feature = torch.cat((acc_feature, gyro_feature), dim=1)
        return self.classifier(fused_feature)


def load_split(data_dir, split):
    path = data_dir / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}；先运行 prepare_uci_har.py")
    with np.load(path, allow_pickle=False) as data:
        x_acc = data["X_acc"].astype(np.float32)
        x_gyro = data["X_gyro"].astype(np.float32)
        y = data["y"].astype(np.int64)
        subject = data["subject"].astype(np.int64)
    if x_acc.shape != x_gyro.shape or x_acc.ndim != 3 or x_acc.shape[1:] != (3, 128):
        raise ValueError(f"{path}: 两路信号应均为 (N,3,128)，得到 {x_acc.shape} 和 {x_gyro.shape}")
    if len(x_acc) != len(y) or len(y) != len(subject):
        raise ValueError(f"{path}: 两路信号、标签、受试者编号未对齐")
    if not np.isin(y, np.arange(6)).all():
        raise ValueError(f"{path}: y 应为 0～5")
    if not np.isfinite(x_acc).all() or not np.isfinite(x_gyro).all():
        raise ValueError(f"{path}: 信号含 NaN 或无穷大")
    return x_acc, x_gyro, y, subject


def make_loader(x_acc, x_gyro, y, batch_size, shuffle=False, generator=None):
    dataset = TensorDataset(torch.from_numpy(x_acc), torch.from_numpy(x_gyro), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator)


def scores(y_true, y_pred):
    cm = np.zeros((6, 6), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    f1_per_class = []
    for i in range(6):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        denominator = 2 * tp + fp + fn
        f1_per_class.append(2 * tp / denominator if denominator else 0.0)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1_per_class)),
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []
    for x_acc, x_gyro, y in loader:
        x_acc, x_gyro, y = x_acc.to(device), x_gyro.to(device), y.to(device)
        logits = model(x_acc, x_gyro)
        total_loss += loss_fn(logits, y).item() * len(y)
        all_true.append(y.cpu().numpy())
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    result = scores(y_true, y_pred)
    result["loss"] = total_loss / len(y_true)
    result["y_true"] = y_true
    result["y_pred"] = y_pred
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs、batch-size、patience 必须大于 0")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cpu")
    data_dir = args.data.expanduser().resolve()
    out_dir = args.output.expanduser().resolve()

    train_acc, train_gyro, train_y, train_subject = load_split(data_dir, "train")
    val_acc, val_gyro, val_y, val_subject = load_split(data_dir, "val")
    test_acc, test_gyro, test_y, test_subject = load_split(data_dir, "test")
    if (set(train_subject) & set(val_subject)
            or set(train_subject) & set(test_subject)
            or set(val_subject) & set(test_subject)):
        raise ValueError("训练、验证、测试集存在相同受试者")
    print(f"Input: accelerometer + gyroscope; train={len(train_y)}, "
          f"val={len(val_y)}, test={len(test_y)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(train_acc, train_gyro, train_y, args.batch_size, True, generator)
    val_loader = make_loader(val_acc, val_gyro, val_y, args.batch_size)
    test_loader = make_loader(test_acc, test_gyro, test_y, args.batch_size)
    model = ConcatFusionCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model_path = out_dir / "best_concat_cnn.pt"
    history_path = out_dir / "history.csv"
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(("epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"))
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_train_loss = 0.0
            for x_acc, x_gyro, y in train_loader:
                x_acc, x_gyro, y = x_acc.to(device), x_gyro.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x_acc, x_gyro), y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(y)
            train_loss = total_train_loss / len(train_y)
            val_result = evaluate(model, val_loader, loss_fn, device)
            writer.writerow((epoch, train_loss, val_result["loss"],
                             val_result["accuracy"], val_result["macro_f1"]))
            history_file.flush()
            print(f"epoch {epoch:02d} | train loss {train_loss:.4f} | "
                  f"val accuracy {val_result['accuracy']:.4f} | "
                  f"val macro-F1 {val_result['macro_f1']:.4f}")
            if val_result["macro_f1"] > best_f1 + 1e-6:
                best_f1 = val_result["macro_f1"]
                best_epoch = epoch
                stale_epochs = 0
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": epoch, "val_macro_f1": best_f1}, model_path)
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"Early stopping after {args.patience} epochs without improvement.")
                    break

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = evaluate(model, test_loader, loss_fn, device)
    result = {
        "experiment": "C_simple_feature_concatenation",
        "fusion_level": "feature",
        "fusion_method": "concatenate 64-D accelerometer feature and 64-D gyroscope feature",
        "best_epoch_by_validation_macro_f1": best_epoch,
        "validation_macro_f1_at_best_epoch": best_f1,
        "test_accuracy": test_result["accuracy"],
        "test_macro_f1": test_result["macro_f1"],
        "test_f1_per_class": dict(zip(CLASS_NAMES, test_result["f1_per_class"])),
        "class_names": CLASS_NAMES,
        "train_samples": len(train_y),
        "val_samples": len(val_y),
        "test_samples": len(test_y),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("true\\pred", *CLASS_NAMES))
        for name, row in zip(CLASS_NAMES, test_result["confusion_matrix"]):
            writer.writerow((name, *row.tolist()))
    with (out_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("subject", "true_label", "predicted_label"))
        writer.writerows(zip(test_subject.tolist(), test_result["y_true"].tolist(),
                             test_result["y_pred"].tolist()))
    print(f"BEST epoch={best_epoch}, validation macro-F1={best_f1:.4f}")
    print(f"TEST accuracy={test_result['accuracy']:.4f}, "
          f"macro-F1={test_result['macro_f1']:.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
