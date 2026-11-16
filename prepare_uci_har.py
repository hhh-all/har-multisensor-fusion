"""Prepare UCI HAR accelerometer/gyroscope data for fusion experiments.

Usage:
  python prepare_uci_har.py --input "D:\\data\\UCI HAR Dataset.zip" --output "D:\\data\\uci_har_ready"
  python prepare_uci_har.py --input "D:\\data\\UCI HAR Dataset" --output "D:\\data\\uci_har_ready"

Requires: numpy
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np


SIGNALS = {
    "acc": ("total_acc_x", "total_acc_y", "total_acc_z"),
    "gyro": ("body_gyro_x", "body_gyro_y", "body_gyro_z"),
}
ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


class DatasetReader:
    def __init__(self, source):
        self.source = Path(source).expanduser().resolve()
        self.archive = None
        if self.source.is_file() and self.source.suffix.lower() == ".zip":
            outer = zipfile.ZipFile(self.source)
            if any(name.endswith("train/Inertial Signals/total_acc_x_train.txt") for name in outer.namelist()):
                self.archive = outer
            else:
                inner_zips = [name for name in outer.namelist() if name.lower().endswith(".zip")]
                if len(inner_zips) != 1:
                    outer.close()
                    raise FileNotFoundError("外层 ZIP 中应包含唯一的 UCI HAR Dataset.zip")
                inner_bytes = io.BytesIO(outer.read(inner_zips[0]))
                outer.close()
                self.archive = zipfile.ZipFile(inner_bytes)
            self.names = self.archive.namelist()
        elif self.source.is_dir():
            hits = list(self.source.rglob("total_acc_x_train.txt"))
            hits = [p for p in hits if p.parent.name == "Inertial Signals" and p.parent.parent.name == "train"]
            if len(hits) != 1:
                raise FileNotFoundError("应能找到唯一的 train/Inertial Signals/total_acc_x_train.txt")
            self.root = hits[0].parent.parent.parent
        else:
            raise FileNotFoundError(f"输入不是现有的 ZIP 或目录：{self.source}")

    def load(self, relative_path, dtype=np.float32):
        if self.archive:
            suffix = "/" + relative_path.replace("\\", "/")
            hits = [name for name in self.names if name == suffix[1:] or name.endswith(suffix)]
            if len(hits) != 1:
                raise FileNotFoundError(f"ZIP 中应有唯一的 {relative_path}")
            with self.archive.open(hits[0]) as stream:
                return np.loadtxt(stream, dtype=dtype, ndmin=2)
        path = self.root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        return np.loadtxt(path, dtype=dtype, ndmin=2)

    def close(self):
        if self.archive:
            self.archive.close()


def load_split(reader, split):
    arrays = {}
    for modality, names in SIGNALS.items():
        channels = []
        for name in names:
            rel = f"{split}/Inertial Signals/{name}_{split}.txt"
            channel = reader.load(rel)
            if channel.shape[1] != 128:
                raise ValueError(f"{rel}: 应为每行 128 个时间点，得到 {channel.shape}")
            channels.append(channel)
        arrays[modality] = np.stack(channels, axis=1)  # (样本数, 3, 128)

    y = reader.load(f"{split}/y_{split}.txt", np.int64).reshape(-1) - 1
    subject = reader.load(f"{split}/subject_{split}.txt", np.int64).reshape(-1)
    counts = {arrays["acc"].shape[0], arrays["gyro"].shape[0], len(y), len(subject)}
    if len(counts) != 1:
        raise ValueError(f"{split}: 六路信号、标签、受试者编号的行数不一致：{counts}")
    if not np.isfinite(arrays["acc"]).all() or not np.isfinite(arrays["gyro"]).all():
        raise ValueError(f"{split}: 信号包含非有限数值")
    if not np.isin(y, np.arange(6)).all():
        raise ValueError(f"{split}: 标签应是原始的 1～6")
    arrays.update(y=y, subject=subject)
    return arrays


def take(data, mask):
    return {key: value[mask] for key, value in data.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UCI HAR ZIP 文件或解压后的目录")
    parser.add_argument("--output", required=True, help="整理结果目录")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reader = DatasetReader(args.input)
    try:
        original_train = load_split(reader, "train")
        test = load_split(reader, "test")
    finally:
        reader.close()

    if np.intersect1d(original_train["subject"], test["subject"]).size:
        raise ValueError("官方训练集与测试集出现相同受试者，请检查数据来源")

    subjects = np.unique(original_train["subject"])
    if len(subjects) < 2:
        raise ValueError("训练集至少需要两名受试者才能划出验证集")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(subjects)
    n_val = max(1, round(len(subjects) * 0.2))
    val_subjects = subjects[:n_val]
    val_mask = np.isin(original_train["subject"], val_subjects)
    train = take(original_train, ~val_mask)
    val = take(original_train, val_mask)

    statistics = {}
    for modality in SIGNALS:
        mean = train[modality].mean(axis=(0, 2), keepdims=True)
        std = train[modality].std(axis=(0, 2), keepdims=True)
        std = np.maximum(std, 1e-6)
        statistics[modality] = {
            "mean": mean.reshape(-1).tolist(),
            "std": std.reshape(-1).tolist(),
        }
        for split in (train, val, test):
            split[modality] = ((split[modality] - mean) / std).astype(np.float32)

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("val", val), ("test", test)):
        np.savez_compressed(out / f"{name}.npz", X_acc=data["acc"], X_gyro=data["gyro"],
                            y=data["y"], subject=data["subject"])
        print(f"{name}: {len(data['y'])} samples, X_acc={data['acc'].shape}, X_gyro={data['gyro'].shape}")

    meta = {
        "source": str(Path(args.input).expanduser().resolve()),
        "seed": args.seed,
        "activity_names_for_labels_0_to_5": ACTIVITIES,
        "validation_subjects": sorted(map(int, val_subjects)),
        "normalization_from_training_only": statistics,
        "note": "X_acc=total_acc_xyz; X_gyro=body_gyro_xyz; 每条数据形状为 (3,128)。",
    }
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
