"""Print a small, reproducible prediction demo with the saved D model.

Example:
  python demo.py
  python demo.py --data ./uci_har_ready --checkpoint ./experiment_D_gated/best_gated_cnn.pt
  python demo.py --missing gyro
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from train_gated_fusion import GatedFusionCNN


NAMES = (
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
)


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=folder / "demo_assets" / "sample_demo.npz",
                        help=".npz 文件，或包含 test.npz 的已整理数据目录")
    parser.add_argument("--checkpoint", type=Path,
                        default=folder / "demo_assets" / "best_gated_cnn.pt")
    parser.add_argument("--index", type=int, default=0, help="第一条展示的测试样本序号，从 0 开始")
    parser.add_argument("--count", type=int, default=6, help="展示多少条连续样本")
    parser.add_argument("--missing", choices=("none", "acc", "gyro"), default="none")
    args = parser.parse_args()

    data_path = args.data.expanduser().resolve()
    if data_path.is_dir():
        data_path = data_path / "test.npz"
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    with np.load(data_path, allow_pickle=False) as data:
        acc = data["X_acc"].astype(np.float32)
        gyro = data["X_gyro"].astype(np.float32)
        y = data["y"].astype(np.int64)
        subject = data["subject"].astype(np.int64)
    if acc.shape != gyro.shape or acc.shape[1:] != (3, 128):
        raise ValueError("X_acc 和 X_gyro 应具有相同的 (N,3,128) 形状")
    if args.index < 0 or args.index >= len(y) or args.count < 1:
        parser.error(f"index 必须在 0～{len(y)-1}，count 必须大于 0")

    torch.set_num_threads(4)
    model = GatedFusionCNN()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stop = min(args.index + args.count, len(y))
    print(f"Saved D model, epoch={checkpoint['epoch']}; mode={args.missing}; samples={args.index}..{stop-1}")
    print("index  subject  true                 predicted            acc_weight  gyro_weight")
    with torch.no_grad():
        for i in range(args.index, stop):
            a = torch.from_numpy(acc[i : i + 1].copy())
            g = torch.from_numpy(gyro[i : i + 1].copy())
            mask_acc = torch.ones((1, 1))
            mask_gyro = torch.ones((1, 1))
            if args.missing == "acc":
                a.zero_()
                mask_acc.zero_()
            elif args.missing == "gyro":
                g.zero_()
                mask_gyro.zero_()
            logits, weight_acc = model(a, g, mask_acc, mask_gyro)
            pred = int(logits.argmax(dim=1).item())
            wa = float(weight_acc.item())
            print(f"{i:5d}  {int(subject[i]):7d}  {NAMES[int(y[i])]:20s} "
                  f"{NAMES[pred]:20s} {wa:10.3f}  {1-wa:11.3f}")


if __name__ == "__main__":
    main()
