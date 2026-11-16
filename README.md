# 基于加速度计与陀螺仪的人体活动识别：特征级融合实验

本项目使用同一部手机的三轴加速度计和三轴陀螺仪信号，识别六种日常活动。包含两个单传感器基线（A、B）、简单特征拼接融合（C）、样本自适应门控融合与传感器随机缺失训练（D），以及传感器缺失和高斯噪声测试。

## 数据来源与任务

- 数据集：[UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)，DOI：[10.24432/C54S4K](https://doi.org/10.24432/C54S4K)。请从官方页面下载 `UCI HAR Dataset.zip`；本仓库不包含完整原始数据。
- 任务：每个长度为 128 个时间点的窗口预测 6 类活动：`WALKING`、`WALKING_UPSTAIRS`、`WALKING_DOWNSTAIRS`、`SITTING`、`STANDING`、`LAYING`。
- 两种传感器：`total_acc_x/y/z` 作为加速度计输入，`body_gyro_x/y/z` 作为陀螺仪输入。`body_acc` 是加速度计信号的派生分量，本实验没有把它当成第三种传感器。
- 数据划分：保留 UCI 官方测试集。脚本以随机种子 42，从官方训练集按**受试者**划出 4 人作验证集；本次划分的验证受试者为 17、25、26、30。训练、验证、测试分别有 5,800、1,552、2,947 个窗口。标准化均值和标准差只由训练集计算。

## 文件说明

```text
README.md
requirements.txt
prepare_uci_har.py          # 读取原始 ZIP，按受试者划分并标准化
train_acc_baseline.py        # A：仅加速度计
train_gyro_baseline.py       # B：仅陀螺仪
train_concat_fusion.py       # C：双分支特征拼接
train_gated_fusion.py        # D：双分支门控融合＋传感器随机缺失训练
evaluate_robustness.py      # A～D 汇总、C/D 缺失与噪声测试
plot_robustness.py          # 被上一脚本自动调用，生成图片
demo.py                     # 六条公开测试样本的命令行演示
demo_assets/
  sample_demo.npz           # 每类一条样本，仅供演示
  best_gated_cnn.pt         # 已训练 D 模型，仅供演示
```

下文命令均在**仓库根目录**的 PowerShell 中运行。Windows 用户可在文件资源管理器中打开存放 `README.md` 的文件夹，点击顶部地址栏，输入 `powershell` 并回车。路径含中文或空格时请加引号。首次复现完整实验需要下载官方数据；只想看小样本演示可直接跳到“演示”一节。

## 运行环境

本次实验在 Windows、Python 3.14.6、NumPy 2.4.6、PyTorch 2.15.0+cpu、Matplotlib 3.11.0 下运行。先确认 `python --version` 能显示 Python 版本，然后安装依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import numpy, torch, matplotlib; print(numpy.__version__, torch.__version__, matplotlib.__version__)"
```

PyTorch 安装命令以其[官方安装说明](https://pytorch.org/get-started/locally/)为准。本项目使用 CPU，不需要 CUDA。若系统使用 `py` 启动 Python，将以下命令中的 `python` 一并换成 `py`。

## 从原始数据复现实验

### 1. 整理数据

将 `--input` 后的路径换成从 UCI 下载的 ZIP 实际位置。脚本也支持外层 ZIP 中再包含 `UCI HAR Dataset.zip` 的情况，以及已解压的目录。

```powershell
python prepare_uci_har.py --input "D:\你的下载目录\UCI HAR Dataset.zip" --output ".\uci_har_ready"
```

完成后会生成 `uci_har_ready/train.npz`、`val.npz`、`test.npz` 和 `metadata.json`。每个 `.npz` 包含 `X_acc`、`X_gyro`、`y`、`subject`；两路信号形状均为 `(样本数, 3, 128)`，`y` 为 0～5。请勿在训练时重新随机划分窗口。

### 2. 训练 A～D

四组实验使用相同的数据划分、种子 42、batch size 64、Adam 学习率 0.001、最多 25 轮。以验证集 Macro-F1 选择最佳轮次；连续 5 轮不提高则提前停止。测试集仅在选出最佳模型后用于最终评估。

```powershell
python train_acc_baseline.py --data ".\uci_har_ready" --output ".\experiment_A_acc"
python train_gyro_baseline.py --data ".\uci_har_ready" --output ".\experiment_B_gyro"
python train_concat_fusion.py --data ".\uci_har_ready" --output ".\experiment_C_concat"
python train_gated_fusion.py --data ".\uci_har_ready" --output ".\experiment_D_gated"
```

- **A/B**：相同的两层 1D CNN，分别输入加速度计或陀螺仪的 3 个通道。
- **C**：两个独立 CNN 各输出 64 维特征，拼接为 128 维后分类；属于**特征级融合**。
- **D**：保留 C 的双分支特征提取器，门控网络按样本输出两路权重，缩放各自的 64 维特征后拼接分类。训练时以 0.30 的总概率随机隐藏一路传感器（隐藏加速度和隐藏陀螺仪各占约 0.15）；不会同时隐藏两路。验证和正常测试时两路均可用。

每个实验目录包含 `best_*.pt`、`metrics.json`、`history.csv`、`confusion_matrix.csv`、`test_predictions.csv`。D 另含 `test_gate_weights.csv`。模型训练和测试代码只使用本次整理出的数据，不依赖外部预训练权重。

### 3. 量化分析与图片

当四个实验目录均已生成后运行：

```powershell
python evaluate_robustness.py --base "."
```

脚本不会重新训练模型。它读取同一官方测试集，先校验 C/D 的正常成绩与保存的 `metrics.json` 一致，然后：

1. **传感器缺失**：将缺失那一路的标准化输入置零；D 同时收到可用性标志，并将缺失一路的权重强制设为 0。
2. **传感器噪声**：只对一路标准化后的信号加入零均值高斯噪声，标准差分别为 0.1、0.3、0.5。每个条件重复 3 次，C 与 D 使用完全相同的扰动。这里的噪声强度是**标准化单位**，不是原始传感器的物理单位。

结果保存在 `experiment_4_robustness/`：

- `baseline_results.csv`：A～D 正常测试的 Accuracy 与 Macro-F1；
- `robustness_summary.csv`：每种扰动下的平均成绩、标准差和相对正常输入的下降百分点；
- `robustness_raw.csv`：各次噪声重复的原始成绩；
- `confusion_matrices.png`、`missing_modality.png`、`noise_curve.png`：报告可用图片；
- `protocol.json`：扰动和随机种子的记录。

`macro_f1_drop_pp = 100 × (正常 Macro-F1 − 扰动后 Macro-F1)`；正值表示下降，负值表示有小幅上升。

## 小样本演示

仓库附有从 UCI 官方测试集中选取的六条样本（每类一条）及 D 的模型权重，下载仓库并安装 PyTorch、NumPy 后即可运行：

```powershell
python demo.py
python demo.py --missing gyro
```

程序会打印真实类别、预测类别和两路门控权重。这个六样本演示仅用于展示推理过程；**表格中的整体成绩由完整测试集计算**。完成数据整理与训练后，也可对完整测试集的前几条样本演示：

```powershell
python demo.py --data ".\uci_har_ready" --checkpoint ".\experiment_D_gated\best_gated_cnn.pt" --index 0 --count 6
```

## 本次实验结果

本表为一次固定种子、固定受试者划分的测试结果；不是多次重新训练的均值。

| 模型 | 最佳验证轮次 | 测试 Accuracy | 测试 Macro-F1 |
| --- | ---: | ---: | ---: |
| A：仅加速度计 | 14 | 85.61% | 85.57% |
| B：仅陀螺仪 | 23 | 75.36% | 75.23% |
| C：特征拼接 | 4 | 87.31% | 87.10% |
| D：门控＋传感器随机缺失训练 | 25 | **91.25%** | **91.16%** |

| 测试条件 | C Macro-F1 | D Macro-F1 |
| --- | ---: | ---: |
| 正常输入 | 87.10% | 91.16% |
| 缺失加速度 | 36.65% | 63.02% |
| 缺失陀螺仪 | 32.62% | 85.20% |
| 加速度噪声标准差 0.5 | 84.97% | 68.65% |

结论：D 在正常输入和单路传感器缺失时优于 C；强加速度噪声下 D 下降更多，不能称为对所有干扰都更鲁棒。D **同时改变了融合结构和训练增强**，本实验只能将提升归因于这一组合，不能单独证明门控模块的贡献。若需要隔离门控作用，应增加相同随机缺失训练条件下的拼接融合消融实验。

## 仓库与提交

上传代码仓库时，把本 README、`requirements.txt`、上述 `.py` 文件和 `demo_assets/` 放在仓库根目录。`uci_har_ready/`、`experiment_A_acc/` 至 `experiment_D_gated/`、`experiment_4_robustness/` 可由命令重新生成；完整 UCI 原始数据请通过上方官方链接下载。提交材料中填写该公开代码仓库的 HTTPS 链接，最终电子压缩包按课程要求另行打包报告、PPT 和链接信息。

## 引用

使用数据集时请引用 UCI 官方页面所列的原始论文和数据集记录：Reyes-Ortiz 等，*Human Activity Recognition Using Smartphones*，UCI Machine Learning Repository，2013，DOI: 10.24432/C54S4K。
