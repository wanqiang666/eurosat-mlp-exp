# EuroSAT-MLP-exp
In this experiment, a three-layer neural network classifier was built and trained from scratch, with the forward propagation and backward propagation implemented manually.
## 1. 项目目标

本项目的目标是：

- 使用纯 NumPy 手写一个简单的神经网络分类器；
- 在 EuroSAT 遥感图像数据集上完成 10 类分类任务；
- 实现完整训练、验证、测试流程；
- 支持保存最优模型、评估混淆矩阵、错误样本分析以及第一层权重可视化。  
  训练脚本会记录训练/验证损失、准确率和学习率曲线，并保存最佳验证集模型与最终测试结果。fileciteturn0file7L67-L112 fileciteturn0file7L114-L238

---

## 2. 项目结构

```text
.
├── data.py          # 数据读取、划分、标准化、DataLoader
├── layers.py        # 手写网络基础层：Linear / ReLU / Sigmoid / Tanh / Sequential
├── model.py         # MLPClassifier 定义
├── loss.py          # SoftmaxCrossEntropyLoss、L2 正则、准确率计算
├── optimizer.py     # 手写 SGD 优化器与学习率衰减
├── train.py         # 模型训练主流程
├── eval.py          # 测试集评估、混淆矩阵与误分类样本导出
├── search.py        # 网格搜索脚本
├── visualize.py     # 第一层权重可视化
└── EuroSAT_RGB/     # 数据集目录（需自行准备）
```

各模块职责如下：

- `data.py`：从 `EuroSAT_RGB` 目录递归读取各类别图像，将子文件夹名视为类别名；支持图像缩放、归一化、展平、训练/验证/测试划分、标准化以及数据划分与 scaler 的保存/加载。fileciteturn0file0L9-L20 fileciteturn0file0L21-L94 fileciteturn0file0L165-L299
- `layers.py`：实现全连接层、ReLU、Sigmoid、Tanh 和顺序容器 `Sequential`，支持前向传播、反向传播、参数与梯度管理。fileciteturn0file2L1-L140
- `model.py`：定义一个两层线性变换、中间带单个激活函数的 MLP 分类器，激活函数支持 `relu / sigmoid / tanh`。fileciteturn0file4L1-L58
- `loss.py`：实现数值稳定版 Softmax + Cross Entropy，同时支持 L2 正则损失与梯度叠加。fileciteturn0file3L1-L52
- `optimizer.py`：实现简易 SGD，支持学习率衰减和最小学习率下限。fileciteturn0file5L1-L45
- `train.py`：完成训练、验证、最优模型保存、曲线保存、最终测试与配置记录。fileciteturn0file7L114-L238
- `eval.py`：基于保存好的划分和 scaler 重新构建测试集，输出测试损失、准确率、混淆矩阵、误分类样本。fileciteturn0file1L1-L113 fileciteturn0file1L115-L239
- `search.py`：对学习率、隐藏层维度、权重衰减和激活函数做网格搜索。fileciteturn0file6L1-L112 fileciteturn0file6L135-L199
- `visualize.py`：对第一层权重进行排序、重塑为 RGB 图像或灰度热力图并保存总览图。fileciteturn0file8L1-L144 fileciteturn0file8L145-L229

---

## 3. 模型说明

项目中的分类器是一个简单的 MLP：

```text
输入图像向量
   ↓
Linear(input_dim, hidden_dim)
   ↓
Activation(ReLU / Sigmoid / Tanh)
   ↓
Linear(hidden_dim, num_classes)
   ↓
logits
```

其中：

- 输入维度 `input_dim = H × W × 3`；
- 默认图像大小为 `64 × 64`；
- 输出类别数为 10；
- 隐藏层维度 `hidden_dim` 可配置；
- 激活函数支持：
  - `relu`
  - `sigmoid`
  - `tanh`。fileciteturn0file4L5-L38

损失函数使用 **SoftmaxCrossEntropyLoss**，并可叠加对所有线性层权重的 **L2 正则项**。fileciteturn0file3L5-L19 fileciteturn0file3L36-L52

---

## 4. 数据集组织方式

代码默认数据集目录结构如下：

```text
EuroSAT_RGB/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

每个子文件夹会被视为一个类别，`data.py` 会自动扫描文件夹名、建立 `class_to_idx` 映射，并按固定顺序收集样本路径以增强可复现性。fileciteturn0file0L9-L20 fileciteturn0file0L34-L64

---

## 5. 环境依赖

本项目依赖较少，主要包括：

- Python 3.9+
- numpy
- matplotlib
- pillow

可使用如下命令安装：

```bash
pip install numpy matplotlib pillow
```

---

## 6. 训练流程

训练主流程位于 `train.py`，大致步骤如下：

1. 读取并展平全部图像数据；
2. 按 `train/val/test` 比例划分数据集；
3. 使用训练集拟合标准化器，并变换验证/测试集；
4. 构建 MLP 模型、损失函数与 SGD 优化器；
5. 按 epoch 迭代训练；
6. 在验证集上评估并保存最优模型；
7. 保存训练历史曲线；
8. 加载最佳模型，在测试集上做最终评估；
9. 将关键配置和结果写入 `config.json`。fileciteturn0file7L114-L238

默认训练参数示例（`train.py` 中主函数入口）：

```python
train(
    root_dir="./EuroSAT_RGB",
    image_size=(64, 64),
    batch_size=64,
    hidden_dim=128,
    activation="relu",
    lr=1e-2,
    lr_decay=0.95,
    min_lr=1e-5,
    weight_decay=1e-4,
    num_epochs=10,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    standardize=True,
    random_seed=42,
    save_dir="./outputs/128_4",
    best_model_name="best_model.npz",
)
```

对应地，训练中会保存：

- `best_model.npz`：最佳模型权重；
- `history.json` / `history.npz`：训练过程记录；
- `loss_curve.png`：损失曲线；
- `accuracy_curve.png`：准确率曲线；
- `lr_curve.png`：学习率变化曲线；
- `split.npz`、`scaler.npz`、`dataset_meta.json`：数据划分与标准化统计；
- `config.json`：训练配置与最终结果。fileciteturn0file7L67-L112 fileciteturn0file7L135-L238

---

## 7. 如何运行

### 7.1 训练模型

```bash
python train.py
```

如需自定义参数，可以直接修改 `train.py` 末尾 `train(...)` 中的配置，或在其他脚本中导入 `train` 函数调用。fileciteturn0file7L240-L256

### 7.2 评估模型

```bash
python eval.py
```

默认会读取：

- 训练输出目录：`./outputs/128_4`
- 评估输出目录：`./eval_outputs/128_4`
- 模型文件名：`best_model.npz`

评估脚本会输出：

- `confusion_matrix.npy`
- `confusion_matrix.txt`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `misclassified_samples.json`
- `eval_summary.json`。fileciteturn0file1L115-L239

### 7.3 网格搜索

```bash
python search.py
```

当前 `search.py` 中定义的网格为：

- `lr`: `[1e-2, 1e-3]`
- `hidden_dim`: `[128, 256]`
- `weight_decay`: `[1e-4, 1e-3]`
- `activation`: `["relu", "tanh"]`

共进行 16 组实验，并将结果保存为：

- `grid_results.json`
- `grid_results.csv`。fileciteturn0file6L135-L199

### 7.4 可视化第一层权重

```bash
python visualize.py
```

脚本会读取训练配置与模型参数，对第一层权重按 `l2` 或 `abs_mean` 进行排序，输出：

- 单个隐藏单元的 RGB 权重图；
- 单个隐藏单元的灰度热力图；
- 多个隐藏单元的总览图；
- `weight_summary.json`。fileciteturn0file8L145-L229

---

## 8. 结果解释

### 8.1 训练阶段

训练过程中会同时记录：

- `train_loss`
- `train_acc`
- `val_loss`
- `val_acc`
- `lr`

这些内容将用于绘制损失曲线、准确率曲线和学习率曲线，便于观察模型收敛情况和过拟合现象。fileciteturn0file7L67-L112

### 8.2 评估阶段

`eval.py` 会加载最佳模型，并在测试集上输出：

- 测试损失 `test_loss`
- 测试准确率 `test_acc`
- 混淆矩阵
- 误分类样本索引、真实标签与预测标签。fileciteturn0file1L21-L57 fileciteturn0file1L115-L239

混淆矩阵可以帮助分析：

- 哪些类别最容易被正确识别；
- 哪些类别之间容易混淆；
- 模型在不同地物类型上的分类偏差。

### 8.3 权重可视化

第一层权重可以被重塑为 `H × W × 3` 的图像形式。虽然这不是卷积核，但依然可以观察模型对输入空间和颜色通道的敏感模式。`visualize.py` 还会记录每个隐藏单元的评分以及 RGB 通道均值，方便做进一步分析。fileciteturn0file8L52-L143 fileciteturn0file8L145-L229

---

## 9. 可复现性设计

项目中包含多处为可复现性服务的设计：

- 数据划分使用固定随机种子；
- 样本文件名按排序顺序读取；
- 训练时保存 `split.npz` 与 `scaler.npz`；
- 评估阶段不重新随机划分，而是直接复用训练阶段保存的划分与标准化统计量。fileciteturn0file0L43-L64 fileciteturn0file0L165-L299 fileciteturn0file1L133-L171

---

## 10. 项目特点

本项目适合作为**机器学习 / 神经网络基础课程作业**，特点包括：

- 纯 NumPy 手写实现，便于理解前向传播与反向传播；
- 模块划分清晰，便于扩展；
- 包含完整训练、验证、测试、搜索、可视化流程；
- 不依赖 PyTorch / TensorFlow，能够更直接地展示神经网络底层机制。fileciteturn0file2L1-L140 fileciteturn0file3L1-L52 fileciteturn0file5L1-L45

---

## 11. 后续可改进方向

后续可以进一步扩展：

- 增加更多隐藏层，构建更深的 MLP；
- 引入 Dropout、Batch Normalization 等机制；
- 增加更丰富的超参数搜索策略；
- 对误分类样本做可视化展示；
- 改为卷积神经网络（CNN）以提升图像分类效果；
- 加入命令行参数解析，使训练与评估更灵活。

---

## 12. 说明

本 README 根据项目源码自动整理生成，内容与当前代码实现保持一致。若你后续修改了默认参数、输出目录或搜索空间，建议同步更新本文档中的对应部分。
