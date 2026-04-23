# EuroSAT-MLP-exp
In this experiment, a three-layer neural network classifier was built and trained from scratch, with the forward propagation and backward propagation implemented manually.

## 1. 项目目标

本项目的目标是：

- 使用纯 NumPy 手写一个简单的神经网络分类器；
- 在 EuroSAT 遥感图像数据集上完成 10 类分类任务；
- 实现完整训练、验证、测试流程；
- 支持保存最优模型、评估混淆矩阵、错误样本分析以及第一层权重可视化。  
  训练脚本会记录训练/验证损失、准确率和学习率曲线，并保存最佳验证集模型与最终测试结果。

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
  - `tanh`。

损失函数使用 **SoftmaxCrossEntropyLoss**，并可叠加对所有线性层权重的 **L2 正则项**。
---

## 4. 环境依赖

- Python 3.9+
- numpy
- matplotlib
- pillow


## 5. 如何运行

### 5.1 训练模型

```bash
python train.py
```

如需自定义参数，可以直接修改 `train.py` 末尾 `train(...)` 中的配置，或在其他脚本中导入 `train` 函数调用。

### 5.2 评估模型

```bash
python eval.py
```


评估脚本会输出：

- `confusion_matrix.npy`
- `confusion_matrix.txt`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `misclassified_samples.json`
- `eval_summary.json`。

### 5.3 网格搜索

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
- `grid_results.csv`。

### 5.4 可视化第一层权重

```bash
python visualize.py

---



