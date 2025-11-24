# 矩阵论课程项目：不同优化器在图像分类任务上的性能比较

## 项目简介

本项目是矩阵论课程的实践作业，旨在比较不同优化器在图像分类任务上的性能表现。通过在STL-10数据集上训练ResNet18模型，我们评估了多种常用优化器的收敛速度、最终准确率和训练时间等指标。

## 项目结构

```
matrix/
├── data/
│   └── stl10_binary/          # STL-10 数据集
├── results/                   # 实验结果目录
│   ├── figures/              # 可视化图表
│   ├── experiment_report.txt # 实验报告
│   └── summary_results.csv   # 结果汇总CSV
├── different_Optimizers.py    # 主代码文件
└── README.md                  # 项目说明文档
```

## 数据集

- STL-10

## 实现的优化器

本项目实现并比较了以下优化器：

| 优化器 | 描述 |
|-------|------|
| SGD | 随机梯度下降 |
| SGD_Nesterov | 带动量的SGD |
| Adam | 自适应矩估计 |
| AdamW | 带权重衰减的Adam |
| RMSprop | 均方根传播 |
| Adagrad | 自适应梯度算法 |
| Adadelta | 自适应学习率算法 |
| NAdam | Nesterov-accelerated Adam |
| Adamax | Adam的无穷范数变体 |
| RAdam | 修正的Adam |

## 实验环境

- Python 3.8+

## 实验设置

- 模型：ResNet（自定义适配STL-10图像大小）
- 批量大小：64
- 训练轮数：50
- 学习率调度器：OneCycleLR
- 数据增强：AutoAugment
- 正则化：CutMix、权重衰减
- 训练设备：GPU（自动检测，若无则使用CPU）

## 运行说明

1. 确保已安装所需依赖：

```bash
pip install torch torchvision matplotlib numpy
```

2. 运行主程序：

```bash
python different_Optimizers.py
```

程序会自动：
- 加载并预处理STL-10数据集
- 初始化并训练模型（使用不同优化器）
- 评估模型性能
- 生成可视化结果
- 保存实验报告
 
本项目仅供学术研究使用。

---
