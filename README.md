# IMDB 情感二分类 —— BERT 微调

**方向**：自然语言处理（NLP）  
**任务**：IMDB 电影评论情感二分类（Positive / Negative）  
**模型**：BERT-base-uncased（微调 3 epoch）  
**最终指标**：Accuracy **94.09%** | F1-Score **94.11%**

---

## 环境配置

| 软件 | 版本 |
|------|------|
| Python | 3.14.5 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Transformers | 4.57.6 |
| Datasets | 5.0.0 |

## 数据准备

使用 HuggingFace Datasets 库自动下载 IMDB 数据集（50,000 条，已分割为 train/test）。

```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```

首次运行时会自动下载，也可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 训练命令

```bash
python src/train.py
```

或使用优化版本：

```bash
python src/train_new.py
```

训练参数（详见报告）：
- learning_rate: 2e-5
- batch_size: 16
- num_train_epochs: 3
- max_length: 512

## 评估命令

```bash
python src/evaluate.py
```

输出 Accuracy、F1-Score、Precision、Recall 及各错误类型分析。

## 推理命令

```bash
python src/inference.py
```

对输入文本进行情感预测（Positive / Negative）。

## 结果说明

| 指标 | 值 |
|------|-----|
| Accuracy | 94.09% |
| F1-Score | 94.11% |
| Negative Precision | 94.42% |
| Negative Recall | 93.71% |
| Positive Precision | 93.76% |
| Positive Recall | 94.46% |

可视化结果（位于 `NLP/results/`）：
- `loss_curve.png` — 训练损失曲线
- `confusion_matrix.png` — 混淆矩阵

详细分析见根目录 `NLP_Wangziyan6210.pdf`。

## 项目结构

```
├── NLP/
│   ├── src/
│   │   ├── train.py          # 训练脚本
│   │   ├── train_new.py      # 优化版训练脚本
│   │   ├── evaluate.py       # 评估脚本
│   │   ├── inference.py      # 推理脚本
│   │   └── data_loader.py    # 数据加载模块
│   ├── results/
│   │   ├── loss_curve.png
│   │   ├── confusion_matrix.png
│   │   └── eval_results.txt
│   └── models/               # 模型权重
├── CV/                       # （未选）
├── MultiModal/               # （未选）
├── requirements.txt
├── NLP_Wangziyan6210.pdf     # 完整考核报告
└── README.md
```

## 学术规范

- 使用 AI 工具（ChatGPT/Claude）辅助查阅资料和代码编写
- 参考文献来源已在报告中注明
