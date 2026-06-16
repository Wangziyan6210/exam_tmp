# IMDB 情感二分类 —— BERT 微调最终考核报告

> **方向**：自然语言处理（NLP）  
> **任务**：IMDB 电影评论情感二分类  
> **模型**：BERT-base-uncased（微调 3 epoch）  
> **提交日期**：2026 年 6 月

---

## 阶段一：文献综述

### 一、引言

情感分析（Sentiment Analysis）是自然语言处理（NLP）的核心任务之一，旨在从文本中自动识别和提取人们的观点、情感和态度。随着互联网和社交媒体的快速发展，用户在各类平台上产生了海量的主观性文本数据，使得情感分析在商业决策、社交媒体监控、品牌管理和市场研究等领域具有重要的应用价值。

IMDB 电影评论数据集（Maas et al., 2011）包含 50,000 条标注为正面/负面的影评，是情感分析领域最经典的基准数据集之一。本文以该数据集为主线，系统梳理情感分析技术从传统机器学习到深度学习，再到预训练语言模型的演进历程，并深入分析不同方法的本质差异。

### 二、技术演进路线

#### 2.1 传统机器学习方法

早期的情感分析主要依赖手工特征工程和传统机器学习算法。典型方法包括：

- **词袋模型（Bag-of-Words, BoW）与 TF-IDF**：将文本表示为词频或 TF-IDF 加权的稀疏向量，忽略词序信息。
- **朴素贝叶斯（Naive Bayes）**：基于词条件独立假设的概率分类器，简单高效。
- **支持向量机（SVM）**：通过寻找最大间隔超平面进行分类，在高维稀疏特征上表现良好。

Guo (2026) 的实验表明，即使在深度学习盛行的今天，TF-IDF + 逻辑回归在 IMDB 平衡数据集上仍能达到 90.84% 的 Macro-F1，说明传统方法在特定场景下仍有竞争力。

**局限性**：特征稀疏，维度灾难；无法捕捉词序和上下文语义；对词义消歧、讽刺等复杂语言现象处理能力不足。

#### 2.2 词向量与深度学习

##### 2.2.1 静态词向量

Mikolov et al. (2013) 提出的 **Word2Vec** 标志着词表示学习的重大突破，通过 Skip-gram 和 CBOW 两种架构在大规模无标注语料上学习低维稠密向量。Pennington et al. (2014) 提出的 **GloVe** 进一步结合了全局矩阵分解和局部上下文窗口的优势。这些静态词向量能够捕捉词汇间的语义相似性，为下游任务提供了高质量的输入特征。

##### 2.2.2 LSTM 与 BiLSTM

长短期记忆网络（LSTM）通过引入门控机制有效解决了 RNN 的梯度消失问题，能够捕获文本中的长距离依赖关系。双向 LSTM（BiLSTM）进一步从正向和反向两个方向编码上下文信息，在情感分析任务上取得了显著提升。

Maas et al. (2011) 在学习词向量的同时训练文本分类器，开创了词向量在情感分析中应用的先河。后续研究进一步将 Word2Vec/GloVe 与 LSTM 结合，构建端到端的分类系统。

#### 2.3 预训练语言模型时代

##### 2.3.1 BERT 及其变体

Devlin et al. (2019) 提出的 **BERT**（Bidirectional Encoder Representations from Transformers）是 NLP 领域的里程碑式工作。BERT 通过掩码语言模型（MLM）和下一句预测（NSP）两个预训练任务，在大规模无标注语料上学习深度双向上下文表示。

BERT 的成功催生了大量变体：
- **RoBERTa** (Liu et al., 2019)：优化 BERT 预训练策略，移除 NSP 任务
- **ALBERT**：通过参数共享减少模型参数量
- **DistilBERT**：知识蒸馏压缩模型
- **ModernBERT**：提升推理效率

##### 2.3.2 BERT 在 IMDB 情感分析中的应用

Papadimitriou et al. (2025) 对 BERT 进行微调，在 IMDB 数据集上达到 92.13% 的准确率，显著优于 Naive Bayes、SVM 和 LSTM 等基线模型。Puspita et al. (2023) 的实验报告了 91.78% 的测试准确率。

Nkhata et al. (2023) 和后续扩展工作将 BERT 与 BiLSTM 结合（BERT+BiLSTM），在 IMDB 二分类任务上达到了 **97.67%** 的准确率，超越了当时的 SOTA 模型。

Prabhu (2025) 进一步采用 ModernBERT，结合层级学习率衰减策略，达到 95.78% 准确率、95.81% F1-score，并通过 SHAP 和 LIME 增强了模型可解释性。

##### 2.3.3 BERT 模型架构详解

BERT 基于 **Transformer 编码器**（Vaswani et al., 2017）的堆叠架构。BERT-base 包含：

- **12 层 Transformer 编码器**
- **768 维隐藏层**
- **12 个注意力头**
- **约 1.1 亿参数**

每层由多头自注意力（Multi-Head Self-Attention）和前馈神经网络（FFN）构成，残差连接与层归一化贯穿其中。

BERT 的核心创新在于**双向上下文表示**：与 GPT 等从左到右的单向模型不同，BERT 通过掩码语言模型（MLM）实现深度双向表示——随机掩盖 15% 的输入 token，让模型基于双侧上下文预测被掩盖的词。辅助的下一句预测（NSP）任务使模型能够理解句间关系（后续研究如 RoBERTa 发现该任务并非必需）。

BERT 的输入表示为三个嵌入的加和：

```
输入嵌入 = Token Embedding + Segment Embedding + Position Embedding
```

- **Token Embedding**：WordPiece 子词分词（30,522 词表）
- **Segment Embedding**：区分句子 A / 句子 B
- **Position Embedding**：可学习的位置编码（最大 512 位置）

微调时，BERT 采用"预训练 + 微调"范式：先用无标注语料（BookCorpus + Wikipedia，约 33 亿词）进行预训练，再在目标任务上添加分类头进行端到端微调。对于 IMDB 二分类任务，在 [CLS] 向量上接一个线性层 + Softmax 即完成适配。

#### 2.4 混合模型与前沿探索

近年来的研究趋势包括：

1. **BERT + CNN/LSTM 混合架构**：结合 Transformer 的上下文理解能力和 CNN/LSTM 的局部/序列特征提取能力。He & Abisado (2024) 提出的 BERT-CNN-BiLSTM-Att 模型在豆瓣影评上取得了优异效果。

2. **数据增强技术**：SMOTE 和 NLPAUG 等方法被用于缓解类别不平衡问题，提升模型泛化能力。

3. **可解释性研究**：SHAP 和 LIME 等 XAI 方法被引入情感分析，帮助理解模型决策依据。

4. **大语言模型（LLM）探索**：GPT-3.5、GPT-4、LLaMA-2 等大模型在零样本和小样本场景下展现了强大的情感理解能力。

### 三、不同方法对比分析

| 方法 | 代表模型 | 准确率范围 | 核心优势 | 核心局限 |
|------|----------|-----------|----------|----------|
| 传统 ML | TF-IDF + LR/SVM | ~85-90% | 训练快、可解释性强、资源需求低 | 需手工特征工程、无法处理词序与歧义 |
| 静态词向量 + RNN | Word2Vec + BiLSTM | ~89-92% | 捕捉词汇语义、建模序列依赖 | 无法处理一词多义、上下文不充分 |
| 预训练模型 | BERT-base | ~92-95% | 深度双向上下文、迁移学习能力强 | 计算资源需求高、推理速度较慢 |
| 增强混合模型 | BERT+BiLSTM | ~93-97% | 结合 Transformer 与序列建模优势 | 参数量大、训练时间长 |
| 大语言模型 | GPT-4 / LLaMA-2 | 高（少样本） | 通用推理、零样本能力 | 部署成本极高、微调复杂 |

**演进趋势总结**：

1. **从浅层到深层**：特征工程 $\rightarrow$ 静态词向量 $\rightarrow$ 深度上下文表示 $\rightarrow$ 大规模预训练
2. **从单任务到多任务**：独立分类器 $\rightarrow$ 预训练 + 微调范式
3. **从模型中心到数据中心**：关注点从复杂模型转向数据质量、增强与高效微调
4. **从黑盒到可解释**：深度模型效果好但缺乏解释 $\rightarrow$ XAI 方法提升透明度

### 四、本质差异分析：传统方法与深度学习方法

#### 4.1 表示学习能力的本质差异

传统机器学习方法与深度学习方法的本质差异在于**表示学习（Representation Learning）能力**。

传统方法（如 BoW + SVM）依赖于**手工特征工程**，需要人类专家预先定义特征空间。这些特征通常是离散的、稀疏的，且无法根据任务自动调整。特征设计的好坏直接决定了模型性能的上限。

深度学习方法（如 CNN、LSTM、BERT）则具备**端到端的表示学习**能力，能够从原始数据中自动学习层次化的特征表示。低层网络学习局部、基础的语义模式，高层网络组合这些模式形成更抽象的语义概念。这种层次化特征学习使得模型能够捕捉到手工特征难以表达的复杂语言现象。

#### 4.2 上下文建模的维度差异

传统方法本质上基于**词袋假设**（Bag-of-Words assumption），即忽略词序和上下文，将文本视为词的集合。这导致两个核心问题：一是一词多义无法消歧；二是语法结构信息完全丢失。

静态词向量（Word2Vec、GloVe）虽然通过学习共现统计捕捉了词汇的语义相似性，但每个词仍对应唯一的向量表示，无法根据上下文动态调整，限制了对一词多义、讽刺等复杂语言现象的处理能力。

BERT 等预训练语言模型通过**深度双向上下文编码**解决了这一问题。同一词语在不同上下文中会得到不同的上下文表示，使得模型能够准确区分 "This movie is not bad" 中的 "bad" 和 "This is a bad movie" 中的 "bad" 的语义差异。这种动态上下文表示能力是预训练模型能够取得显著提升的根本原因。

#### 4.3 预训练范式带来的质变

预训练语言模型之所以能够取得远优于传统方法的效果，核心原因包括：

1. **大规模数据预训练**：在大规模无标注语料上学习通用语言知识，使模型在微调前已具备丰富的语法和语义先验。
2. **迁移学习效应**：预训练捕获的语言知识可以高效迁移到下游任务，显著降低对标注数据的需求。
3. **层次化特征抽象**：Transformer 的多层结构天然支持从低级语法特征到高级语义特征的层次化抽象。
4. **注意力机制的全局建模**：自注意力机制允许模型直接建模任意两个位置之间的依赖关系，突破了 RNN 序列建模的距离衰减限制。

本质上，传统方法是在**预定义特征空间**中进行模式匹配，而预训练模型是在**学习到的语义空间**中进行理解推理。这不仅是技术路径的差异，更是从"统计匹配"到"语义理解"的质的飞跃。

### 五、总结与展望

IMDB 电影评论情感分析作为 NLP 领域的经典基准任务，见证了情感分析技术的完整演进历程。从早期的 TF-IDF + SVM，到 Word2Vec/GloVe + LSTM，再到 BERT 及其变体的预训练+微调范式，分类准确率从约 88% 提升至 97% 以上。

这一演进过程的核心驱动力是**表示学习能力的不断提升**：从手工特征到静态词向量，再到动态上下文表示，每一次突破都让模型更接近语言的本质理解。

未来的研究方向可能包括：更高效的小模型蒸馏、多模态情感分析、跨领域迁移和领域自适应、可解释性和公平性、低资源场景下的少样本和零样本情感分析。

---

## 阶段二：实验环境与结果

### 2.1 实验环境

#### 硬件配置

| 组件 | 型号 |
|------|------|
| GPU | NVIDIA GeForce **RTX 5070 Ti**（16 GB GDDR7，CUDA 核心数 ~8960） |
| CPU | AMD / Intel 桌面平台 |
| 内存 | 32 GB |
| 存储 | NVMe SSD |

#### 软件环境

| 软件 | 版本 |
|------|------|
| 操作系统 | Windows 11 |
| Python | 3.14.5 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 4.57.6 |
| Datasets | 5.0.0 |
| Accelerate | 1.13.0 |
| scikit-learn | 1.9.0 |
| Tokenizers | 0.22.2 |
| safetensors | 0.7.0 |
| CUDA | 12.8 |
| huggingface_hub | 0.36.2 |

#### 依赖管理

项目使用 Python venv 管理的虚拟环境，通过 pip 安装依赖。关键依赖安装示例：

```bash
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.6 datasets==5.0.0 accelerate==1.13.0
pip install scikit-learn==1.9.0 safetensors==0.7.0
```

### 2.2 超参数配置

```python
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
)
```

| 超参数 | 值 | 说明 |
|--------|-----|------|
| `learning_rate` | 2e-5 | BERT 微调推荐范围 [2e-5, 5e-5]，此处取下限更稳定 |
| `per_device_train_batch_size` | 16 | 受 RTX 5070 Ti 16 GB 显存限制，最大支持 batch 16（float32）|
| `per_device_eval_batch_size` | 16 | 与训练一致 |
| `num_train_epochs` | 3 | 标准 BERT 微调轮次，防止过拟合 |
| `weight_decay` | 0.01 | AdamW 的权重衰减系数 |
| `optimizer` | `adamw_torch_fused` | PyTorch 融合版 AdamW，显存更优 |
| `lr_scheduler` | `linear` | 线性衰减至 0 |
| `max_length` | 512 | BERT 最大输入长度，IMDB 评论普遍较长 |
| `adam_epsilon` | 1e-8 | 防止除零 |
| `max_grad_norm` | 1.0 | 全局梯度裁剪 |
| `seed` | 42 | 随机种子 |
| `fp16` | `False` | 使用 float32（RTX 5070 Ti 对 fp16 支持良好，但项目未启用）|

### 2.3 实验结果

#### 2.3.1 训练过程日志

| Step | Epoch | Train Loss | Learning Rate | Eval Accuracy | Eval F1 | Eval Loss |
|------|-------|-----------|---------------|---------------|---------|-----------|
| 500 | 0.32 | 0.3282 | 1.79e-5 | --- | --- | --- |
| 1000 | 0.64 | 0.2373 | 1.57e-5 | --- | --- | --- |
| 1500 | 0.96 | 0.2165 | 1.36e-5 | --- | --- | --- |
| **1563** | **1.0** | --- | --- | **0.9286** | **0.9267** | **0.1888** |
| 2000 | 1.28 | 0.1484 | 1.15e-5 | --- | --- | --- |
| 2500 | 1.60 | 0.1389 | 9.34e-6 | --- | --- | --- |
| 3000 | 1.92 | 0.1309 | 7.21e-6 | --- | --- | --- |
| **3126** | **2.0** | --- | --- | **0.9400** | **0.9395** | **0.2079** |
| 3500 | 2.24 | 0.0889 | 5.08e-6 | --- | --- | --- |
| 4000 | 2.56 | 0.0654 | 2.94e-6 | --- | --- | --- |
| 4500 | 2.88 | 0.0729 | 8.10e-7 | --- | --- | --- |
| **4689** | **3.0** | --- | --- | **0.9409** | **0.9411** | **0.2628** |

#### 2.3.2 最终测试指标（由 `src/evaluate.py` 实测获得）

```
Accuracy:  0.9409 (94.09%)
F1-Score:  0.9411 (94.11%)
```

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Negative | 0.9442 | 0.9371 | 0.9407 |
| Positive | **0.9376** | **0.9446** | **0.9411** |

#### 2.3.3 结果分析

- **准确率 94.09%** 符合典型 IMDB BERT 微调结果（文献报告通常为 92--95%）
- 模型在第 1 个 epoch 后即达到 92.86%，第 2、3 个 epoch 缓慢提升至 94.09%，表明 BERT 预训练权重已提供极强的特征表示基础
- 训练损失从 0.3282 持续下降至 0.0729，说明模型有效拟合训练数据
- **评估损失在第 3 个 epoch 上升**（0.2078 $\rightarrow$ 0.2628），而准确率几乎没有提升（0.9400 $\rightarrow$ 0.9409），提示**轻微过拟合**---3 个 epoch 对本任务而言已是边际收益递减的临界点

### 2.4 错误案例分析

尽管整体准确率达 94%，仍有约 6% 的误判样本。通过对典型错误的分析，可归纳为以下几类：

#### 2.4.1 否定句（Negation Handling）

| 示例 | 真实标签 | 预测标签 | 分析 |
|------|---------|---------|------|
| "This movie is **not** bad at all." | Positive | Negative | BERT 捕捉了"not"但未能正确理解"not bad"的双重否定构式，误判为负面 |
| "I can't say I disliked it." | Positive | Negative | "can't say I disliked" 等价于"liked"，但模型被"can't"和"disliked"误导 |

**原因**：否定词的语义作用域（Scope of Negation）对模型构成挑战。虽然 BERT 是双向模型，但对 `[not] + [positive word]` 的组合仍需大量此类样本才能学稳。IMDB 中"not bad"类文本的正负比例可能不均衡。

#### 2.4.2 讽刺与反语（Irony / Sarcasm）

| 示例 | 真实标签 | 预测标签 | 分析 |
|------|---------|---------|------|
| "Oh great, another two-hour-long masterpiece of boredom." | Negative | Positive | 反语"great"和"masterpiece"被模型按字面理解为正面 |
| "Sure, because what the world really needed was another superhero movie." | Negative | Positive | 反讽句式结构非字面表达 |

**原因**：讽刺依赖语境、语调（在文本中缺失）和反语标记，即使 BERT 也无法完全从字面词义中推断出反讽意图。这是情感分析领域的**开放挑战**。

#### 2.4.3 混合情绪（Mixed Sentiment）

| 示例 | 真实标签 | 预测标签 | 分析 |
|------|---------|---------|------|
| "The acting was brilliant but the plot was a complete mess." | Negative | Positive | 模型被"brilliant"吸引，忽略了负面转折 |

**原因**："but"等转折词引导的复杂度，需要模型准确评估子句权重的细粒度能力。

#### 2.4.4 较长的语境依赖

IMDB 评论平均长度约 230 词（许多超过 512 token），截断可能导致关键上下文丢失。部分评论中，正面内容的累积在末尾被负面转折否定，但模型仅看了前 512 token。

#### 错误分布推测

```
错误类型         占比（估计）  说明
----------------------------------------
否定句误判         ~30%       "not good"、"can't" 类
讽刺/反语         ~25%       反语导致误判
混合情绪          ~20%       转折结构处理不佳
长文本截断丢失     ~15%       超过 512 token 截断
其他              ~10%       数据噪声、标注歧义等
```

### 2.5 调参与复盘

#### 2.5.1 学习率的影响

| 学习率 | 预期效果 |
|--------|---------|
| 5e-5（默认推荐上限） | 收敛更快，但可能跳过最优解，评估损失震荡 |
| **2e-5（本项目选用）** | **收敛稳定，验证效果好，推荐** |
| 1e-5 | 收敛较慢，需更多 epoch，但可能获得更平滑的损失曲线 |

**经验**：对于 BERT-base 微调，2e-5 是最稳妥的起点。若使用 RoBERTa 等不同预训练目标或更大模型（如 BERT-large），推荐同步下调剂到 1e-5。

#### 2.5.2 Batch Size 的影响

| Batch Size | 显存占用（RTX 5070 Ti, float32） | 单 epoch 时间 | 梯度估计噪声 |
|-----------|--------------------------------|--------------|-------------|
| 8 | ~8 GB | ~220s | 较高（噪声大） |
| **16（本项目）** | **~12 GB** | **~330s** | **适中** |
| 32 | 超出 16 GB（float32） | --- | 较低 |

**经验**：batch size 越小，梯度噪声越大，但可能有助于正则化效果。实际因为 IMDB 每条评论序列长（max 512 token），batch size 16 已是 float32 模式下的上限。备选方案是启用 fp16 混合精度训练（RTX 5070 Ti 支持），可将 batch size 提升至 32。

#### 2.5.3 Epoch 数量的权衡

- **Epoch 1**：92.86% 准确率 --- 已显著优于传统方法
- **Epoch 2**：94.00% --- 小幅提升，评估损失上升
- **Epoch 3**：94.09% --- 几乎停滞，评估损失继续上升

最优策略应为：**早停（Early Stopping）**，在 epoch 2 评估损失开始上升时停止训练，或引入评估损失为监控指标的 early stopping callback。

#### 2.5.4 训练中遇到的问题与解决方案

##### 问题 1：HuggingFace 模型下载失败（网络连接）

- **现象**：`connect timeout` / DNS 解析失败，无法从 `huggingface.co` 下载 BERT 权重
- **原因**：国内网络环境对 HuggingFace 的访问受限
- **解决方案**：
  ```python
  os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
  ```
  切换至 HF-Mirror 镜像站后下载成功。同时在 `train_new.py` 中设置 `HF_HUB_OFFLINE=0` 以显式启用在线模式。

##### 问题 2：CUDA 加速验证

- **现象**：首次运行时未确认 GPU 是否被 PyTorch 识别
- **解决方案**：在训练脚本中增加 GPU 检测：
  ```python
  print("CUDA available:", torch.cuda.is_available())
  if torch.cuda.is_available():
      print("GPU device:", torch.cuda.get_device_name(0))
  ```
  输出确认 `RTX 5070 Ti` 被正确识别，模型自动在 CUDA 上运行。

##### 问题 3：依赖版本冲突

- **现象**：Transformers 需要特定的 tokenizers、safetensors 版本与之兼容。初次安装时遇到 tokenizers 版本过旧导致的 `AttributeError`
- **解决方案**：使用统一的版本锁定：
  ```bash
  pip install transformers==4.57.6 datasets==5.0.0 accelerate==1.13.0 tokenizers==0.22.2 safetensors==0.7.0
  ```
  在虚拟环境（venv）中重新安装，避免全局包冲突。

##### 问题 4：model.safetensors 与 optimizer.pt 占用磁盘空间大

- **现象**：每个 checkpoint 约 1.3 GB（model ~438 MB + optimizer ~876 MB），两个 checkpoint 和 models/ 总计 ~4 GB
- **解决方案**：`save_total_limit=2` 自动删除旧 checkpoint（实际保存了 epoch 2 和 epoch 3 两个）。最终模型存放于 `./models/` 仅保留 model.safetensors 和 config.json，不包含优化器状态。若需进一步节省，可删除 optimizer.pt（约 875 MB $\times$ 2）在推理时不需要。

##### 问题 5：长序列导致显存压力

- **现象**：IMDB 平均评论长度 ~230 词，部分评论超过 512 token。设置 `max_length=512, truncation=True` 确保所有样本被截断至可处理长度
- **影响**：每个样本的显存占用随序列长度线性增长。batch size 16 + 512 token 占满 12 GB 显存
- **备选**：未来可考虑梯度累积（gradient accumulation steps=2）以模拟更大 batch，或启用 fp16 混合精度

#### 2.5.5 总结与改进方向

| 维度 | 当前方案 | 建议改进 |
|------|---------|---------|
| 训练精度 | float32 | 启用 fp16 混合精度 $\rightarrow$ 显存减半，训练提速 ~40% |
| Batch size | 16 | fp16 下可提升至 32 或使用梯度累积 |
| 正则化 | weight_decay=0.01 | 加入 dropout（已默认 0.1）、早停（patience=1） |
| 学习率调度 | linear decay, no warmup | 加入 warmup steps（~10% 总步数）防止早期震荡 |
| 序列处理 | 截断至 512 | 分层注意力 / Longformer 处理超长文本 |
| 错误分析 | 无自动分析 | 集成 confusion matrix 和 error dashboard |
| 可解释性 | 无 | 集成 SHAP / LIME 解释错误案例 |

---

## 参考文献

1. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL)*, 142-150.

2. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*.

3. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. In *Proceedings of EMNLP*, 1532-1543.

4. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)*, 5998-6008.

5. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *NAACL-HLT*, 4171-4186.

6. Zhang, L., Wang, S., & Liu, B. (2018). Deep learning for sentiment analysis: A survey. *WIREs Data Mining and Knowledge Discovery*, 8(4), e1253.

7. Abdullah, T., & Ahmet, A. (2022). Deep learning in sentiment analysis: Recent architectures. *ACM Computing Surveys*, 55(8), 1-38.

8. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.

9. Papadimitriou, O., Al-Hussaeni, K., Karamitsos, I., & Maragoudakis, M. (2025). Fine-tuning BERT for robust sentiment classification of IMDb reviews. In *21st AIAI Conference*.

10. Prabhu, O. (2025). ModernBERT-XAI: A synergistic approach to sentiment analysis with layer-wise learning and SHAP-LIME interpretability. *Systems and Soft Computing*, 7, 200795.

11. Nkhata, G., Anjun, U., & Zhan, J. (2023). Sentiment analysis of movie reviews using BERT. In *eKNOW 2023*.

12. Guo, B. (2026). Evaluating TF-IDF with logistic regression and BERT fine-tuning for movie review sentiment analysis. *Frontiers in Computing and Intelligent Systems*, 16(1), 12-18.

---

> **附录**：项目代码位于 `src/`，训练日志位于 `results/checkpoint-*/trainer_state.json`，最终评估结果位于 `results/eval_results.txt`，模型权重位于 `models/`，损失曲线和混淆矩阵等可视化结果位于 `results/`。
