# MIA 实验项目

**评估微调大语言模型中的成员推断风险**  
*Evaluating Membership Inference Risks in Fine-tuned Large Language Models*

---

## 📁 项目结构

```
final/
├── config/                        # 实验配置模块
│   └── config.py
├── data/                          # 数据目录
│   ├── raw/
│   └── processed/
├── models/                        # 微调模型与缓存（默认忽略版本控制）
├── results/                       # 实验结果
│   ├── tables/
│   ├── metrics/
│   ├── plots/
│   └── scores/
├── scripts/                       # 单次实验脚本
│   ├── data_preparation.py
│   ├── train.py
│   ├── attack.py
│   └── evaluate.py
├── scripts_pack/                  # 批处理与绘图脚本
├── requirements.txt               # Python 依赖
└── README.md                      # 本文件
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

```bash
python scripts/data_preparation.py
```

**输出**:
- `data/processed/member_train.jsonl` (1000条)
- `data/processed/nonmember.jsonl` (1000条)
- `data/processed/split_indices.json`

### 3. 模型微调

**Full Fine-tuning:**
```bash
python scripts/train.py --strategy full
```

**LoRA Fine-tuning:**
```bash
python scripts/train.py --strategy lora
```

**Head Fine-tuning:**
```bash
python scripts/train.py --strategy head
```

每个策略会保存 5 个 epoch 的检查点。

### 4. MIA 攻击

**示例：对 Full FT Epoch 5 进行 Loss-based 攻击**
```bash
python scripts/attack.py \
    --model_path models/pythia-70m-full/checkpoint-XXX \
    --attack loss \
    --output_name full_epoch5_loss
```

**三种攻击方法：**
- `loss`: Loss-based Attack
- `mink`: Min-K% Probability Attack
- `ratio`: Ratio Attack (需要预训练模型作为参考)

### 5. 结果评估

```bash
python scripts/evaluate.py \
    --score_files results/scores/full_epoch1_loss.json results/scores/full_epoch5_loss.json \
    --names "Full-Epoch1" "Full-Epoch5" \
    --output_prefix full_comparison
```

**输出**:
- `results/metrics/full_comparison_metrics.json`
- `results/plots/full_comparison_roc.png`
- `results/plots/full_comparison_auc_comparison.png`

---

## ⚙️ 配置说明

主要配置在 `config/config.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `EleutherAI/pythia-70m` | 使用的模型 |
| `MEMBER_TRAIN_SIZE` | 1000 | 成员训练集大小 |
| `NONMEMBER_SIZE` | 1000 | 非成员集大小 |
| `MAX_LENGTH` | 128 | 最大序列长度 |
| `BATCH_SIZE` | 4 | 批大小 |
| `NUM_EPOCHS` | 5 | 训练轮数 |
| `LEARNING_RATE` | 2e-5 | 学习率 |
| `LORA_R` | 8 | LoRA rank |
| `MINK_PERCENT` | 10 | Min-K% 攻击的 K 值 |

---

## 📊 实验流程

```mermaid
graph LR
    A[AG News] --> B[数据准备]
    B --> C[Member 1000条]
    B --> D[Non-member 1000条]
    C --> E[微调]
    E --> F[Full FT]
    E --> G[LoRA FT]
    E --> H[Head FT]
    F --> I[MIA 攻击]
    G --> I
    H --> I
    C --> I
    D --> I
    I --> J[评估 AUC/TPR]
    J --> K[可视化]
```

---

## 📈 预期成果

1. **模型检查点**: 每个微调策略 × 5 个 epoch = 15 个检查点
2. **攻击得分**: 每个检查点 × 3 种攻击方法 = 45 组攻击结果
3. **评估指标**: AUC-ROC, TPR @ 1% FPR
4. **可视化图表**:
   - ROC 曲线对比
   - AUC 随 epoch 变化曲线
   - 不同微调策略对比

---

## 🎯 研究问题 (RQ)

1. **RQ1**: 不同微调策略（Full/LoRA/Head）是否导致不同程度的 MIA 风险？
2. **RQ2**: MIA 风险是否随训练 epoch 增加而"涌现"？
3. **RQ3**: 参数高效微调（LoRA, Head）是否比 Full FT 更安全？

---


## 📚 参考文献

1. Carlini, N., et al. (2022). Membership Inference Attacks From First Principles.
2. Yeom, S., et al. (2018). Privacy Risk in Machine Learning.
3. Shokri, R., et al. (2017). Membership Inference Attacks Against ML Models.


