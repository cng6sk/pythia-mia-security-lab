"""
实验配置文件
Configuration for MIA experiments on fine-tuned LLMs
"""

import torch

# ============================================================================
# 模型配置 (Model Configuration)
# ============================================================================
MODEL_NAME = "EleutherAI/pythia-70m"
MODEL_CACHE_DIR = "./models/cache"  # 模型缓存目录

# ============================================================================
# 数据配置 (Data Configuration)
# ============================================================================
DATASET_NAME = "ag_news"
MEMBER_TRAIN_SIZE = 5000      # 成员训练集大小（原来 1000，现扩展到 5000）
NONMEMBER_SIZE = 5000         # 非成员集大小（与成员保持一致）
MAX_LENGTH = 128              # 最大序列长度
RANDOM_SEED = 42              # 随机种子，保证可复现

# 数据路径
DATA_RAW_DIR = "./data/raw"
DATA_PROCESSED_DIR = "./data/processed"
MEMBER_TRAIN_FILE = f"{DATA_PROCESSED_DIR}/member_train.jsonl"
NONMEMBER_FILE = f"{DATA_PROCESSED_DIR}/nonmember.jsonl"
SPLIT_INDICES_FILE = f"{DATA_PROCESSED_DIR}/split_indices.json"

# ============================================================================
# 训练配置 (Training Configuration)
# ============================================================================
# 基础训练参数
BATCH_SIZE = 4                      # 批大小（针对6GB显存优化）
GRADIENT_ACCUMULATION_STEPS = 4     # 梯度累积步数（实际batch=16）
LEARNING_RATE = 2e-5                # 学习率
NUM_EPOCHS = 5                      # 训练轮数
WARMUP_STEPS = 100                  # 预热步数
WEIGHT_DECAY = 0.01                 # 权重衰减
# 说明：
#   - 对于小模型（如 pythia-70m），在 RTX 3060 6GB 上使用 FP32 训练也足够
#   - 为避免 Accelerate 在某些配置下出现 “Attempting to unscale FP16 gradients” 错误，
#     这里默认关闭 fp16，由 Trainer 全程使用 FP32 训练，稳定性更好
FP16 = False                        # 是否使用混合精度训练（关闭可避免 FP16 梯度缩放相关报错）

# LoRA 配置
LORA_R = 8                          # LoRA rank
LORA_ALPHA = 16                     # LoRA alpha
LORA_DROPOUT = 0.05                 # LoRA dropout
LORA_TARGET_MODULES = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

# 模型保存路径
MODELS_DIR = "./models"
PRETRAINED_DIR = f"{MODELS_DIR}/pythia-70m-pretrained"
FULL_FT_DIR = f"{MODELS_DIR}/pythia-70m-full"
LORA_FT_DIR = f"{MODELS_DIR}/pythia-70m-lora"
HEAD_FT_DIR = f"{MODELS_DIR}/pythia-70m-head"

# ============================================================================
# MIA 攻击配置 (MIA Attack Configuration)
# ============================================================================
# Min-K% 攻击参数
MINK_PERCENT = 10  # 选择最低10%的token概率

# 攻击方法列表
ATTACK_METHODS = ["loss", "ratio", "mink"]

# ============================================================================
# 评估配置 (Evaluation Configuration)
# ============================================================================
# 结果保存路径
RESULTS_DIR = "./results"
SCORES_DIR = f"{RESULTS_DIR}/scores"
METRICS_DIR = f"{RESULTS_DIR}/metrics"
PLOTS_DIR = f"{RESULTS_DIR}/plots"

# 评估指标
FPR_THRESHOLD = 0.01  # TPR @ 1% FPR

# ============================================================================
# 设备配置 (Device Configuration)
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # DataLoader workers

# ============================================================================
# 日志配置 (Logging Configuration)
# ============================================================================
LOG_INTERVAL = 50  # 每50步打印一次日志
SAVE_STEPS = 500   # 每500步保存一次检查点

# ============================================================================
# 微调策略配置 (Fine-tuning Strategies)
# ============================================================================
FINETUNING_STRATEGIES = {
    "full": {
        "name": "Full Fine-tuning",
        "save_dir": FULL_FT_DIR,
        "trainable_params": "all"
    },
    "lora": {
        "name": "LoRA Fine-tuning",
        "save_dir": LORA_FT_DIR,
        "trainable_params": "lora"
    },
    "head": {
        "name": "Head Fine-tuning",
        "save_dir": HEAD_FT_DIR,
        "trainable_params": "lm_head"
    }
}

# ============================================================================
# 实验配置总结 (Experiment Summary)
# ============================================================================
def print_config():
    """打印当前配置"""
    print("=" * 80)
    print("实验配置总览 (Experiment Configuration)")
    print("=" * 80)
    print(f"模型: {MODEL_NAME}")
    print(f"设备: {DEVICE}")
    print(f"数据集: {DATASET_NAME}")
    print(f"成员样本数: {MEMBER_TRAIN_SIZE}")
    print(f"非成员样本数: {NONMEMBER_SIZE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"批大小: {BATCH_SIZE} (累积后: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"学习率: {LEARNING_RATE}")
    print(f"混合精度: {FP16}")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()

