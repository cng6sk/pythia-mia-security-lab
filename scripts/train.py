"""
模型微调脚本
Fine-tuning Script for Different Strategies

支持三种微调策略：
1. Full Fine-tuning: 全参数微调
2. LoRA Fine-tuning: 低秩适配微调
3. Head Fine-tuning: 仅微调输出层
"""

import os
import sys
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import argparse

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import *


def load_training_data(tokenizer):
    """加载训练数据"""
    print("\n" + "="*80)
    print("加载训练数据")
    print("="*80)
    
    # 读取成员训练数据
    samples = []
    with open(MEMBER_TRAIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"✓ 加载了 {len(samples)} 条训练样本")
    
    # 转换为 Dataset 格式
    texts = [s['text'] for s in samples]
    
    # Tokenize
    print("正在 tokenize...")
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors=None
    )
    
    dataset = Dataset.from_dict(tokenized)
    print(f"✓ 数据集准备完成，样本数: {len(dataset)}")
    
    return dataset


def prepare_model_full_ft(model_name):
    """准备 Full Fine-tuning 模型"""
    print("\n配置: Full Fine-tuning（全参数微调）")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.float16 if FP16 else torch.float32,
    ).to(DEVICE)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  - 总参数: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,} (100%)")
    
    return model


def prepare_model_lora_ft(model_name):
    """准备 LoRA Fine-tuning 模型"""
    print("\n配置: LoRA Fine-tuning（低秩适配）")
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.float16 if FP16 else torch.float32,
    )
    
    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none"
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    
    # 统计参数
    model.print_trainable_parameters()
    
    return model


def prepare_model_head_ft(model_name):
    """准备 Head Fine-tuning 模型（仅微调输出层）"""
    print("\n配置: Head Fine-tuning（仅输出层）")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.float16 if FP16 else torch.float32,
    ).to(DEVICE)
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 仅解冻输出层（Pythia/GPTNeoX 使用 embed_out）
    # 尝试找到输出层（不同模型架构名称不同）
    if hasattr(model, 'lm_head'):
        # GPT-2, GPT-J 等
        for param in model.lm_head.parameters():
            param.requires_grad = True
        print("  - 解冻层: lm_head")
    elif hasattr(model, 'embed_out'):
        # Pythia (GPTNeoX)
        for param in model.embed_out.parameters():
            param.requires_grad = True
        print("  - 解冻层: embed_out")
    else:
        # 其他架构：解冻最后的 Linear 层
        print("  - 搜索输出层...")
        found = False
        for name, module in model.named_modules():
            if 'head' in name.lower() or 'output' in name.lower():
                for param in module.parameters():
                    param.requires_grad = True
                print(f"  - 解冻层: {name}")
                found = True
                break
        if not found:
            raise ValueError(f"无法找到输出层！模型架构: {model.__class__.__name__}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  - 总参数: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def train_model(model, tokenizer, train_dataset, output_dir, strategy_name):
    """训练模型"""
    print("\n" + "="*80)
    print(f"开始训练: {strategy_name}")
    print("="*80)
    
    # Data collator (用于 language modeling)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 不使用 masked language modeling
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOG_INTERVAL,
        save_strategy="epoch",  # 每个 epoch 保存一次
        save_total_limit=NUM_EPOCHS,  # 保留所有 epoch
        fp16=FP16,
        dataloader_num_workers=0,  # Windows 上设为 0
        remove_unused_columns=False,
        report_to="none"  # 不使用 wandb 等工具
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print(f"\n训练配置:")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - 保存路径: {output_dir}")
    print()
    
    trainer.train()
    
    # 保存最终模型
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    print(f"\n✓ 训练完成！最终模型保存在: {final_dir}")
    
    return trainer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="微调 LLM")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["full", "lora", "head"],
        help="微调策略: full, lora, 或 head"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HuggingFace 模型名称（如 EleutherAI/pythia-31m）。如果不指定，使用 config.py 中的 MODEL_NAME"
    )
    args = parser.parse_args()
    
    # 确定使用的模型名称
    model_name = args.model_name if args.model_name else MODEL_NAME
    
    strategy = args.strategy
    strategy_config = FINETUNING_STRATEGIES[strategy]
    
    print("\n" + "="*80)
    print("LLM 微调脚本")
    print("="*80)
    print(f"模型: {model_name}")
    print(f"策略: {strategy_config['name']}")
    print(f"设备: {DEVICE}")
    print("="*80)
    
    # 1. 加载 tokenizer
    print("\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_DIR
    )
    # GPT-Neo/Pythia 需要设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer 加载完成")
    
    # 2. 加载训练数据
    train_dataset = load_training_data(tokenizer)
    
    # 3. 准备模型
    if strategy == "full":
        model = prepare_model_full_ft(model_name)
    elif strategy == "lora":
        model = prepare_model_lora_ft(model_name)
    elif strategy == "head":
        model = prepare_model_head_ft(model_name)
    
    # 4. 训练
    # 根据模型名称动态生成保存路径
    model_short = model_name.split("/")[-1]  # 例如 "pythia-31m"
    output_dir = os.path.join(MODELS_DIR, f"{model_short}-{strategy}")
    train_model(model, tokenizer, train_dataset, output_dir, strategy_config['name'])
    
    print("\n" + "="*80)
    print("✓ 所有任务完成！")
    print("="*80)
    print(f"\n模型检查点保存在: {output_dir}")
    print(f"每个 epoch 的模型都已保存 (checkpoint-X)")
    print("\n下一步: 运行 python scripts/attack.py 进行 MIA 攻击")


if __name__ == "__main__":
    main()

