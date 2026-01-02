"""
成员推断攻击脚本
Membership Inference Attack (MIA) Implementation

实现三种攻击方法：
1. Loss-based Attack: 基于负对数似然
2. Min-K% Probability Attack: 基于最低概率的token
3. Ratio Attack: 使用预训练模型作为参考
"""

import os
import sys
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import *


def load_model_and_tokenizer(model_path):
    """加载模型和 tokenizer"""
    print(f"\n加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if FP16 else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    print("✓ 模型加载完成")
    return model, tokenizer


def load_test_data():
    """加载测试数据（成员 + 非成员）"""
    print("\n" + "="*80)
    print("加载测试数据")
    print("="*80)
    
    # 加载成员样本
    member_samples = []
    with open(MEMBER_TRAIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            member_samples.append(json.loads(line))
    
    # 加载非成员样本
    nonmember_samples = []
    with open(NONMEMBER_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            nonmember_samples.append(json.loads(line))
    
    print(f"✓ 成员样本: {len(member_samples)} 条")
    print(f"✓ 非成员样本: {len(nonmember_samples)} 条")
    
    # 创建标签 (1=member, 0=non-member)
    texts = [s['text'] for s in member_samples] + [s['text'] for s in nonmember_samples]
    labels = [1] * len(member_samples) + [0] * len(nonmember_samples)
    
    return texts, labels


@torch.no_grad()
def compute_loss_scores(model, tokenizer, texts):
    """
    Loss-based Attack
    计算每个文本的负对数似然（NLL）
    Score = -NLL (越高越可能是成员)
    """
    print("\n执行 Loss-based Attack...")
    scores = []
    
    for text in tqdm(texts, desc="计算 Loss"):
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        ).to(model.device)
        
        # 计算 loss
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        
        # Score = -loss (loss 越低，越可能是成员)
        scores.append(-loss)
    
    return np.array(scores)


@torch.no_grad()
def compute_mink_scores(model, tokenizer, texts, k_percent=10):
    """
    Min-K% Probability Attack
    选择概率最低的 K% token，计算其平均对数概率
    """
    print(f"\n执行 Min-K% Probability Attack (K={k_percent}%)...")
    scores = []
    
    for text in tqdm(texts, desc="计算 Min-K%"):
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        ).to(model.device)
        
        # 获取 logits
        outputs = model(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # 计算每个 token 的 log probability
        input_ids = inputs["input_ids"][0]  # [seq_len]
        log_probs = []
        
        for i in range(len(input_ids) - 1):
            token_logits = logits[0, i, :]  # [vocab_size]
            token_log_probs = torch.log_softmax(token_logits, dim=-1)
            next_token = input_ids[i + 1]
            log_prob = token_log_probs[next_token].item()
            log_probs.append(log_prob)
        
        if len(log_probs) == 0:
            scores.append(0.0)
            continue
        
        # 选择最低的 K%
        k = max(1, int(len(log_probs) * k_percent / 100))
        lowest_k = sorted(log_probs)[:k]
        
        # 平均值作为 score
        score = np.mean(lowest_k)
        scores.append(score)
    
    return np.array(scores)


@torch.no_grad()
def compute_ratio_scores(target_model, reference_model, tokenizer, texts):
    """
    Ratio Attack
    Score = NLL_reference - NLL_target
    """
    print("\n执行 Ratio Attack...")
    scores = []
    
    for text in tqdm(texts, desc="计算 Ratio"):
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )
        
        # Target model loss
        inputs_target = inputs.to(target_model.device)
        outputs_target = target_model(**inputs_target, labels=inputs_target["input_ids"])
        loss_target = outputs_target.loss.item()
        
        # Reference model loss
        inputs_ref = inputs.to(reference_model.device)
        outputs_ref = reference_model(**inputs_ref, labels=inputs_ref["input_ids"])
        loss_ref = outputs_ref.loss.item()
        
        # Ratio score
        score = loss_ref - loss_target
        scores.append(score)
    
    return np.array(scores)


def save_scores(scores, labels, save_path):
    """保存攻击得分"""
    # 确保目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir:  # 如果路径包含目录
        os.makedirs(save_dir, exist_ok=True)
    
    results = {
        "scores": scores.tolist(),
        "labels": labels
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ 结果已保存: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MIA 攻击")
    parser.add_argument("--model_path", type=str, required=True, help="微调模型路径")
    parser.add_argument("--attack", type=str, required=True, 
                        choices=["loss", "mink", "ratio"], help="攻击方法")
    parser.add_argument("--output_name", type=str, required=True, help="输出文件名（不含扩展名）")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="输出目录（可选，默认使用 config 中的 SCORES_DIR）")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("成员推断攻击 (MIA)")
    print("="*80)
    print(f"模型: {args.model_path}")
    print(f"攻击方法: {args.attack}")
    print("="*80)
    
    # 1. 加载测试数据
    texts, labels = load_test_data()
    
    # 2. 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # 3. 执行攻击
    if args.attack == "loss":
        scores = compute_loss_scores(model, tokenizer, texts)
    
    elif args.attack == "mink":
        scores = compute_mink_scores(model, tokenizer, texts, MINK_PERCENT)
    
    elif args.attack == "ratio":
        # 需要加载参考模型（预训练模型）
        print("\n加载参考模型（预训练）...")
        reference_model, _ = load_model_and_tokenizer(MODEL_NAME)
        scores = compute_ratio_scores(model, reference_model, tokenizer, texts)
    
    # 4. 保存结果
    # 如果指定了 output_dir，使用它；否则使用配置中的 SCORES_DIR
    output_dir = args.output_dir if args.output_dir else SCORES_DIR
    save_path = os.path.join(output_dir, f"{args.output_name}.json")
    save_scores(scores, labels, save_path)
    
    print("\n" + "="*80)
    print("✓ 攻击完成！")
    print("="*80)
    print("\n下一步: 运行 python scripts/evaluate.py 进行评估")


if __name__ == "__main__":
    main()

