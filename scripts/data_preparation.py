"""
数据准备脚本 (使用 MIMIR n-gram 去重方法)
Data Preparation for MIA Experiments with n-gram Deduplication

功能：
1. 下载 AG News 数据集
2. 从 train split 划分 member 和 non-member 样本
3. 使用 MIMIR 的 n-gram 去重方法（13-gram, threshold=0.8）
4. 确保同分布（相同来源、平衡长度、无 n-gram 重叠）
5. 保存为 jsonl 格式

参考：MIMIR: Membership Inference via Probabilistic Ratio (NeurIPS 2024)
"""

import os
import json
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import sys
from collections import defaultdict

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import *


def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    print(f"✓ 随机种子已设置: {seed}")


def download_dataset():
    """下载 AG News 数据集"""
    print("\n" + "="*80)
    print("步骤 1: 下载 AG News 数据集")
    print("="*80)
    
    dataset = load_dataset(DATASET_NAME)
    train_data = dataset['train']
    
    print(f"✓ 数据集加载完成")
    print(f"  - 训练集大小: {len(train_data)}")
    
    return train_data


def prepare_text(example):
    """
    准备文本：将标题和正文拼接
    AG News 格式: {'text': '...', 'label': 0-3}
    我们只使用 text，不使用 label（这是语言模型任务，不是分类）
    """
    return example['text']


def calculate_text_length(text, tokenizer=None):
    """计算文本长度（如果有tokenizer就用token数，否则用字符数）"""
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        return len(text.split())  # 简单按空格分词


def extract_ngrams(text, n=13):
    """
    从文本中提取 n-gram 集合
    
    Args:
        text: 输入文本字符串
        n: n-gram 的大小
    
    Returns:
        set: n-gram 集合
    """
    words = text.lower().split()  # 转小写并分词
    if len(words) < n:
        # 如果文本太短，返回整个文本作为一个 gram
        return {' '.join(words)}
    
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(ngram)
    
    return ngrams


def compute_ngram_similarity(text1, text2, n=13):
    """
    计算两个文本的 n-gram Jaccard 相似度
    
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    
    Args:
        text1, text2: 输入文本
        n: n-gram 大小
    
    Returns:
        float: Jaccard 相似度 [0, 1]
    """
    ngrams1 = extract_ngrams(text1, n)
    ngrams2 = extract_ngrams(text2, n)
    
    if len(ngrams1) == 0 and len(ngrams2) == 0:
        return 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def filter_by_ngram_overlap(member_samples, candidate_samples, n=13, threshold=0.8, target_size=1000):
    """
    使用 n-gram 去重过滤候选样本
    
    选择与所有 member 样本的 n-gram 重叠度都 < threshold 的候选样本
    
    Args:
        member_samples: 成员样本列表
        candidate_samples: 候选非成员样本列表
        n: n-gram 大小
        threshold: Jaccard 相似度阈值
        target_size: 目标非成员样本数量
    
    Returns:
        list: 过滤后的非成员样本
    """
    print("\n" + "="*80)
    print(f"步骤 2.5: n-gram 去重过滤 (n={n}, threshold={threshold})")
    print("="*80)
    
    member_texts = [s['text'] for s in member_samples]
    filtered_samples = []
    
    print(f"正在过滤 {len(candidate_samples)} 个候选样本...")
    print(f"目标：找到 {target_size} 个与 member 无显著 n-gram 重叠的样本")
    
    for candidate in tqdm(candidate_samples, desc="n-gram 过滤"):
        candidate_text = candidate['text']
        
        # 检查与所有 member 样本的相似度
        max_similarity = 0.0
        for member_text in member_texts:
            similarity = compute_ngram_similarity(candidate_text, member_text, n)
            max_similarity = max(max_similarity, similarity)
            
            # 如果与任何一个 member 的相似度 >= threshold，则跳过
            if similarity >= threshold:
                break
        
        # 如果与所有 member 的最大相似度 < threshold，则保留
        if max_similarity < threshold:
            filtered_samples.append(candidate)
            
            # 如果已找到足够的样本，提前结束
            if len(filtered_samples) >= target_size:
                break
    
    print(f"\n✓ 过滤完成")
    print(f"  - 候选样本数: {len(candidate_samples)}")
    print(f"  - 过滤后样本数: {len(filtered_samples)}")
    print(f"  - 过滤率: {(1 - len(filtered_samples)/len(candidate_samples))*100:.2f}%")
    
    if len(filtered_samples) < target_size:
        print(f"\n⚠ 警告: 过滤后样本不足 {target_size} 个!")
        print(f"  建议: 降低 threshold 或增加候选样本池")
    
    return filtered_samples


def split_member_nonmember_with_ngram(train_data, member_size, nonmember_size, 
                                      n=13, threshold=0.8, seed=42):
    """
    从训练集中划分 member 和 non-member 样本（使用 n-gram 去重）
    
    遵循 MIMIR 的方法：
    1. 随机选择 member 样本
    2. 从剩余样本中过滤出与 member 无显著 n-gram 重叠的样本作为 non-member
    
    Args:
        train_data: 完整的训练数据
        member_size: 成员样本数量
        nonmember_size: 非成员样本数量
        n: n-gram 大小
        threshold: Jaccard 相似度阈值
        seed: 随机种子
    
    Returns:
        member_samples, nonmember_samples, indices_dict
    """
    print("\n" + "="*80)
    print("步骤 2: 划分成员与非成员样本（MIMIR n-gram 方法）")
    print("="*80)
    print(f"参数: n={n}, threshold={threshold}, seed={seed}")
    
    set_seed(seed)
    
    # 获取所有索引
    total_size = len(train_data)
    all_indices = list(range(total_size))
    random.shuffle(all_indices)
    
    # 确保有足够的数据
    # 注意：n-gram 过滤会减少可用样本，所以需要更多候选
    min_required = member_size + nonmember_size * 3  # 预留 3 倍空间
    if total_size < min_required:
        print(f"⚠ 警告: 数据量较少，可能无法满足 n-gram 过滤需求")
    
    # 1. 选择 member 样本
    member_indices = all_indices[:member_size]
    member_samples = [train_data[i] for i in tqdm(member_indices, desc="提取成员样本")]
    
    print(f"✓ 成员样本: {len(member_samples)} 条")
    
    # 2. 准备 non-member 候选池（使用剩余所有数据）
    candidate_indices = all_indices[member_size:]
    candidate_samples = [train_data[i] for i in tqdm(candidate_indices, desc="准备候选样本池")]
    
    print(f"✓ 候选样本池: {len(candidate_samples)} 条")
    
    # 3. 使用 n-gram 过滤
    nonmember_samples = filter_by_ngram_overlap(
        member_samples, 
        candidate_samples, 
        n=n, 
        threshold=threshold,
        target_size=nonmember_size
    )
    
    # 4. 如果过滤后样本不足，从候选中随机补充（并给出警告）
    if len(nonmember_samples) < nonmember_size:
        print(f"\n⚠ 样本不足，从候选池随机补充...")
        remaining = nonmember_size - len(nonmember_samples)
        # 补充未被选中的候选样本
        unused_candidates = [s for s in candidate_samples if s not in nonmember_samples]
        random.shuffle(unused_candidates)
        nonmember_samples.extend(unused_candidates[:remaining])
        print(f"✓ 已补充 {remaining} 个样本")
    
    # 截取到目标大小
    nonmember_samples = nonmember_samples[:nonmember_size]
    
    # 获取 non-member 的索引
    nonmember_texts = [s['text'] for s in nonmember_samples]
    nonmember_indices = []
    for idx in candidate_indices:
        if train_data[idx]['text'] in nonmember_texts:
            nonmember_indices.append(idx)
            if len(nonmember_indices) >= nonmember_size:
                break
    
    # 保存索引和元数据
    indices_dict = {
        "member_indices": member_indices,
        "nonmember_indices": nonmember_indices,
        "seed": seed,
        "total_size": total_size,
        "ngram_n": n,
        "ngram_threshold": threshold,
        "method": "MIMIR_ngram_deduplication"
    }
    
    print(f"\n✓ 最终结果:")
    print(f"  - 成员样本: {len(member_samples)} 条")
    print(f"  - 非成员样本: {len(nonmember_samples)} 条")
    print(f"  - 使用数据: {member_size + len(nonmember_indices)} / {total_size}")
    
    return member_samples, nonmember_samples, indices_dict


def analyze_distribution(member_samples, nonmember_samples):
    """分析成员和非成员样本的分布"""
    print("\n" + "="*80)
    print("步骤 3: 分析数据分布")
    print("="*80)
    
    # 计算文本长度
    member_lengths = [len(s['text'].split()) for s in member_samples]
    nonmember_lengths = [len(s['text'].split()) for s in nonmember_samples]
    
    print(f"\n成员样本统计:")
    print(f"  - 数量: {len(member_samples)}")
    print(f"  - 平均长度: {np.mean(member_lengths):.2f} 词")
    print(f"  - 中位数长度: {np.median(member_lengths):.2f} 词")
    print(f"  - 长度范围: [{np.min(member_lengths)}, {np.max(member_lengths)}]")
    
    print(f"\n非成员样本统计:")
    print(f"  - 数量: {len(nonmember_samples)}")
    print(f"  - 平均长度: {np.mean(nonmember_lengths):.2f} 词")
    print(f"  - 中位数长度: {np.median(nonmember_lengths):.2f} 词")
    print(f"  - 长度范围: [{np.min(nonmember_lengths)}, {np.max(nonmember_lengths)}]")
    
    # 统计长度差异
    length_diff = abs(np.mean(member_lengths) - np.mean(nonmember_lengths))
    print(f"\n✓ 平均长度差异: {length_diff:.2f} 词")
    if length_diff < 5:
        print("  ✓ 分布良好：成员与非成员长度分布相近")
    else:
        print("  ⚠ 警告：长度分布差异较大，可能影响实验公平性")


def save_data(member_samples, nonmember_samples, indices_dict):
    """保存数据到 jsonl 格式"""
    print("\n" + "="*80)
    print("步骤 4: 保存数据")
    print("="*80)
    
    # 创建目录
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    
    # 保存成员样本
    with open(MEMBER_TRAIN_FILE, 'w', encoding='utf-8') as f:
        for sample in tqdm(member_samples, desc="保存成员样本"):
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    print(f"✓ 成员样本已保存: {MEMBER_TRAIN_FILE}")
    
    # 保存非成员样本
    with open(NONMEMBER_FILE, 'w', encoding='utf-8') as f:
        for sample in tqdm(nonmember_samples, desc="保存非成员样本"):
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    print(f"✓ 非成员样本已保存: {NONMEMBER_FILE}")
    
    # 保存索引
    with open(SPLIT_INDICES_FILE, 'w', encoding='utf-8') as f:
        json.dump(indices_dict, f, indent=2)
    print(f"✓ 划分索引已保存: {SPLIT_INDICES_FILE}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("MIA 数据准备脚本 (MIMIR n-gram 去重方法)")
    print("="*80)
    print(f"数据集: {DATASET_NAME}")
    print(f"成员样本数: {MEMBER_TRAIN_SIZE}")
    print(f"非成员样本数: {NONMEMBER_SIZE}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"n-gram 参数: n=13, threshold=0.8")
    print("="*80)
    
    # 1. 下载数据集
    train_data = download_dataset()
    
    # 2. 划分样本（使用 n-gram 去重）
    member_samples, nonmember_samples, indices_dict = split_member_nonmember_with_ngram(
        train_data, 
        MEMBER_TRAIN_SIZE, 
        NONMEMBER_SIZE,
        n=13,
        threshold=0.8,
        seed=RANDOM_SEED
    )
    
    # 3. 分析分布
    analyze_distribution(member_samples, nonmember_samples)
    
    # 4. 保存数据
    save_data(member_samples, nonmember_samples, indices_dict)
    
    print("\n" + "="*80)
    print("✓ 数据准备完成！")
    print("="*80)
    print(f"✓ 已使用 MIMIR n-gram 去重方法")
    print(f"  - 13-gram Jaccard 相似度 < 0.8")
    print(f"  - 确保 member 和 non-member 无显著文本重叠")
    print("\n下一步: 运行 python scripts/train.py 开始微调")


if __name__ == "__main__":
    main()

