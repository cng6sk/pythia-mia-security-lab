"""
评估脚本
Evaluation and Visualization of MIA Results

功能：
1. 计算 AUC-ROC
2. 计算 TPR @ 1% FPR
3. 生成可视化图表
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
import argparse

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import *

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def load_scores(score_file):
    """加载攻击得分"""
    with open(score_file, 'r') as f:
        data = json.load(f)
    
    scores = np.array(data['scores'])
    labels = np.array(data['labels'])
    
    return scores, labels


def compute_metrics(scores, labels):
    """计算评估指标"""
    # AUC-ROC
    auc_score = roc_auc_score(labels, scores)
    
    # ROC 曲线
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # TPR @ 1% FPR
    tpr_at_low_fpr = 0.0
    for i in range(len(fpr)):
        if fpr[i] <= FPR_THRESHOLD:
            tpr_at_low_fpr = tpr[i]
        else:
            break
    
    metrics = {
        "auc": auc_score,
        "tpr_at_1fpr": tpr_at_low_fpr,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }
    
    return metrics


def plot_roc_curve(metrics_dict, save_path):
    """绘制 ROC 曲线对比图"""
    plt.figure(figsize=(10, 8))
    
    for name, metrics in metrics_dict.items():
        fpr = metrics['fpr']
        tpr = metrics['tpr']
        auc_score = metrics['auc']
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})', linewidth=2)
    
    # 随机猜测基线
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves for MIA', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC 曲线已保存: {save_path}")
    plt.close()


def plot_auc_comparison(results_dict, save_path):
    """绘制 AUC 对比柱状图"""
    names = list(results_dict.keys())
    aucs = [results_dict[name]['auc'] for name in names]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(names)), aucs, color='steelblue', alpha=0.8)
    
    # 添加数值标签
    for i, (bar, auc_val) in enumerate(zip(bars, aucs)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc_val:.3f}', ha='center', va='bottom', fontsize=11)
    
    # 添加随机猜测基线
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random (0.5)')
    
    plt.xlabel('Model / Epoch', fontsize=14)
    plt.ylabel('AUC-ROC', fontsize=14)
    plt.title('MIA Attack Performance (AUC Comparison)', fontsize=16)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylim([0, 1.0])
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ AUC 对比图已保存: {save_path}")
    plt.close()


def plot_epoch_curve(epoch_metrics, strategy_name, save_path):
    """绘制不同 epoch 的 AUC 变化曲线"""
    epochs = sorted(epoch_metrics.keys())
    aucs = [epoch_metrics[e]['auc'] for e in epochs]
    tprs = [epoch_metrics[e]['tpr_at_1fpr'] for e in epochs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # AUC 曲线
    ax1.plot(epochs, aucs, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Random')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('AUC-ROC', fontsize=14)
    ax1.set_title(f'{strategy_name}: AUC vs Epoch', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_ylim([0.45, 1.0])
    
    # TPR @ 1% FPR 曲线
    ax2.plot(epochs, tprs, marker='s', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('TPR @ 1% FPR', fontsize=14)
    ax2.set_title(f'{strategy_name}: TPR@1%FPR vs Epoch', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Epoch 变化曲线已保存: {save_path}")
    plt.close()


def print_summary(results_dict):
    """打印评估摘要"""
    print("\n" + "="*80)
    print("评估结果摘要")
    print("="*80)
    
    for name, metrics in results_dict.items():
        print(f"\n{name}:")
        print(f"  - AUC-ROC: {metrics['auc']:.4f}")
        print(f"  - TPR @ 1% FPR: {metrics['tpr_at_1fpr']:.4f}")
        
        if metrics['auc'] >= 0.8:
            print(f"  ⚠ 高风险：显著的成员推断风险")
        elif metrics['auc'] >= 0.6:
            print(f"  ⚡ 中等风险：存在一定的隐私泄露")
        else:
            print(f"  ✓ 低风险：接近随机猜测")
    
    print("\n" + "="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估 MIA 结果")
    parser.add_argument("--score_files", nargs='+', required=True, 
                        help="攻击得分文件列表")
    parser.add_argument("--names", nargs='+', required=True,
                        help="对应的模型名称")
    parser.add_argument("--output_prefix", type=str, required=True,
                        help="输出文件前缀")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MIA 评估与可视化")
    print("="*80)
    
    # 加载并评估所有结果
    results_dict = {}
    for score_file, name in zip(args.score_files, args.names):
        print(f"\n处理: {name}")
        scores, labels = load_scores(score_file)
        metrics = compute_metrics(scores, labels)
        results_dict[name] = metrics
        print(f"  ✓ AUC: {metrics['auc']:.4f}")
        print(f"  ✓ TPR@1%FPR: {metrics['tpr_at_1fpr']:.4f}")
    
    # 保存指标
    metrics_file = os.path.join(METRICS_DIR, f"{args.output_prefix}_metrics.json")
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(metrics_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✓ 指标已保存: {metrics_file}")
    
    # 生成图表
    print("\n生成可视化...")
    
    # ROC 曲线
    roc_path = os.path.join(PLOTS_DIR, f"{args.output_prefix}_roc.png")
    plot_roc_curve(results_dict, roc_path)
    
    # AUC 对比
    auc_path = os.path.join(PLOTS_DIR, f"{args.output_prefix}_auc_comparison.png")
    plot_auc_comparison(results_dict, auc_path)
    
    # 打印摘要
    print_summary(results_dict)
    
    print("\n" + "="*80)
    print("✓ 评估完成！")
    print("="*80)


if __name__ == "__main__":
    main()

