"""
按 epoch 绘制 AUC / TPR 折线图
Epoch-wise Line Plots for MIA Metrics

用途：
- 更直观地展示“随 epoch 变化，模型受攻击程度的上升”
- 支持两种对比模式（一次只能选一种，避免图太乱）：
  1）固定微调策略，比较不同攻击方法的强度
      - 例：strategy=full，attacks=loss ratio mink
      - 图：X=epoch, Y=AUC-ROC，多条线=不同攻击
  2）固定攻击方法，比较不同微调策略的“防御力”
      - 例：strategies=full lora head, attacks=loss
      - 图：X=epoch, Y=AUC-ROC，多条线=不同策略

数据来源：
- 直接读取 scores JSON（由 scripts/attack.py 生成），重新计算指标
- 文件命名约定（与 batch_attack.py 保持一致）：
    results/scores/<prefix>/<prefix>_<strategy>_epoch<E>_<attack>.json
  例如：
    results/scores/pythia-70m/pythia-70m_full_epoch3_loss.json

用法示例（在项目根目录执行）：

1. 固定策略 full，看不同攻击随 epoch 的变化：
python -m scripts_pack.epoch_lineplot `
  --model-prefix pythia-70m `
  --mode fixed_strategy `
  --fixed full `
  --strategies full `
  --attacks loss ratio mink `
  --epochs 1 2 3 4 5 `
  --metric auc

2. 固定攻击 loss，看不同策略随 epoch 的变化：
python -m scripts_pack.epoch_lineplot `
  --model-prefix pythia-70m `
  --mode fixed_strategy `
  --fixed full `
  --strategies full `
  --attacks loss ratio mink `
  --epochs 1 2 3 4 5 `
  --metric auc
"""

import argparse
import os
import sys
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import SCORES_DIR, PLOTS_DIR  # noqa: E402
from scripts.evaluate import load_scores, compute_metrics  # noqa: E402


# 统一绘图风格（与 evaluate.py 保持一致）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def build_score_path(prefix: str, strategy: str, epoch: int, attack: str) -> str:
    """构造 score 文件路径。"""
    fname = f"{prefix}_{strategy}_epoch{epoch}_{attack}.json"
    return os.path.join(SCORES_DIR, prefix, fname)


def collect_metrics_over_epochs(
    prefix: str,
    strategies: List[str],
    attacks: List[str],
    epochs: List[int],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    收集 (line_key, epoch) -> metrics (auc, tpr@1%)。

    line_key 视模式而定：
      - 如果固定 strategy，多攻击对比：line_key = attack 名
      - 如果固定 attack，多策略对比：line_key = strategy 名
    这里不判断模式，只生成完整的 (strategy, attack) 组合，后面再选。
    """
    data: Dict[str, Dict[int, Dict[str, float]]] = {}

    for strategy in strategies:
        for attack in attacks:
            line_key = f"{strategy}:{attack}"
            data[line_key] = {}

            for e in epochs:
                score_path = build_score_path(prefix, strategy, e, attack)
                if not os.path.isfile(score_path):
                    print(f"⚠ 缺少文件，跳过: {score_path}")
                    continue

                scores, labels = load_scores(score_path)
                metrics = compute_metrics(scores, labels)
                data[line_key][e] = metrics

    return data


def plot_lines_over_epochs(
    metrics_data: Dict[str, Dict[int, Dict[str, float]]],
    mode: str,
    fixed: str,
    metric: str,
    prefix: str,
    output_name: str,
) -> None:
    """
    绘制折线图：
      - X: epoch
      - Y: metric (auc or tpr)
      - 多条线: 不同攻击 或 不同策略

    Args:
        metrics_data: collect_metrics_over_epochs 的输出
        mode: "fixed_strategy" 或 "fixed_attack"
        fixed: 固定的策略名或攻击名
        metric: "auc" 或 "tpr"（tpr 表示 TPR@1%FPR）
        prefix: 模型前缀（如 pythia-70m）
        output_name: 输出文件前缀
    """
    plt.figure(figsize=(10, 7))

    # 解析 line_key = "strategy:attack"
    line_series: Dict[str, List[Tuple[int, float]]] = {}

    for key, epoch_dict in metrics_data.items():
        strategy, attack = key.split(":", maxsplit=1)

        if mode == "fixed_strategy":
            if strategy != fixed:
                continue
            line_label = attack
        elif mode == "fixed_attack":
            if attack != fixed:
                continue
            line_label = strategy
        else:
            continue

        # 收集 (epoch, value)
        points: List[Tuple[int, float]] = []
        for e, m in epoch_dict.items():
            val = m["auc"] if metric == "auc" else m["tpr_at_1fpr"]
            points.append((e, val))

        if not points:
            continue

        points.sort(key=lambda x: x[0])
        line_series[line_label] = points

    if not line_series:
        print("⚠ 未找到任何可绘制的数据，请检查 scores 文件是否齐全。")
        plt.close()
        return

    # 绘制每条线
    for label, pts in line_series.items():
        xs = [e for e, _ in pts]
        ys = [v for _, v in pts]
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=8, label=label)

    # Y 轴标签
    if metric == "auc":
        plt.ylabel("AUC-ROC", fontsize=14)
        plt.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Random (0.5)")
        plt.ylim([0.45, 1.0])
    else:
        plt.ylabel("TPR @ 1% FPR", fontsize=14)
        plt.ylim([0.0, 1.0])

    plt.xlabel("Epoch", fontsize=14)

    # 标题
    if mode == "fixed_strategy":
        title = f"{prefix} | Strategy={fixed} | {metric.upper()} vs Epoch"
    else:
        title = f"{prefix} | Attack={fixed} | {metric.upper()} vs Epoch"
    plt.title(title, fontsize=16)

    plt.legend(title="攻击" if mode == "fixed_strategy" else "策略", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"{output_name}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✓ 折线图已保存: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="按 epoch 绘制 AUC / TPR 折线图")
    parser.add_argument(
        "--model-prefix",
        type=str,
        required=True,
        help="模型前缀，如 pythia-70m（对应 results/scores/<prefix>/...）",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["full"],
        help="策略列表：full, lora, head 等",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["loss"],
        help="攻击列表：loss, ratio, mink 等",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="epoch 列表（默认: 1 2 3 4 5）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fixed_strategy", "fixed_attack"],
        required=True,
        help="对比模式：fixed_strategy 或 fixed_attack",
    )
    parser.add_argument(
        "--fixed",
        type=str,
        required=True,
        help="固定的策略名（当 mode=fixed_strategy）或攻击名（当 mode=fixed_attack）",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["auc", "tpr"],
        default="auc",
        help="纵轴指标：auc（AUC-ROC）或 tpr（TPR@1%FPR）",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="输出文件前缀（可选，不含路径和扩展名）。若不指定则自动生成。",
    )

    args = parser.parse_args()

    prefix = args.model_prefix
    strategies = args.strategies
    attacks = args.attacks
    epochs = [int(e) for e in args.epochs]

    print("\n" + "=" * 80)
    print("Epoch 折线图脚本")
    print("=" * 80)
    print(f"模型前缀: {prefix}")
    print(f"策略列表: {strategies}")
    print(f"攻击列表: {attacks}")
    print(f"Epoch 列表: {epochs}")
    print(f"模式: {args.mode}")
    print(f"固定项: {args.fixed}")
    print(f"纵轴指标: {args.metric}")
    print("=" * 80)

    # 模式合法性检查
    if args.mode == "fixed_strategy" and args.fixed not in strategies:
        print(f"⚠ mode=fixed_strategy 但 fixed={args.fixed} 不在 strategies 中")
    if args.mode == "fixed_attack" and args.fixed not in attacks:
        print(f"⚠ mode=fixed_attack 但 fixed={args.fixed} 不在 attacks 中")

    # 收集所有 (strategy, attack) 的指标
    metrics_data = collect_metrics_over_epochs(prefix, strategies, attacks, epochs)

    # 输出前缀
    if args.output_prefix:
        out_prefix = args.output_prefix
    else:
        if args.mode == "fixed_strategy":
            out_prefix = f"{prefix}_{args.fixed}_attacks_{args.metric}_vs_epoch"
        else:
            out_prefix = f"{prefix}_{args.fixed}_strategies_{args.metric}_vs_epoch"

    # 绘制
    plot_lines_over_epochs(
        metrics_data=metrics_data,
        mode=args.mode,
        fixed=args.fixed,
        metric=args.metric,
        prefix=prefix,
        output_name=out_prefix,
    )


if __name__ == "__main__":
    main()


