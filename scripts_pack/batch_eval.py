"""
批量评估与画图脚本
Batch Evaluation & Plotting for MIA Experiments

功能：
- 基于已有的 scores JSON（由 scripts/attack.py 生成）
- 自动调用 scripts/evaluate.py 生成多组关键图表和指标
- 默认针对单一模型前缀（如 pythia-70m），支持预训练 + Full + LoRA + Head

图像设计（对应论文中的主要图）：
1. 策略对比（epoch=5, loss 攻击）:
   - Pretrained vs Full vs LoRA vs Head
2. Full / Head / LoRA 的“涌现”曲线（AUC & TPR@1%FPR vs epoch, loss）
3. Full-Epoch5 在不同攻击方法下的对比（loss / ratio / mink）

用法示例（在项目根目录执行）：

1. 仅干跑（查看会调用哪些命令）：
   python -m scripts_pack.batch_eval --model-prefix pythia-70m --dry-run

2. 实际生成所有图表：
   python -m scripts_pack.batch_eval --model-prefix pythia-70m

注意：
- 假设 scores 保存在：results/scores/<model_prefix>/...json
- 假设文件命名遵循 batch_attack.py 的约定：
    - 预训练: {prefix}_pretrained_{attack}.json
    - Full:   {prefix}_full_epoch{E}_{attack}.json
    - LoRA:   {prefix}_lora_epoch{E}_{attack}.json
    - Head:   {prefix}_head_epoch{E}_{attack}.json
  其中 E = 1..5
"""

import argparse
import os
import subprocess
from typing import List, Dict

from config.config import SCORES_DIR, METRICS_DIR, PLOTS_DIR


def run_evaluate(
    score_files: List[str],
    names: List[str],
    output_prefix: str,
    dry_run: bool = False,
) -> None:
    """
    调用 scripts/evaluate.py 生成一组评估结果和图表。
    """
    cmd = [
        "python",
        "scripts/evaluate.py",
        "--score_files",
        *score_files,
        "--names",
        *names,
        "--output_prefix",
        output_prefix,
    ]

    print("\n" + "=" * 80)
    print(f"评估前缀: {output_prefix}")
    print("命令:", " ".join(cmd))
    print("=" * 80)

    if dry_run:
        print("(dry-run 模式，不实际执行)")
        return

    # 确保结果目录存在
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ 评估失败: {output_prefix}")
    else:
        print(f"✓ 评估完成: {output_prefix}")


def make_path(prefix: str, fname: str) -> str:
    """辅助函数：构造某个模型前缀下的 scores 路径。"""
    return os.path.join(SCORES_DIR, prefix, fname)


def main():
    parser = argparse.ArgumentParser(description="批量评估并生成 MIA 图像")
    parser.add_argument(
        "--model-prefix",
        type=str,
        required=True,
        help="模型前缀，如 pythia-70m（对应 results/scores/pythia-70m/）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="用于涌现分析的 epoch 列表（默认: 1 2 3 4 5）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的命令，而不真正运行 evaluate.py",
    )

    args = parser.parse_args()
    prefix = args.model_prefix
    epochs = args.epochs

    print("\n" + "=" * 80)
    print("批量评估与画图脚本")
    print("=" * 80)
    print(f"模型前缀: {prefix}")
    print(f"Epoch 列表: {epochs}")
    print(f"dry-run: {args.dry_run}")
    print("=" * 80)

    # 1. 策略对比图（loss, epoch=最后一个）
    last_epoch = max(epochs)
    loss_strategy_files = [
        make_path(prefix, f"{prefix}_pretrained_loss.json"),
        make_path(prefix, f"{prefix}_full_epoch{last_epoch}_loss.json"),
        make_path(prefix, f"{prefix}_lora_epoch{last_epoch}_loss.json"),
        make_path(prefix, f"{prefix}_head_epoch{last_epoch}_loss.json"),
    ]
    loss_strategy_names = [
        "Pretrained",
        f"Full-E{last_epoch}",
        f"LoRA-E{last_epoch}",
        f"Head-E{last_epoch}",
    ]
    run_evaluate(
        score_files=loss_strategy_files,
        names=loss_strategy_names,
        output_prefix=f"{prefix}_strategy_loss_epoch{last_epoch}",
        dry_run=args.dry_run,
    )

    # 2. Full / LoRA / Head 的涌现曲线（loss）
    def emergence_for_strategy(strategy: str) -> None:
        files = [
            make_path(prefix, f"{prefix}_{strategy}_epoch{e}_loss.json") for e in epochs
        ]
        names = [f"E{e}" for e in epochs]
        run_evaluate(
            score_files=files,
            names=names,
            output_prefix=f"{prefix}_{strategy}_loss_emergence",
            dry_run=args.dry_run,
        )

    for strategy in ["full", "lora", "head"]:
        emergence_for_strategy(strategy)

    # 3. Full-EpochLast 不同攻击方法对比（loss / ratio / mink）
    full_attack_files = [
        make_path(prefix, f"{prefix}_full_epoch{last_epoch}_loss.json"),
        make_path(prefix, f"{prefix}_full_epoch{last_epoch}_ratio.json"),
        make_path(prefix, f"{prefix}_full_epoch{last_epoch}_mink.json"),
    ]
    full_attack_names = ["Loss", "Ratio", "MinK"]
    run_evaluate(
        score_files=full_attack_files,
        names=full_attack_names,
        output_prefix=f"{prefix}_full_epoch{last_epoch}_attack_compare",
        dry_run=args.dry_run,
    )

    # 4. （可选）Ratio 攻击下的策略对比（epoch=last）
    ratio_strategy_files = [
        make_path(prefix, f"{prefix}_pretrained_ratio.json"),
        make_path(prefix, f"{prefix}_full_epoch{last_epoch}_ratio.json"),
        make_path(prefix, f"{prefix}_lora_epoch{last_epoch}_ratio.json"),
        make_path(prefix, f"{prefix}_head_epoch{last_epoch}_ratio.json"),
    ]
    ratio_strategy_names = [
        "Pretrained",
        f"Full-E{last_epoch}",
        f"LoRA-E{last_epoch}",
        f"Head-E{last_epoch}",
    ]
    run_evaluate(
        score_files=ratio_strategy_files,
        names=ratio_strategy_names,
        output_prefix=f"{prefix}_strategy_ratio_epoch{last_epoch}",
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 80)
    if args.dry_run:
        print("✓ dry-run 完成：以上为将要执行的所有评估命令。")
    else:
        print("✓ 所有评估任务已完成。图像和指标已保存到 results/metrics 与 results/plots。")
    print("=" * 80)


if __name__ == "__main__":
    main()


