"""
批量训练脚本
Batch Training Script for Multiple Fine-tuning Strategies

功能：
- 自动对指定模型运行多种微调策略（full, lora, head）
- 支持自定义模型名称（如 pythia-31m, pythia-70m）
- 自动创建对应的模型保存目录

用法示例（在项目根目录执行）：

1. 训练 pythia-31m 的所有策略：
   python -m scripts_pack.batch_train --model-name EleutherAI/pythia-31m --strategies full lora head

2. 只训练 full 和 lora：
   python -m scripts_pack.batch_train --model-name EleutherAI/pythia-31m --strategies full lora

3. 使用默认模型（config.py 中的 MODEL_NAME）：
   python -m scripts_pack.batch_train --strategies full lora head

注意：
- 训练时间较长，建议先用 --dry-run 查看将要执行的命令
- 每个策略的训练会依次进行（不是并行），避免显存冲突
"""

import argparse
import os
import subprocess
import sys
from typing import List

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import MODELS_DIR, MODEL_NAME as DEFAULT_MODEL_NAME


def run_training(
    model_name: str,
    strategy: str,
    dry_run: bool = False,
) -> bool:
    """
    调用 scripts/train.py 对指定模型运行一次微调。

    Args:
        model_name: HuggingFace 模型名称（如 "EleutherAI/pythia-31m"）
        strategy: 微调策略（full / lora / head）
        dry_run: 如果为 True，仅打印命令不实际执行

    Returns:
        bool: 训练是否成功
    """
    cmd = [
        "python",
        "scripts/train.py",
        "--strategy",
        strategy,
        "--model-name",
        model_name,
    ]

    print("\n" + "=" * 80)
    print(f"训练策略: {strategy.upper()} | 模型: {model_name}")
    print("=" * 80)
    print("命令:", " ".join(cmd))
    print("=" * 80)

    if dry_run:
        print("(dry-run 模式，不实际执行)")
        return True
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"❌ 训练失败: {strategy}")
        return False
    else:
        print(f"✓ 训练完成: {strategy}")
        return True




def main():
    parser = argparse.ArgumentParser(description="批量训练多种微调策略")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HuggingFace 模型名称（如 EleutherAI/pythia-31m）。如果不指定，使用 config.py 中的 MODEL_NAME",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["full", "lora", "head"],
        choices=["full", "lora", "head"],
        help="要训练的微调策略列表（默认: full lora head）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的命令，而不真正运行训练",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="如果模型目录已存在，则跳过对应策略的训练",
    )

    args = parser.parse_args()

    # 确定使用的模型名称
    if args.model_name:
        model_name = args.model_name
        # 提取模型短名称（用于目录命名）
        model_short = model_name.split("/")[-1]  # "pythia-31m"
    else:
        model_name = DEFAULT_MODEL_NAME
        model_short = model_name.split("/")[-1]
        print(f"使用 config.py 中的默认模型: {model_name}")

    print("\n" + "=" * 80)
    print("批量微调训练脚本")
    print("=" * 80)
    print(f"模型: {model_name}")
    print(f"策略列表: {args.strategies}")
    print(f"仅打印命令(dry-run): {args.dry_run}")
    print(f"跳过已存在模型: {args.skip_existing}")
    print("=" * 80)

    # 依次训练每个策略
    results = {}
    for strategy in args.strategies:
        # 检查是否已存在
        if args.skip_existing:
            model_dir = os.path.join(MODELS_DIR, f"{model_short}-{strategy}")
            if os.path.isdir(model_dir) and os.listdir(model_dir):
                print(f"\n⚠ 模型目录已存在，跳过: {model_dir}")
                results[strategy] = "skipped"
                continue

        # 运行训练
        success = run_training(
            model_name=model_name,
            strategy=strategy,
            dry_run=args.dry_run,
        )
        results[strategy] = "success" if success else "failed"

    # 打印总结
    print("\n" + "=" * 80)
    print("训练总结")
    print("=" * 80)
    for strategy, status in results.items():
        if status == "success":
            print(f"✓ {strategy.upper()}: 训练完成")
        elif status == "skipped":
            print(f"⏭ {strategy.upper()}: 已跳过（目录已存在）")
        else:
            print(f"❌ {strategy.upper()}: 训练失败")
    print("=" * 80)


if __name__ == "__main__":
    main()

