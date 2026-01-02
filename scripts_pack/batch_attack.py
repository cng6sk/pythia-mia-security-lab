"""
批量成员推断攻击脚本
Batch MIA Attacks for All Checkpoints of a Given Model Prefix

功能：
- 自动遍历 models/ 下指定前缀的模型目录（如：models/pythia-70m-*）
- 在每个模型目录中查找所有 checkpoint-* 子目录
- 对每个 checkpoint 运行 scripts/attack.py，计算指定攻击方法的得分
- 自动为每个结果生成有意义的名称（包含策略名和 epoch/step 信息）

用法示例（在项目根目录执行）：

1. 仅对 pythia-70m 的所有策略 (full/lora/head) 做 loss 攻击：

   python -m scripts_pack.batch_attack --model-prefix pythia-70m --attacks loss

2. 对单一策略（例如 full）做多种攻击（loss + ratio）：

   python -m scripts_pack.batch_attack --model-prefix pythia-70m --strategies full --attacks loss ratio

注意：
- 本脚本不会修改任何已有结果，只是追加新的攻击结果文件。
- 需要确保对应的 checkpoint 已经通过 scripts/train.py 训练生成。
"""

import argparse
import json
import os
import subprocess
from typing import Dict, List, Tuple

from config.config import MODELS_DIR, SCORES_DIR


def find_model_variants(model_prefix: str, strategies: List[str]) -> Dict[str, str]:
    """
    在 models/ 目录下查找所有以给定前缀开头的模型子目录，并按策略名过滤。

    约定：
    - 模型目录命名为：<model_prefix>-<strategy>，例如：
        - pythia-70m-full
        - pythia-70m-lora
        - pythia-70m-head

    Returns:
        dict[str, str]: {strategy_name: absolute_model_dir}
    """
    model_variants: Dict[str, str] = {}

    if not os.path.isdir(MODELS_DIR):
        print(f"⚠ 未找到模型目录: {MODELS_DIR}")
        return model_variants

    for entry in os.listdir(MODELS_DIR):
        full_path = os.path.join(MODELS_DIR, entry)
        if not os.path.isdir(full_path):
            continue

        # 期望目录名格式: "<prefix>-<strategy>"
        if not entry.startswith(model_prefix):
            continue

        # 例：pythia-70m-full -> strategy = "full"
        parts = entry.split("-", maxsplit=2)
        if len(parts) < 3:
            # 可能是 cache 等目录，跳过
            continue

        strategy = parts[2]
        if strategies and strategy not in strategies:
            continue

        model_variants[strategy] = full_path

    return model_variants


def find_checkpoints(model_dir: str) -> List[str]:
    """
    在给定的模型目录下查找所有 checkpoint-* 子目录，按步数从小到大排序。

    Returns:
        List[str]: checkpoint 目录的绝对路径列表
    """
    checkpoints: List[Tuple[int, str]] = []

    if not os.path.isdir(model_dir):
        return []

    for entry in os.listdir(model_dir):
        if not entry.startswith("checkpoint-"):
            continue
        full_path = os.path.join(model_dir, entry)
        if not os.path.isdir(full_path):
            continue

        # checkpoint-1234 -> step = 1234
        try:
            step = int(entry.split("-")[-1])
        except ValueError:
            step = -1
        checkpoints.append((step, full_path))

    # 按步数排序，保证顺序一致
    checkpoints.sort(key=lambda x: x[0])
    return [path for _, path in checkpoints]


def infer_epoch_from_checkpoint(ckpt_dir: str) -> str:
    """
    从 checkpoint 目录中的 trainer_state.json 推断 epoch 信息（整数）。

    如果无法获取，返回基于 step 的占位字符串。
    """
    trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
    ckpt_name = os.path.basename(ckpt_dir)

    if not os.path.isfile(trainer_state_path):
        # 回退到使用 checkpoint 名称
        return ckpt_name.replace("checkpoint-", "step")

    try:
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        # trainer_state["log_history"] 中通常包含若干条 dict，其中含有 "epoch"
        epochs = [
            float(item["epoch"])
            for item in state.get("log_history", [])
            if isinstance(item, dict) and "epoch" in item
        ]
        if not epochs:
            return ckpt_name.replace("checkpoint-", "step")
        # 取最后一次记录的 epoch 并转换为整数
        last_epoch = int(round(epochs[-1]))
        return f"epoch{last_epoch}"
    except Exception:
        return ckpt_name.replace("checkpoint-", "step")


def run_attack(
    model_path: str,
    attack: str,
    output_name: str,
    output_dir: str,
    dry_run: bool = False,
) -> None:
    """
    调用 scripts/attack.py 对指定模型运行一次攻击。

    Args:
        model_path: 模型 checkpoint 目录路径（相对或绝对）
        attack: 攻击方法名（loss / mink / ratio）
        output_name: 结果文件前缀（不含扩展名）
        output_dir: 结果文件保存目录（已包含模型前缀子目录）
        dry_run: 如果为 True，仅打印命令不实际执行
    """
    cmd = [
        "python",
        "scripts/attack.py",
        "--model_path",
        model_path,
        "--attack",
        attack,
        "--output_name",
        output_name,
        "--output_dir",
        output_dir,
    ]

    print("\n" + "=" * 80)
    print(f"运行攻击：{attack} | 模型: {model_path}")
    print(f"输出文件前缀: {output_name}")
    print(f"保存目录: {output_dir}")
    print("命令:", " ".join(cmd))
    print("=" * 80)

    if dry_run:
        return

    # 确保结果目录存在
    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ 攻击失败: {output_name}")
    else:
        print(f"✓ 攻击完成: {output_name}")


def main():
    parser = argparse.ArgumentParser(description="批量对所有 checkpoint 执行 MIA 攻击")
    parser.add_argument(
        "--model-prefix",
        type=str,
        required=True,
        help="模型前缀，如：pythia-70m（将匹配 models/ 下以该前缀开头的子目录）",
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=["full", "lora", "head"],
        help="需要处理的微调策略子目录名（默认: full lora head）",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["loss"],
        help="要运行的攻击方法列表，例如：loss mink ratio（默认: loss）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的命令，而不真正运行 attack.py",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="如果结果文件已存在，则跳过对应攻击（默认: 开启）",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("批量成员推断攻击 (MIA) - 批处理脚本")
    print("=" * 80)
    print(f"模型前缀: {args.model_prefix}")
    print(f"策略列表: {args.strategies}")
    print(f"攻击方法: {args.attacks}")
    print(f"仅打印命令(dry-run): {args.dry_run}")
    print(f"跳过已存在结果: {args.skip_existing}")
    print("=" * 80)

    # 1. 清理策略列表（去除可能的换行符和空白）
    strategies_clean = [s.strip() for s in args.strategies if s.strip()]
    
    # 2. 检查是否包含预训练模型
    include_pretrained = "pretrained" in strategies_clean
    strategies_to_find = [s for s in strategies_clean if s != "pretrained"]
    
    # 3. 查找所有策略对应的模型目录
    model_variants = find_model_variants(args.model_prefix, strategies_to_find)
    
    # 4. 如果包含预训练模型，添加到列表
    if include_pretrained:
        # 添加预训练模型（使用 config 中的 MODEL_NAME）
        from config.config import MODEL_NAME
        model_variants["pretrained"] = MODEL_NAME  # 存储 HuggingFace 模型名称
    
    # 5. 检查是否有任何模型需要处理
    if not model_variants:
        if include_pretrained:
            # 如果只有 pretrained，应该已经添加了，不应该到这里
            print(f"⚠ 预训练模型配置异常，请检查 config.py 中的 MODEL_NAME。")
        else:
            print(f"⚠ 未找到任何匹配前缀 '{args.model_prefix}' 的模型目录，请检查 models/ 下的命名。")
        return

    print("\n发现以下模型变体：")
    for strategy, path in model_variants.items():
        if strategy == "pretrained":
            print(f"- {strategy}: {path} (HuggingFace 模型)")
        else:
            print(f"- {strategy}: {path}")

    # 3. 遍历每个策略
    for strategy, model_dir in model_variants.items():
        print("\n" + "-" * 80)
        if strategy == "pretrained":
            print(f"处理策略: {strategy} | 模型: {model_dir}")
        else:
            print(f"处理策略: {strategy} | 目录: {model_dir}")
        print("-" * 80)

        # 预训练模型没有 checkpoint，直接攻击
        if strategy == "pretrained":
            for attack_name in args.attacks:
                # 结果文件名：模型前缀 + pretrained + 攻击名
                output_name = f"{args.model_prefix}_pretrained_{attack_name}"

                # 创建模型前缀子目录
                model_scores_dir = os.path.join(SCORES_DIR, args.model_prefix)
                os.makedirs(model_scores_dir, exist_ok=True)

                # 如果需要跳过已存在的结果
                if args.skip_existing:
                    out_path = os.path.join(model_scores_dir, f"{output_name}.json")
                    if os.path.exists(out_path):
                        print(f"⚠ 结果已存在，跳过: {out_path}")
                        continue

                # 对预训练模型运行攻击（使用 HuggingFace 模型名称）
                run_attack(
                    model_path=model_dir,  # HuggingFace 模型名称，如 "EleutherAI/pythia-70m"
                    attack=attack_name,
                    output_name=output_name,
                    output_dir=model_scores_dir,
                    dry_run=args.dry_run,
                )
            continue

        # 微调模型：查找所有 checkpoint
        checkpoints = find_checkpoints(model_dir)
        if not checkpoints:
            print(f"⚠ 未在 {model_dir} 下找到任何 checkpoint-* 目录，跳过。")
            continue

        print(f"✓ 发现 {len(checkpoints)} 个 checkpoint：")
        for ckpt_dir in checkpoints:
            print(f"  - {os.path.basename(ckpt_dir)}")

        # 4. 对每个 checkpoint 运行指定的攻击方法
        for ckpt_dir in checkpoints:
            ckpt_name = os.path.basename(ckpt_dir)  # 例如 "checkpoint-1565"
            epoch_tag = infer_epoch_from_checkpoint(ckpt_dir)  # 例如 "epoch1" 或 "step-1565"

            for attack_name in args.attacks:
                # 结果文件名建议包含：模型前缀 + 策略 + epoch/step + 攻击名
                output_name = f"{args.model_prefix}_{strategy}_{epoch_tag}_{attack_name}"

                # 创建模型前缀子目录（例如 results/scores/pythia-70m/）
                model_scores_dir = os.path.join(SCORES_DIR, args.model_prefix)
                os.makedirs(model_scores_dir, exist_ok=True)

                # 如果需要跳过已存在的结果
                if args.skip_existing:
                    out_path = os.path.join(model_scores_dir, f"{output_name}.json")
                    if os.path.exists(out_path):
                        print(f"⚠ 结果已存在，跳过: {out_path}")
                        continue

                # model_path 使用相对路径，便于在项目根目录下执行
                rel_model_path = os.path.relpath(ckpt_dir, ".")
                run_attack(
                    model_path=rel_model_path,
                    attack=attack_name,
                    output_name=output_name,
                    output_dir=model_scores_dir,
                    dry_run=args.dry_run,
                )

    print("\n" + "=" * 80)
    if args.dry_run:
        print("✓ dry-run 完成：以上为将要执行的所有命令。")
    else:
        print("✓ 所有批量攻击任务已完成（或已跳过已有结果）。")
    print("=" * 80)


if __name__ == "__main__":
    main()


