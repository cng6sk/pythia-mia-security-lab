import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import METRICS_DIR, PLOTS_DIR, SCORES_DIR  # noqa: E402
from scripts.evaluate import compute_metrics, load_scores  # noqa: E402

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)


PREFERRED_ATTACK_ORDER = ["loss", "mink", "ratio"]
PREFERRED_STRATEGY_ORDER = ["pretrained", "lora", "head", "full"]


def order_items(items: List[str], preferred: List[str]) -> List[str]:
    """将 items 按 preferred 优先级排序，其余未列出的项按原顺序追加。"""
    ordered: List[str] = []
    seen = set()

    for x in preferred:
        if x in items and x not in seen:
            ordered.append(x)
            seen.add(x)

    for x in items:
        if x not in seen:
            ordered.append(x)
            seen.add(x)

    return ordered


def _score_path(prefix: str, strategy: str, epoch: int, attack: str) -> str:
    if strategy == "pretrained":
        fname = f"{prefix}_pretrained_{attack}.json"
    else:
        fname = f"{prefix}_{strategy}_epoch{epoch}_{attack}.json"
    return os.path.join(SCORES_DIR, prefix, fname)


def _metrics_bundle_path(prefix: str, attack: str, epoch: int) -> str:
    fname = f"{prefix}_strategy_{attack}_epoch{epoch}_metrics.json"
    return os.path.join(METRICS_DIR, fname)


def _load_scores_if_exists(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not os.path.isfile(path):
        return None
    return load_scores(path)


def _load_metrics_bundle(prefix: str, attack: str, epoch: int) -> Optional[Dict[str, Dict[str, object]]]:
    fp = _metrics_bundle_path(prefix, attack, epoch)
    if not os.path.isfile(fp):
        return None
    with open(fp, "r", encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except Exception:
            return None
    if not isinstance(raw, dict):
        return None
    return raw


def collect_curves(
    prefix: str,
    strategies: List[str],
    attacks: List[str],
    epoch: int,
    include_pretrained: bool,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """
    返回 curves[attack][strategy] = {fpr, tpr, auc, tpr_at_1fpr}
    """
    result: Dict[str, Dict[str, Dict[str, object]]] = {}

    strat_list = strategies.copy()
    if include_pretrained and "pretrained" not in strat_list:
        strat_list = ["pretrained"] + strat_list

    for attack in attacks:
        result[attack] = {}
        bundle = _load_metrics_bundle(prefix, attack, epoch)

        for strat in strat_list:
            score_fp = _score_path(prefix, strat, epoch, attack)
            loaded = _load_scores_if_exists(score_fp)
            if loaded is not None:
                scores, labels = loaded
                result[attack][strat] = compute_metrics(scores, labels)
                continue

            if bundle and strat in bundle:
                result[attack][strat] = bundle[strat]
                continue

            print(f"⚠ 缺少数据，跳过: attack={attack}, strategy={strat}, file={score_fp}")

        if not result[attack]:
            result.pop(attack, None)

    return result


def plot_attack_facets(
    curves: Dict[str, Dict[str, Dict[str, object]]],
    attacks: List[str],
    strategies: List[str],
    prefix: str,
    epoch: int,
    dpi: int,
    out_dir: str,
) -> None:
    if not curves:
        print("⚠ 没有可绘制的曲线数据")
        return

    n_attacks = len(attacks)
    # 更宽的比例，减小子图压缩；height 控制在 6 保持清晰
    fig, axes = plt.subplots(1, n_attacks, figsize=(18, 6), sharex=False, sharey=True)
    if n_attacks == 1:
        axes = [axes]

    fpr_threshold = 0.01

    for ax, attack in zip(axes, attacks):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.axvline(x=fpr_threshold, color="gray", linestyle=":", linewidth=1.5, alpha=0.9)

        if attack not in curves:
            ax.set_title(f"{attack} (no data)")
            continue

        metrics_lines: List[str] = []

        for strat in strategies:
            if strat not in curves[attack]:
                continue
            m = curves[attack][strat]
            fpr = np.array(m["fpr"], dtype=float)
            tpr = np.array(m["tpr"], dtype=float)
            auc = float(m["auc"])
            tpr_at = float(m.get("tpr_at_1fpr", np.interp(fpr_threshold, fpr, tpr)))

            (line,) = ax.plot(fpr, tpr, linewidth=2.2, label=strat)
            color = line.get_color()
            ax.scatter(
                [fpr_threshold],
                [tpr_at],
                s=70,
                color=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
            )

            metrics_lines.append(f"{strat}: AUC={auc:.3f}, TPR@1%={tpr_at:.3f}")

        if metrics_lines:
            ax.text(
                0.98,
                0.02,
                "\n".join(metrics_lines),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=11,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "gray", "alpha": 0.88},
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")  # 保持比例，避免曲线“瘦长”扭曲
        ax.set_xlabel("FPR", fontsize=14)
        ax.set_title(f"Attack: {attack}", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)

    axes[0].set_ylabel("TPR", fontsize=14)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=12, framealpha=0.9)
        plt.subplots_adjust(bottom=0.18, wspace=0.22)
    else:
        plt.tight_layout()

    out_path = os.path.join(out_dir, f"{prefix}_6_2_attack_facets_epoch{epoch}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    pdf_path = out_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"✓ 1x{n_attacks} ROC 面板已保存: {out_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 6.2: 1x3 ROC（固定攻击，比较策略）")
    parser.add_argument("--model-prefix", type=str, required=True, help="如 pythia-70m")
    parser.add_argument("--epoch", type=int, default=5, help="使用的 epoch（默认 5）")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["full", "lora", "head"],
        help="要对比的微调策略（默认: full lora head）",
    )
    parser.add_argument(
        "--include-pretrained",
        action="store_true",
        help="是否包含预训练 baseline（加入单独曲线）",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["loss", "ratio", "mink"],
        help="攻击方法列表，顺序即面板顺序",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录（默认 results/plots/section6_2/<prefix>/）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=500,
        help="导出图片的 dpi（默认 500，可设 600）",
    )

    args = parser.parse_args()

    prefix = args.model_prefix
    epoch = int(args.epoch)
    strategies = [s.strip() for s in args.strategies if s.strip()]
    attacks = [a.strip() for a in args.attacks if a.strip()]

    # 统一排序：攻击 loss/mink/ratio（左到右），策略 pretrained/lora/head/full（用于曲线与文本框顺序）
    attacks = order_items(attacks, PREFERRED_ATTACK_ORDER)
    base_strategies = strategies
    if args.include_pretrained and "pretrained" not in base_strategies:
        base_strategies = ["pretrained"] + base_strategies
    plot_strategies = order_items(base_strategies, PREFERRED_STRATEGY_ORDER)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(PLOTS_DIR, "section6_2", prefix)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Section 6.2 绘图 (1x3 ROC: 固定攻击，比较策略)")
    print("=" * 80)
    print(f"模型前缀: {prefix}")
    print(f"Epoch: {epoch}")
    print(f"策略: {strategies} (包含预训练: {args.include_pretrained})")
    print(f"攻击: {attacks}")
    print(f"输出目录: {out_dir}")
    print("=" * 80)

    curves = collect_curves(
        prefix=prefix,
        strategies=strategies,
        attacks=attacks,
        epoch=epoch,
        include_pretrained=args.include_pretrained,
    )

    plot_attack_facets(
        curves=curves,
        attacks=attacks,
        strategies=plot_strategies,
        prefix=prefix,
        epoch=epoch,
        dpi=args.dpi,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
