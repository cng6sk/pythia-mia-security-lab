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
from scripts.evaluate import compute_metrics, load_scores, plot_roc_curve  # noqa: E402


plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def _score_path(prefix: str, variant: str, epoch: Optional[int], attack: str) -> str:
    if variant == "pretrained":
        fname = f"{prefix}_pretrained_{attack}.json"
    else:
        if epoch is None:
            raise ValueError("epoch is required for finetuned variants")
        fname = f"{prefix}_{variant}_epoch{epoch}_{attack}.json"
    return os.path.join(SCORES_DIR, prefix, fname)


def _metrics_path(prefix: str, attack: str, epoch: int) -> str:
    fname = f"{prefix}_strategy_{attack}_epoch{epoch}_metrics.json"
    return os.path.join(METRICS_DIR, fname)


def _load_scores_if_exists(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not os.path.isfile(path):
        return None
    scores, labels = load_scores(path)
    return scores, labels


def _load_metrics_bundle_if_exists(prefix: str, attack: str, epoch: int) -> Optional[Dict[str, Dict[str, object]]]:
    metrics_fp = _metrics_path(prefix, attack, epoch)
    if not os.path.isfile(metrics_fp):
        return None
    with open(metrics_fp, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return None
    return raw


def _build_variant_order(epoch: int) -> List[Tuple[str, str]]:
    return [
        ("Pretrained", "pretrained"),
        (f"Full-E{epoch}", "full"),
        (f"LoRA-E{epoch}", "lora"),
        (f"Head-E{epoch}", "head"),
    ]


def plot_roc_for_attack(prefix: str, attack: str, epoch: int, out_dir: str) -> None:
    variants = _build_variant_order(epoch)

    metrics_dict: Dict[str, Dict[str, object]] = {}
    found_any_scores = False

    for display_name, variant in variants:
        score_fp = _score_path(prefix, variant, epoch if variant != "pretrained" else None, attack)
        loaded = _load_scores_if_exists(score_fp)
        if loaded is None:
            continue
        found_any_scores = True
        scores, labels = loaded
        metrics_dict[display_name] = compute_metrics(scores, labels)

    if not metrics_dict:
        metrics_fp = _metrics_path(prefix, attack, epoch)
        if os.path.isfile(metrics_fp):
            with open(metrics_fp, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for display_name, _ in variants:
                if display_name in raw:
                    metrics_dict[display_name] = raw[display_name]
        if not metrics_dict:
            print(f"⚠ 未找到可用于 ROC 绘图的数据: prefix={prefix}, attack={attack}, epoch={epoch}")
            return

    out_path = os.path.join(out_dir, f"{prefix}_6_1_{attack}_roc_epoch{epoch}.png")
    plot_roc_curve(metrics_dict, out_path)

    if not found_any_scores:
        print("(提示) ROC 曲线使用 metrics 文件绘制；若要生成 score 分布图，需要 scores JSON。")


def _scores_to_df(scores: np.ndarray, labels: np.ndarray, variant_name: str) -> "object":
    import pandas as pd

    membership = np.where(labels == 1, "member", "non-member")
    return pd.DataFrame({"score": scores, "membership": membership, "variant": variant_name})


def plot_pair_roc(
    prefix: str,
    attack: str,
    finetune_strategy: str,
    epoch: int,
    out_dir: str,
    dpi: int,
) -> None:
    variants = _build_variant_order(epoch)
    display_map = {v: d for d, v in variants}

    pretrained_display = display_map["pretrained"]
    finetune_display = display_map[finetune_strategy]

    metrics_dict: Dict[str, Dict[str, object]] = {}
    found_any_scores = False

    pretrained_fp = _score_path(prefix, "pretrained", None, attack)
    finetune_fp = _score_path(prefix, finetune_strategy, epoch, attack)

    pretrained_loaded = _load_scores_if_exists(pretrained_fp)
    finetune_loaded = _load_scores_if_exists(finetune_fp)

    if pretrained_loaded is not None:
        found_any_scores = True
        scores, labels = pretrained_loaded
        metrics_dict[pretrained_display] = compute_metrics(scores, labels)

    if finetune_loaded is not None:
        found_any_scores = True
        scores, labels = finetune_loaded
        metrics_dict[finetune_display] = compute_metrics(scores, labels)

    if len(metrics_dict) != 2:
        bundle = _load_metrics_bundle_if_exists(prefix, attack, epoch)
        if bundle is not None:
            if pretrained_display in bundle:
                metrics_dict[pretrained_display] = bundle[pretrained_display]
            if finetune_display in bundle:
                metrics_dict[finetune_display] = bundle[finetune_display]

    if len(metrics_dict) != 2:
        print(
            "⚠ 无法生成 pair ROC：需要 pretrained 与 finetuned 两条曲线。\n"
            f"  pretrained: {pretrained_fp}\n"
            f"  finetuned: {finetune_fp}\n"
        )
        return

    fpr_threshold = 0.01

    # 10x8 inches @ dpi=300 -> 3000x2400 px, 满足论文清晰度
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Random Guess", linewidth=1)
    ax.axvline(x=fpr_threshold, color="gray", linestyle=":", linewidth=2.2, alpha=0.9, zorder=1)
    ax.text(
        fpr_threshold,
        0.03,
        "FPR=1%",
        rotation=90,
        ha="right",
        va="bottom",
        color="gray",
        fontsize=11,
    )

    curve_cache: List[Tuple[str, np.ndarray, np.ndarray, float, float, str]] = []

    for name, metrics in metrics_dict.items():
        fpr = np.array(metrics["fpr"], dtype=float)
        tpr = np.array(metrics["tpr"], dtype=float)
        auc_val = float(metrics["auc"])
        if "tpr_at_1fpr" in metrics:
            tpr_at = float(metrics["tpr_at_1fpr"])
        else:
            tpr_at = float(np.interp(fpr_threshold, fpr, tpr))

        (line,) = ax.plot(
            fpr,
            tpr,
            linewidth=2.6,
            label=f"{name} (AUC={auc_val:.3f}, TPR@1%={tpr_at:.3f})",
            zorder=2,
        )
        curve_cache.append((name, fpr, tpr, auc_val, tpr_at, line.get_color()))

    for i, (name, _fpr, _tpr, _auc, tpr_at, color) in enumerate(curve_cache):
        ax.scatter(
            [fpr_threshold],
            [tpr_at],
            s=90,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            f"TPR@1%={tpr_at:.3f}",
            xy=(fpr_threshold, tpr_at),
            xytext=(18, 18 if i == 0 else -18),
            textcoords="offset points",
            color=color,
            fontsize=12,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.9},
        )

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=16)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    # 不加大标题，方便论文 caption 描述
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = os.path.join(
        out_dir,
        f"{prefix}_6_1_{attack}_roc_pretrained_vs_{finetune_strategy}_epoch{epoch}.png",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    # 同步导出 PDF（矢量更清晰）
    pdf_path = out_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"✓ ROC 对照图已保存: {out_path}")
    plt.close()

    if not found_any_scores:
        print("(提示) ROC 曲线使用 metrics 文件绘制；若要生成分布图，需要 scores JSON。")


def plot_pair_score_distribution(
    prefix: str,
    attack: str,
    finetune_strategy: str,
    epoch: int,
    out_dir: str,
    dpi: int,
) -> None:
    variants = _build_variant_order(epoch)
    display_map = {v: d for d, v in variants}

    pretrained_display = display_map["pretrained"]
    finetune_display = display_map[finetune_strategy]

    pretrained_fp = _score_path(prefix, "pretrained", None, attack)
    finetune_fp = _score_path(prefix, finetune_strategy, epoch, attack)

    pretrained_loaded = _load_scores_if_exists(pretrained_fp)
    finetune_loaded = _load_scores_if_exists(finetune_fp)

    if pretrained_loaded is None or finetune_loaded is None:
        print(
            "⚠ 无法生成分布图：需要 pretrained 与 finetuned 的 scores JSON。\n"
            f"  pretrained: {pretrained_fp}\n"
            f"  finetuned: {finetune_fp}\n"
        )
        return

    import pandas as pd

    pre_scores, pre_labels = pretrained_loaded
    ft_scores, ft_labels = finetune_loaded

    df_pre = _scores_to_df(pre_scores, pre_labels, pretrained_display)
    df_ft = _scores_to_df(ft_scores, ft_labels, finetune_display)

    # 放大尺寸，便于 300+ dpi 导出（14x5 -> 4900x1750 @350dpi）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, sub_df, title in [
        (axes[0], df_pre, pretrained_display),
        (axes[1], df_ft, finetune_display),
    ]:
        try:
            sns.kdeplot(
                data=sub_df,
                x="score",
                hue="membership",
                fill=True,
                common_norm=False,
                alpha=0.35,
                linewidth=2,
                ax=ax,
            )
        except Exception:
            sns.histplot(
                data=sub_df,
                x="score",
                hue="membership",
                stat="density",
                element="step",
                common_norm=False,
                ax=ax,
            )
        ax.set_title("")  # 去掉大标题，论文 caption 负责描述
        ax.set_xlabel("Attack score", fontsize=15)
        ax.set_ylabel("Density", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        leg = ax.get_legend()
        if leg:
            leg.set_title("")
            leg.set_bbox_to_anchor((1.01, 1.0))
            frame = leg.get_frame()
            if frame is not None:
                frame.set_alpha(0.9)
        n_member = int((sub_df["membership"] == "member").sum())
        n_non = int((sub_df["membership"] == "non-member").sum())
        ax.text(
            0.98,
            0.98,
            f"n(member)={n_member}\n n(non)={n_non}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )

    # 不加 suptitle，交给 LaTeX caption 描述
    plt.tight_layout(rect=[0, 0, 1, 1])

    out_path = os.path.join(
        out_dir,
        f"{prefix}_6_1_{attack}_dist_pretrained_vs_{finetune_strategy}_epoch{epoch}.png",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    pdf_path = out_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"✓ 分布对照图已保存: {out_path}")
    plt.close()


def plot_score_distributions_for_attack(prefix: str, attack: str, epoch: int, out_dir: str) -> None:
    variants = _build_variant_order(epoch)

    dfs = []
    for display_name, variant in variants:
        score_fp = _score_path(prefix, variant, epoch if variant != "pretrained" else None, attack)
        loaded = _load_scores_if_exists(score_fp)
        if loaded is None:
            print(f"⚠ 缺少 score 文件，跳过分布子图: {score_fp}")
            continue
        scores, labels = loaded
        dfs.append(_scores_to_df(scores, labels, display_name))

    if not dfs:
        print(f"⚠ 没有任何 scores JSON，无法绘制分布图: prefix={prefix}, attack={attack}, epoch={epoch}")
        return

    import pandas as pd

    df = pd.concat(dfs, ignore_index=True)
    variants_present = [name for name, _ in variants if name in set(df["variant"].tolist())]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for i, vname in enumerate(variants_present[:4]):
        ax = axes[i]
        sub = df[df["variant"] == vname]
        try:
            sns.kdeplot(
                data=sub,
                x="score",
                hue="membership",
                fill=True,
                common_norm=False,
                alpha=0.35,
                linewidth=2,
                ax=ax,
            )
        except Exception:
            sns.histplot(
                data=sub,
                x="score",
                hue="membership",
                stat="density",
                element="step",
                common_norm=False,
                ax=ax,
            )

        ax.set_title(vname)
        ax.set_ylabel("Density")

        n_member = int((sub["membership"] == "member").sum())
        n_non = int((sub["membership"] == "non-member").sum())
        ax.text(
            0.98,
            0.98,
            f"n(member)={n_member}\n n(non)={n_non}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )

        if i != 0:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    for j in range(len(variants_present), 4):
        axes[j].axis("off")

    fig.suptitle(f"{prefix} | Attack={attack} | Score Distributions (Epoch={epoch})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(out_dir, f"{prefix}_6_1_{attack}_score_dist_epoch{epoch}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✓ 分布图已保存: {out_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 6.1: Pretrained vs Fine-tuned MIA risk plotting")
    parser.add_argument("--model-prefix", type=str, required=True, help="如 pythia-70m")
    parser.add_argument("--epoch", type=int, default=5, help="用于对照的微调 epoch（默认 5）")
    parser.add_argument(
        "--pair",
        action="store_true",
        help="仅绘制 Pretrained vs 指定微调策略/epoch（生成论文 6.1 的两张核心图）",
    )
    parser.add_argument(
        "--finetune",
        type=str,
        choices=["full", "lora", "head"],
        default="full",
        help="pair 模式下选择对比的微调策略（默认: full）",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["loss", "ratio", "mink"],
        help="攻击方法列表（默认: loss ratio mink）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录（默认 results/plots/section6_1/<prefix>/）",
    )
    parser.add_argument(
        "--skip-dist",
        action="store_true",
        help="仅画 ROC（跳过分布图）",
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
    attacks = [a.strip() for a in args.attacks if a.strip()]

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(PLOTS_DIR, "section6_1", prefix)

    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Section 6.1 绘图")
    print("=" * 80)
    print(f"模型前缀: {prefix}")
    print(f"Epoch: {epoch}")
    print(f"Attacks: {attacks}")
    print(f"输出目录: {out_dir}")
    print("=" * 80)

    for attack in attacks:
        if args.pair:
            plot_pair_roc(
                prefix=prefix,
                attack=attack,
                finetune_strategy=args.finetune,
                epoch=epoch,
                out_dir=out_dir,
                dpi=args.dpi,
            )
            if not args.skip_dist:
                plot_pair_score_distribution(
                    prefix=prefix,
                    attack=attack,
                    finetune_strategy=args.finetune,
                    epoch=epoch,
                    out_dir=out_dir,
                    dpi=args.dpi,
                )
        else:
            plot_roc_for_attack(prefix, attack, epoch, out_dir)
            if not args.skip_dist:
                plot_score_distributions_for_attack(prefix, attack, epoch, out_dir)


if __name__ == "__main__":
    main()
