import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
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


def parse_params_from_prefix(prefix: str) -> Optional[int]:
    m = re.search(r"(\d+(?:\.\d+)?)([mb])\b", prefix.lower())
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return int(num * 1_000_000)
    if unit == "b":
        return int(num * 1_000_000_000)
    return None


def _metrics_bundle_path(prefix: str, attack: str, epoch: int) -> str:
    fname = f"{prefix}_strategy_{attack}_epoch{epoch}_metrics.json"
    return os.path.join(METRICS_DIR, fname)


def _score_path(prefix: str, strategy: str, epoch: int, attack: str) -> str:
    if strategy == "pretrained":
        fname = f"{prefix}_pretrained_{attack}.json"
    else:
        fname = f"{prefix}_{strategy}_epoch{epoch}_{attack}.json"
    return os.path.join(SCORES_DIR, prefix, fname)


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


def metric_value(m: Dict[str, object], metric: str) -> Optional[float]:
    try:
        if metric == "auc":
            return float(m["auc"])
        return float(m.get("tpr_at_1fpr", 0.0))
    except Exception:
        return None


def collect_risk_by_size(
    model_prefixes: List[str],
    strategies: List[str],
    attack: str,
    epoch: int,
    metric: str,
    include_pretrained: bool,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}

    strat_list = strategies.copy()
    if include_pretrained and "pretrained" not in strat_list:
        strat_list = ["pretrained"] + strat_list

    for prefix in model_prefixes:
        out[prefix] = {}
        bundle = _load_metrics_bundle(prefix, attack, epoch)

        for strat in strat_list:
            score_fp = _score_path(prefix, strat, epoch, attack)
            loaded = _load_scores_if_exists(score_fp)
            if loaded is not None:
                scores, labels = loaded
                m = compute_metrics(scores, labels)
                v = metric_value(m, metric)
                if v is not None:
                    out[prefix][strat] = v
                continue

            if bundle and strat in bundle:
                v = metric_value(bundle[strat], metric)
                if v is not None:
                    out[prefix][strat] = v
                continue

    for p in list(out.keys()):
        if not out[p]:
            out.pop(p, None)

    return out


def collect_roc_by_size(
    model_prefixes: List[str],
    strategy: str,
    attack: str,
    epoch: int,
) -> Dict[str, Dict[str, object]]:
    """返回 curves[prefix] = metrics dict（含 fpr/tpr/auc/tpr_at_1fpr）。"""
    curves: Dict[str, Dict[str, object]] = {}

    for prefix in model_prefixes:
        bundle = _load_metrics_bundle(prefix, attack, epoch)
        score_fp = _score_path(prefix, strategy, epoch, attack)
        loaded = _load_scores_if_exists(score_fp)
        if loaded is not None:
            scores, labels = loaded
            curves[prefix] = compute_metrics(scores, labels)
            continue

        if bundle and strategy in bundle:
            if isinstance(bundle[strategy], dict):
                curves[prefix] = bundle[strategy]
            continue

    return curves


def build_size_axis(model_prefixes: List[str], x_mode: str) -> Tuple[List[str], List[float], List[str]]:
    items: List[Tuple[str, int]] = []
    for p in model_prefixes:
        params = parse_params_from_prefix(p)
        if params is None:
            params = 0
        items.append((p, params))

    items.sort(key=lambda x: (x[1], x[0]))
    prefixes_sorted = [p for p, _ in items]

    if x_mode == "categorical":
        x = list(range(len(prefixes_sorted)))
        tick_labels = prefixes_sorted
        return prefixes_sorted, [float(v) for v in x], tick_labels

    xs: List[float] = []
    tick_labels: List[str] = []
    for p, params in items:
        if params <= 0:
            xs.append(float("nan"))
        else:
            xs.append(float(np.log10(float(params))))
        tick_labels.append(p)

    return prefixes_sorted, xs, tick_labels


def plot_scale_risk_curve(
    risk: Dict[str, Dict[str, float]],
    model_prefixes: List[str],
    strategies: List[str],
    attack: str,
    epoch: int,
    metric: str,
    x_mode: str,
    dpi: int,
    out_dir: str,
    output_prefix: str,
) -> None:
    prefixes_sorted, x_all, tick_labels = build_size_axis(model_prefixes, x_mode)

    fig, ax = plt.subplots(figsize=(10, 6))

    for strat in strategies:
        xs: List[float] = []
        ys: List[float] = []
        labels_for_points: List[str] = []

        for p, x in zip(prefixes_sorted, x_all):
            if p not in risk or strat not in risk[p]:
                continue
            if x_mode == "log_params" and (x is None or np.isnan(x)):
                continue
            xs.append(float(x))
            ys.append(float(risk[p][strat]))
            labels_for_points.append(p)

        if not xs:
            continue

        (line,) = ax.plot(xs, ys, linewidth=2.2, alpha=0.95, label=strat)
        ax.scatter(xs, ys, s=70, color=line.get_color(), edgecolor="white", linewidth=1.0, zorder=5)

    if metric == "auc":
        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_ylabel("AUC-ROC", fontsize=16)
    else:
        ax.set_ylabel("TPR @ 1% FPR", fontsize=16)

    ax.set_xlabel("Model size" if x_mode == "categorical" else "log10(#params)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)

    if x_mode == "categorical":
        ax.set_xticks(list(range(len(prefixes_sorted))))
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        valid_ticks: List[float] = []
        valid_labels: List[str] = []
        for x, lbl in zip(x_all, tick_labels):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                continue
            valid_ticks.append(float(x))
            valid_labels.append(lbl)
        if valid_ticks:
            ax.set_xticks(valid_ticks)
            ax.set_xticklabels(valid_labels, rotation=0)

    ax.legend(loc="best", fontsize=13, framealpha=0.9)

    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{output_prefix}_{attack}_epoch{epoch}.png")
    pdf_path = png_path.replace(".png", ".pdf")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"✓ 6.4-1 规模-风险曲线已保存: {png_path}")


def plot_size_epoch_heatmap(
    model_prefixes: List[str],
    strategies: List[str],
    attacks: List[str],
    epochs: List[int],
    metric: str,
    heatmap_strategy: str,
    heatmap_attack: str,
    x_mode: str,
    dpi: int,
    out_dir: str,
    output_prefix: str,
) -> None:
    prefixes_sorted, _x_all, _tick_labels = build_size_axis(model_prefixes, x_mode)

    rows: List[Dict[str, object]] = []
    for prefix in prefixes_sorted:
        for e in epochs:
            risk = collect_risk_by_size(
                model_prefixes=[prefix],
                strategies=strategies,
                attack=heatmap_attack,
                epoch=e,
                metric=metric,
                include_pretrained=False,
            )
            v = None
            if prefix in risk and heatmap_strategy in risk[prefix]:
                v = float(risk[prefix][heatmap_strategy])
            rows.append({"prefix": prefix, "epoch": int(e), "value": v})

    df = pd.DataFrame(rows)
    if df.empty or df["value"].dropna().empty:
        print("⚠ 6.4-2 heatmap：没有可绘制的数据")
        return

    pivot = df.pivot(index="prefix", columns="epoch", values="value").reindex(index=prefixes_sorted)

    fig, ax = plt.subplots(figsize=(11, max(3.2, 0.55 * len(prefixes_sorted))))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="viridis",
        annot=False,
        cbar_kws={"label": "AUC-ROC" if metric == "auc" else "TPR @ 1% FPR"},
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Model size", fontsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="both", labelsize=13)

    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{output_prefix}_heatmap_{heatmap_attack}_{heatmap_strategy}_{metric}.png")
    pdf_path = png_path.replace(".png", ".pdf")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"✓ 6.4-2 风险地形图已保存: {png_path}")


def plot_roc_facets_by_size(
    model_prefixes: List[str],
    attacks: List[str],
    strategy: str,
    epoch: int,
    dpi: int,
    out_dir: str,
    output_prefix: str,
) -> None:
    prefixes_sorted, _x_all, _tick_labels = build_size_axis(model_prefixes, "log_params")
    n_attacks = len(attacks)
    fig, axes = plt.subplots(1, n_attacks, figsize=(18, 6), sharex=False, sharey=True)
    if n_attacks == 1:
        axes = [axes]

    fpr_threshold = 0.01

    for ax, attack in zip(axes, attacks):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.axvline(x=fpr_threshold, color="gray", linestyle=":", linewidth=1.5, alpha=0.9)

        curves = collect_roc_by_size(
            model_prefixes=prefixes_sorted,
            strategy=strategy,
            attack=attack,
            epoch=epoch,
        )

        if not curves:
            ax.set_title(f"Attack: {attack} (no data)", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            continue

        for prefix in prefixes_sorted:
            if prefix not in curves:
                continue
            m = curves[prefix]
            try:
                fpr = np.asarray(m["fpr"], dtype=float)
                tpr = np.asarray(m["tpr"], dtype=float)
                auc_val = float(m["auc"])
            except Exception:
                continue

            ax.plot(
                fpr,
                tpr,
                linewidth=2.1,
                label=f"{prefix} (AUC={auc_val:.3f})",
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("FPR", fontsize=14)
        ax.set_title(f"Attack: {attack}", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)

    axes[0].set_ylabel("TPR", fontsize=14)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(5, len(labels)),
            fontsize=11,
            framealpha=0.9,
        )
        plt.subplots_adjust(bottom=0.20, wspace=0.22)
    else:
        plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{output_prefix}_roc_by_size_{strategy}_epoch{epoch}.png")
    pdf_path = out_path.replace(".png", ".pdf")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"✓ 6.4 ROC(AUC) 曲线已保存: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 6.4: 模型规模对成员推断风险的影响（RQ3）")
    parser.add_argument(
        "--model-prefixes",
        nargs="+",
        default=["pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m"],
        help="模型前缀列表（默认 Pythia: 14m 31m 70m 160m 410m）",
    )
    parser.add_argument("--epoch", type=int, default=5, help="使用哪个 epoch（默认 5）")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["full", "lora", "head"],
        help="对比的策略列表（默认 full lora head）",
    )
    parser.add_argument(
        "--include-pretrained",
        action="store_true",
        help="是否把 pretrained 也加入对比（注意：pretrained 不走 epoch 文件名）",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["loss", "ratio", "mink"],
        help="攻击列表（每个 attack 输出一张 6.4-1 图）",
    )
    parser.add_argument(
        "--metric",
        choices=["auc", "tpr"],
        default="auc",
        help="纵轴指标：auc 或 tpr（TPR@1%FPR）",
    )
    parser.add_argument(
        "--x-mode",
        choices=["log_params", "categorical"],
        default="log_params",
        help="横轴：log_params（默认）或 categorical（按 size 作为有序类别）",
    )
    parser.add_argument("--dpi", type=int, default=600, help="导出 dpi（默认 600）")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录（默认 results/plots/section6_4/）",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="输出文件名前缀（不含扩展名），可选",
    )

    parser.add_argument(
        "--roc-curve",
        action="store_true",
        help="是否额外生成 ROC(AUC) 曲线：固定 strategy+epoch，对比不同模型规模，按 attack 分面",
    )
    parser.add_argument(
        "--roc-strategy",
        type=str,
        default="full",
        help="ROC(AUC) 曲线使用的策略（默认 full）",
    )

    parser.add_argument("--heatmap", action="store_true", help="是否额外生成 6.4-2 热力图")
    parser.add_argument(
        "--heatmap-strategy",
        type=str,
        default="full",
        help="热力图固定的策略（默认 full）",
    )
    parser.add_argument(
        "--heatmap-attack",
        type=str,
        default="loss",
        help="热力图固定的攻击（默认 loss）",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="热力图使用的 epoch 列表（默认 1 2 3 4 5）",
    )

    args = parser.parse_args()

    model_prefixes = [p.strip() for p in args.model_prefixes if p.strip()]
    attacks = order_items([a.strip() for a in args.attacks if a.strip()], PREFERRED_ATTACK_ORDER)

    base_strategies = [s.strip() for s in args.strategies if s.strip()]
    if args.include_pretrained and "pretrained" not in base_strategies:
        base_strategies = ["pretrained"] + base_strategies
    strategies = order_items(base_strategies, PREFERRED_STRATEGY_ORDER)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(PLOTS_DIR, "section6_4")

    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = f"6_4_{args.metric}_{args.x_mode}"

    print("\n" + "=" * 80)
    print("Section 6.4 绘图 (RQ3: 模型规模影响)")
    print("=" * 80)
    print(f"模型前缀: {model_prefixes}")
    print(f"epoch: {args.epoch}")
    print(f"攻击: {attacks}")
    print(f"策略: {strategies}")
    print(f"metric: {args.metric}")
    print(f"x-mode: {args.x_mode}")
    print(f"dpi: {args.dpi}")
    print(f"输出目录: {out_dir}")
    print("=" * 80)

    for attack in attacks:
        risk = collect_risk_by_size(
            model_prefixes=model_prefixes,
            strategies=strategies,
            attack=attack,
            epoch=int(args.epoch),
            metric=args.metric,
            include_pretrained=args.include_pretrained,
        )
        if not risk:
            print(f"⚠ 6.4-1：attack={attack} 没有可绘制的数据")
            continue

        plot_scale_risk_curve(
            risk=risk,
            model_prefixes=model_prefixes,
            strategies=strategies,
            attack=attack,
            epoch=int(args.epoch),
            metric=args.metric,
            x_mode=args.x_mode,
            dpi=int(args.dpi),
            out_dir=out_dir,
            output_prefix=output_prefix,
        )

    if args.heatmap:
        plot_size_epoch_heatmap(
            model_prefixes=model_prefixes,
            strategies=strategies,
            attacks=attacks,
            epochs=[int(e) for e in args.epochs],
            metric=args.metric,
            heatmap_strategy=args.heatmap_strategy,
            heatmap_attack=args.heatmap_attack,
            x_mode=args.x_mode,
            dpi=int(args.dpi),
            out_dir=out_dir,
            output_prefix=output_prefix,
        )

    if args.roc_curve:
        plot_roc_facets_by_size(
            model_prefixes=model_prefixes,
            attacks=attacks,
            strategy=str(args.roc_strategy).strip(),
            epoch=int(args.epoch),
            dpi=int(args.dpi),
            out_dir=out_dir,
            output_prefix=output_prefix,
        )


if __name__ == "__main__":
    main()
