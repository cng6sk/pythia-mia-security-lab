import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


def _score_path(prefix: str, strategy: str, epoch: int, attack: str) -> str:
    fname = f"{prefix}_{strategy}_epoch{epoch}_{attack}.json"
    return os.path.join(SCORES_DIR, prefix, fname)


def _emergence_metrics_path(prefix: str, strategy: str, attack: str) -> str:
    # batch_eval.py 里 emergence 输出前缀形如: {prefix}_{strategy}_loss_emergence
    fname = f"{prefix}_{strategy}_{attack}_emergence_metrics.json"
    return os.path.join(METRICS_DIR, fname)


def _load_scores_if_exists(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not os.path.isfile(path):
        return None
    return load_scores(path)


def _load_emergence_metrics_if_exists(prefix: str, strategy: str, attack: str) -> Optional[Dict[str, Dict[str, object]]]:
    fp = _emergence_metrics_path(prefix, strategy, attack)
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


def collect_over_epochs(
    prefix: str,
    strategies: List[str],
    attack: str,
    epochs: List[int],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """返回 data[strategy][epoch] = metrics dict（含 auc, tpr_at_1fpr）。"""
    data: Dict[str, Dict[int, Dict[str, float]]] = {s: {} for s in strategies}

    # 先尝试从 scores 逐个 epoch 计算
    any_scores = False
    for strategy in strategies:
        for e in epochs:
            fp = _score_path(prefix, strategy, e, attack)
            loaded = _load_scores_if_exists(fp)
            if loaded is None:
                continue
            any_scores = True
            scores, labels = loaded
            data[strategy][e] = compute_metrics(scores, labels)

    if any_scores:
        # 清掉完全空的策略
        for s in list(data.keys()):
            if not data[s]:
                data.pop(s, None)
        return data

    # 若没有任何 scores，则尝试 emergence metrics 文件（evaluate.py 生成的 bundle）
    any_metrics = False
    for strategy in list(data.keys()):
        bundle = _load_emergence_metrics_if_exists(prefix, strategy, attack)
        if bundle is None:
            data.pop(strategy, None)
            continue
        # bundle 的 key 一般为 "E1" / "E2"...
        for e in epochs:
            k = f"E{e}"
            if k in bundle:
                any_metrics = True
                data[strategy][e] = bundle[k]

        if not data[strategy]:
            data.pop(strategy, None)

    if any_metrics:
        return data

    return {}


def plot_lines(
    data: Dict[str, Dict[int, Dict[str, float]]],
    prefix: str,
    attack: str,
    metric: str,
    epochs: List[int],
    dpi: int,
    out_dir: str,
    output_prefix: str,
) -> None:
    if not data:
        print("⚠ 没有可绘制的数据：请检查 results/scores 或 results/metrics 是否存在对应文件")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    all_y: List[float] = []

    plotted_epochs: List[int] = []

    for strategy, edict in data.items():
        xs = [e for e in epochs if e in edict]
        if not xs:
            continue
        ys = []
        for e in xs:
            m = edict[e]
            ys.append(float(m["auc"]) if metric == "auc" else float(m.get("tpr_at_1fpr", 0.0)))

        all_y.extend(ys)
        plotted_epochs.extend(xs)

        (line,) = ax.plot(xs, ys, marker="o", linewidth=2.4, markersize=7, label=strategy)

        # 标出该折线的最大值点
        max_i = int(np.argmax(np.asarray(ys, dtype=float)))
        x_max = int(xs[max_i])
        y_max = float(ys[max_i])
        ax.scatter(
            [x_max],
            [y_max],
            s=90,
            marker="D",
            color=line.get_color(),
            edgecolor="white",
            linewidth=1.2,
            zorder=6,
        )
        ax.annotate(
            f"{y_max:.3f}",
            xy=(x_max, y_max),
            xytext=(8, 8),
            textcoords="offset points",
            color=line.get_color(),
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": line.get_color(), "alpha": 0.9},
            zorder=7,
        )

    if all_y:
        y_min = float(min(all_y))
        y_max = float(max(all_y))
        if y_min == y_max:
            pad = 0.02 if metric == "auc" else 0.05
        else:
            pad = 0.08 * (y_max - y_min)
        y_low = y_min - pad
        y_high = y_max + pad
        if metric == "tpr":
            y_low = max(0.0, y_low)
            y_high = min(1.0, y_high)

    if metric == "auc":
        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.2)
        if all_y:
            ax.set_ylim(max(0.45, y_low), min(1.0, y_high))
        else:
            ax.set_ylim(0.45, 1.0)
        ax.set_ylabel("AUC-ROC", fontsize=16)
    else:
        if all_y:
            ax.set_ylim(y_low, y_high)
        else:
            ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("TPR @ 1% FPR", fontsize=16)

    ax.set_xlabel("Epoch", fontsize=16)
    # epoch 只能是整数：禁止出现 1.5 之类的小数刻度
    if plotted_epochs:
        xticks = sorted(set(plotted_epochs))
    else:
        xticks = sorted(set(epochs))
    if xticks:
        ax.set_xticks(xticks)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)

    # 不加大标题，交给论文 caption
    ax.legend(loc="best", fontsize=13, framealpha=0.9)

    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{output_prefix}.png")
    pdf_path = os.path.join(out_dir, f"{output_prefix}.pdf")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"✓ 6.3 折线图已保存: {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 6.3: 风险随 epoch 演化（RQ2）")
    parser.add_argument("--model-prefix", type=str, required=True, help="如 pythia-70m")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["full", "lora", "head"],
        help="对比策略（默认 full lora head）",
    )
    parser.add_argument("--attack", type=str, default="loss", help="攻击方法（默认 loss）")
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="epoch 列表（默认 1 2 3 4 5）",
    )
    parser.add_argument(
        "--metric",
        choices=["auc", "tpr"],
        default="auc",
        help="纵轴指标：auc 或 tpr（TPR@1%FPR）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="导出 dpi（默认 600）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录（默认 results/plots/section6_3/<prefix>/）",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="输出文件名前缀（不含扩展名），可选",
    )

    args = parser.parse_args()

    prefix = args.model_prefix
    strategies = [s.strip() for s in args.strategies if s.strip()]
    attack = args.attack.strip()
    epochs = [int(e) for e in args.epochs]

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(PLOTS_DIR, "section6_3", prefix)

    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = f"{prefix}_6_3_{attack}_{args.metric}_vs_epoch"

    print("\n" + "=" * 80)
    print("Section 6.3 绘图 (RQ2: 随 epoch 演化)")
    print("=" * 80)
    print(f"模型前缀: {prefix}")
    print(f"攻击方法: {attack}")
    print(f"策略: {strategies}")
    print(f"epochs: {epochs}")
    print(f"metric: {args.metric}")
    print(f"dpi: {args.dpi}")
    print(f"输出目录: {out_dir}")
    print("=" * 80)

    data = collect_over_epochs(prefix, strategies, attack, epochs)

    plot_lines(
        data=data,
        prefix=prefix,
        attack=attack,
        metric=args.metric,
        epochs=epochs,
        dpi=args.dpi,
        out_dir=out_dir,
        output_prefix=output_prefix,
    )


if __name__ == "__main__":
    main()
