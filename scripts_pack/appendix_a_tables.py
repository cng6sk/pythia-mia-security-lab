import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import METRICS_DIR, RESULTS_DIR, SCORES_DIR  # noqa: E402
from scripts.evaluate import compute_metrics, load_scores  # noqa: E402


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


def _extract_auc_tpr(m: Dict[str, object]) -> Tuple[Optional[float], Optional[float]]:
    try:
        auc_val = float(m["auc"])
    except Exception:
        auc_val = None
    try:
        tpr_val = float(m.get("tpr_at_1fpr", None))
    except Exception:
        tpr_val = None
    return auc_val, tpr_val


def load_metrics_for_one(prefix: str, strategy: str, attack: str, epoch: int) -> Optional[Dict[str, object]]:
    score_fp = _score_path(prefix, strategy, epoch, attack)
    loaded = _load_scores_if_exists(score_fp)
    if loaded is not None:
        scores, labels = loaded
        return compute_metrics(scores, labels)

    bundle = _load_metrics_bundle(prefix, attack, epoch)
    if bundle and strategy in bundle and isinstance(bundle[strategy], dict):
        return bundle[strategy]

    return None


def collect_appendix_records(
    model_prefixes: List[str],
    attacks: List[str],
    strategies: List[str],
    epochs: List[int],
    include_pretrained: bool,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    strat_list = strategies.copy()
    if include_pretrained and "pretrained" not in strat_list:
        strat_list = ["pretrained"] + strat_list
    strat_list = order_items(strat_list, PREFERRED_STRATEGY_ORDER)

    last_epoch = max(epochs) if epochs else 1

    for prefix in model_prefixes:
        for attack in attacks:
            for strat in strat_list:
                if strat == "pretrained":
                    m = load_metrics_for_one(prefix, strat, attack, epoch=last_epoch)
                    auc_val, tpr_val = _extract_auc_tpr(m) if m is not None else (None, None)
                    records.append(
                        {
                            "model": prefix,
                            "attack": attack,
                            "epoch": "pretrained",
                            "strategy": strat,
                            "auc": auc_val,
                            "tpr": tpr_val,
                        }
                    )
                    continue

                for e in epochs:
                    m = load_metrics_for_one(prefix, strat, attack, epoch=int(e))
                    auc_val, tpr_val = _extract_auc_tpr(m) if m is not None else (None, None)
                    records.append(
                        {
                            "model": prefix,
                            "attack": attack,
                            "epoch": int(e),
                            "strategy": strat,
                            "auc": auc_val,
                            "tpr": tpr_val,
                        }
                    )

    return pd.DataFrame.from_records(records)


def make_wide_table(df: pd.DataFrame, model: str, attack: str, strategies: List[str]) -> pd.DataFrame:
    sub = df[(df["model"] == model) & (df["attack"] == attack)].copy()
    if sub.empty:
        return pd.DataFrame()

    epochs_present = [x for x in sub["epoch"].unique().tolist() if x != "pretrained"]
    epochs_present_int = sorted([int(x) for x in epochs_present if isinstance(x, (int, np.integer)) or str(x).isdigit()])
    epoch_order: List[object] = (["pretrained"] if (sub["epoch"] == "pretrained").any() else []) + epochs_present_int
    sub["epoch"] = pd.Categorical(sub["epoch"], categories=epoch_order, ordered=True)

    wide_auc = sub.pivot_table(index="epoch", columns="strategy", values="auc", aggfunc="first", observed=False)
    wide_tpr = sub.pivot_table(index="epoch", columns="strategy", values="tpr", aggfunc="first", observed=False)

    out = pd.DataFrame(index=wide_auc.index)
    cols: List[str] = []
    for s in strategies:
        out[f"{s}_AUC"] = wide_auc[s] if s in wide_auc.columns else np.nan
        out[f"{s}_TPR@1%"] = wide_tpr[s] if s in wide_tpr.columns else np.nan
        cols.append(f"{s}_AUC")
        cols.append(f"{s}_TPR@1%")

    return out[cols]


def make_merged_model_table(
    df: pd.DataFrame,
    model: str,
    attacks: List[str],
    strategies: List[str],
    epochs: List[int],
    include_pretrained: bool,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for attack in attacks:
        wide = make_wide_table(df, model=model, attack=attack, strategies=strategies)
        if wide.empty:
            continue
        wide2 = wide.reset_index().rename(columns={"index": "epoch"})
        wide2.insert(0, "attack", attack)
        frames.append(wide2)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)

    merged["attack"] = pd.Categorical(merged["attack"], categories=attacks, ordered=True)
    epoch_order: List[object] = (["pretrained"] if include_pretrained else []) + [int(e) for e in epochs]
    merged["epoch"] = pd.Categorical(merged["epoch"], categories=epoch_order, ordered=True)
    merged = merged.sort_values(["attack", "epoch"], kind="stable")

    merged = merged.set_index(["attack", "epoch"])
    return merged


def df_to_longtable_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    return df.to_latex(
        longtable=True,
        escape=False,
        na_rep="--",
        float_format=lambda x: f"{x:.3f}",
        caption=caption,
        label=label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Appendix A tables: full numerical results (AUC / TPR@1%FPR)")
    parser.add_argument(
        "--model-prefixes",
        nargs="+",
        default=["pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m"],
        help="模型规模列表（默认 Pythia 14m/31m/70m/160m/410m）",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["loss", "ratio", "mink"],
        help="攻击列表（默认 loss ratio mink）",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["lora", "head", "full"],
        help="策略列表（默认 lora head full）",
    )
    parser.add_argument(
        "--include-pretrained",
        action="store_true",
        help="是否在表中加入 pretrained 作为 baseline 行",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="epoch 列表（默认 1 2 3 4 5）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录（默认 results/tables/appendix_a）",
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
        out_dir = os.path.join(RESULTS_DIR, "tables", "appendix_a")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Appendix A: full numerical tables")
    print("=" * 80)
    print(f"models: {model_prefixes}")
    print(f"attacks: {attacks}")
    print(f"strategies: {strategies}")
    print(f"epochs: {args.epochs}")
    print(f"out_dir: {out_dir}")
    print("=" * 80)

    df_long = collect_appendix_records(
        model_prefixes=model_prefixes,
        attacks=attacks,
        strategies=strategies,
        epochs=[int(e) for e in args.epochs],
        include_pretrained=bool(args.include_pretrained),
    )

    long_csv_path = os.path.join(out_dir, "appendix_a_full_long.csv")
    df_long.to_csv(long_csv_path, index=False)
    print(f"✓ long CSV saved: {long_csv_path}")

    latex_blocks: List[str] = []
    for model in model_prefixes:
        # 仍保留每个 attack 的宽表 CSV（便于单独查看）
        for attack in attacks:
            wide_single = make_wide_table(df_long, model=model, attack=attack, strategies=strategies)
            if wide_single.empty:
                continue
            wide_csv_path = os.path.join(out_dir, f"appendix_a_{model}_{attack}_wide.csv")
            wide_single.to_csv(wide_csv_path)

        # 合并为每个模型一张“大表”：行=attack×epoch，列=各策略(AUC/TPR)
        merged = make_merged_model_table(
            df_long,
            model=model,
            attacks=attacks,
            strategies=strategies,
            epochs=[int(e) for e in args.epochs],
            include_pretrained=bool(args.include_pretrained),
        )
        if merged.empty:
            continue

        merged_csv_path = os.path.join(out_dir, f"appendix_a_{model}_merged.csv")
        merged.to_csv(merged_csv_path)

        caption = f"Appendix A: {model} | attacks={','.join(attacks)} | AUC-ROC and TPR@1%FPR over epochs"
        label = f"tab:appendixA:{model}:merged"
        latex_blocks.append(df_to_longtable_latex(merged, caption=caption, label=label))

    latex_path = os.path.join(out_dir, "appendix_a_tables.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(latex_blocks))

    print(f"✓ LaTeX saved: {latex_path}")


if __name__ == "__main__":
    main()