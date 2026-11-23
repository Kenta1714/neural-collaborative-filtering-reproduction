"""保存済み NeuMF-pre の人気度の偏りを分析し、比較図を docs/ に保存する。

全作品から学習済み作品を除いて推薦し、Smooth xQuAD の係数を val で選ぶ。
モデルの再学習は行わず、test での精度と低人気作品の推薦機会を比較する。
"""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
from scipy.special import expit
import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
# ファイルを直接実行した場合も、プロジェクト内のモジュールを読み込めるようにする。
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_DIR))

from ncf_reproduction.data import load_data
from ncf_reproduction.model import NeuMF
from popularity_reranking.reranking import (
    interaction_groups,
    profile_distribution,
    rerank_smooth,
    select_operating_points,
    top_indices,
)

STRENGTH_GRID = [
    0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3,
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0,
]
TOP_K = 10
CANDIDATE_COUNT = 100


@torch.inference_mode()
def predict_catalog(model, user_num, catalog, seen, batch_size=8192):
    """全作品を推論し、ユーザーごとの学習済み作品だけを候補から除く。"""
    result = np.empty((user_num, len(catalog)), dtype=np.float64)
    total = result.size
    for start in range(0, total, batch_size):
        positions = np.arange(start, min(start + batch_size, total))
        users = positions // len(catalog)
        items = catalog[positions % len(catalog)]
        logits = model(torch.from_numpy(users), torch.from_numpy(items)).cpu().numpy()
        result.ravel()[positions] = expit(logits.astype(float))
    # val・test の作品も通常の候補とし、正解を上位候補に強制挿入しない。
    users, items = seen.nonzero()
    column = np.full(seen.shape[1], -1, dtype=int)
    column[catalog] = np.arange(len(catalog))
    valid = column[items] >= 0
    result[users[valid], column[items[valid]]] = -np.inf
    return result


def rank(initial, initial_relevance, groups, preferences, strength):
    """関連度上位100件から Smooth xQuAD で10件を選ぶ。"""
    positions = rerank_smooth(
        initial_relevance, groups[initial], preferences, strength, k=TOP_K
    )
    return np.take_along_axis(initial, positions, axis=1)


def evaluate(recommended, positives, groups):
    """同じ推薦リストから精度と低人気作品の推薦枠・カバレッジを計算する。"""
    matches = recommended == positives[:, None]
    hit = matches.any(axis=1)
    ndcg = np.where(hit, 1 / np.log2(matches.argmax(axis=1) + 2), 0)
    unique_tail = int((groups[np.unique(recommended)] == 1).sum())
    return {
        "hr": hit.mean(),
        "ndcg": ndcg.mean(),
        "aplt": (groups[recommended] == 1).mean(),
        "unique_tail_items": unique_tail,
        "tail_coverage": unique_tail / int((groups == 1).sum()),
    }


def save_figures(preferences, base, smooth, docs_dir):
    """履歴と推薦の偏り、および再ランキング前後の比較図を保存する。"""
    docs_dir.mkdir(parents=True, exist_ok=True)
    history_tail = preferences[:, 1].mean()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["Train history", "NeuMF"], [history_tail, base["aplt"]], color=["C0", "C1"])
    ax.set(title="Popularity bias", ylabel="Tail share", ylim=(0, 0.2))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    fig.tight_layout()
    fig.savefig(docs_dir / "popularity_bias.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    plots = [
        ("aplt", "Tail share"),
        ("unique_tail_items", "Unique tail items"),
        ("hr", "HR@10"),
        ("ndcg", "NDCG@10"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), plots):
        values = [base[metric], smooth[metric]]
        ax.bar(["Baseline", "Smooth xQuAD"], values, color=["C0", "C1"])
        ax.set(title=title, ylim=(0, max(values) * 1.15))
        if metric in ["aplt", "hr"]:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
        elif metric == "unique_tail_items":
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(docs_dir / "reranking_results.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_analysis(run_dir, data_dir, batch_size=8192):
    """保存済みモデルを評価し、係数選択と図の保存まで実行する。"""
    cfg = yaml.safe_load((run_dir / "NeuMF-pre/.hydra/config.yaml").read_text())
    seed = int(cfg["seed"])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)
    train_rows, val_rows, test_rows, user_num, item_num, train_mat = load_data(
        data_dir, seed
    )
    train = np.asarray(train_rows, dtype=np.int64)
    seen = train_mat.tocsr()
    popularity = np.bincount(train[:, 1], minlength=item_num)
    groups, _ = interaction_groups(popularity, [0.8])
    preferences = profile_distribution(train, groups, user_num, 2)
    positives = {
        split: np.asarray(rows, dtype=np.int64).reshape(user_num, 100, 2)[:, 0, 1]
        for split, rows in [("val", val_rows), ("test", test_rows)]
    }
    for items in positives.values():
        assert not np.asarray(seen[np.arange(user_num), items]).any()
    settings = cfg["model"]
    model = NeuMF(
        user_num, item_num,
        settings["factor_num"], settings["num_layers"], settings["dropout"],
    )
    checkpoint = torch.load(
        run_dir / "NeuMF-pre/checkpoint.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"{user_num:,}ユーザー × {item_num:,}作品を推論中…", flush=True)
    relevance = predict_catalog(model, user_num, np.arange(item_num), seen, batch_size)
    initial = top_indices(relevance, CANDIDATE_COUNT)
    initial_relevance = np.take_along_axis(relevance, initial, axis=1)
    base = evaluate(initial[:, :TOP_K], positives["test"], groups)
    print("推論完了", flush=True)

    # モデルは固定し、後処理の係数だけを val で選ぶ。
    sweep = pd.DataFrame([
        {
            "method": "Smooth", "strength": strength,
            **evaluate(
                rank(initial, initial_relevance, groups, preferences, strength),
                positives["val"], groups,
            ),
        }
        for strength in STRENGTH_GRID
    ])
    selected = select_operating_points(sweep)
    strength = float(selected.iloc[0].strength)
    print(f"val で選んだ補正係数：λ = {strength:g}")

    recommended = rank(initial, initial_relevance, groups, preferences, strength)
    assert not (np.diff(np.sort(recommended, axis=1), axis=1) == 0).any()
    assert not np.asarray(
        seen[np.repeat(np.arange(user_num), TOP_K), recommended.ravel()]
    ).any()
    smooth = evaluate(recommended, positives["test"], groups)
    docs_dir = PROJECT_DIR / "docs"
    save_figures(preferences, base, smooth, docs_dir)
    print(f"低人気作品の推薦枠: {base['aplt']:.2%} → {smooth['aplt']:.2%}")
    print(
        f"推薦された低人気作品: {base['unique_tail_items']:,}作品 → "
        f"{smooth['unique_tail_items']:,}作品"
    )
    print(f"全体の HR@10: {base['hr']:.2%} → {smooth['hr']:.2%}")
    print(f"全体の NDCG@10: {base['ndcg']:.5f} → {smooth['ndcg']:.5f}")
    print(f"図の保存先: {docs_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="NeuMF-pre/ を含む学習結果のディレクトリ",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_DIR,
        help="data/ml-1m.* を含むディレクトリ（既定: プロジェクト直下）",
    )
    parser.add_argument("--batch-size", type=int, default=8192, help="推論バッチサイズ")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size は1以上にしてください")
    run_analysis(args.run_dir, args.data_dir, args.batch_size)


if __name__ == "__main__":
    main()
