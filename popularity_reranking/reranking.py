"""全件推薦に Smooth xQuAD を適用するための関数。

元の NCF 再現実装とは分けて、人気度補正と val での係数選択を扱う。
推論・集計・図の保存は notebook で行う。
式の出典：https://cdn.aaai.org/ocs/18199/18199-78818-1-PB.pdf
"""

import numpy as np
import pandas as pd


def interaction_groups(popularity, boundaries):
    """学習時の出現回数を累積し、同じ人気度をまとめてカテゴリに分ける。

    境界を超えた作品と同じ人気度の作品は、まとめて人気側に入れる。
    このため、各カテゴリの出現回数の比率は指定値と厳密には一致しない。
    出現回数0の作品は最後のカテゴリに入れる。val・test の履歴は使わない。
    """
    popularity = np.asarray(popularity)
    boundaries = np.asarray(boundaries, dtype=float)
    if (popularity.ndim != 1 or not np.isfinite(popularity).all()
            or (popularity < 0).any() or popularity.sum() <= 0):
        raise ValueError("人気度は非負の1次元配列とし、合計を正にしてください")
    if (boundaries.ndim != 1 or (boundaries <= 0).any()
            or (boundaries >= 1).any() or (np.diff(boundaries) <= 0).any()):
        raise ValueError("境界は0より大きく1未満の値を昇順に指定してください")
    ordered = np.sort(popularity)[::-1]
    thresholds = ordered[np.searchsorted(np.cumsum(ordered), boundaries * ordered.sum())]
    groups = (popularity[:, None] < thresholds).sum(axis=1)
    return groups, thresholds


def profile_distribution(train, item_groups, user_num, group_num):
    """ユーザーごとの学習履歴から、人気度カテゴリの件数比率を求める。"""
    train = np.asarray(train, dtype=np.int64)
    counts = np.zeros((user_num, group_num), dtype=float)
    # 同じユーザー・カテゴリに対応する履歴を加算する。
    np.add.at(counts, (train[:, 0], item_groups[train[:, 1]]), 1)
    if (counts.sum(axis=1) == 0).any():
        raise ValueError("各ユーザーに1件以上の学習履歴が必要です")
    return counts / counts.sum(axis=1, keepdims=True)




def top_indices(scores, k):
    """スコアの降順で候補位置を返す。同点は元の列順、-inf は対象外とする。"""
    scores = np.asarray(scores)
    if (scores.ndim != 2 or not 0 < k <= scores.shape[1]
            or np.isnan(scores).any() or np.isposinf(scores).any()
            or (np.isfinite(scores).sum(axis=1) < k).any()):
        raise ValueError("各行に推薦数 k 以上の有限スコアを持つ候補が必要です")
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def rerank_smooth(relevance, groups, preferences, strength, k=10):
    """関連度とカテゴリの補正を足し合わせ、推薦する候補位置を順に選ぶ。

    元論文の式(1)・(4)に基づく Smooth xQuAD。
    推薦済みのカテゴリ比率から (1 - n_d / |S|) ** |S| を計算する。
    まだ何も推薦していないとき、この値は1とする。
    """
    relevance = np.asarray(relevance, dtype=float)
    groups = np.asarray(groups)
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("補正の強さは有限の非負値にしてください")
    _validate_groups(relevance, groups, preferences, 2)
    if not 0 < k <= relevance.shape[1]:
        raise ValueError("推薦数 k は1以上、候補数以下にしてください")
    order = top_indices(relevance, relevance.shape[1])
    # 補正なしでは元の関連度順をそのまま使う。
    if strength == 0:
        return order[:, :k]
    scores = np.take_along_axis(relevance, order, axis=1)
    groups = np.take_along_axis(groups, order, axis=1)
    available = np.ones_like(groups, dtype=bool)
    counts = np.zeros((len(relevance), 2), dtype=int)
    output = np.empty((len(relevance), k), dtype=int)
    rows = np.arange(len(relevance))
    for step in range(k):
        # すでに十分に推薦したカテゴリほど、追加のボーナスを小さくする。
        uncovered = np.ones_like(counts, dtype=float) if step == 0 else (1 - counts / step) ** step
        bonus = preferences * uncovered
        objective = scores + strength * np.take_along_axis(bonus, groups, axis=1)
        chosen = np.argmax(np.where(available, objective, -np.inf), axis=1)
        output[:, step] = order[rows, chosen]
        available[rows, chosen] = False
        counts[rows, groups[rows, chosen]] += 1
    return output


def _validate_groups(relevance, groups, preferences, group_num):
    if relevance.ndim != 2 or groups.shape != relevance.shape or not np.isfinite(relevance).all():
        raise ValueError("関連度は有限値の2次元配列とし、カテゴリの配列と形状をそろえてください")
    if not np.issubdtype(groups.dtype, np.integer) or (groups < 0).any() or (groups >= group_num).any():
        raise ValueError("カテゴリIDの型または値が不正です")
    if (preferences.shape != (len(relevance), group_num) or (preferences < 0).any()
            or not np.isfinite(preferences).all() or not np.allclose(preferences.sum(axis=1), 1)):
        raise ValueError("履歴のカテゴリ比率は非負かつ合計1にしてください")


def select_operating_points(sweep, hr_budget=0.005, ndcg_budget=0.005):
    """val の精度低下を許容範囲に収め、低人気作品の推薦枠を最大にする。

    本実験独自の選択ルール。test の指標は係数選択に使わない。
    """
    selected = []
    for method, table in sweep.groupby("method", sort=False):
        baseline = table.loc[table.strength == 0].iloc[0]
        feasible = table[(table.hr >= baseline.hr - hr_budget - 1e-12)
                         & (table.ndcg >= baseline.ndcg - ndcg_budget - 1e-12)]
        # 低人気枠が同じなら NDCG、HR、小さい係数の順で選ぶ。
        best = feasible.sort_values(["aplt", "ndcg", "hr", "strength"],
                                    ascending=[False, False, False, True]).iloc[0]
        selected.append({"method": method, "strength": float(best.strength),
                         "objective": "aplt", "val_objective": best.aplt,
                         "val_hr": best.hr, "val_ndcg": best.ndcg})
    return pd.DataFrame(selected)
