"""Smooth xQuAD の計算と val での係数選択を独立した計算で検証する。"""

import unittest

import numpy as np
import pandas as pd

from popularity_reranking.reranking import (
    interaction_groups,
    profile_distribution, rerank_smooth, select_operating_points, top_indices,
)


def literal_smooth(relevance, groups, preference, strength, k):
    """論文の式を候補ごとに計算し、一括計算した実装との照合に使う。"""
    selected = []
    for _ in range(k):
        values = []
        for item, score in enumerate(relevance):
            if item in selected:
                values.append(-np.inf)
                continue
            category = groups[item]
            factors = [1 - sum(groups[j] == category for j in selected) / len(selected)
                       for _ in selected]
            values.append(score + strength * preference[category] * np.prod(factors))
        selected.append(int(np.argmax(values)))
    return selected


class RerankingTests(unittest.TestCase):
    def test_train_interaction_groups_keep_popularity_ties(self):
        groups, thresholds = interaction_groups([80, 10, 5, 5, 0], [.8])
        np.testing.assert_array_equal(groups, [0, 1, 1, 1, 1])
        np.testing.assert_array_equal(thresholds, [80])
        groups, _ = interaction_groups([4, 2, 2, 1, 1], [.5, .8])
        np.testing.assert_array_equal(groups, [0, 0, 0, 2, 2])

    def test_profile_uses_each_users_training_interactions(self):
        train = np.array([[0, 0], [0, 1], [0, 2], [1, 1]])
        result = profile_distribution(train, np.array([0, 1, 1]), 2, 2)
        np.testing.assert_allclose(result, [[1 / 3, 2 / 3], [0, 1]])
        with self.assertRaises(ValueError):
            profile_distribution(train, np.array([0, 1, 1]), 3, 2)

    def test_greedy_matches_literal_paper_objective(self):
        rng = np.random.default_rng(7)
        for _ in range(60):
            relevance = rng.random(8)
            groups = rng.integers(2, size=8)
            preference = rng.dirichlet(np.ones(2))
            strength = rng.uniform(.1, 1.)
            expected = literal_smooth(relevance, groups, preference, strength, 5)
            actual = rerank_smooth(relevance[None], groups[None], preference[None], strength, 5)[0]
            np.testing.assert_array_equal(actual, expected)

    def test_zero_strength_and_ties_preserve_base_order(self):
        r = np.array([[.8, .8, .6]])
        g = np.array([[0, 1, 1]])
        p = np.array([[.5, .5]])
        np.testing.assert_array_equal(rerank_smooth(r, g, p, 0, 2), [[0, 1]])
        np.testing.assert_array_equal(rerank_smooth(r, g, p, .3, 3), [[0, 1, 2]])

    def test_candidate_permutation_without_ties(self):
        r = np.array([[.8, .72, .57, .41]])
        g = np.array([[0, 1, 1, 0]])
        p = np.array([[.2, .8]])
        permutation = np.array([2, 0, 3, 1])
        original = rerank_smooth(r, g, p, .7, 3)
        shuffled = permutation[rerank_smooth(r[:, permutation], g[:, permutation], p, .7, 3)]
        np.testing.assert_array_equal(original, shuffled)

    def test_unavailable_candidates_and_insufficient_pool(self):
        np.testing.assert_array_equal(top_indices([[.2, -np.inf, .8]], 2), [[2, 0]])
        with self.assertRaises(ValueError):
            top_indices([[.2, -np.inf]], 2)
        with self.assertRaises(ValueError):
            rerank_smooth(np.array([[.2, .8]]), np.array([[0, 1]]), np.array([[.5, .5]]), -.1)

    def test_validation_selection_obeys_accuracy_budgets(self):
        sweep = pd.DataFrame([
            dict(method="Smooth", strength=0, hr=.7, ndcg=.5, aplt=.1),
            dict(method="Smooth", strength=.3, hr=.699, ndcg=.499, aplt=.2),
            dict(method="Smooth", strength=.9, hr=.6, ndcg=.4, aplt=.9),
        ])
        self.assertEqual(select_operating_points(sweep).iloc[0].strength, .3)
        self.assertEqual(select_operating_points(sweep, hr_budget=0, ndcg_budget=0).iloc[0].strength, 0)


if __name__ == "__main__":
    unittest.main()
