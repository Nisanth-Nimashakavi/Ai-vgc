"""Regression coverage for native snapshot-league bookkeeping."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("torch")

from ai_vgc.nn.rl import _native_record_snapshot_results, _native_snapshot_choice


def test_native_pfsp_prioritizes_snapshots_the_learner_loses_to():
    pool = ["/models/easy.pt", "/models/hard.pt", "/models/fresh.pt"]
    record = {
        # The learner wins 90% here, so it should have the smallest weight.
        pool[0]: {"games": 10.0, "reward": 9.0},
        # The learner wins 20% here, so it should be sampled most often.
        pool[1]: {"games": 10.0, "reward": 2.0},
    }
    with patch("ai_vgc.nn.rl.random.choices", return_value=[pool[1]]) as choices:
        assert _native_snapshot_choice(pool, record) == pool[1]

    population, weights = choices.call_args.args
    assert population == pool
    assert weights == pytest.approx([0.01, 0.64, 0.25])


def test_native_pfsp_records_only_labelled_live_snapshots():
    pool = ["/models/a.pt", "/models/b.pt"]
    record = {pool[0]: {"games": 2.0, "reward": 1.0}}
    results = {
        "self": [1.0, 0.0],
        f"pool:{pool[0]}": [1.0, 0.5, 0.0],
        f"pool:{pool[1]}": [0.0, 0.0],
        "pool:/models/no-longer-live.pt": [1.0],
    }

    _native_record_snapshot_results(results, pool, record)

    assert record == {
        pool[0]: {"games": 5.0, "reward": 2.5},
        pool[1]: {"games": 2.0, "reward": 0.0},
    }


def test_native_pfsp_rejects_an_empty_snapshot_pool():
    with pytest.raises(ValueError, match="snapshot pool is empty"):
        _native_snapshot_choice([], {})
