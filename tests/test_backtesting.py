import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.analytics.backtesting import run_backtest


def series(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="B"))


def test_hand_worked_high_and_low_events_and_baseline():
    result = run_backtest(
        series([0, 1, 2, 5, 2, -2, 1]), window=3, threshold=1.5, horizon=1, sample="all"
    )
    assert result.outcomes["direction"].tolist() == ["high", "low"]
    assert result.outcomes["forward_change_bps"].tolist() == [-3, 3]
    assert result.outcomes["convergence_bps"].tolist() == [3, 3]
    assert result.outcomes["hit"].all()
    assert result.summary.loc["all", "valid_signals"] == 2
    assert result.summary.loc["all", "hit_rate_pct"] == 100
    assert result.summary.loc["all", "mean_forward_change_bps"] == 0
    assert result.summary.loc["all", "mean_convergence_bps"] == 3
    assert result.baseline.loc["high", "evaluated_signals"] == 6
    assert result.baseline.loc["high", "hit_rate_pct"] == pytest.approx(100 / 3)
    assert result.baseline.loc["low", "hit_rate_pct"] == pytest.approx(200 / 3)
    assert result.baseline.loc["low", "mean_forward_change_bps"] == pytest.approx(1 / 6)


def test_no_available_future_excluded_but_valid_signals_counted():
    result = run_backtest(series(range(8)), window=2, threshold=1, horizon=3, sample="all")
    assert result.summary.loc["all", "valid_signals"] == 6
    assert result.summary.loc["all", "evaluated_signals"] == 1
    assert result.signals.iloc[-1]["status"] == "excluded_no_future"
    assert pd.isna(result.signals.iloc[-1]["end_date"])


def test_greedy_nonoverlap_allows_shared_endpoint():
    data = series(range(13))
    result = run_backtest(data, window=2, threshold=1, horizon=3, sample="all")
    assert result.outcomes["date"].tolist() == data.index[[2, 5, 8]].tolist()
    assert result.outcomes["end_date"].tolist() == data.index[[5, 8, 11]].tolist()
    overlap = run_backtest(
        data, window=2, threshold=1, horizon=3, sample="all", non_overlapping=False
    )
    assert len(overlap.outcomes) == 8
    assert len(overlap.signals) == len(result.signals)
    # Baseline population does not silently change with event overlap option.
    assert_frame_equal(overlap.baseline, result.baseline)


def test_training_purge_and_testing_cutoff_no_boundary_crossing():
    data = series(range(20))
    train = run_backtest(
        data,
        window=2,
        threshold=1,
        horizon=3,
        train_fraction=0.6,
        sample="train",
        non_overlapping=False,
    )
    test = run_backtest(
        data,
        window=2,
        threshold=1,
        horizon=3,
        train_fraction=0.6,
        sample="test",
        non_overlapping=False,
    )
    cutoff = data.index[12]
    assert train.parameters["split_date"] == cutoff.date().isoformat()
    assert (train.outcomes["end_date"] < cutoff).all()
    assert (test.signals["date"] >= cutoff).all()
    assert (train.signals["status"] == "excluded_cross_split").sum() == 3
    assert len(train.baseline) == 2
    assert train.baseline.loc["high", "evaluated_signals"] == 9


def test_selected_start_warmup_end_cap_and_split_before_signal_filter():
    data = series(range(30))
    result = run_backtest(
        data,
        window=2,
        threshold=1,
        horizon=2,
        start=data.index[10],
        end=data.index[19],
        train_fraction=0.6,
        sample="test",
        non_overlapping=False,
    )
    assert result.parameters["split_date"] == data.index[16].date().isoformat()
    assert result.signals.iloc[0]["date"] == data.index[16]
    assert result.outcomes["end_date"].max() == data.index[19]
    assert result.signals.iloc[-1]["status"] == "excluded_no_future"
    changed = data.copy()
    changed.iloc[22:] = np.nan
    assert_frame_equal(
        result.summary,
        run_backtest(
            changed,
            window=2,
            threshold=1,
            horizon=2,
            start=data.index[10],
            end=data.index[19],
            train_fraction=0.6,
            sample="test",
            non_overlapping=False,
        ).summary,
    )


def test_missing_intermediate_or_endpoint_excludes_event():
    data = series([0, 1, 2, 5, np.nan, 3, 2, 1])
    result = run_backtest(data, window=3, threshold=1.5, horizon=2, sample="all")
    assert result.signals.iloc[0]["status"] == "excluded_missing_path"
    assert result.outcomes.empty
    # Intermediate NaN cannot be skipped to find a different future observation.
    assert result.baseline.loc["high", "evaluated_signals"] == 3


def test_future_perturbation_preserves_past_signal_values():
    data = series([0, 1, 2, 5, 2, -2, 1, 8, 9, 3, 2, 0])
    changed = data.copy()
    changed.iloc[8:] = [100, -200, 50, 700]
    cols = ["date", "spread_bps", "zscore", "direction"]
    original = run_backtest(data, window=3, threshold=1, horizon=1, sample="all")
    perturbed = run_backtest(changed, window=3, threshold=1, horizon=1, sample="all")
    prefix = run_backtest(data.iloc[:8], window=3, threshold=1, horizon=1, sample="all")
    before = original.signals[original.signals["date"] < data.index[8]][cols]
    assert_frame_equal(before, perturbed.signals[perturbed.signals["date"] < data.index[8]][cols])
    assert_frame_equal(before, prefix.signals[cols])


@pytest.mark.parametrize("values,window", [([], 3), ([1, 2], 3), ([2] * 30, 3)])
def test_empty_insufficient_constant_no_fabricated_signals(values, window):
    result = run_backtest(series(values), window=window, horizon=2, sample="all")
    assert result.signals.empty
    assert result.outcomes.empty
    assert result.summary.loc["all", "valid_signals"] == 0
    assert pd.isna(result.summary.loc["all", "hit_rate_pct"])


def test_flat_outcome_is_not_a_hit():
    result = run_backtest(series([0, 1, 2, 5, 5]), window=3, horizon=1, sample="all")
    assert result.outcomes.iloc[0]["convergence_bps"] == 0
    assert not result.outcomes.iloc[0]["hit"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"horizon": 1.5},
        {"threshold": 0},
        {"threshold": np.inf},
        {"train_fraction": 1},
        {"sample": "random"},
        {"start": "2020-02-01", "end": "2020-01-01"},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        run_backtest(series(range(20)), window=3, **kwargs)


def test_test_data_cannot_change_train_signals_or_evaluated_outcomes():
    data = series(np.arange(40, dtype=float))
    changed = data.copy()
    changed.iloc[28:] = [np.nan, *np.linspace(-1000, 1000, 11)]
    options = dict(window=3, threshold=1, horizon=3, sample="train", non_overlapping=False)
    train = run_backtest(data, **options)
    perturbed = run_backtest(changed, **options)
    assert_frame_equal(train.signals, perturbed.signals)
    assert_frame_equal(train.outcomes, perturbed.outcomes)
    assert_frame_equal(train.baseline, perturbed.baseline)
    assert train.outcomes["end_date"].max() == data.index[27]
    assert (
        train.signals[train.signals["status"] == "excluded_cross_split"]["date"].min()
        == data.index[25]
    )


def test_split_uses_selected_calendar_rows_even_when_values_missing():
    data = series(np.arange(30, dtype=float))
    changed = data.copy()
    changed.iloc[10:16] = np.nan
    options = dict(
        window=2,
        threshold=1,
        horizon=2,
        start=data.index[10],
        end=data.index[19],
        train_fraction=0.6,
        sample="test",
    )
    original = run_backtest(data, **options)
    missing = run_backtest(changed, **options)
    assert original.parameters["split_date"] == missing.parameters["split_date"]
    assert missing.parameters["split_date"] == data.index[16].date().isoformat()
    assert (missing.signals["date"] >= data.index[16]).all()


@pytest.mark.parametrize(
    "missing_position,expected_status",
    [
        (4, "excluded_missing_path"),  # Inside the t=2..5 path.
        (5, "excluded_missing_path"),  # Exact endpoint.
        (6, "evaluated"),  # First observation after the endpoint must not matter.
    ],
)
def test_missing_at_or_after_endpoint_boundary(missing_position, expected_status):
    data = series(np.arange(20, dtype=float))
    data.iloc[missing_position] = np.nan
    result = run_backtest(data, window=2, threshold=1, horizon=3, sample="all")
    first = result.signals.loc[result.signals["date"] == data.index[2]].iloc[0]
    assert first["end_date"] == data.index[5]
    assert first["status"] == expected_status


def test_baseline_independent_of_statistical_threshold_and_reference_window():
    data = series([0, 1, 2, 5, 2, -2, 1, 3, 8, 2, -1, 7])
    options = dict(horizon=2, sample="all")
    short = run_backtest(data, window=2, threshold=1, **options)
    impossible = run_backtest(data, window=120, threshold=100, **options)
    assert impossible.signals.empty
    assert_frame_equal(short.baseline, impossible.baseline)
    changes = data.shift(-2).sub(data).dropna()
    assert short.baseline.loc["low", "evaluated_signals"] == len(data) - 2
    assert short.baseline.loc["low", "mean_forward_change_bps"] == pytest.approx(changes.mean())
    assert short.baseline.loc["high", "mean_convergence_bps"] == pytest.approx(-changes.mean())


def test_range_outside_data_and_all_missing_sample_have_explicit_empty_outputs():
    data = series([np.nan] * 8)
    for options in ({}, {"start": "2030-01-01", "end": "2030-12-31"}):
        result = run_backtest(data, window=2, horizon=1, sample="all", **options)
        assert result.signals.empty
        assert result.outcomes.empty
        assert (result.baseline["evaluated_signals"] == 0).all()
        assert result.baseline["hit_rate_pct"].isna().all()
