import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.analytics.relative_value import rolling_statistics


def series(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="B"))


def test_prior_statistics_and_midrank_percentiles_hand_worked():
    stats = rolling_statistics(series([1, 2, 3, 2]), window=3)
    row = stats.iloc[3]
    assert row["mean_bps"] == 2
    assert row["std_bps"] == 1
    assert row["zscore"] == 0
    assert row["deviation_bps"] == 0
    assert row["percentile"] == 50
    extreme = rolling_statistics(series([1, 2, 3, 4]), window=3).iloc[3]
    assert extreme["zscore"] == 2
    assert extreme["mean_bps"] == 2  # Current value must not enter the mean.
    assert extreme["percentile"] == 100


def test_zero_variance_has_no_signal_but_has_percentile():
    stats = rolling_statistics(series([2, 2, 2, 2, 3]), window=3)
    assert stats["zscore"].isna().all()
    assert stats.iloc[3]["percentile"] == 50
    assert stats.iloc[4]["percentile"] == 100


def test_missing_current_and_prior_preserve_periods():
    stats = rolling_statistics(series([1, 2, np.nan, 4, 5, 6, 7]), window=3)
    assert stats.iloc[:6]["zscore"].isna().all()
    assert stats.iloc[6]["mean_bps"] == 5
    assert rolling_statistics(series([1, 2]), window=3)["zscore"].isna().all()


def test_future_perturbation_and_prefix_invariance():
    original = series([1, 2, 3, 5, 2, 1, 3, 4])
    changed = original.copy()
    changed.iloc[5:] = [900, -800, 400]
    assert_frame_equal(
        rolling_statistics(original, 3).iloc[:5], rolling_statistics(changed, 3).iloc[:5]
    )
    assert_frame_equal(
        rolling_statistics(original.iloc[:5], 3), rolling_statistics(original, 3).iloc[:5]
    )


@pytest.mark.parametrize("window", [0, 1, 1.5, True])
def test_bad_window_rejected(window):
    with pytest.raises(ValueError):
        rolling_statistics(series([1, 2, 3]), window)


def test_ambiguous_dates_and_infinity_rejected():
    for data in [
        pd.Series([1, 2]),
        series([1, 2]).iloc[::-1],
        pd.Series([1, 2], index=pd.to_datetime(["2020-01-01"] * 2)),
        series([1, np.inf]),
    ]:
        with pytest.raises(ValueError):
            rolling_statistics(data, 2)
