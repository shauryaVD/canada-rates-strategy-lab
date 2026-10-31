import numpy as np
import pandas as pd
import pytest

from src.analytics.curves import classify_curve, term_spreads


def test_term_spreads_hand_worked_positive_negative_missing():
    result = term_spreads(
        pd.DataFrame({"2Y": [3.20, 4.0, np.nan], "5Y": [3.4, 3.5, 3.2], "10Y": [3.65, 3.0, 3.0]})
    )
    assert result.iloc[0].to_list() == pytest.approx([45, 20, 25])
    assert result.iloc[1].to_list() == pytest.approx([-100, -50, -50])
    assert pd.isna(result.iloc[2]["2s10s"])
    assert result.iloc[2]["5s10s"] == pytest.approx(-20)
    assert term_spreads(pd.DataFrame({"2Y": [3]})).isna().all().all()


@pytest.mark.parametrize(
    "short,long,expected",
    [
        (-10, -5, "Bull steepening"),
        (-5, -10, "Bull flattening"),
        (5, 10, "Bear steepening"),
        (10, 5, "Bear flattening"),
        (-5, 5, "Mixed/indeterminate"),
        (5, -5, "Mixed/indeterminate"),
        (0, 5, "Mixed/indeterminate"),
        (5, 5, "Mixed/indeterminate"),
        (np.nan, 5, "Mixed/indeterminate"),
        (5, np.inf, "Mixed/indeterminate"),
    ],
)
def test_classification(short, long, expected):
    assert classify_curve(short, long) == expected
