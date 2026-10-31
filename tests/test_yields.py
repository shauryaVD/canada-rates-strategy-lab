import numpy as np
import pandas as pd
import pytest

from src.analytics.yields import movement_table, to_bps


def test_percentage_points_to_bps():
    assert to_bps(3.65 - 3.20) == pytest.approx(45)
    assert to_bps(-0.025) == pytest.approx(-2.5)


def test_observation_changes_hand_worked():
    frame = pd.DataFrame({"2Y": np.arange(22) / 100 + 3.0})
    row = movement_table(frame).loc["2Y"]
    assert row["current"] == pytest.approx(3.21)
    assert row["prior"] == pytest.approx(3.20)
    assert row["change_1_bps"] == pytest.approx(1)
    assert row["change_5_bps"] == pytest.approx(5)
    assert row["change_20_bps"] == pytest.approx(20)


def test_missing_rows_are_not_skipped_and_short_history_is_missing():
    row = movement_table(pd.DataFrame({"2Y": [3.0, np.nan, 3.2]})).loc["2Y"]
    assert pd.isna(row["prior"])
    assert pd.isna(row["change_1_bps"])
    assert pd.isna(row["change_5_bps"])
    assert movement_table(pd.DataFrame(columns=["2Y"])).loc["2Y"].isna().all()
