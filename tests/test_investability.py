import numpy as np
import pandas as pd

from neural_factors.investability import realized_turnover


def test_turnover_uses_post_return_nav_weights():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    weights = pd.DataFrame({
        "date": [dates[0], dates[1]],
        "id": [1, 1],
        "scaled_weight": [1.0, 1.0],
        "ret_fwd": [0.10, 0.00],
    })
    turnover = realized_turnover(weights)
    assert np.isclose(turnover.loc[0, "trade_notional"], 1.0)
    assert np.isclose(turnover.loc[1, "trade_notional"], 0.0, atol=1e-12)


def test_turnover_counts_entry_and_exit():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    weights = pd.DataFrame({
        "date": [dates[0], dates[1]],
        "id": [1, 2],
        "scaled_weight": [0.5, 0.5],
        "ret_fwd": [0.0, 0.0],
    })
    turnover = realized_turnover(weights)
    assert np.isclose(turnover.loc[1, "trade_notional"], 1.0)
