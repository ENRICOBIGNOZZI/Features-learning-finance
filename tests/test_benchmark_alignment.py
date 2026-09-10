import pandas as pd

from neural_factors.benchmarks import payoff_month_from_formation


def test_formation_month_is_shifted_to_realized_payoff_month():
    dates = pd.Series(pd.to_datetime(["2019-12-31", "2020-01-31", "2020-02-29"]))
    payoff = payoff_month_from_formation(dates)
    expected = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    assert payoff.tolist() == expected.tolist()
