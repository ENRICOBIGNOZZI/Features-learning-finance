import pandas as pd

from neural_factors.train import SplitConfig


def payoff_month(date_string: str) -> pd.Timestamp:
    return pd.Timestamp(date_string) + pd.offsets.MonthEnd(1)


def test_default_split_has_disjoint_adjacent_payoff_periods():
    split = SplitConfig()
    assert payoff_month(split.train_end) == pd.Timestamp("2004-12-31")
    assert payoff_month(split.val_start) == pd.Timestamp("2005-01-31")
    assert payoff_month(split.val_end) == pd.Timestamp("2014-12-31")
    assert payoff_month(split.test_start) == pd.Timestamp("2015-01-31")
    assert payoff_month(split.test_end) == pd.Timestamp("2024-12-31")
