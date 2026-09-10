import numpy as np
import pandas as pd

from neural_factors.data import JKPUSData


def test_missing_future_return_does_not_change_formation_universe(tmp_path):
    frame = pd.DataFrame({
        "id": [1, 2],
        "eom": pd.to_datetime(["2020-01-31", "2020-01-31"]),
        "me": [100.0, 50.0],
        "ret_exc_lead1m": [0.10, np.nan],
        "signal": [0.2, -0.2],
    })
    frame.to_parquet(tmp_path / "JKP_USA_clean_2020.parquet", index=False)
    data = JKPUSData(tmp_path, add_log_me=False)
    scaler = (np.zeros(len(data.state_columns), dtype=np.float32),
              np.ones(len(data.state_columns), dtype=np.float32))
    panel = next(data.iter_year(2020, "2020-01-01", "2020-01-31", scaler))
    assert len(panel.ids) == 2
    assert panel.return_observed.tolist() == [True, False]
    assert np.allclose(panel.r, [0.10, 0.0])


def test_current_stock_count_does_not_use_future_return_availability(tmp_path):
    frame = pd.DataFrame({
        "id": [1, 2, 3],
        "eom": pd.to_datetime(["2020-01-31"] * 3),
        "me": [100.0, 50.0, 25.0],
        "ret_exc_lead1m": [0.10, np.nan, np.nan],
        "signal": [0.2, -0.2, 0.0],
    })
    frame.to_parquet(tmp_path / "JKP_USA_clean_2020.parquet", index=False)
    data = JKPUSData(tmp_path, add_log_me=False)
    assert int(data.monthly_state.loc[pd.Timestamp("2020-01-31"), "n_stocks"]) == 3
    assert np.isclose(
        data.monthly_state.loc[pd.Timestamp("2020-01-31"), "log_n"], np.log(3.0)
    )
