# Research snapshot - 10 September 2026

This folder contains the verified result summaries used in the attached paper draft.

The data pipeline is causal at portfolio formation: the characteristic universe is selected on the 1963-2004 training sample, next-month return availability does not determine the month-t stock universe, state variables are lagged correctly, and validation/test splits are defined in payoff time.

Completed evidence in this snapshot:
- 60/60 pilot fits for Kmax = 1,2,4,8,16,32 with 10 random seeds each.
- 10/10 full-sample Kmax=32 fits, train 1963-2004, validation 2005-2014, test 2015-2024.
- 10-seed ensemble and comparison with the linear managed-portfolio baseline.
- representative-seed factor interpretation, spanning, truncation, policy spectrum, stability, turnover and cost diagnostics.
- automated unit tests for timing, no-look-ahead, invariance and batched/ragged model evaluation.

The full 1963-2024 K-grid, architecture ablations and annual expanding-window walk-forward run are still executing and are deliberately not presented as completed results in this snapshot.
