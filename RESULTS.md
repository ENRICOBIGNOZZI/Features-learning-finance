# Results Snapshot

This file summarizes the admissible September 10, 2026 empirical snapshot. All numbers use the causal JKP U.S. cache and payoff-time sample definitions in `AUDIT.md`.

## Capacity and random initialization

The full 60-run battery uses ten seeds for each $K_{max}\in\{1,2,4,8,16,32\}$. Mean evaluation Sharpe ratios are 1.43, 1.61, 1.69, 1.72, 1.72, and 1.79, respectively. The corresponding ten-seed ensembles have Sharpe 1.51, 1.74, 1.87, 1.87, 1.89, and 1.90. The linear managed-portfolio benchmark has evaluation Sharpe 1.51.

The paired 12-month circular-block bootstrap estimate for the $K=32$ ensemble minus the linear benchmark is +0.39 Sharpe points, with 95% interval [-0.51, 1.13] and bootstrap probability of a positive difference of about 0.80. We therefore treat the mean performance improvement as economically suggestive, not statistically decisive.

## Economic dimension

The representative $K=32$ run is selected using validation Sharpe only: it is the lowest-index run among the two seeds bracketing the median validation Sharpe. Its evaluation Sharpe is 1.68. The policy entropy rank is 1.11; the first policy eigen-direction accounts for 98.0% of spectrum mass and the first two account for 99.7%.

Out-of-sample canonical truncation gives Sharpe 1.28 using one direction and 1.68 using two. With two directions, correlation with the full payoff is 0.997 and relative payoff MSE is 0.53%. This is the key distinction between nominal representation capacity and effective investment dimension.

## What the network learns

The first canonical factor combines cash-flow and profitability signals, recent and intermediate-horizon return information, investment and accounting changes, and a strong negative tilt toward extreme recent winners and high-volatility stocks. Important nonlinear interactions include one-month return x extreme-return/volatility, one-month return x earnings-change information, one-month return x operating profitability, and extreme-return/volatility x operating profitability.

The second factor is economically distinct, with stronger size, volatility, accrual, investment, and liquidity structure. The leading allocation declines when lagged market volatility and cross-sectional return dispersion are high. The factor definitions are descriptive neural functions, not causal structural factors.

## Spanning and portfolio value

For the leading factor, annualized evaluation-period spanning alpha and Newey-West t-statistics are approximately 8.64% (4.97) versus FF5+MOM, 8.39% (3.32) versus q5, 6.96% (2.15) versus QMJ, 4.87% (2.43) versus JKP mispricing, and 5.66% (3.54) versus the broad JKP-core benchmark.

Using pre-2015 tangency weights, adding the learned factor raises evaluation Sharpe by 1.12 relative to FF5+MOM, 0.62 relative to q5, 0.51 relative to QMJ, 0.20 relative to JKP mispricing, and 0.37 relative to JKP core. These are portfolio improvements conditional on the specific pre-test estimation procedure, not population Sharpe claims.

## Costs, ablations, and recursion

The representative portfolio, scaled with lagged volatility, has Sharpe 1.63 gross, 1.55 at 5 bps, 1.46 at 10 bps, and 1.20 at 25 bps. Average half-L1 turnover is 0.82 per month; average gross exposure is 2.81. The 10 bps plus 100 bps annual borrow scenario has Sharpe 1.35.

Five-seed ablations give mean evaluation Sharpe 1.83 for the full model, 1.53 without aggregate state, 1.86 without cross-sectional pooling, 1.46 with static allocation, 1.71 with a linear characteristic map, and 1.71 for a direct conditional neural stock policy. Thus aggregate state and dynamic allocation matter materially; the learned cross-sectional pooling block is not necessary for the main result.

A five-seed annual expanding-window exercise produces individual Sharpe ratios from 1.24 to 2.23, with mean 1.97. The equal-weighted recursive ensemble has Sharpe 2.21 over 120 monthly payoffs and a 12-month circular-block 95% interval [1.57, 2.97]. A recursively re-estimated linear managed-portfolio benchmark reaches Sharpe 1.57. The neural-minus-linear difference is 0.64, with paired 12-month block-bootstrap interval [-0.22, 1.39] and 92% of bootstrap draws positive. The neural ensemble has positive annual Sharpe in every payoff year from 2015 through 2024.

## Interpretation discipline

`K_{max}` is model capacity. Policy rank is the effective separable dimension of the learned stock-weight rule. Canonical allocation rank orders risk-standardized traded directions. Neither is interpreted as the number of equilibrium risk factors in the economy. Spanning results, cost results, and recursive performance are empirical diagnostics of this learned investment policy, not claims of causal identification.
