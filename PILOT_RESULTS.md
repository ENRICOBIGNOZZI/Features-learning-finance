# Pilot Results — JKP USA

These are architecture diagnostics, not paper results. All models below use the cleaned U.S. JKP stock panel, a single seed, only five training epochs, and no transaction-cost or leverage constraints.

Pilot dates: train 2000--2014, validation 2015--2019, test 2020--2024.

| Model | K | Validation Sharpe | Test Sharpe |
|---|---:|---:|---:|
| Full state + cross-section | 1 | 1.542 | 1.307 |
| Full state + cross-section | 4 | **1.623** | **1.511** |
| Full state + cross-section | 8 | 1.509 | 1.478 |
| No lagged market state | 4 | 1.603 | 1.276 |
| Mean rather than learned cross-section summary | 4 | 1.281 | 1.453 |
| Static factor allocator | 4 | 0.868 | 0.835 |

The K=4 full model is the strongest short pilot. The comparison is deliberately preliminary because five epochs are not enough to establish equal optimization quality across variants.

## Endogenous factor dimension

The K=4 model was allowed four latent factors, but the invariant economic spectrum is highly concentrated:

| Canonical direction | Economic share |
|---|---:|
| 1 | 99.9843% |
| 2 | 0.0157% |
| 3--4 | negligible |

Its entropy effective rank is 1.0015. In this pilot, the NN therefore uses a rich nonlinear characteristic map but organizes the final tangency portfolio around essentially one dominant economic direction. This is why the architecture treats K as maximum capacity and measures effective economic dimension ex post rather than forcing a factor count.

## What the dominant factor looks like

On the 2020--2024 test period, the dominant canonical factor loads most strongly on cash-based profitability, net operating assets, gross-margin/sales dynamics, extreme-return/volatility, seasonality, asset tangibility, turnover, gross profitability, and revenue surprise.

Its strongest top-minus-bottom characteristic tilts are operating leverage, the JKP mispricing-performance composite, cash-based operating profitability, asset/capital turnover, gross profitability, lagged cash profitability, operating profitability, QMJ profitability, operating cash flow, and the mispricing-management composite.

A useful first economic description is therefore **profitability / operating efficiency / quality**, enriched by investment, mispricing, seasonality, and tail-return information. That label is descriptive only; the NN is not restricted to an additive factor model.

The dominant factor is genuinely nonlinear. The largest pairwise permutation interaction diagnostics include:

- cash profitability × net operating assets;
- cash profitability × extreme-return/volatility;
- cash profitability × change in gross margin relative to sales;
- cash profitability × annual seasonality;
- cash profitability × gross profitability;
- gross-margin/sales dynamics × extreme-return/volatility.

These interaction scores are diagnostic contrasts, not structural causal effects.

## Conditional pricing of the factor

The allocation to the dominant factor is strongly related in the test sample to lagged equal-weighted market return (correlation 0.69), lagged 12-month equal-weighted market return (0.66), and lagged value-weighted market return (0.61). This suggests that the network is learning a comparatively stable characteristic factor and changing its price/exposure with market state.

## Longer 1963--2024 prototype

A second single-seed run used the paper split (1963--2004 train, 2005--2014 validation, 2015--2024 test), K=4, and ten epochs. It reached validation Sharpe 3.07 and test Sharpe 1.81 before costs. The number is not an investable backtest: raw scale is unconstrained and no trading costs, turnover, capacity, or repeated-seed uncertainty are charged.

The same low-dimensional result survives the longer sample. Canonical direction 1 receives 99.9255% of the training-sample economic spectrum; direction 2 receives 0.0745%; the rest is negligible. On the 2015--2024 test period, direction 1 accounts for about 95.4% of absolute portfolio contribution and has standalone Sharpe 1.66.

The longer-sample factor is economically broader than the short pilot. Its strongest sensitivities include residual momentum (`resff3_12_1`), one-month return, earnings surprise, three-year sales growth, extreme-return/volatility, sales growth, investment growth, operating profitability, seasonality, and cash-based profitability.

Its top-minus-bottom profile combines positive performance/momentum and quality/cash-flow characteristics with a strong negative loading on recent extreme winners. The largest nonlinear interactions are residual momentum × short-term return, residual momentum × earnings surprise, residual momentum × extreme-return/volatility, residual momentum × sales growth, and short-term return × earnings surprise.

The dominant factor's conditional allocation is lower when lagged cross-sectional dispersion and lagged market volatility are high (test correlations about -0.82 and -0.77) and is mildly higher with the prior 12-month equal-weighted market trend (0.36). This is a natural candidate interpretation: a nonlinear momentum-quality factor whose exposure is reduced in high-dispersion/high-volatility states.
