# Empirical Audit

The project treats formation information and payoff information as different objects.
The current admissible pipeline is `data/JKP_USA_causal` plus the code on the
working tree after the September 10 audit. Earlier result folders remain only for
forensics and must not be quoted as final evidence.

## Timing rules

A JKP row dated month t contains characteristics observable at formation and
`ret_exc_lead1m`, the payoff realized in t+1. Splits are therefore defined so
that no payoff month straddles train, validation, and evaluation samples.
The main formation windows are 1963-01--2004-11, 2004-12--2014-11, and
2014-12--2024-11.

Availability of `ret_exc_lead1m` never determines whether a stock belongs to
the formation universe. Unresolved payoffs are retained, set to zero in the
baseline accounting, recorded with a mask, and stressed adversely in the
implementability analysis.

## Feature selection

Characteristic coverage is screened using the training sample only through
November 2004. The resulting universe has 123 JKP characteristics; the network
also receives a cross-sectional log-size rank. Nine characteristics that would
pass only when future coverage is used are excluded from the causal cache.

## State variables

Current stock count is computed from the formation universe, not from the count
of nonmissing future returns. Historical market-return and dispersion states
are lagged so their underlying returns have been realized by formation time.
All state standardization uses training observations only.

## Portfolio implementation

Volatility targeting is ex post and uses lagged realized volatility. The cap of
three applies to the volatility-scaling multiplier, not to gross exposure.
Turnover compares target positions to previous positions after stock-return
drift and portfolio-NAV normalization. Cost diagnostics include trading costs,
short-borrow scenarios, and an adverse unresolved-return stress.

## Dimension language

`Kmax` is maximum latent/separable factor capacity. The policy spectrum measures
the effective separable rank of the learned characteristic-state stock-weight
rule. The return-based canonical allocation spectrum orders risk-standardized
traded factor directions. Neither object is interpreted as the number of
fundamental or equilibrium factors in the market.

The decisive low-dimensionality diagnostic is out-of-sample truncation: keep
the first r canonical contributions and measure Sharpe, correlation with the
full strategy, and payoff reconstruction error as r grows.

## Superseded results

Folders prefixed `SUPERSEDED_` used an earlier state construction in which the
current stock-count state was based on the number of available next-month
returns. The effect was numerically tiny because unresolved returns affect well
below one percent of the cross-section, but those results are not admissible as
final paper evidence. The final battery is rerun after this correction.

## Final admissible result directories

The paper snapshot uses `results/causal_v2_k_grid_paper` for the 60-run capacity battery, `results/causal_v2_k_grid_paper/k32_seed04` for representative factor interpretation and implementability diagnostics, `results/causal_v2_ablations_paper` for five-seed architecture ablations, `results/causal_v2_linear_managed` for the linear benchmark, and `results/causal_v2_walkforward_seed00` through `seed04` for recursive estimation. `results/causal_v2_walkforward_ensemble5` contains the deterministic equal-weighted recursive ensemble summary.

The representative run is chosen exclusively from the validation Sharpe distribution: seed 04 is one of the two observations bracketing the median validation Sharpe, with the lower seed index used as a deterministic tie-break. Factor profiles, spanning tests, and cost diagnostics therefore do not select a favorable evaluation-period seed.

All equity-line comparisons in the paper use realized 2015--2024 payoffs. When curves are placed on a common risk scale for visualization, volatility matching is performed ex post and is explicitly labeled; it does not alter the reported Sharpe ratios or feed back into model selection.
