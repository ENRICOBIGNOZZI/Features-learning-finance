# The Investment Dimension of the Characteristic Zoo

This repository contains the code and empirical replication files for a neural-factor approach to conditional portfolio choice. The economic question is not whether a neural network can forecast individual stock returns, but how much of a large characteristic set survives into the tangency portfolio once nonlinear feature learning is allowed.

For stock $i$ at formation month $t$, a neural map produces $K$ characteristic scores $\phi_k(z_{i,t})$. They define traded managed portfolios,

$$F_{k,t+1}=\frac{1}{N_t}\sum_i \widetilde\phi_{k,t}(z_{i,t})R^e_{i,t+1},$$

and a conditional allocator uses lagged market state and an optional permutation-invariant cross-sectional summary to choose $b_t$. The final payoff is

$$R^p_{t+1}=b_t'F_{t+1},\qquad w_{i,t}=N_t^{-1}\widetilde\phi_t(z_{i,t})'b_t.$$

No long-short, market-neutrality, or gross-exposure restriction is imposed in the baseline training problem. Score RMS normalization fixes scale only. Training maximizes a block analogue of the tangency criterion $E[R^p]/\sqrt{E[(R^p)^2]}$, which is monotone in Sharpe for positive-mean payoffs.

## Causal empirical design

The admissible data cache is `data/JKP_USA_causal`. JKP rows dated $t$ contain a next-month payoff, so train/validation/evaluation splits are defined in payoff time. The main formation windows are 1963-01--2004-11, 2004-12--2014-11, and 2014-12--2024-11. Future return availability never determines formation-universe membership. Characteristic coverage is screened using training data only, leaving 123 JKP characteristics plus a cross-sectional size rank, for 124 network inputs.

## Current paper evidence

The final fixed-split $K_{max}\times$ seed battery contains 60 models: $K_{max}\in\{1,2,4,8,16,32\}$ and ten seeds at each capacity. Mean 2015--2024 Sharpe rises from 1.43 at $K=1$ to 1.79 at $K=32$; the ten-seed $K=32$ ensemble reaches 1.90, versus 1.51 for the linear managed-portfolio benchmark. The representative $K=32$ run is chosen from validation performance only and has test Sharpe 1.68.

The representative policy is highly concentrated. Its rotation-invariant policy entropy rank is 1.11 and the first policy direction contains 98.0% of spectrum mass. Two canonical traded directions reproduce the full test payoff with correlation 0.997 and relative payoff MSE 0.53%.

The leading factor has positive spanning alphas against FF5+MOM, q5, QMJ, JKP mispricing, and a broad JKP-core specification. At 10 bps per dollar traded, the representative volatility-targeted portfolio retains Sharpe 1.46. A five-seed annual expanding-window ensemble reaches Sharpe 2.21 over 2015--2024 versus 1.57 for a recursively re-estimated linear characteristic benchmark; its own 12-month circular-block bootstrap interval is [1.57, 2.97]. See `RESULTS.md` and the paper for the full interpretation and qualifications.

## Reproduction

Install with `python3 -m pip install -e .`. Core scripts are `robustness_grid.py` for the $K\times$seed battery, `summarize_seed_ensembles.py` for ensembles, `run_ablations.py` for architecture tests, `walk_forward.py` for recursive estimation, `analyze_spanning.py` for factor spanning and marginal Sharpe, `analyze_investability.py` for turnover/costs, `analyze_factors.py` and `analyze_stability.py` for economic interpretation, and `make_finance_figures.py` for the paper figure suite.

Data and large model checkpoints are intentionally excluded from Git. `snapshot_results/` contains the small result files needed to audit the paper tables and figures. Earlier folders described in Git history or local `results/` directories are not admissible unless they use the causal cache and payoff-time split documented in `AUDIT.md`.
