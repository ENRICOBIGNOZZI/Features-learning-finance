# Neural Factor Discovery for Asset Pricing

This repository studies a finance-first question: can a neural network discover the nonlinear characteristic factors that matter for the conditional tangency portfolio?

The model is one end-to-end neural network. It is not trained to forecast every stock return and it does not impose a hand-built factor definition.

For stock $i$ at month $t$, let $z_{i,t}$ denote the vector of firm characteristics. The network learns $K$ nonlinear characteristic maps

$$
\phi_k(z_{i,t}), \qquad k=1,\ldots,K,
$$

and the corresponding managed factor returns

$$
F_{k,t+1}=\frac{1}{N_t}\sum_{i=1}^{N_t}\phi_k(z_{i,t})R^e_{i,t+1}.
$$

A second part of the same network observes lagged market state $M_t$ and a permutation-invariant summary $C_t$ of the current stock cross-section, and produces factor allocations $b_t$.

The final portfolio is

$$
R^p_{t+1}=b_t'F_{t+1},
$$

or, equivalently,

$$
w_{i,t}=\frac{1}{N_t}\sum_{k=1}^K b_{k,t}\phi_k(z_{i,t}).
$$

This gives a low-rank nonlinear characteristic-by-state representation of portfolio weights. Firm-level nonlinearities create interactions among characteristics; the JKP characteristics are monthly cross-sectional ranks, so the inputs are already relative to the contemporaneous stock universe. Aggregate cross-sectional and market conditions price the learned characteristic factors dynamically through $b_t$.

## Economic objective

Training minimizes

$$
\frac{1}{T}\sum_t (1-R^p_{t+1})^2,
$$

which selects the tangency-portfolio direction without optimizing a noisy sample Sharpe ratio directly.

No market-neutrality, long/short, gross-exposure, or leverage constraint is imposed in the baseline. Consequently raw return scale is not an investability claim. Sharpe is scale invariant; portfolio plots should be rescaled ex post to a common target volatility when comparing strategies.

## Factor identification

The latent factor coordinates are not economically identified: any invertible rotation can represent the same portfolio. We therefore do not interpret neural units directly.

After training, the code whitens the learned factor-return space and diagonalizes an economic allocation operator. This produces an invariant ordered basis of canonical factor directions and an endogenous effective factor dimension $K_{eff}$.

Thus $K$ is maximum model capacity. The economically active number of factors is an empirical output.

## Data

The implementation uses the cleaned U.S. JKP stock-level panel already available locally. The current cache contains 132 characteristics and annual files from 1963 through 2024. Data are not committed to GitHub.

## Reproduce the pilot

```bash
python3 -m pip install -e .
python3 run_experiment.py \
  --data-dir /Users/enrico/Desktop/PHD/portfolio/paper_codice/data/JKP_USA_clean \
  --output-dir results/pilot_k4 \
  --preset pilot --variant full --hidden 32 --factors 4 \
  --context-heads 2 --epochs 20 --device mps
```

The pilot split is 2000--2014 train, 2015--2019 validation, and 2020--2024 test. The paper preset uses 1963--2004, 2005--2014, and 2015--2024.

Interpret the learned canonical factors with

```bash
python3 analyze_factors.py \
  --data-dir /Users/enrico/Desktop/PHD/portfolio/paper_codice/data/JKP_USA_clean \
  --result-dir results/pilot_k4 --start 2020-01-01 --end 2024-12-31
```

The interpretation output includes characteristic sensitivities, nonlinear pairwise interactions, top-minus-bottom characteristic profiles, and factor-allocation links to lagged market state.

## Research status

The checked-in code is a first falsifiable prototype. The pilot numbers are pre-cost, single-seed diagnostics. A finance result requires multiple seeds, walk-forward retraining, turnover and transaction costs, capacity, spanning against standard factors, and stability across subperiods.
