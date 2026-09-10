# Paper result snapshot

This directory contains the small, auditable outputs used by the September 2026 paper snapshot. Raw JKP data and neural-network checkpoints are deliberately excluded.

`k_grid/` contains the 60-run factor-capacity battery and ten-seed ensembles. `linear/` contains the linear characteristic benchmark. `ablations/` contains the five-seed architecture comparison. `walk_forward/` contains the five recursive yearly-update runs summarized as an equal-weighted ensemble. `representative/` contains the validation-selected representative model's policy spectrum, canonical truncation, factor interpretation, spanning, and implementability outputs.

All admissible numbers use the causal data construction and payoff-time splits documented in `../AUDIT.md`. Earlier exploratory outputs are intentionally not included here.