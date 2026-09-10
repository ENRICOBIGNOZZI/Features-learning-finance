import copy
import numpy as np
import torch

from neural_factors.data import MonthPanel
from neural_factors.model import NeuralFactorModel
from neural_factors.train import _global_objective_step, tangency_score


def _panels():
    rng = np.random.default_rng(7)
    out = []
    for month in range(5):
        n = 8 + month
        out.append(MonthPanel(
            date=np.datetime64(f"2020-{month+1:02d}-28"),
            x=rng.normal(size=(n, 4)).astype(np.float32),
            r=(0.02 * rng.normal(size=n)).astype(np.float32),
            ids=np.arange(n), me=np.ones(n, dtype=np.float32),
            state=rng.normal(size=2).astype(np.float32),
            return_observed=np.ones(n, dtype=bool),
        ))
    return out


def test_memory_efficient_global_gradient_matches_direct_objective():
    torch.manual_seed(9)
    model_a = NeuralFactorModel(4, 2, hidden=8, factors=2, context_heads=1)
    model_b = copy.deepcopy(model_a)
    panels = _panels()
    opt_a = torch.optim.SGD(model_a.parameters(), lr=1e-3)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=1e-3)
    _global_objective_step(model_a, panels, opt_a, torch.device("cpu"), 1e9)
    values = []
    for p in panels:
        values.append(model_b(torch.tensor(p.x), torch.tensor(p.state), torch.tensor(p.r)).portfolio_return)
    loss = -tangency_score(torch.stack(values))
    loss.backward(); opt_b.step()
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb, atol=2e-6, rtol=2e-5)
