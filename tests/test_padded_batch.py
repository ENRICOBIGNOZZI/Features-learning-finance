import torch

from neural_factors.model import NeuralFactorModel


def test_padded_forward_matches_month_by_month():
    torch.manual_seed(7)
    model = NeuralFactorModel(
        input_dim=6, state_dim=3, hidden=12, factors=4, context_heads=2
    )
    ns = [11, 7, 13]
    xs = [torch.randn(n, 6) for n in ns]
    states = [torch.randn(3) for _ in ns]
    returns = [0.03 * torch.randn(n) for n in ns]
    individual = [model(x, s, r) for x, s, r in zip(xs, states, returns)]

    max_n = max(ns)
    xpad = torch.zeros(len(ns), max_n, 6)
    rpad = torch.zeros(len(ns), max_n)
    mask = torch.zeros(len(ns), max_n, dtype=torch.bool)
    for i, n in enumerate(ns):
        xpad[i, :n] = xs[i]
        rpad[i, :n] = returns[i]
        mask[i, :n] = True
    batched = model.forward_padded(xpad, torch.stack(states), rpad, mask)
    expected_portfolio = torch.stack([out.portfolio_return for out in individual])
    expected_factors = torch.stack([out.factor_returns for out in individual])
    expected_allocator = torch.stack([out.allocator for out in individual])
    assert torch.allclose(batched.portfolio_return, expected_portfolio, atol=1e-6)
    assert torch.allclose(batched.factor_returns, expected_factors, atol=1e-6)
    assert torch.allclose(batched.allocator, expected_allocator, atol=1e-6)
    for i, n in enumerate(ns):
        assert torch.allclose(
            batched.stock_weights[i, :n], individual[i].stock_weights, atol=1e-6
        )
        assert torch.count_nonzero(batched.stock_weights[i, n:]) == 0
