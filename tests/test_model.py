import torch

from neural_factors.model import NeuralFactorModel


def test_portfolio_identity_and_permutation_invariance():
    torch.manual_seed(1)
    model = NeuralFactorModel(
        input_dim=7, state_dim=3, hidden=16, factors=5, context_heads=3
    )
    x = torch.randn(50, 7)
    state = torch.randn(3)
    returns = torch.randn(50) * 0.05

    out = model(x, state, returns)
    direct = torch.dot(out.stock_weights, returns)
    assert torch.allclose(out.portfolio_return, direct, atol=1e-6)
    assert out.factor_scores.shape == (50, 5)
    assert out.factor_returns.shape == (5,)

    permutation = torch.randperm(50)
    permuted = model(x[permutation], state, returns[permutation])
    assert torch.allclose(out.portfolio_return, permuted.portfolio_return, atol=1e-6)
    assert torch.allclose(out.factor_returns, permuted.factor_returns, atol=1e-6)
    assert torch.allclose(out.allocator, permuted.allocator, atol=1e-6)
