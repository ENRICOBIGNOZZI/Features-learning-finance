import torch

from neural_factors.model import NeuralFactorModel


def test_ragged_forward_matches_sequential():
    torch.manual_seed(19)
    model = NeuralFactorModel(
        input_dim=5, state_dim=4, hidden=14, factors=3, context_heads=2
    )
    ns = [9, 13, 6, 11]
    xs = [torch.randn(n, 5) for n in ns]
    states = [torch.randn(4) for _ in ns]
    returns = [0.02 * torch.randn(n) for n in ns]
    sequential = [model(x, s, r) for x, s, r in zip(xs, states, returns)]

    offsets = [0]
    for n in ns:
        offsets.append(offsets[-1] + n)
    ragged = model.forward_ragged(
        torch.cat(xs), torch.stack(states), torch.cat(returns), offsets
    )
    assert torch.allclose(
        ragged.portfolio_return,
        torch.stack([out.portfolio_return for out in sequential]), atol=1e-6,
    )
    assert torch.allclose(
        ragged.factor_returns,
        torch.stack([out.factor_returns for out in sequential]), atol=1e-6,
    )
    assert torch.allclose(
        ragged.allocator,
        torch.stack([out.allocator for out in sequential]), atol=1e-6,
    )
    cursor = 0
    for row, n in enumerate(ns):
        assert torch.allclose(
            ragged.stock_weights[cursor:cursor+n],
            sequential[row].stock_weights,
            atol=1e-6,
        )
        cursor += n


def test_ragged_conditional_score_policy_matches_sequential():
    torch.manual_seed(23)
    model = NeuralFactorModel(
        input_dim=4, state_dim=3, hidden=10, factors=1, context_heads=2,
        use_context=True, use_state=True, static_allocator=True,
        conditional_scores=True,
    )
    ns = [8, 5, 12]
    xs = [torch.randn(n, 4) for n in ns]
    states = [torch.randn(3) for _ in ns]
    returns = [0.03 * torch.randn(n) for n in ns]
    sequential = [model(x, s, r) for x, s, r in zip(xs, states, returns)]
    offsets = [0]
    for n in ns:
        offsets.append(offsets[-1] + n)
    ragged = model.forward_ragged(
        torch.cat(xs), torch.stack(states), torch.cat(returns), offsets
    )
    assert torch.allclose(
        ragged.portfolio_return,
        torch.stack([out.portfolio_return for out in sequential]), atol=1e-6,
    )
