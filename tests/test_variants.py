import torch

from neural_factors.model import NeuralFactorModel


def test_no_context_allocator_is_stock_cross_section_invariant_given_state():
    torch.manual_seed(3)
    model = NeuralFactorModel(
        input_dim=5, state_dim=3, hidden=12, factors=4,
        use_context=False, use_state=True,
    )
    state = torch.randn(3)
    x1, x2 = torch.randn(20, 5), torch.randn(37, 5)
    r1, r2 = torch.randn(20), torch.randn(37)
    b1 = model(x1, state, r1).allocator
    b2 = model(x2, state, r2).allocator
    assert torch.allclose(b1, b2, atol=1e-7)


def test_linear_characteristic_variant_is_affine_in_raw_characteristics():
    torch.manual_seed(4)
    model = NeuralFactorModel(
        input_dim=5, state_dim=3, hidden=12, factors=3,
        linear_characteristics=True,
    )
    assert isinstance(model.factor_head, torch.nn.Linear)
