from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.SiLU(), nn.LayerNorm(hidden)]
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPool(nn.Module):
    """Permutation-invariant learned summaries of the current stock cross-section."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.hidden = hidden
        self.score = nn.Linear(hidden, heads, bias=False)

    @property
    def out_dim(self) -> int:
        return self.heads * self.hidden

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.score(h) / math.sqrt(max(h.shape[-1], 1))
        alpha = torch.softmax(logits, dim=0)
        pooled = alpha.transpose(0, 1) @ h
        return pooled.reshape(-1)


@dataclass
class FactorOutput:
    portfolio_return: torch.Tensor
    factor_returns: torch.Tensor
    factor_scores: torch.Tensor
    allocator: torch.Tensor
    stock_weights: torch.Tensor
    context: torch.Tensor


class NeuralFactorModel(nn.Module):
    """One NN: characteristics -> latent factors -> conditional tangency portfolio."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden: int = 64,
        factors: int = 16,
        context_heads: int = 8,
        use_context: bool = True,
        use_state: bool = True,
        static_allocator: bool = False,
    ):
        super().__init__()
        self.factors = factors
        self.use_context = use_context
        self.use_state = use_state
        self.static_allocator = static_allocator
        self.encoder = MLP(input_dim, hidden, hidden, depth=2)
        self.pool = AttentionPool(hidden, context_heads)
        self.context_dim = self.pool.out_dim if use_context else hidden
        self.state_hidden = max(8, hidden // 4)
        if use_state:
            self.state_encoder = MLP(state_dim, self.state_hidden, self.state_hidden, depth=1)
        else:
            self.state_encoder = None

        # Factor construction is state-invariant: phi_k depends only on stock characteristics.
        # Market/cross-sectional state enters through the allocator, so K measures the
        # rank of the conditional characteristic-by-state interaction.
        score_dim = hidden
        allocator_dim = self.context_dim + (self.state_hidden if use_state else 0)

        self.factor_head = MLP(score_dim, hidden, factors, depth=1)
        if static_allocator:
            self.static_b = nn.Parameter(torch.zeros(factors))
            self.allocator = None
        else:
            self.static_b = None
            self.allocator = MLP(allocator_dim, hidden, factors, depth=1)

    def _context(self, h: torch.Tensor) -> torch.Tensor:
        if self.use_context:
            return self.pool(h)
        return h.mean(dim=0)

    def _state(self, state: torch.Tensor) -> torch.Tensor | None:
        if not self.use_state:
            return None
        return self.state_encoder(state.unsqueeze(0)).squeeze(0)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        returns: torch.Tensor,
    ) -> FactorOutput:
        h = self.encoder(x)
        context = self._context(h)
        state_h = self._state(state)

        factor_scores = self.factor_head(h)

        if self.static_allocator:
            allocator = self.static_b
        else:
            allocator_parts = [context]
            if state_h is not None:
                allocator_parts.append(state_h)
            allocator = self.allocator(torch.cat(allocator_parts).unsqueeze(0)).squeeze(0)

        n = max(int(x.shape[0]), 1)
        factor_returns = factor_scores.transpose(0, 1) @ returns / n
        portfolio_return = torch.dot(allocator, factor_returns)
        stock_weights = factor_scores @ allocator / n
        return FactorOutput(
            portfolio_return=portfolio_return,
            factor_returns=factor_returns,
            factor_scores=factor_scores,
            allocator=allocator,
            stock_weights=stock_weights,
            context=context,
        )
