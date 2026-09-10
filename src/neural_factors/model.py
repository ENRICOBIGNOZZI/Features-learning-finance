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

    def forward_padded(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool B padded cross-sections; mask is True for observed stocks."""
        logits = self.score(h) / math.sqrt(max(h.shape[-1], 1))
        logits = logits.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        alpha = torch.softmax(logits, dim=1)
        pooled = torch.einsum("bnh,bnd->bhd", alpha, h)
        return pooled.reshape(h.shape[0], -1)


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
        linear_characteristics: bool = False,
        conditional_scores: bool = False,
    ):
        super().__init__()
        self.factors = factors
        self.use_context = use_context
        self.use_state = use_state
        self.linear_characteristics = linear_characteristics
        self.conditional_scores = conditional_scores
        self.encoder = MLP(input_dim, hidden, hidden, depth=2)
        self.pool = AttentionPool(hidden, context_heads)
        self.context_dim = self.pool.out_dim if use_context else 0
        self.state_hidden = max(8, hidden // 4)
        if use_state:
            self.state_encoder = MLP(state_dim, self.state_hidden, self.state_hidden, depth=1)
        else:
            self.state_encoder = None

        # The characteristic factors are stable functions of firm characteristics.
        # Cross-sectional ranks already make characteristics relative to the current
        # stock universe; aggregate cross-sectional context and market state determine
        # the time-varying factor allocation, not the factor identities themselves.
        allocator_dim = self.context_dim + (self.state_hidden if use_state else 0)

        score_base_dim = input_dim if linear_characteristics else hidden
        score_dim = score_base_dim
        if conditional_scores:
            score_dim += self.context_dim + (self.state_hidden if use_state else 0)
        if linear_characteristics and not conditional_scores:
            self.factor_head = nn.Linear(input_dim, factors, bias=True)
        else:
            self.factor_head = MLP(score_dim, hidden, factors, depth=2)
        self.static_allocator = static_allocator or allocator_dim == 0
        if self.static_allocator:
            self.static_b = nn.Parameter(0.05 * torch.randn(factors))
            self.allocator = None
        else:
            self.static_b = None
            self.allocator = MLP(allocator_dim, hidden, factors, depth=1)

    def _context(self, h: torch.Tensor) -> torch.Tensor:
        if self.use_context:
            return self.pool(h)
        return h.new_empty((0,))

    def _state(self, state: torch.Tensor) -> torch.Tensor | None:
        if not self.use_state:
            return None
        if state.ndim == 1:
            return self.state_encoder(state.unsqueeze(0)).squeeze(0)
        return self.state_encoder(state)

    def _score_input_single(
        self, x: torch.Tensor, h: torch.Tensor, context: torch.Tensor,
        state_h: torch.Tensor | None,
    ) -> torch.Tensor:
        base = x if self.linear_characteristics else h
        if not self.conditional_scores:
            return base
        parts = [base]
        if self.use_context:
            parts.append(context.unsqueeze(0).expand(base.shape[0], -1))
        if state_h is not None:
            parts.append(state_h.unsqueeze(0).expand(base.shape[0], -1))
        return torch.cat(parts, dim=1)

    def _score_input_padded(
        self, x: torch.Tensor, h: torch.Tensor, context: torch.Tensor,
        state_h: torch.Tensor | None,
    ) -> torch.Tensor:
        base = x if self.linear_characteristics else h
        if not self.conditional_scores:
            return base
        parts = [base]
        if self.use_context:
            parts.append(context[:, None, :].expand(-1, base.shape[1], -1))
        if state_h is not None:
            parts.append(state_h[:, None, :].expand(-1, base.shape[1], -1))
        return torch.cat(parts, dim=2)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        returns: torch.Tensor,
    ) -> FactorOutput:
        h = self.encoder(x)
        context = self._context(h)
        state_h = self._state(state)

        score_input = self._score_input_single(x, h, context, state_h)
        factor_scores = self.factor_head(score_input)
        factor_scores = factor_scores / torch.sqrt(
            factor_scores.square().mean(dim=0, keepdim=True) + 1e-6
        )

        if self.static_allocator:
            allocator = self.static_b
        else:
            allocator_parts = []
            if self.use_context:
                allocator_parts.append(context)
            if state_h is not None:
                allocator_parts.append(state_h)
            allocator = self.allocator(torch.cat(allocator_parts).unsqueeze(0)).squeeze(0)

        # Overall allocation scale is unidentified under a Sharpe objective. Fix it
        # numerically without restricting relative factor exposures.
        allocator = allocator / torch.sqrt(allocator.square().mean() + 1e-6)

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

    def forward_padded(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor,
    ) -> FactorOutput:
        """Vectorized forward for padded monthly cross-sections [B,N,D]."""
        h = self.encoder(x)
        if self.use_context:
            context = self.pool.forward_padded(h, mask)
        else:
            context = h.new_empty((h.shape[0], 0))
        state_h = self._state(state)
        score_input = self._score_input_padded(x, h, context, state_h)
        factor_scores = self.factor_head(score_input)
        active = mask.unsqueeze(-1).to(factor_scores.dtype)
        counts = active.sum(dim=1).clamp_min(1.0)
        rms = torch.sqrt((factor_scores.square() * active).sum(dim=1) / counts + 1e-6)
        factor_scores = factor_scores / rms.unsqueeze(1)
        factor_scores = factor_scores * active

        if self.static_allocator:
            allocator = self.static_b.unsqueeze(0).expand(h.shape[0], -1)
        else:
            allocator_parts = []
            if self.use_context:
                allocator_parts.append(context)
            if state_h is not None:
                allocator_parts.append(state_h)
            allocator = self.allocator(torch.cat(allocator_parts, dim=1))
        allocator = allocator / torch.sqrt(
            allocator.square().mean(dim=1, keepdim=True) + 1e-6
        )

        counts_1d = counts.squeeze(-1)
        factor_returns = torch.einsum("bnk,bn->bk", factor_scores, returns) / counts_1d.unsqueeze(-1)
        portfolio_return = (allocator * factor_returns).sum(dim=1)
        stock_weights = torch.einsum("bnk,bk->bn", factor_scores, allocator) / counts_1d.unsqueeze(-1)
        stock_weights = stock_weights * mask.to(stock_weights.dtype)
        return FactorOutput(
            portfolio_return=portfolio_return,
            factor_returns=factor_returns,
            factor_scores=factor_scores,
            allocator=allocator,
            stock_weights=stock_weights,
            context=context,
        )

    def forward_ragged(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        returns: torch.Tensor,
        offsets: list[int] | tuple[int, ...],
    ) -> FactorOutput:
        """Vectorized stock encoder for a ragged block of monthly cross-sections."""
        h = self.encoder(x)
        lengths = [end - start for start, end in zip(offsets[:-1], offsets[1:])]
        batch = len(lengths)
        if self.use_context:
            context = torch.stack([
                self.pool(h[start:end]) for start, end in zip(offsets[:-1], offsets[1:])
            ], dim=0)
        else:
            context = h.new_empty((batch, 0))
        state_h = self._state(state)

        base = x if self.linear_characteristics else h
        if self.conditional_scores:
            parts = [base]
            counts = torch.as_tensor(lengths, device=x.device, dtype=torch.long)
            if self.use_context:
                parts.append(torch.repeat_interleave(context, counts, dim=0))
            if state_h is not None:
                parts.append(torch.repeat_interleave(state_h, counts, dim=0))
            raw_scores = self.factor_head(torch.cat(parts, dim=1))
        else:
            raw_scores = self.factor_head(base)

        scores_by_month = []
        factor_returns = []
        for start, end in zip(offsets[:-1], offsets[1:]):
            raw_m = raw_scores[start:end]
            rms = torch.sqrt(raw_m.square().mean(dim=0, keepdim=True) + 1e-6)
            score_m = raw_m / rms
            scores_by_month.append(score_m)
            r_m = returns[start:end]
            factor_returns.append(score_m.transpose(0, 1) @ r_m / max(end - start, 1))
        factor_returns_t = torch.stack(factor_returns, dim=0)
        if self.static_allocator:
            allocator = self.static_b.unsqueeze(0).expand(batch, -1)
        else:
            allocator_parts = []
            if self.use_context:
                allocator_parts.append(context)
            if state_h is not None:
                allocator_parts.append(state_h)
            allocator = self.allocator(torch.cat(allocator_parts, dim=1))
        allocator = allocator / torch.sqrt(
            allocator.square().mean(dim=1, keepdim=True) + 1e-6
        )
        portfolio_return = (allocator * factor_returns_t).sum(dim=1)
        stock_weights = torch.cat([
            score_m @ allocator[row] / max(offsets[row + 1] - offsets[row], 1)
            for row, score_m in enumerate(scores_by_month)
        ])
        return FactorOutput(
            portfolio_return=portfolio_return,
            factor_returns=factor_returns_t,
            factor_scores=torch.cat(scores_by_month, dim=0),
            allocator=allocator,
            stock_weights=stock_weights,
            context=context,
        )
