"""
Graph encoders and auxiliary heads for the Phase-1 discrete temporal
affordance-progress graph experiment.

This file is intentionally task-agnostic. ``DiscreteTemporalGraphEncoder`` and
``GraphAuxiliaryHeads`` only depend on the graph schema (head_dims / graph_dim),
not on any StackCube-specific name.

Two encoder modes:
  - "mlp" (default): a small MLP over the current-step one-hot graph vector.
  - "temporal_transformer": a TransformerEncoder over the last K+1 graphs,
    inspired by STTran's spatial-temporal modeling of dynamic scene graphs.
    Phase-1 keeps this as an optional toggle; the MLP mode is the default and
    should not regress baseline PPO behavior.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class DiscreteTemporalGraphEncoder(nn.Module):
    """
    Encode the discrete graph for the critic.

    Inspired by STTran (Cong et al.): the temporal-transformer mode applies a
    self-attention encoder over a length-(K+1) sequence of per-step graph
    one-hots, with a learned positional embedding. The MLP mode is a strict
    subset that conditions the critic on just the current-step graph.
    """

    def __init__(
        self,
        graph_dim: int = 36,
        embed_dim: int = 128,
        encoder_type: str = "mlp",
        k: int = 5,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 256,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()
        if encoder_type not in ("mlp", "temporal_transformer"):
            raise ValueError(f"unknown encoder_type={encoder_type!r}")
        self.encoder_type = encoder_type
        self.graph_dim = graph_dim
        self.embed_dim = embed_dim
        self.k = k

        if encoder_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(graph_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_proj = nn.Linear(graph_dim, embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, k + 1, embed_dim))
            nn.init.normal_(self.pos_embed, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=transformer_dropout,
                batch_first=True,
                activation="relu",
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_layers
            )

    @property
    def out_features(self) -> int:
        return self.embed_dim

    def forward(
        self,
        graph_onehot: torch.Tensor,
        graph_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          graph_onehot: [B, graph_dim] current-step graph. Used by MLP mode.
          graph_history: [B, K+1, graph_dim] last K+1 graphs (most recent last).
            Required by temporal_transformer mode; ignored by MLP mode.
        Returns:
          [B, embed_dim] critic-graph latent.
        """
        if self.encoder_type == "mlp":
            return self.mlp(graph_onehot)
        if graph_history is None:
            raise ValueError(
                "temporal_transformer encoder requires graph_history of shape [B, K+1, graph_dim]"
            )
        x = self.input_proj(graph_history)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.transformer(x)
        return x.mean(dim=1)


class GraphAuxiliaryHeads(nn.Module):
    """
    Per-head linear classifiers predicting graph labels from the *normal*
    visual/proprio observation latent.

    These heads only consume the obs latent (no graph input), so training them
    shapes the encoder to surface affordance-relevant features. The actor's
    decision boundary is unaffected by the heads beyond shared-encoder
    representation learning, and the heads are never used at inference.
    """

    def __init__(self, obs_latent_dim: int, head_dims: Dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(obs_latent_dim, dim) for name, dim in head_dims.items()}
        )

    def forward(self, obs_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(obs_latent) for name, head in self.heads.items()}
