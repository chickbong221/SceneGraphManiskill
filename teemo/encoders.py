"""
Graph encoders for TEEMO (draft §Method).

Provides three things:
  1. RelationGraphEncoder — relation-aware message passing over the 4 nodes and
     the typed edges, producing a pooled graph feature g_t for actor/critic.
  2. CausalRelationMask — straight-through Gumbel-Sigmoid binary mask over
     relation variables (draft §"Binary causal relation mask"). Optional.
  3. GraphAuxiliaryHeads — per-relation CE heads predicting graph labels from the
     observation latent (representation-shaping aux loss).

The encoder consumes the structured graph dict from graph_builder.build_graph
(node features + per-(pair,field) class ids). It embeds each relation by its
label, attaches it to its endpoint nodes, runs message passing, and pools.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from teemo import vocab
from teemo.graph_builder import PAIRS, SLOT_ORDER, TeemoGraphSpec


class RelationGraphEncoder(nn.Module):
    """
    Relation-aware message passing. Nodes carry features; each (pair, field)
    relation is embedded from its discrete label and routed along its edge.
    Optionally gated by a per-relation binary mask M_t (draft causal mask).
    """

    def __init__(self, spec: TeemoGraphSpec, node_in_dim: int,
                 hidden: int = 128, layers: int = 2):
        super().__init__()
        self.spec = spec
        self.hidden = hidden
        self.layers = layers
        self.node_names = spec.node_names
        self.node_index = {n: i for i, n in enumerate(spec.node_names)}

        self.node_proj = nn.Linear(node_in_dim, hidden)
        # one embedding per relation field, sized to that field's class count
        self.rel_embed = nn.ModuleDict({
            fld: nn.Embedding(vocab.RELATION_NUM_CLASSES[fld], hidden)
            for fld in vocab.RELATION_NUM_CLASSES
        })
        self.msg = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        ) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.GRUCell(hidden, hidden) for _ in range(layers)])
        self.out_dim = hidden

    def forward(self, graph: Dict, mask: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        nodes = graph["nodes"]
        N = next(iter(nodes.values())).shape[0]
        device = next(iter(nodes.values())).device
        # node hidden states (N, num_nodes, hidden)
        h = torch.stack([self.node_proj(nodes[n]) for n in self.node_names], dim=1)

        # precompute relation messages per (pair, field)
        edges: List[Tuple[int, int, torch.Tensor, str]] = []
        for pkey, tA, tB, objA, objB, fields in PAIRS:
            ia, ib = self.node_index[objA], self.node_index[objB]
            for fld in fields:
                lab = graph["targets"][f"{pkey}:{fld}"]
                re = self.rel_embed[fld](lab)               # (N, hidden)
                if mask is not None and f"{pkey}:{fld}" in mask:
                    re = re * mask[f"{pkey}:{fld}"].unsqueeze(-1)
                edges.append((ia, ib, re, fld))

        for L in range(self.layers):
            agg = torch.zeros_like(h)
            deg = torch.zeros(N, h.shape[1], 1, device=device)
            for ia, ib, re, _ in edges:
                # message a->b and b->a
                m_ab = self.msg[L](torch.cat([h[:, ia], h[:, ib], re], dim=-1))
                m_ba = self.msg[L](torch.cat([h[:, ib], h[:, ia], re], dim=-1))
                agg[:, ib] = agg[:, ib] + m_ab
                agg[:, ia] = agg[:, ia] + m_ba
                deg[:, ib] += 1; deg[:, ia] += 1
            agg = agg / deg.clamp_min(1.0)
            new_h = []
            for n in range(h.shape[1]):
                new_h.append(self.upd[L](agg[:, n], h[:, n]))
            h = torch.stack(new_h, dim=1)

        return h.mean(dim=1)                                # (N, hidden)


class CausalRelationMask(nn.Module):
    """
    Per-relation Bernoulli mask with straight-through Gumbel-Sigmoid (draft).
    Produces a dict[(pair:field)] -> (N,) gate in {0,1} (soft in train via STE).
    Trained through the downstream loss + L1 penalty on probabilities.
    """

    def __init__(self, spec: TeemoGraphSpec, rel_feat_dim: int, hidden: int = 64):
        super().__init__()
        self.slots = [f"{p}:{f}" for (p, f) in SLOT_ORDER]
        self.decoder = nn.Sequential(
            nn.Linear(rel_feat_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, len(self.slots)),
        )

    def forward(self, rel_feat: torch.Tensor, tau: float = 1.0, hard: bool = True):
        logits = self.decoder(rel_feat)                     # (N, num_slots)
        probs = torch.sigmoid(logits)
        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            g = torch.log(u) - torch.log(1 - u)
            y = torch.sigmoid((logits + g) / tau)
            if hard:
                y_hard = (y > 0.5).float()
                y = y_hard + y - y.detach()
        else:
            y = (probs > 0.5).float()
        mask = {self.slots[i]: y[:, i] for i in range(len(self.slots))}
        l1 = probs.mean()
        return mask, l1


class GraphAuxiliaryHeads(nn.Module):
    """Per-relation CE heads predicting graph labels from the obs latent."""

    def __init__(self, obs_latent_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({
            f"{p}:{f}": nn.Linear(obs_latent_dim, vocab.RELATION_NUM_CLASSES[f])
            for (p, f) in SLOT_ORDER
        })

    def forward(self, obs_latent):
        return {k: head(obs_latent) for k, head in self.heads.items()}

    @staticmethod
    def loss(logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        total = 0.0
        for k, lg in logits.items():
            total = total + F.cross_entropy(lg, targets[k])
        return total / max(len(logits), 1)
