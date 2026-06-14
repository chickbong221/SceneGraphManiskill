"""
Per-env K+1 ring buffer of privileged state, for K=5 temporal labels.

Handles ManiSkillVectorEnv auto-reset: push(state, just_reset) invalidates an
env's history on the step it reset, so temporal labels never cross episode
boundaries. get_window() returns the full (K+1, N, ...) stack oldest..newest and
a valid mask (env has >= K+1 in-episode pushes).

Stores ALL fields needed by temporal_labels (poses, gripper width, contact
forces, booleans) so binary window-stability debounce can be computed.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


class GraphHistory:
    FIELDS = (
        "tcp_p", "tcp_q", "gripper_width",
        "cubeA_p", "cubeA_q", "cubeB_p", "cubeB_q",
        "contact_gA", "contact_gB", "contact_AB",
        "is_grasped", "is_on",
    )

    def __init__(self, num_envs: int, k: int, device):
        self.num_envs = num_envs
        self.k = k
        self.device = device
        self.size = k + 1
        self.head = 0
        self.valid_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.buf: Dict[str, torch.Tensor] = {}

    def _init(self, state):
        if self.buf:
            return
        for f in self.FIELDS:
            v = state[f]
            self.buf[f] = torch.zeros((self.size,) + tuple(v.shape),
                                      dtype=v.dtype, device=v.device)

    def push(self, state, just_reset):
        self._init(state)
        for f in self.FIELDS:
            self.buf[f][self.head] = state[f]
        self.head = (self.head + 1) % self.size
        size_t = torch.tensor(self.size, dtype=torch.long, device=self.device)
        nc = torch.minimum(self.valid_count + 1, size_t)
        self.valid_count = torch.where(just_reset.bool(),
                                       torch.ones_like(nc), nc)

    def get_window(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Return window dict field -> (K+1, N, ...) ordered oldest..newest, plus
        valid mask (>= K+1 pushes). Oldest = state K steps before newest.
        """
        order = [(self.head + i) % self.size for i in range(self.size)]
        idx = torch.tensor(order, device=self.device)
        window = {f: self.buf[f][idx] for f in self.FIELDS}
        valid = self.valid_count >= self.size
        return window, valid

    def reset_all(self):
        self.valid_count.zero_()
        self.head = 0
