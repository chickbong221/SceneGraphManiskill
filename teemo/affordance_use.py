"""
Runtime affordance bank: loads per-class object-frame candidates and computes
tcp-alignment error and aperture error against the nearest candidate.

Candidates were produced offline by affordance_extract.py. If a class .npz is
missing, errors return a large/zero sentinel so the corresponding relation falls
to its sentinel bin (graceful degradation; schema stays intact).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch


def quat_geodesic(q1, q2):
    """Batched angle (rad) between wxyz quats. q1 (...,4), q2 (...,4)."""
    dot = torch.abs((q1 * q2).sum(-1)).clamp(-1.0, 1.0)
    return 2.0 * torch.arccos(dot)


def quat_mul(a, b):
    """Hamilton product, wxyz, batched (...,4)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def quat_conj(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_rotate(q, v):
    """Rotate vector v (...,3) by quat q (...,4)."""
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]


class AffordanceBank:
    def __init__(self, aff_dir: str, device):
        self.device = device
        self.cand: Dict[str, Dict[str, torch.Tensor]] = {}
        for cls in ("cubeA", "cubeB"):
            p = os.path.join(aff_dir, f"{cls}.npz")
            if os.path.exists(p):
                d = np.load(p)
                self.cand[cls] = {
                    "pos": torch.tensor(d["pos"], dtype=torch.float32, device=device),
                    "quat": torch.tensor(d["quat"], dtype=torch.float32, device=device),
                    "aper": torch.tensor(d["aper"], dtype=torch.float32, device=device),
                }

    def _nearest(self, cls, tcp_p, tcp_q, obj_p, obj_q, lambda_rot):
        """
        Return (a_err (N,), best_idx (N,)) for nearest candidate.
        Candidates are object-frame; map to world via current object pose.
        """
        c = self.cand.get(cls)
        N = tcp_p.shape[0]
        if c is None:
            return (torch.full((N,), 1e3, device=self.device),
                    torch.zeros(N, dtype=torch.long, device=self.device))
        M = c["pos"].shape[0]
        # world candidate positions: obj_p + R(obj_q) @ cand_pos
        cp = c["pos"].unsqueeze(0).expand(N, M, 3)          # (N,M,3)
        oq = obj_q.unsqueeze(1).expand(N, M, 4)
        world_p = obj_p.unsqueeze(1) + quat_rotate(oq, cp)  # (N,M,3)
        e_pos = torch.linalg.norm(world_p - tcp_p.unsqueeze(1), dim=-1)  # (N,M)
        # current tcp in object frame: q_obj^{-1} ⊗ q_tcp
        tcp_obj_q = quat_mul(quat_conj(obj_q), tcp_q)        # (N,4)
        cq = c["quat"].unsqueeze(0).expand(N, M, 4)
        e_rot = quat_geodesic(tcp_obj_q.unsqueeze(1).expand(N, M, 4), cq)  # (N,M)
        a = e_pos + lambda_rot * e_rot
        best = torch.argmin(a, dim=1)
        a_err = a.gather(1, best.unsqueeze(1)).squeeze(1)
        return a_err, best

    def alignment_error(self, cls, tcp_p, tcp_q, obj_p, obj_q, lambda_rot):
        a_err, _ = self._nearest(cls, tcp_p, tcp_q, obj_p, obj_q, lambda_rot)
        return a_err

    def aperture_error(self, cls, width, tcp_p, tcp_q, obj_p, obj_q, lambda_rot):
        c = self.cand.get(cls)
        N = width.shape[0]
        if c is None:
            return torch.zeros(N, device=self.device)
        _, best = self._nearest(cls, tcp_p, tcp_q, obj_p, obj_q, lambda_rot)
        cand_aper = c["aper"][best]                          # (N,)
        return width - cand_aper
