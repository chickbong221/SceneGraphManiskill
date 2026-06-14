"""
TEEMO spatio-temporal semantic graph builder (full draft design), StackCube-v1.

Faithful to draft §3 + Tables 1-2:
  - Persistent nodes with feature [type, class_emb_id, tau, z_i(masked RGB)].
  - Absolute relations: distance, height (5-way); tcp_align, aperture (5-way via
    affordance set); contact, grasp, support (2-way). containment DEFER(const).
  - Temporal relations over K=5 with the draft transition rule (window-stable
    debounce for binary; signed magnitude bins for continuous).
  - Eligibility enforced against vocab.ELIGIBLE_PAIRS.

GT oracle demo: relations from privileged state; z_i from GT segmentation mask.
tau = 0 always (oracle persistence). No goal node.

Outputs per step (batched, num_envs leading dim):
  nodes:   dict of node-feature tensors
  rel:     dict[field] -> (num_envs,) long class id, one entry per eligible pair
  onehot:  (num_envs, GRAPH_DIM) concatenated one-hots (for critic/aux)
  targets: dict[field] -> (num_envs,) long  (for aux CE heads)

Pairs in this StackCube realization (instantiating the eligible-pair table):
  tcp-cubeA, tcp-cubeB                : distance, height, tcp_align, *_change
  gripper-cubeA, gripper-cubeB        : aperture, contact, grasp, aperture_change,
                                        contact_transition, grasp_transition
  cubeA-cubeB                         : contact, support, containment(const),
                                        contact_transition, support_transition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from teemo import vocab
from teemo.affordance_use import AffordanceBank   # see affordance_use.py


# ---------------------------------------------------------------------------
# Pair instantiation for StackCube (object instances bound to eligible types)
# ---------------------------------------------------------------------------
# Each entry: (pair_key, typeA, typeB, objA, objB, relation_fields)
# objA/objB index into the node set by name.
PAIRS: List[Tuple[str, str, str, str, str, Tuple[str, ...]]] = [
    ("tcp-cubeA", "tcp", "object", "tcp", "cubeA",
        ("distance", "height", "tcp_align",
         "distance_change", "height_change", "alignment_change")),
    ("tcp-cubeB", "tcp", "object", "tcp", "cubeB",
        ("distance", "height", "tcp_align",
         "distance_change", "height_change", "alignment_change")),
    ("grip-cubeA", "gripper", "object", "gripper", "cubeA",
        ("aperture", "contact", "grasp",
         "aperture_change", "contact_transition", "grasp_transition")),
    ("grip-cubeB", "gripper", "object", "gripper", "cubeB",
        ("aperture", "contact", "grasp",
         "aperture_change", "contact_transition", "grasp_transition")),
    ("cubeA-cubeB", "object", "object", "cubeA", "cubeB",
        ("contact", "support", "containment",
         "contact_transition", "support_transition", "containment_transition")),
]


def _assert_eligibility():
    for pkey, tA, tB, _, _, fields in PAIRS:
        for fld in fields:
            allowed = vocab.ELIGIBLE_PAIRS[fld]
            assert (tA, tB) in allowed or (tB, tA) in allowed, \
                f"pair {pkey} type ({tA},{tB}) not eligible for relation {fld}"


_assert_eligibility()

# Stable serialization order: every (pair, field) absolute+temporal slot.
SLOT_ORDER: List[Tuple[str, str]] = []
for pkey, *_rest, fields in PAIRS:
    for fld in fields:
        SLOT_ORDER.append((pkey, fld))

GRAPH_DIM = sum(vocab.RELATION_NUM_CLASSES[fld] for _, fld in SLOT_ORDER)


@dataclass
class TeemoGraphSpec:
    node_names: Tuple[str, ...] = ("tcp", "gripper", "cubeA", "cubeB")
    node_types: Tuple[str, ...] = ("tcp", "gripper", "object", "object")
    node_classes: Tuple[str, ...] = ("none", "none", "cubeA", "cubeB")
    temporal_k: int = 5
    graph_dim: int = GRAPH_DIM
    slot_order: Tuple[Tuple[str, str], ...] = tuple(SLOT_ORDER)
    cube_half_size: float = 0.02
    z_feat_dim: int = 3      # mean-pooled RGB per object node (live)


# ---------------------------------------------------------------------------
# Binning (consumes calibrated thresholds dict)
# ---------------------------------------------------------------------------

def _bucketize(x, edges):
    b = torch.as_tensor(edges, device=x.device, dtype=x.dtype)
    return torch.bucketize(x, b).long()


def bin_distance(d, th):
    return _bucketize(d, th["distance_edges"])           # 5 bins


def bin_height(h, th):
    return _bucketize(h, th["height_edges"])             # 5 bins


def bin_align(a, th):
    if th.get("align_edges") is None:
        return torch.zeros_like(a).long()
    return _bucketize(a, th["align_edges"])              # 5 bins


def bin_aperture(ap, th):
    if th.get("aperture_edges") is None:
        return torch.full_like(ap, 2).long()             # "fit" sentinel
    return _bucketize(ap, th["aperture_edges"])          # 5 bins


def cont_change_7way(delta, deadband, speed_edges, improve_is_negative=True):
    """
    Map signed K-window change to the shared 7-way code.
    improve_is_negative: distance/alignment/aperture improve when delta<0;
                         height "up" when delta>0 (set False there).
    Returns long class ids using vocab CONT_* encoding.
    """
    absd = torch.abs(delta)
    rate = absd / 5.0
    se = torch.as_tensor(speed_edges, device=delta.device, dtype=delta.dtype)
    speed = torch.bucketize(rate, se).long()             # 0/1/2
    stable = absd <= deadband
    if improve_is_negative:
        improving = delta < 0
    else:
        improving = delta > 0
    cls = torch.where(improving, speed, speed + 4)       # 0-2 improve, 4-6 worsen
    cls = torch.where(stable, torch.full_like(cls, vocab.CONT_STABLE), cls)
    return cls


# ---------------------------------------------------------------------------
# Privileged state extraction (StackCube) — GT, works in any obs mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_state(base_env) -> Dict[str, torch.Tensor]:
    info = base_env.evaluate()                            # read-only, all envs
    agent = base_env.agent
    return {
        "tcp_p": agent.tcp.pose.p,
        "tcp_q": agent.tcp.pose.q,
        "cubeA_p": base_env.cubeA.pose.p,
        "cubeA_q": base_env.cubeA.pose.q,
        "cubeB_p": base_env.cubeB.pose.p,
        "cubeB_q": base_env.cubeB.pose.q,
        "cubeA_vel": base_env.cubeA.linear_velocity,
        "gripper_width": agent.robot.get_qpos()[..., -1] + agent.robot.get_qpos()[..., -2],
        "is_grasped": info["is_cubeA_grasped"],
        "is_on": info["is_cubeA_on_cubeB"],
        "contact_AB": base_env.scene.get_pairwise_contact_forces(
            base_env.cubeA, base_env.cubeB).norm(dim=-1),
        "contact_gA": base_env.scene.get_pairwise_contact_forces(
            base_env.agent.finger1_link, base_env.cubeA).norm(dim=-1),
        "contact_gB": base_env.scene.get_pairwise_contact_forces(
            base_env.agent.finger1_link, base_env.cubeB).norm(dim=-1),
    }


@torch.no_grad()
def extract_z_features(obs, base_env, spec: TeemoGraphSpec) -> Dict[str, torch.Tensor]:
    """
    Live masked-RGB evidence z_i per OBJECT node from GT segmentation.
    Requires obs_mode to include segmentation; returns mean-pooled RGB (N,3).
    If segmentation absent, returns zeros (z stays in schema, inert).
    """
    device = base_env.device
    N = base_env.num_envs
    zeros = {n: torch.zeros(N, spec.z_feat_dim, device=device)
             for n in ("cubeA", "cubeB")}
    sd = obs.get("sensor_data") if isinstance(obs, dict) else None
    if not sd:
        return zeros
    cam = next(iter(sd.values()))
    if "segmentation" not in cam or "rgb" not in cam:
        return zeros
    seg = cam["segmentation"]            # (N,H,W,1) int16
    rgb = cam["rgb"].float() / 255.0     # (N,H,W,3)
    out = {}
    for name, actor in (("cubeA", base_env.cubeA), ("cubeB", base_env.cubeB)):
        pid = actor.per_scene_id          # (N,) per-env id
        mask = (seg[..., 0] == pid.view(N, 1, 1))     # (N,H,W)
        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=(1, 2)).clamp_min(1.0)
        pooled = (rgb * m).sum(dim=(1, 2)) / denom    # (N,3)
        out[name] = pooled
    return out


# ---------------------------------------------------------------------------
# Node features  x_i = [type_onehot(3), class_id, tau, z_i]
# ---------------------------------------------------------------------------

def build_node_features(state, z_feats, spec: TeemoGraphSpec, class_embed: torch.nn.Embedding):
    N = state["tcp_p"].shape[0]
    device = state["tcp_p"].device
    feats = {}
    pos = {"tcp": state["tcp_p"], "gripper": state["tcp_p"],
           "cubeA": state["cubeA_p"], "cubeB": state["cubeB_p"]}
    for name, ntype, ncls in zip(spec.node_names, spec.node_types, spec.node_classes):
        type_id = vocab.NODE_TYPE_ID[ntype]
        type_oh = torch.zeros(N, len(vocab.NODE_TYPES), device=device)
        type_oh[:, type_id] = 1.0
        cls_id = torch.full((N,), vocab.SEMANTIC_CLASS_ID[ncls], device=device, dtype=torch.long)
        cls_emb = class_embed(cls_id)                       # (N, emb)
        tau = torch.zeros(N, 1, device=device)              # oracle persistence
        if name in z_feats:
            z = z_feats[name]
        else:
            z = torch.zeros(N, spec.z_feat_dim, device=device)
        feats[name] = torch.cat([type_oh, cls_emb, tau, z, pos[name]], dim=-1)
    return feats


# ---------------------------------------------------------------------------
# Absolute relation labels
# ---------------------------------------------------------------------------

@torch.no_grad()
def absolute_labels(state, th, aff: AffordanceBank, spec) -> Dict[str, torch.Tensor]:
    tcp_p, tcp_q = state["tcp_p"], state["tcp_q"]
    A_p, A_q = state["cubeA_p"], state["cubeA_q"]
    B_p, B_q = state["cubeB_p"], state["cubeB_q"]
    w = state["gripper_width"]
    eps = th["contact_force_eps"]
    out = {}

    for obj, (op, oq) in (("cubeA", (A_p, A_q)), ("cubeB", (B_p, B_q))):
        d = torch.linalg.norm(tcp_p - op, dim=-1)
        h = tcp_p[:, 2] - op[:, 2]
        out[f"tcp-{obj}:distance"] = bin_distance(d, th)
        out[f"tcp-{obj}:height"] = bin_height(h, th)
        a_err = aff.alignment_error(obj, tcp_p, tcp_q, op, oq, th["lambda_rot"])
        out[f"tcp-{obj}:tcp_align"] = bin_align(a_err, th)
        ap_err = aff.aperture_error(obj, w, tcp_p, tcp_q, op, oq, th["lambda_rot"])
        out[f"grip-{obj}:aperture"] = bin_aperture(ap_err, th)

    out["grip-cubeA:contact"] = (state["contact_gA"] > eps).long()
    out["grip-cubeB:contact"] = (state["contact_gB"] > eps).long()
    out["grip-cubeA:grasp"] = state["is_grasped"].long()
    out["grip-cubeB:grasp"] = torch.zeros_like(state["is_grasped"]).long()

    out["cubeA-cubeB:contact"] = (state["contact_AB"] > eps).long()
    # support: contact + vertical order + xy-align + stable (draft definition)
    offset = A_p - B_p
    xy = torch.linalg.norm(offset[:, :2], dim=-1)
    z = torch.abs(offset[:, 2] - 2 * spec.cube_half_size)
    stable = torch.linalg.norm(state["cubeA_vel"], dim=-1) < 1e-2
    support = (state["contact_AB"] > eps) & (xy < 0.025) & (z < 0.01) & stable
    out["cubeA-cubeB:support"] = support.long()
    out["cubeA-cubeB:containment"] = torch.zeros_like(support).long()  # DEFER const
    return out


# ---------------------------------------------------------------------------
# Temporal relation labels (K=5 window-stability debounce)
# ---------------------------------------------------------------------------

@torch.no_grad()
def temporal_labels(cur, prev_window, valid, th, aff, spec) -> Dict[str, torch.Tensor]:
    """
    cur: current state dict.
    prev_window: dict field -> tensor (K+1, N, ...) ring of past states (oldest..newest)
                 or None.
    valid: (N,) bool, True iff full K+1 in-episode history available.
    Continuous change uses net delta over K; binary uses window-stability rule.
    """
    N = cur["tcp_p"].shape[0]
    device = cur["tcp_p"].device
    stable_cont = torch.full((N,), vocab.CONT_STABLE, dtype=torch.long, device=device)
    out = {}

    def cont(fld, cur_val, k_ago_val, deadband, speed, neg):
        if prev_window is None:
            return stable_cont.clone()
        delta = cur_val - k_ago_val
        cls = cont_change_7way(delta, deadband, speed, improve_is_negative=neg)
        return torch.where(valid, cls, stable_cont)

    if prev_window is not None:
        kp = {f: prev_window[f][0] for f in prev_window}   # K steps ago = oldest
    else:
        kp = None

    for obj in ("cubeA", "cubeB"):
        op = cur[f"{obj}_p"]; oq = cur[f"{obj}_q"]
        d = torch.linalg.norm(cur["tcp_p"] - op, dim=-1)
        h = cur["tcp_p"][:, 2] - op[:, 2]
        if kp is not None:
            pop = kp[f"{obj}_p"]
            pd = torch.linalg.norm(kp["tcp_p"] - pop, dim=-1)
            ph = kp["tcp_p"][:, 2] - pop[:, 2]
        else:
            pd = d; ph = h
        out[f"tcp-{obj}:distance_change"] = cont(
            "distance_change", d, pd, th["deadband_distance"], th["speed_distance"], neg=True)
        out[f"tcp-{obj}:height_change"] = cont(
            "height_change", h, ph, th["deadband_height"], th["speed_height"], neg=False)
        # alignment change
        a_cur = aff.alignment_error(obj, cur["tcp_p"], cur["tcp_q"], op, oq, th["lambda_rot"])
        if kp is not None:
            a_prev = aff.alignment_error(obj, kp["tcp_p"], kp["tcp_q"], pop, kp[f"{obj}_q"], th["lambda_rot"])
        else:
            a_prev = a_cur
        out[f"tcp-{obj}:alignment_change"] = cont(
            "alignment_change", a_cur, a_prev, th["deadband_align"], th["speed_align"], neg=True)
        # aperture change
        ap_cur = aff.aperture_error(obj, cur["gripper_width"], cur["tcp_p"], cur["tcp_q"], op, oq, th["lambda_rot"])
        if kp is not None:
            ap_prev = aff.aperture_error(obj, kp["gripper_width"], kp["tcp_p"], kp["tcp_q"], pop, kp[f"{obj}_q"], th["lambda_rot"])
        else:
            ap_prev = ap_cur
        out[f"grip-{obj}:aperture_change"] = cont(
            "aperture_change", ap_cur, ap_prev, th["deadband_aperture"], th["speed_aperture"], neg=True)

    # --- binary transitions with window-stability debounce ---
    out.update(_binary_transition("grip-cubeA", "contact_transition",
                                  _binary_trace(prev_window, cur, "contact_gA", th), valid, N, device))
    out.update(_binary_transition("grip-cubeB", "contact_transition",
                                  _binary_trace(prev_window, cur, "contact_gB", th), valid, N, device))
    out.update(_binary_transition("cubeA-cubeB", "contact_transition",
                                  _binary_trace(prev_window, cur, "contact_AB", th), valid, N, device))
    out.update(_binary_transition("grip-cubeA", "grasp_transition",
                                  _binary_trace_bool(prev_window, cur, "is_grasped"), valid, N, device))
    out.update(_binary_transition("grip-cubeB", "grasp_transition",
                                  None, valid, N, device, const=vocab.TR_MAINTAIN_NO))
    out.update(_binary_transition("cubeA-cubeB", "support_transition",
                                  _binary_trace_bool(prev_window, cur, "is_on"), valid, N, device))
    out.update(_binary_transition("cubeA-cubeB", "containment_transition",
                                  None, valid, N, device, const=vocab.TR_MAINTAIN_NO))
    return out


def _binary_trace(prev_window, cur, force_field, th):
    """Return (K+1, N) bool trace from contact forces, or None."""
    if prev_window is None or force_field not in prev_window:
        return None
    eps = th["contact_force_eps"]
    past = prev_window[force_field] > eps          # (K+1, N) but newest is cur
    return past


def _binary_trace_bool(prev_window, cur, field):
    if prev_window is None or field not in prev_window:
        return None
    return prev_window[field].bool()


def _binary_transition(pkey, fld, trace, valid, N, device, const=None):
    key = f"{pkey}:{fld}"
    if const is not None or trace is None:
        c = vocab.TR_MAINTAIN_NO if const is None else const
        return {key: torch.full((N,), c, dtype=torch.long, device=device)}
    # trace: (K+1, N) bool, oldest..newest
    start_stable = trace[0] == trace[1]
    end_stable = trace[-2] == trace[-1]
    both = start_stable & end_stable & valid
    b0 = trace[0]; b1 = trace[-1]
    cls = torch.full((N,), vocab.TR_MAINTAIN_NO, dtype=torch.long, device=device)
    cls = torch.where(b0 & b1, torch.full_like(cls, vocab.TR_MAINTAIN), cls)
    cls = torch.where(~b0 & b1, torch.full_like(cls, vocab.TR_GAIN), cls)
    cls = torch.where(b0 & ~b1, torch.full_like(cls, vocab.TR_LOSE), cls)
    # unstable windows -> excluded; we encode as MAINTAIN_NO (background) + a mask
    cls = torch.where(both, cls, torch.full_like(cls, vocab.TR_MAINTAIN_NO))
    return {key: cls}


# ---------------------------------------------------------------------------
# Serialize to one-hot + targets
# ---------------------------------------------------------------------------

def serialize(abs_labels, temp_labels, device, N):
    targets = {}
    parts = []
    for pkey, fld in SLOT_ORDER:
        key = f"{pkey}:{fld}"
        lab = abs_labels.get(key, temp_labels.get(key))
        nc = vocab.RELATION_NUM_CLASSES[fld]
        oh = torch.nn.functional.one_hot(lab.long(), nc).float()
        parts.append(oh)
        targets[key] = lab.long()
    return torch.cat(parts, dim=-1), targets


@torch.no_grad()
def build_graph(state, z_feats, prev_window, valid, th, aff, spec, class_embed):
    abs_l = absolute_labels(state, th, aff, spec)
    temp_l = temporal_labels(state, prev_window, valid, th, aff, spec)
    N = state["tcp_p"].shape[0]
    onehot, targets = serialize(abs_l, temp_l, state["tcp_p"].device, N)
    nodes = build_node_features(state, z_feats, spec, class_embed)
    return {"nodes": nodes, "abs": abs_l, "temp": temp_l,
            "onehot": onehot, "targets": targets}
