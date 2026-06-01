"""
Discrete temporal affordance-progress graph for StackCube-v1.

Phase-1 design:
  - The "graph" is oracle privileged supervision. For Phase 1 it is fed to:
      (a) the critic (when --use_graph_critic) and
      (b) an auxiliary classification head on top of the visual/proprio latent
          (when --use_graph_aux). The actor never receives the graph, so the
          policy remains deployable from normal observations only.
  - Node naming is generic ("end_effector", "gripper", "target_object",
    "support_object", "goal") so the same schema can be reused across other
    manipulation tasks via task-specific state extractors.
  - Only discrete labels are exposed. Continuous values are *not* included in
    the graph vector (they live only inside the binning helpers).

Graph schema (total 36-dim one-hot):
  Absolute (15):
    ee_target_dist                  3
    target_goal_xy                  3
    target_goal_z                   3
    gripper_target_grasp            2
    target_support_contact_or_on    2
    target_motion_state             2
  Temporal change over K=5 steps (21):
    ee_target_distance_change       7
    target_goal_distance_change     7
    target_support_z_offset_change  7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch


# -----------------------------------------------------------------------------
# Schema / spec
# -----------------------------------------------------------------------------

# Stable order used everywhere we serialize the graph to a flat one-hot vector.
HEAD_ORDER: Tuple[str, ...] = (
    "ee_target_dist",
    "target_goal_xy",
    "target_goal_z",
    "gripper_target_grasp",
    "target_support_contact_or_on",
    "target_motion_state",
    "ee_target_distance_change",
    "target_goal_distance_change",
    "target_support_z_offset_change",
)

HEAD_DIMS: Dict[str, int] = {
    "ee_target_dist": 3,
    "target_goal_xy": 3,
    "target_goal_z": 3,
    "gripper_target_grasp": 2,
    "target_support_contact_or_on": 2,
    "target_motion_state": 2,
    "ee_target_distance_change": 7,
    "target_goal_distance_change": 7,
    "target_support_z_offset_change": 7,
}

# Class enums (kept as module-level constants for readability)
class DistanceChange:
    IMPROVING_SLOW = 0
    IMPROVING_MID = 1
    IMPROVING_FAST = 2
    WORSENING_SLOW = 3
    WORSENING_MID = 4
    WORSENING_FAST = 5
    STABLE = 6


class SignedChange:
    INCREASING_SLOW = 0
    INCREASING_MID = 1
    INCREASING_FAST = 2
    DECREASING_SLOW = 3
    DECREASING_MID = 4
    DECREASING_FAST = 5
    STABLE = 6


@dataclass
class StackCubeGraphSpec:
    """Static description of the StackCube graph schema."""

    num_nodes: int = 5
    node_names: Tuple[str, ...] = (
        "end_effector",
        "gripper",
        "target_object",
        "support_object",
        "goal",
    )
    graph_dim: int = 36
    temporal_k: int = 5
    head_dims: Dict[str, int] = field(default_factory=lambda: dict(HEAD_DIMS))
    head_order: Tuple[str, ...] = HEAD_ORDER


# -----------------------------------------------------------------------------
# Binning helpers (all operate on batched tensors and return long class indices)
# -----------------------------------------------------------------------------

def onehot(idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(idx.long(), num_classes=num_classes).float()


def bin_distance(d: torch.Tensor, close: float = 0.04, mid: float = 0.12) -> torch.Tensor:
    """3-way bin: 0=close, 1=mid, 2=far."""
    boundaries = torch.tensor([close, mid], device=d.device, dtype=d.dtype)
    return torch.bucketize(d, boundaries).long()


def bin_xy_alignment(d: torch.Tensor, aligned: float = 0.025, near: float = 0.08) -> torch.Tensor:
    """3-way bin: 0=aligned, 1=near, 2=far."""
    boundaries = torch.tensor([aligned, near], device=d.device, dtype=d.dtype)
    return torch.bucketize(d, boundaries).long()


def bin_z_to_goal(z_err: torch.Tensor, tol: float = 0.015) -> torch.Tensor:
    """3-way bin: 0=below_goal, 1=at_goal_height, 2=above_goal."""
    abs_z = torch.abs(z_err)
    above = (z_err > 0).long()
    # base: below=0 / above=2
    out = above * 2
    out = torch.where(abs_z < tol, torch.ones_like(out), out)
    return out


def bool_label(b: torch.Tensor) -> torch.Tensor:
    """2-way bin: 0=false, 1=true."""
    return b.long()


def distance_change_7way(
    delta: torch.Tensor,
    eps: float = 0.003,
    slow: float = 0.01,
    mid: float = 0.03,
) -> torch.Tensor:
    """
    Classify a distance-like potential's signed change over K steps.

    delta = potential_t - potential_{t-K}
      delta < 0  -> improving (distance shrank)
      delta > 0  -> worsening (distance grew)
      |delta| <= eps -> stable

    Magnitude bands within improving/worsening:
      |delta| <= slow  -> slow
      slow < |delta| <= mid  -> mid
      |delta| > mid  -> fast
    """
    abs_d = torch.abs(delta)
    boundaries = torch.tensor([slow, mid], device=delta.device, dtype=delta.dtype)
    mag_cls = torch.bucketize(abs_d, boundaries).long()  # 0/1/2
    is_worsening = (delta > 0).long()
    cls = mag_cls + is_worsening * 3  # 0..2 improving, 3..5 worsening
    cls = torch.where(abs_d <= eps, torch.full_like(cls, DistanceChange.STABLE), cls)
    return cls


def signed_change_7way(
    delta: torch.Tensor,
    eps: float = 0.002,
    slow: float = 0.006,
    mid: float = 0.02,
) -> torch.Tensor:
    """
    Classify a signed potential's change over K steps (e.g. z offset).

    delta > 0  -> increasing
    delta < 0  -> decreasing
    |delta| <= eps -> stable
    """
    abs_d = torch.abs(delta)
    boundaries = torch.tensor([slow, mid], device=delta.device, dtype=delta.dtype)
    mag_cls = torch.bucketize(abs_d, boundaries).long()
    is_decreasing = (delta < 0).long()
    cls = mag_cls + is_decreasing * 3
    cls = torch.where(abs_d <= eps, torch.full_like(cls, SignedChange.STABLE), cls)
    return cls


# -----------------------------------------------------------------------------
# Privileged state extraction (StackCube-specific; generic names internally)
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_stackcube_state(base_env) -> Dict[str, torch.Tensor]:
    """
    Read the privileged state needed to build the StackCube graph.

    Returns a dict keyed by generic node-relation names:
      tcp_pos       [B,3]   end_effector world position
      cubeA_pos     [B,3]   target_object world position
      cubeB_pos     [B,3]   support_object world position
      is_grasped    [B]     gripper_target_grasp
      is_on         [B]     target_support_contact_or_on
      is_static     [B]     target_motion_state (1 = static)

    NOTE: in rgb obs mode the obs extra dict only carries tcp_pose, so we read
    from the env directly. ``base_env.evaluate()`` is read-only and returns
    consistent labels for *all* envs (including ones that just auto-reset),
    which avoids the missing-keys-after-auto-reset issue in info.
    """
    info = base_env.evaluate()
    return {
        "tcp_pos": base_env.agent.tcp.pose.p,
        "cubeA_pos": base_env.cubeA.pose.p,
        "cubeB_pos": base_env.cubeB.pose.p,
        "is_grasped": info["is_cubeA_grasped"],
        "is_on": info["is_cubeA_on_cubeB"],
        "is_static": info["is_cubeA_static"],
    }


# -----------------------------------------------------------------------------
# Temporal history buffer
# -----------------------------------------------------------------------------

class StackCubeGraphHistory:
    """
    Per-env ring buffer over the last K+1 privileged-state snapshots.

    Supports the vectorized auto-reset semantics of ManiSkillVectorEnv: pushing
    a state with ``just_reset[i] = True`` invalidates env i's prior history so
    we never compare across episode boundaries.

    Only stores the position fields needed for temporal labels. Absolute
    boolean labels are computed fresh from the current state each step.
    """

    _FIELDS = ("tcp_pos", "cubeA_pos", "cubeB_pos")

    def __init__(self, num_envs: int, k: int, device):
        self.num_envs = num_envs
        self.k = k
        self.device = device
        self._size = k + 1
        self._head = 0  # index where the next push will be written
        self.valid_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.buffers: Dict[str, torch.Tensor] = {}

    def _maybe_init_buffers(self, state: Dict[str, torch.Tensor]):
        if self.buffers:
            return
        for f in self._FIELDS:
            v = state[f]
            self.buffers[f] = torch.zeros(
                (self._size,) + tuple(v.shape), dtype=v.dtype, device=v.device
            )

    def push(self, state: Dict[str, torch.Tensor], just_reset: torch.Tensor):
        """
        Append a new state snapshot.

        just_reset[i] = True means env i was auto-reset right before producing
        this state, so its previous history is no longer in the same episode
        and must be invalidated.
        """
        self._maybe_init_buffers(state)
        for f in self._FIELDS:
            self.buffers[f][self._head] = state[f]
        self._head = (self._head + 1) % self._size

        size_t = torch.tensor(self._size, dtype=torch.long, device=self.device)
        new_count = torch.minimum(self.valid_count + 1, size_t)
        # For freshly-reset envs, only the just-pushed state is in the new episode.
        new_count = torch.where(just_reset.bool(), torch.ones_like(new_count), new_count)
        self.valid_count = new_count

    def get_prev_k_state(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Return (state pushed k steps before the most recent push, valid_mask).

        valid_mask[i] is True iff env i has at least k+1 in-episode pushes, so
        the prev state is in the same episode as the current state.
        """
        idx_prev = (self._head - 1 - self.k) % self._size
        prev_state = {f: self.buffers[f][idx_prev] for f in self._FIELDS}
        valid_mask = self.valid_count >= (self.k + 1)
        return prev_state, valid_mask

    def reset_all(self):
        self.valid_count.zero_()
        self._head = 0


# -----------------------------------------------------------------------------
# Graph builder
# -----------------------------------------------------------------------------

@torch.no_grad()
def build_stackcube_graph(
    state: Dict[str, torch.Tensor],
    prev_state: Optional[Dict[str, torch.Tensor]] = None,
    prev_valid: Optional[torch.Tensor] = None,
    *,
    z_stack_offset: float = 0.04,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Build the absolute + temporal discrete graph for StackCube-v1.

    Inputs:
      state: dict from ``extract_stackcube_state`` (current time t).
      prev_state: same shape dict at time t-K, or None if no prev available.
      prev_valid: [B] bool. True iff prev_state[i] is in the same episode as
        state[i]. Where False, temporal labels default to "stable".
      z_stack_offset: vertical offset for the derived goal node above
        support_object. For StackCube-v1 this is 2 * cube_half_size = 0.04.

    Returns:
      graph_onehot: [B, 36] float tensor.
      graph_targets: dict[str, [B] long]. Integer class labels for the aux
        classification heads.

    Generic node mapping for StackCube:
      end_effector  <- tcp
      gripper       <- gripper subgraph (gripper_target_grasp edge encodes its
                        per-step grasp relation with the target object)
      target_object <- cubeA (red)
      support_object<- cubeB (green)
      goal          <- support_object_pos + [0, 0, z_stack_offset]
    """
    tcp = state["tcp_pos"]
    A = state["cubeA_pos"]
    B = state["cubeB_pos"]
    device = tcp.device

    # Derived goal node.
    goal = B.clone()
    goal[:, 2] = goal[:, 2] + z_stack_offset

    # --- Absolute labels -----------------------------------------------------
    ee_target_d = torch.linalg.norm(tcp - A, dim=-1)
    ee_target_dist_cls = bin_distance(ee_target_d)

    goal_xy_d = torch.linalg.norm((A - goal)[:, :2], dim=-1)
    target_goal_xy_cls = bin_xy_alignment(goal_xy_d)

    z_err = A[:, 2] - goal[:, 2]
    target_goal_z_cls = bin_z_to_goal(z_err)

    gripper_target_grasp_cls = bool_label(state["is_grasped"])
    target_support_contact_cls = bool_label(state["is_on"])
    target_motion_state_cls = bool_label(state["is_static"])

    # --- Temporal labels -----------------------------------------------------
    B_dim = tcp.shape[0]
    stable_dist = torch.full((B_dim,), DistanceChange.STABLE, dtype=torch.long, device=device)
    stable_signed = torch.full((B_dim,), SignedChange.STABLE, dtype=torch.long, device=device)

    if prev_state is None:
        ee_change_cls = stable_dist
        target_goal_change_cls = stable_dist
        z_change_cls = stable_signed
    else:
        prev_tcp = prev_state["tcp_pos"]
        prev_A = prev_state["cubeA_pos"]
        prev_B = prev_state["cubeB_pos"]

        # ee <-> target distance change
        prev_ee_target = torch.linalg.norm(prev_tcp - prev_A, dim=-1)
        delta_ee = ee_target_d - prev_ee_target
        ee_change_cls = distance_change_7way(delta_ee)

        # target <-> goal distance change (goal_t-K derived from prev_B too)
        prev_goal = prev_B.clone()
        prev_goal[:, 2] = prev_goal[:, 2] + z_stack_offset
        prev_target_goal_d = torch.linalg.norm(prev_A - prev_goal, dim=-1)
        cur_target_goal_d = torch.linalg.norm(A - goal, dim=-1)
        delta_tg = cur_target_goal_d - prev_target_goal_d
        target_goal_change_cls = distance_change_7way(delta_tg)

        # signed target_z - support_z change
        prev_rel_z = prev_A[:, 2] - prev_B[:, 2]
        cur_rel_z = A[:, 2] - B[:, 2]
        delta_z = cur_rel_z - prev_rel_z
        z_change_cls = signed_change_7way(delta_z)

        if prev_valid is not None:
            v = prev_valid.bool()
            ee_change_cls = torch.where(v, ee_change_cls, stable_dist)
            target_goal_change_cls = torch.where(v, target_goal_change_cls, stable_dist)
            z_change_cls = torch.where(v, z_change_cls, stable_signed)

    targets: Dict[str, torch.Tensor] = {
        "ee_target_dist": ee_target_dist_cls,
        "target_goal_xy": target_goal_xy_cls,
        "target_goal_z": target_goal_z_cls,
        "gripper_target_grasp": gripper_target_grasp_cls,
        "target_support_contact_or_on": target_support_contact_cls,
        "target_motion_state": target_motion_state_cls,
        "ee_target_distance_change": ee_change_cls,
        "target_goal_distance_change": target_goal_change_cls,
        "target_support_z_offset_change": z_change_cls,
    }

    parts = [onehot(targets[h], HEAD_DIMS[h]) for h in HEAD_ORDER]
    graph_onehot = torch.cat(parts, dim=-1)
    return graph_onehot, targets


# -----------------------------------------------------------------------------
# Relation proposal (Graph R-CNN-inspired)
# -----------------------------------------------------------------------------

class StackCubeRelationProposal:
    """
    Fixed task-relevant edge proposal for StackCube-v1.

    Inspired by Graph R-CNN's RePN (Relation Proposal Network): instead of
    scoring every O(N^2) object pair, RePN proposes a small set of edges that
    are likely to carry meaningful relations. Here we use a hand-curated set
    of 4 task-relevant edges. A future revision can replace this with a
    learned top-K object-pair scorer driven by node features.
    """

    DEFAULT_EDGES = (
        ("end_effector", "target_object"),
        ("gripper", "target_object"),
        ("target_object", "support_object"),
        ("target_object", "goal"),
    )

    def __init__(self, edges=None):
        self.edges = tuple(edges) if edges is not None else self.DEFAULT_EDGES

    def forward(self, node_features=None, graph_context=None):
        # node_features / graph_context unused in Phase 1; kept for forward
        # compatibility with a learned proposal network.
        return list(self.edges)

    __call__ = forward


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    device = torch.device("cpu")
    # Single-env synthetic check.
    def to_state(tcp, A, B, g=False, on=False, st=False):
        return {
            "tcp_pos": torch.tensor([tcp], dtype=torch.float32, device=device),
            "cubeA_pos": torch.tensor([A], dtype=torch.float32, device=device),
            "cubeB_pos": torch.tensor([B], dtype=torch.float32, device=device),
            "is_grasped": torch.tensor([g], dtype=torch.bool, device=device),
            "is_on": torch.tensor([on], dtype=torch.bool, device=device),
            "is_static": torch.tensor([st], dtype=torch.bool, device=device),
        }

    # Distance improving: prev far from target, current close.
    prev = to_state(tcp=[0.0, 0.0, 0.3], A=[0.0, 0.0, 0.02], B=[0.1, 0.0, 0.02])
    cur = to_state(tcp=[0.0, 0.0, 0.03], A=[0.0, 0.0, 0.02], B=[0.1, 0.0, 0.02], g=True)
    valid = torch.tensor([True], device=device)
    g, t = build_stackcube_graph(cur, prev, valid)
    assert g.shape == (1, 36), g.shape
    # one-hot groups sum to 1
    sums = []
    off = 0
    for h in HEAD_ORDER:
        d = HEAD_DIMS[h]
        sums.append(g[:, off:off + d].sum(dim=-1))
        off += d
    assert torch.allclose(torch.stack(sums), torch.ones_like(torch.stack(sums))), "one-hot groups must sum to 1"
    assert t["ee_target_distance_change"].item() in (0, 1, 2), "expected improving class for shrinking distance"
    assert t["gripper_target_grasp"].item() == 1
    assert t["ee_target_dist"].item() == 0, "tcp very close to cubeA -> close class"

    # Distance worsening
    prev = to_state(tcp=[0.0, 0.0, 0.03], A=[0.0, 0.0, 0.02], B=[0.1, 0.0, 0.02])
    cur = to_state(tcp=[0.0, 0.0, 0.3], A=[0.0, 0.0, 0.02], B=[0.1, 0.0, 0.02])
    g, t = build_stackcube_graph(cur, prev, valid)
    assert t["ee_target_distance_change"].item() in (3, 4, 5), "expected worsening class"

    # Stable distance
    prev = to_state(tcp=[0.0, 0.0, 0.05], A=[0.0, 0.0, 0.02], B=[0.1, 0.0, 0.02])
    cur = to_state(tcp=[0.0, 0.0, 0.0505], A=[0.0, 0.0, 0.02], B=[0.1, 0.0, 0.02])
    g, t = build_stackcube_graph(cur, prev, valid)
    assert t["ee_target_distance_change"].item() == DistanceChange.STABLE

    # Z increasing (target lifted)
    prev = to_state(tcp=[0, 0, 0.05], A=[0.1, 0, 0.02], B=[0.1, 0, 0.02])
    cur = to_state(tcp=[0, 0, 0.05], A=[0.1, 0, 0.08], B=[0.1, 0, 0.02])
    g, t = build_stackcube_graph(cur, prev, valid)
    assert t["target_support_z_offset_change"].item() in (0, 1, 2), "expected increasing class for lifted target"

    # Z decreasing
    prev = to_state(tcp=[0, 0, 0.05], A=[0.1, 0, 0.08], B=[0.1, 0, 0.02])
    cur = to_state(tcp=[0, 0, 0.05], A=[0.1, 0, 0.02], B=[0.1, 0, 0.02])
    g, t = build_stackcube_graph(cur, prev, valid)
    assert t["target_support_z_offset_change"].item() in (3, 4, 5), "expected decreasing class"

    # Invalid prev -> stable for all temporal labels
    invalid = torch.tensor([False], device=device)
    g, t = build_stackcube_graph(cur, prev, invalid)
    for h in ("ee_target_distance_change", "target_goal_distance_change", "target_support_z_offset_change"):
        assert t[h].item() == 6, f"{h} should be stable when prev is invalid"

    # No prev at all -> stable
    g, t = build_stackcube_graph(cur)
    for h in ("ee_target_distance_change", "target_goal_distance_change", "target_support_z_offset_change"):
        assert t[h].item() == 6, f"{h} should be stable when prev_state is None"

    # Class index ranges
    for h, dim in HEAD_DIMS.items():
        assert 0 <= t[h].min().item() and t[h].max().item() < dim

    # Relation proposal
    prop = StackCubeRelationProposal()
    edges = prop()
    assert len(edges) == 4
    assert ("end_effector", "target_object") in edges

    print("stackcube_graph self-test passed.")


if __name__ == "__main__":
    _self_test()
