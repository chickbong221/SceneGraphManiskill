"""TEEMO graph self-test (no training, no offline assets required).

Builds a StackCube-v1 env, runs ~20 random-action steps, exercises the full
build_graph pipeline at every step, and asserts the contract held by the rest
of the package:

  - graph["onehot"].shape == (num_envs, GRAPH_DIM)
  - every per-(pair, field) one-hot group sums to exactly 1 on every env
  - every integer target lies in [0, num_classes_for_that_field)
  - GraphHistory's valid mask matches the per-env push count (>= K+1 pushes
    in-episode, accounting for any auto-resets observed during the 20 steps)

Run AFTER deps are installed and BEFORE training, with thresholds_default.json
present and an (optionally empty) teemo/affordances dir:

    python -m teemo.selftest

If affordance .npz files are missing, AffordanceBank falls back to sentinel
values and the alignment/aperture bins will collapse to constants — that is
expected and does not fail this self-test.
"""

from __future__ import annotations

import json
import os
import sys

import torch

import gymnasium as gym
import mani_skill.envs  # noqa: F401  register envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from teemo import vocab
from teemo.graph_builder import (
    GRAPH_DIM,
    SLOT_ORDER,
    TeemoGraphSpec,
    build_graph,
    extract_state,
    extract_z_features,
)
from teemo.history import GraphHistory
from teemo.affordance_use import AffordanceBank


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    th_path = os.path.join(here, "thresholds_default.json")
    aff_dir = os.path.join(here, "affordances")

    with open(th_path) as f:
        th = json.load(f)

    num_envs = 4
    env_id = "StackCube-v1"

    # ManiSkillVectorEnv handles auto-reset semantics consistently with the
    # PPO training loop; we keep terminations active so the validity-mask
    # assertion exercises the just_reset branch as well.
    raw_env = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode="rgb+segmentation",
        control_mode="pd_joint_delta_pos",
    )
    envs = ManiSkillVectorEnv(
        raw_env, num_envs, ignore_terminations=False, record_metrics=False
    )
    base = envs.unwrapped
    device = base.device

    spec = TeemoGraphSpec()
    aff = AffordanceBank(aff_dir, device)  # empty dir => sentinel behavior
    class_embed = torch.nn.Embedding(len(vocab.SEMANTIC_CLASSES), 8).to(device)
    hist = GraphHistory(num_envs, spec.temporal_k, device)

    obs, _ = envs.reset(seed=0)
    seed_state = extract_state(base)
    hist.reset_all()
    hist.push(seed_state, just_reset=torch.ones(num_envs, dtype=torch.bool, device=device))

    # mirror GraphHistory's valid_count for an independent check
    K1 = spec.temporal_k + 1
    expected_count = torch.ones(num_envs, dtype=torch.long, device=device)

    steps = 20
    for step in range(steps):
        action = envs.action_space.sample()
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, device=device)
        obs, rew, term, trunc, info = envs.step(action)
        just_reset = (term.bool() | trunc.bool())

        state = extract_state(base)
        hist.push(state, just_reset=just_reset)
        window, valid = hist.get_window()
        z = extract_z_features(obs, base, spec)
        graph = build_graph(state, z, window, valid, th, aff, spec, class_embed)

        # mirror the validity-count rule
        expected_count = torch.where(
            just_reset,
            torch.ones_like(expected_count),
            torch.minimum(expected_count + 1,
                          torch.tensor(K1, dtype=torch.long, device=device)),
        )
        expected_valid = expected_count >= K1

        # ---- assertions ---------------------------------------------------
        oh = graph["onehot"]
        assert oh.shape == (num_envs, GRAPH_DIM), (
            f"step {step}: onehot shape {tuple(oh.shape)} != ({num_envs}, {GRAPH_DIM})"
        )
        assert torch.isfinite(oh).all(), f"step {step}: non-finite onehot"

        off = 0
        for pkey, fld in SLOT_ORDER:
            nc = vocab.RELATION_NUM_CLASSES[fld]
            group_sum = oh[:, off:off + nc].sum(dim=-1)
            assert torch.allclose(group_sum, torch.ones_like(group_sum)), (
                f"step {step}: slot {pkey}:{fld} one-hot group does not sum to 1; "
                f"got {group_sum.tolist()}"
            )
            off += nc
        assert off == GRAPH_DIM, f"slot tally {off} != GRAPH_DIM {GRAPH_DIM}"

        for slot_key, t in graph["targets"].items():
            fld = slot_key.split(":")[1]
            nc = vocab.RELATION_NUM_CLASSES[fld]
            assert t.dtype == torch.long, f"target {slot_key} not long"
            assert int(t.min().item()) >= 0 and int(t.max().item()) < nc, (
                f"target {slot_key} out of range [0,{nc}): "
                f"min={int(t.min().item())} max={int(t.max().item())}"
            )

        assert (valid == expected_valid).all(), (
            f"step {step}: hist valid mask {valid.tolist()} disagrees with "
            f"expected {expected_valid.tolist()} (counts {expected_count.tolist()})"
        )

        # node features sanity: every node returns the expected shape
        for name in spec.node_names:
            nf = graph["nodes"][name]
            assert nf.shape[0] == num_envs and nf.dim() == 2, (
                f"node {name} shape {tuple(nf.shape)} unexpected"
            )

    print("teemo.selftest PASS")
    print(f"  graph_dim={GRAPH_DIM}  num_slots={len(SLOT_ORDER)}  num_envs={num_envs}")
    envs.close()


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("teemo.selftest FAIL:", e, file=sys.stderr)
        raise
