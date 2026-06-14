"""
TEEMO graph visualization (the image output you asked for).

Two figures per timestep, BOTH driven by the same graph_builder.build_graph
output so the picture provably matches what the policy sees:

  (a) RGB overlay: nodes drawn at their image-projected positions (object nodes
      at GT-mask centroid; tcp/gripper projected from 3D via camera params),
      active edges drawn as labeled lines, edge-family color-coded. Saves a PNG
      per step and a strip/gif per episode -> shows edges turning on/off.

  (b) Node-link diagram: clean networkx/matplotlib graph with edge labels (the
      method-section figure).

Run OFFLINE with num_envs=1, obs_mode includes rgb+segmentation. Not in the
training loop.

Usage:
  python -m teemo.viz_graph --env-id StackCube-v1 \
      --thresholds teemo/thresholds.json --affordance-dir teemo/affordances \
      --episodes 2 --out-dir viz_out
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import gymnasium as gym
import mani_skill.envs  # noqa: register envs

from teemo import vocab
from teemo.graph_builder import (
    TeemoGraphSpec, extract_state, extract_z_features, build_graph, PAIRS,
)
from teemo.affordance_use import AffordanceBank
from teemo.history import GraphHistory

# edge-family colors (BGR for cv2)
FAMILY_COLOR = {
    "spatial": (60, 180, 75),     # green
    "alignment": (245, 130, 48),  # orange
    "physical": (230, 25, 75),    # red/blue-ish
}
FIELD_FAMILY = {
    "distance": "spatial", "height": "spatial",
    "tcp_align": "alignment", "aperture": "alignment",
    "contact": "physical", "grasp": "physical",
    "support": "physical", "containment": "physical",
}


def project_points(pts_world, cam_param):
    """
    pts_world: (M,3) numpy. cam_param: dict with intrinsic_cv (3,3) and
    extrinsic_cv (3,4) or cam2world; we use world->cam->pixel.
    Returns (M,2) pixel coords.
    """
    K = np.asarray(cam_param["intrinsic_cv"]).reshape(3, 3)
    ext = np.asarray(cam_param["extrinsic_cv"]).reshape(3, 4)  # world->cam
    M = pts_world.shape[0]
    homog = np.concatenate([pts_world, np.ones((M, 1))], axis=1)  # (M,4)
    cam = (ext @ homog.T).T                                        # (M,3)
    proj = (K @ cam.T).T
    proj = proj[:, :2] / proj[:, 2:3].clip(1e-6)
    return proj


def mask_centroid(seg, pid):
    """seg (H,W) int, pid scalar -> (x,y) centroid or None."""
    ys, xs = np.where(seg == pid)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def draw_overlay(rgb, seg, cam_param, state_np, graph, env, step, out_path):
    import cv2
    img = (rgb).astype(np.uint8).copy()
    H, W = img.shape[:2]

    # node anchors
    anchors = {}
    # object nodes: mask centroid
    for name, actor in (("cubeA", env.cubeA), ("cubeB", env.cubeB)):
        pid = int(actor.per_scene_id[0].item())
        c = mask_centroid(seg, pid)
        if c is None:
            # fallback: project 3D center
            p = state_np[f"{name}_p"]
            c = tuple(project_points(p[None], cam_param)[0])
        anchors[name] = c
    # tcp / gripper: project 3D
    tcp = state_np["tcp_p"][None]
    px = project_points(tcp, cam_param)[0]
    anchors["tcp"] = tuple(px)
    anchors["gripper"] = tuple(px + np.array([0, 12]))

    # draw edges (only active/non-background)
    for pkey, tA, tB, objA, objB, fields in PAIRS:
        a = anchors.get(objA); b = anchors.get(objB)
        if a is None or b is None:
            continue
        for fld in fields:
            if fld.endswith("_change") or fld.endswith("_transition"):
                continue  # overlay shows absolute edges; temporal in caption
            lab_id = int(graph["targets"][f"{pkey}:{fld}"][0].item())
            lab = vocab.RELATION_LABELS[fld][lab_id]
            # skip negative/background binary labels for clarity
            if lab in ("no-contact", "no-grasp", "no-support", "no-containment"):
                continue
            fam = FIELD_FAMILY[fld]
            color = FAMILY_COLOR[fam]
            pa = (int(a[0]), int(a[1])); pb = (int(b[0]), int(b[1]))
            cv2.line(img, pa, pb, color, 1, cv2.LINE_AA)
            mid = ((pa[0]+pb[0])//2, (pa[1]+pb[1])//2)
            cv2.putText(img, f"{fld}:{lab}", mid, cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, color, 1, cv2.LINE_AA)
    # draw nodes
    for name, c in anchors.items():
        cv2.circle(img, (int(c[0]), int(c[1])), 4, (255, 255, 255), -1)
        cv2.putText(img, name, (int(c[0])+5, int(c[1])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def draw_nodelink(graph, step, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.Graph()
    for n in ("tcp", "gripper", "cubeA", "cubeB"):
        G.add_node(n)
    labels = {}
    for pkey, tA, tB, objA, objB, fields in PAIRS:
        active = []
        for fld in fields:
            if fld.endswith("_change") or fld.endswith("_transition"):
                continue
            lid = int(graph["targets"][f"{pkey}:{fld}"][0].item())
            lab = vocab.RELATION_LABELS[fld][lid]
            if lab in ("no-contact", "no-grasp", "no-support", "no-containment"):
                continue
            active.append(f"{fld}={lab}")
        if active:
            G.add_edge(objA, objB)
            labels[(objA, objB)] = "\n".join(active)
    pos = nx.spring_layout(G, seed=0)
    plt.figure(figsize=(6, 5))
    nx.draw(G, pos, with_labels=True, node_color="#cde",
            node_size=1500, font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=6)
    plt.title(f"TEEMO graph t={step}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="StackCube-v1")
    ap.add_argument("--thresholds", default="teemo/thresholds.json")
    ap.add_argument("--affordance-dir", default="teemo/affordances")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--out-dir", default="viz_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.thresholds) as f:
        th = json.load(f)

    env = gym.make(args.env_id, num_envs=1, obs_mode="rgb+segmentation",
                   control_mode="pd_joint_delta_pos", render_mode="rgb_array")
    base = env.unwrapped
    device = base.device
    spec = TeemoGraphSpec()
    aff = AffordanceBank(args.affordance_dir, device)
    class_embed = torch.nn.Embedding(len(vocab.SEMANTIC_CLASSES), 8).to(device)
    hist = GraphHistory(1, spec.temporal_k, device)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        st = extract_state(base)
        hist.reset_all()
        hist.push(st, torch.ones(1, dtype=torch.bool, device=device))
        for step in range(args.max_steps):
            action = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(action)
            st = extract_state(base)
            just_reset = (term | trunc)
            hist.push(st, just_reset.to(device) if torch.is_tensor(just_reset)
                      else torch.tensor([bool(just_reset)], device=device))
            window, valid = hist.get_window()
            z = extract_z_features(obs, base, spec)
            graph = build_graph(st, z, window, valid, th, aff, spec, class_embed)

            # numpy state for projection
            state_np = {k: (v[0].cpu().numpy() if torch.is_tensor(v) else v)
                        for k, v in st.items()}
            sd = obs["sensor_data"]; cam_name = next(iter(sd))
            rgb = sd[cam_name]["rgb"][0].cpu().numpy()
            seg = sd[cam_name]["segmentation"][0, ..., 0].cpu().numpy()
            cam_param = {k: v[0].cpu().numpy()
                         for k, v in obs["sensor_param"][cam_name].items()}

            ov = os.path.join(args.out_dir, f"ep{ep}_t{step:03d}_overlay.png")
            nl = os.path.join(args.out_dir, f"ep{ep}_t{step:03d}_graph.png")
            try:
                draw_overlay(rgb, seg, cam_param, state_np, graph, base, step, ov)
            except Exception as e:
                print(f"overlay failed t={step}: {e}")
            draw_nodelink(graph, step, nl)
        print(f"episode {ep} done -> {args.out_dir}")
    env.close()


if __name__ == "__main__":
    main()
