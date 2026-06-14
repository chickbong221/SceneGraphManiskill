"""
Threshold calibration (data-driven, no magic numbers).

Measures the empirical distribution of every continuous quantity over
motion-planning SUCCESS episodes, then sets each bin edge from landmarks +
quantiles per teemo_threshold_calibration.md. Writes teemo/thresholds.json,
consumed by graph_builder at runtime and by viz for matching annotations.

Run AFTER affordance_extract.py (alignment needs the candidate sets).

Usage:
  python -m teemo.calibrate_thresholds \
      --env-id StackCube-v1 --num-episodes 200 \
      --affordance-dir teemo/affordances --out teemo/thresholds.json

This rolls out the env under the motion-planner-equivalent state replay, OR
(simpler) re-collects rollouts by replaying a state .h5. To stay dependency-light
we read the SAME replayed STATE .h5 used for affordances.
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from teemo.affordance_extract import (
    split_raw_pose, gripper_width_from_qpos, object_frame_tcp,
)
from transforms3d.quaternions import quat2mat


# --- geometry helpers -------------------------------------------------------

def quat_geodesic_angle(q1, q2):
    """Angle (rad) between two wxyz quaternions, batched (N,4)."""
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def nearest_candidate_error(tcp_p, tcp_q, obj_p, obj_q, cand_pos, cand_quat,
                            cand_aper, width, lambda_rot):
    """
    For each timestep, compute alignment error `a` to nearest candidate and the
    aperture error `ap` for that candidate.

    cand_* are object-frame candidates (M,*). We map them to world via current
    object pose, then take argmin over M of (e_pos + lambda_rot*e_rot).
    Returns a (T,), ap (T,).
    """
    T = tcp_p.shape[0]
    M = cand_pos.shape[0]
    a_out = np.empty(T, np.float32)
    ap_out = np.empty(T, np.float32)
    for t in range(T):
        R = quat2mat(obj_q[t])
        world_p = (R @ cand_pos.T).T + obj_p[t]            # (M,3)
        # candidate world quat = obj_q ⊗ cand_quat (approx via matrix compose)
        # cheaper: compare orientation error in object frame directly.
        e_pos = np.linalg.norm(world_p - tcp_p[t], axis=-1)            # (M,)
        # express current tcp in object frame, compare to cand_quat
        _, tcp_obj_q = object_frame_tcp(obj_p[t], obj_q[t], tcp_p[t], tcp_q[t])
        e_rot = quat_geodesic_angle(np.tile(tcp_obj_q, (M, 1)), cand_quat)  # (M,)
        a_all = e_pos + lambda_rot * e_rot
        j = int(np.argmin(a_all))
        a_out[t] = a_all[j]
        ap_out[t] = width[t] - cand_aper[j]
    return a_out, ap_out


# --- edge-setting rules -----------------------------------------------------

def quantile_edges(x, qs):
    return [float(np.quantile(x, q)) for q in qs]


def calibrate(traj_path, aff_dir, k_window=5):
    if h5py is None:
        raise RuntimeError("pip install h5py")

    # load affordance candidates per class
    aff = {}
    for cls in ("cubeA", "cubeB"):
        p = os.path.join(aff_dir, f"{cls}.npz")
        if os.path.exists(p):
            d = np.load(p)
            aff[cls] = (d["pos"], d["quat"], d["aper"])

    # accumulators
    D, H = [], []                          # distance, signed height
    A_err, AP_err = [], []                 # alignment, aperture
    e_pos_all, e_rot_all = [], []          # for lambda_rot scale matching
    # per-step deltas and K-window changes
    dist_dK, height_dK, align_dK, aper_dK = [], [], [], []
    stationary_dist_dK, stationary_height_dK = [], []
    fc = []                                # contact force placeholder (if stored)

    # First pass for lambda_rot needs e_pos/e_rot stds -> do a pre-pass with
    # lambda_rot=1, collect stds, then recompute alignment with proper lambda.
    def iter_eps(fn):
        with h5py.File(traj_path, "r") as f:
            for ek in [k for k in f.keys() if k.startswith("traj_")]:
                ep = f[ek]
                obs = ep["obs"] if "obs" in ep else ep
                try:
                    tcp_p, tcp_q = split_raw_pose(np.asarray(obs["extra"]["tcp_pose"]))
                    A_p, A_q = split_raw_pose(np.asarray(obs["extra"]["cubeA_pose"]))
                    B_p, B_q = split_raw_pose(np.asarray(obs["extra"]["cubeB_pose"]))
                    qpos = np.asarray(obs["agent"]["qpos"])
                except KeyError:
                    continue
                fn(tcp_p, tcp_q, A_p, A_q, B_p, B_q, gripper_width_from_qpos(qpos))

    # pre-pass: collect e_pos / e_rot separately (lambda=0 -> a=e_pos; track e_rot)
    def prepass(tcp_p, tcp_q, A_p, A_q, B_p, B_q, width):
        if "cubeA" not in aff:
            return
        cp, cq, ca = aff["cubeA"]
        for t in range(tcp_p.shape[0]):
            R = quat2mat(A_q[t]); world_p = (R @ cp.T).T + A_p[t]
            e_pos = np.linalg.norm(world_p - tcp_p[t], axis=-1)
            _, tcp_obj_q = object_frame_tcp(A_p[t], A_q[t], tcp_p[t], tcp_q[t])
            e_rot = quat_geodesic_angle(np.tile(tcp_obj_q, (cp.shape[0], 1)), cq)
            e_pos_all.append(e_pos.min()); e_rot_all.append(e_rot.min())
    iter_eps(prepass)
    if e_pos_all and e_rot_all and np.std(e_rot_all) > 1e-6:
        lambda_rot = float(np.std(e_pos_all) / np.std(e_rot_all))
    else:
        lambda_rot = 0.05
    print(f"lambda_rot = {lambda_rot:.4f}")

    # main pass
    def mainpass(tcp_p, tcp_q, A_p, A_q, B_p, B_q, width):
        d = np.linalg.norm(tcp_p - A_p, axis=-1)
        h = tcp_p[:, 2] - A_p[:, 2]
        D.extend(d.tolist()); H.extend(h.tolist())

        if "cubeA" in aff:
            cp, cq, ca = aff["cubeA"]
            a_err, ap_err = nearest_candidate_error(
                tcp_p, tcp_q, A_p, A_q, cp, cq, ca, width, lambda_rot)
            A_err.extend(a_err.tolist()); AP_err.extend(ap_err.tolist())
        else:
            a_err = np.zeros_like(d); ap_err = np.zeros_like(d)

        T = d.shape[0]
        for t in range(k_window, T):
            ddist = d[t] - d[t - k_window]
            dheight = h[t] - h[t - k_window]
            dist_dK.append(ddist); height_dK.append(dheight)
            if "cubeA" in aff:
                align_dK.append(a_err[t] - a_err[t - k_window])
                aper_dK.append(ap_err[t] - ap_err[t - k_window])
            # stationary windows: tcp barely moved -> calibrate deadband
            if np.linalg.norm(tcp_p[t] - tcp_p[t - k_window]) < 1e-3:
                stationary_dist_dK.append(abs(ddist))
                stationary_height_dK.append(abs(dheight))
    iter_eps(mainpass)

    D = np.array(D); H = np.array(H)
    out = {"lambda_rot": lambda_rot, "k_window": k_window}

    # --- distance: very-near edge = grasp-range landmark; rest = quantiles ---
    # grasp-range proxy: small percentile of distances when very close
    d_grasp = float(np.quantile(D, 0.05))            # near-contact landmark
    upper = D[D > d_grasp]
    e2, e3, e4 = quantile_edges(upper, [0.33, 0.66, 0.90]) if upper.size else (0.1, 0.2, 0.3)
    out["distance_edges"] = [d_grasp, e2, e3, e4]    # 4 edges -> 5 bins

    # --- height: inner ±half-cube landmark, outer ±90pct ---
    h_level = 0.02                                   # cube half-size (StackCube)
    q90 = float(np.quantile(np.abs(H), 0.90))
    out["height_edges"] = [-q90, -h_level, h_level, q90]

    # --- alignment (abs): aligned edge = small pct, rest quantiles ---
    if A_err:
        A_err = np.array(A_err)
        a_aligned = float(np.quantile(A_err, 0.05))
        up = A_err[A_err > a_aligned]
        a2, a3, a4 = quantile_edges(up, [0.33, 0.66, 0.90]) if up.size else (0.05, 0.1, 0.2)
        out["align_edges"] = [a_aligned, a2, a3, a4]
    else:
        out["align_edges"] = None

    # --- aperture-fit (abs): fit straddles 0, inner = grasp-band, outer ±90 ---
    if AP_err:
        AP_err = np.array(AP_err)
        ap_fit = float(np.std(AP_err[np.abs(AP_err) < np.quantile(np.abs(AP_err), 0.3)]) + 1e-4)
        ap90 = float(np.quantile(np.abs(AP_err), 0.90))
        out["aperture_edges"] = [-ap90, -ap_fit, ap_fit, ap90]
    else:
        out["aperture_edges"] = None

    # --- temporal deadbands: 95pct of |dK| on stationary windows ---
    out["deadband_distance"] = float(np.quantile(stationary_dist_dK, 0.95)) if stationary_dist_dK else 0.004
    out["deadband_height"] = float(np.quantile(stationary_height_dK, 0.95)) if stationary_height_dK else 0.004
    out["deadband_align"] = out["deadband_distance"]      # reuse scale
    out["deadband_aperture"] = 0.002

    # --- temporal speed edges: 50/85 pct of per-step RATE among moving windows ---
    def speed_edges(dK_list, deadband):
        arr = np.abs(np.array(dK_list))
        moving = arr[arr > deadband] / k_window
        if moving.size == 0:
            return [0.005, 0.02]
        return [float(np.quantile(moving, 0.50)), float(np.quantile(moving, 0.85))]
    out["speed_distance"] = speed_edges(dist_dK, out["deadband_distance"])
    out["speed_height"] = speed_edges(height_dK, out["deadband_height"])
    out["speed_align"] = speed_edges(align_dK, out["deadband_align"]) if align_dK else [0.01, 0.04]
    out["speed_aperture"] = speed_edges(aper_dK, out["deadband_aperture"]) if aper_dK else [0.002, 0.008]

    # --- contact force eps: needs stored forces; default floor if absent ---
    out["contact_force_eps"] = float(np.quantile(fc, 0.99)) if fc else 1e-3

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="replayed STATE .h5")
    ap.add_argument("--affordance-dir", default="teemo/affordances")
    ap.add_argument("--k-window", type=int, default=5)
    ap.add_argument("--out", default="teemo/thresholds.json")
    args = ap.parse_args()

    th = calibrate(args.traj, args.affordance_dir, args.k_window)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(th, f, indent=2)
    print(f"wrote {args.out}")
    print(json.dumps(th, indent=2))


if __name__ == "__main__":
    main()
