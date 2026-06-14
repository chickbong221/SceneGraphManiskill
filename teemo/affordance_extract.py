"""
Affordance pose set extraction (draft §3.2), per semantic class, object frame.

Drives the alignment relations (tcp_align, aperture) and their temporal changes.

Pipeline:
  1. Collect SUCCESS episodes with the built-in motion planner:
       python -m mani_skill.examples.motionplanning.panda.run \
           -e StackCube-v1 -n 300 --save-traj --only-count-success
     (motion planner requires control_mode pd_joint_pos.)
  2. Replay to recover per-step STATE (poses, qpos):
       python -m mani_skill.trajectory.replay_trajectory \
           --traj-path demos/StackCube-v1/motionplanning/trajectory.h5 \
           --use-first-env-state -o state --save-traj -b cpu
     -> produces trajectory.state.pd_joint_pos.physx_cpu.h5
  3. Run THIS script on the replayed .h5 to extract object-frame candidates at
     the grasp-onset event (and place/support-onset event), per class.

Output: teemo/affordances/<class>.npz with arrays:
    pos   (M,3)   object-frame TCP position
    quat  (M,4)   object-frame TCP orientation (wxyz)
    aper  (M,)    gripper width at the event (meters)
Optionally k-means-reduced to a handful of representative candidates per class.

This is an OFFLINE script; the live graph builder only loads the .npz.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

# transforms3d is a ManiSkill dependency.
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse


# --- pose helpers (wxyz quaternions, as ManiSkill/SAPIEN use) ---------------

def pose_to_mat(p, q):
    """p (3,), q (4,) wxyz -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = quat2mat(q)
    T[:3, 3] = p
    return T


def mat_to_pose(T):
    return T[:3, 3].copy(), mat2quat(T[:3, :3])


def invert_pose(p, q):
    """Return object-frame transform inverse: world->object."""
    R = quat2mat(q)
    Rinv = R.T
    pinv = -Rinv @ p
    return pinv, mat2quat(Rinv)


def object_frame_tcp(obj_p, obj_q, tcp_p, tcp_q):
    """
    Express TCP pose in the object's frame:  T_obj^{-1} @ T_tcp.
    Returns (pos(3,), quat(4,) wxyz).
    """
    inv_p, inv_q = invert_pose(obj_p, obj_q)
    # compose: T_cand = inv(T_obj) @ T_tcp
    T = pose_to_mat(inv_p, inv_q) @ pose_to_mat(tcp_p, tcp_q)
    return mat_to_pose(T)


# --- raw-pose parsing -------------------------------------------------------
# ManiSkill raw_pose layout is [px,py,pz, qw,qx,qy,qz].

def split_raw_pose(raw):
    raw = np.asarray(raw)
    return raw[..., :3], raw[..., 3:7]


def gripper_width_from_qpos(qpos):
    """Panda: last two qpos entries are finger joints (mimic). width = sum."""
    return qpos[..., -1] + qpos[..., -2]


# --- event detection within an episode -------------------------------------

def find_first_true(mask: np.ndarray):
    """Index of first True, or -1."""
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else -1


def extract_from_episode(ep_grp, obs_grp) -> Dict[str, List[np.ndarray]]:
    """
    Pull object-frame TCP candidates at grasp-onset (and support-onset) for the
    target object (cubeA). Returns dict: class_name -> list of [pos|quat|aper].

    Requires the replayed STATE trajectory which stores, per step, env_states
    or obs containing cubeA_pose, cubeB_pose, tcp_pose, and agent qpos.

    NOTE: exact h5 key paths depend on the replay format. We try the common
    'obs' dict keys first; if your replay nests differently, adjust KEYS below.
    """
    out: Dict[str, List[np.ndarray]] = {"cubeA": [], "cubeB": []}

    # --- pull per-step arrays (adjust keys to your replay layout) ---
    # Expected state-mode obs keys (StackCube _get_obs_extra under state):
    #   extra/tcp_pose      (T,7)
    #   extra/cubeA_pose    (T,7)
    #   extra/cubeB_pose    (T,7)
    #   agent/qpos          (T,nq)
    def get(path):
        node = obs_grp
        for k in path.split("/"):
            node = node[k]
        return np.asarray(node)

    try:
        tcp_raw = get("extra/tcp_pose")
        A_raw = get("extra/cubeA_pose")
        B_raw = get("extra/cubeB_pose")
        qpos = get("agent/qpos")
    except KeyError:
        # Fallback: many replays store a flat 'obs' (T,D). In that case you must
        # know the slice layout; we skip such episodes rather than guess.
        return out

    tcp_p, tcp_q = split_raw_pose(tcp_raw)
    A_p, A_q = split_raw_pose(A_raw)
    B_p, B_q = split_raw_pose(B_raw)
    width = gripper_width_from_qpos(qpos)

    T = tcp_p.shape[0]

    # grasp-onset proxy: gripper has closed AND tcp very near cubeA.
    # We don't have is_grasping stored; approximate via geometry + closed width.
    d_tcp_A = np.linalg.norm(tcp_p - A_p, axis=-1)
    closed = width < 0.039          # below near-fully-open
    near_A = d_tcp_A < 0.03
    grasp_mask = closed & near_A
    g = find_first_true(grasp_mask)
    if g >= 0:
        pos, quat = object_frame_tcp(A_p[g], A_q[g], tcp_p[g], tcp_q[g])
        out["cubeA"].append(np.concatenate([pos, quat, [width[g]]]))

    # support/place-onset proxy: cubeA above cubeB by ~2*half_size and xy-aligned.
    offset = A_p - B_p
    xy = np.linalg.norm(offset[:, :2], axis=-1)
    z = np.abs(offset[:, 2] - 0.04)
    place_mask = (xy < 0.025) & (z < 0.01)
    s = find_first_true(place_mask)
    if s >= 0:
        # candidate expressed relative to the SUPPORT object (cubeB) frame for place
        pos, quat = object_frame_tcp(B_p[s], B_q[s], tcp_p[s], tcp_q[s])
        out["cubeB"].append(np.concatenate([pos, quat, [width[s]]]))

    return out


def maybe_kmeans(arr: np.ndarray, k: int) -> np.ndarray:
    """Reduce M candidates to k representatives. Falls back to all if M<=k."""
    if arr.shape[0] <= k:
        return arr
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(arr[:, :3])
        # take the medoid (nearest real sample) of each cluster on full vector
        reps = []
        for c in range(k):
            members = arr[km.labels_ == c]
            if len(members) == 0:
                continue
            center = members[:, :3].mean(0)
            j = np.argmin(np.linalg.norm(members[:, :3] - center, axis=1))
            reps.append(members[j])
        return np.stack(reps)
    except ImportError:
        # no sklearn: random subsample
        idx = np.random.RandomState(0).choice(arr.shape[0], k, replace=False)
        return arr[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True,
                    help="replayed STATE .h5 (trajectory.state.*.h5)")
    ap.add_argument("--out-dir", default="teemo/affordances")
    ap.add_argument("--kmeans-k", type=int, default=12,
                    help="representatives per class (<=0 keeps all)")
    args = ap.parse_args()

    if h5py is None:
        raise RuntimeError("pip install h5py")

    os.makedirs(args.out_dir, exist_ok=True)
    accum: Dict[str, List[np.ndarray]] = {"cubeA": [], "cubeB": []}

    with h5py.File(args.traj, "r") as f:
        ep_keys = [k for k in f.keys() if k.startswith("traj_")]
        print(f"{len(ep_keys)} episodes in {args.traj}")
        for ek in ep_keys:
            ep = f[ek]
            obs = ep["obs"] if "obs" in ep else ep
            got = extract_from_episode(ep, obs)
            for cls, lst in got.items():
                accum[cls].extend(lst)

    for cls, lst in accum.items():
        if not lst:
            print(f"[warn] no candidates for class {cls}")
            continue
        arr = np.stack(lst).astype(np.float32)
        if args.kmeans_k > 0:
            arr = maybe_kmeans(arr, args.kmeans_k)
        out = os.path.join(args.out_dir, f"{cls}.npz")
        np.savez(out, pos=arr[:, :3], quat=arr[:, 3:7], aper=arr[:, 7])
        print(f"wrote {out}: {arr.shape[0]} candidates")


if __name__ == "__main__":
    main()
