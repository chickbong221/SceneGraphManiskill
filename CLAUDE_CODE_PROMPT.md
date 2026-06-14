# Claude Code Task — Integrate full TEEMO graph into ppo_rgb.py (StackCube-v1)

You are working in the repo `SceneGraphManiskill` (a ManiSkill3 source fork).
I am giving you a `teemo/` package (8 files) implementing the FULL TEEMO
spatio-temporal semantic graph from our paper draft. Your job is to (1) place it,
(2) build the affordance + threshold artifacts, (3) wire it into the existing
`examples/baselines/ppo/ppo_rgb.py` as a critic/aux conditioning signal, and
(4) produce visualization figures. Do NOT redesign the graph — the design in
`teemo/` is authoritative and must be preserved exactly.

IMPORTANT CONTEXT — the repo already contains an OLD, REDUCED graph prototype:
`examples/baselines/ppo/stackcube_graph.py`, `graph_modules.py`,
`readme_ppo_graph.md`, and graph imports already in `ppo_rgb.py`. That old
prototype is a 36-dim subset and is NOT what we want. REPLACE its role with the
new `teemo/` package. Keep the old files for reference but stop importing them.

## Ground-truth facts about this repo (already verified — do not re-derive wrong)
- Task `StackCube-v1`, `max_episode_steps=50`, robot `panda_wristcam`.
- Actors: `base_env.cubeA`, `base_env.cubeB`; `cube_half_size=0.02` (goal stack
  offset = 0.04).
- `base_env.evaluate()` returns `is_cubeA_grasped`, `is_cubeA_on_cubeB`,
  `is_cubeA_static`, `success` — read-only, safe to call each step for ALL envs.
- TCP: `base_env.agent.tcp.pose` (.p, .q). Fingers: `agent.finger1_link`,
  `agent.finger2_link`. Gripper width = last two `agent.robot.get_qpos()` entries
  (panda_finger_joint1/2, mimic-coupled, each 0..0.04).
- Contact: `base_env.scene.get_pairwise_contact_forces(a, b)` -> (N,3).
- Segmentation: obs_mode `rgb+segmentation`; per-object mask = seg == `actor.per_scene_id`.
- Camera sensor name is `base_camera`, 128x128.
- Motion-planner solver for StackCube requires control_mode `pd_joint_pos`.
- `ManiSkillVectorEnv` auto-resets inside `step`; use the provided `GraphHistory`
  + `base_env.evaluate()` pattern to avoid crossing episode boundaries. For done
  envs at GAE bootstrap, use a ZERO graph (the old readme used the same choice).

## Step 0 — Place the package
- Copy the `teemo/` directory to repo root (so `import teemo` works) OR into
  `examples/baselines/ppo/teemo/` and run training from that dir. Pick one and be
  consistent. Ensure `teemo/__init__.py` exists.
- `pip install h5py networkx scikit-learn` if missing (calibration/affordance/viz).

## Step 1 — Self-test the graph in isolation (no training)
Write `teemo/selftest.py` that:
- makes `StackCube-v1` with num_envs=4, obs_mode `rgb+segmentation`,
  control_mode `pd_joint_delta_pos`;
- builds a default thresholds dict (use teemo/thresholds_default.json that you
  create with reasonable placeholder edges so the pipeline runs BEFORE calibration);
- steps 20 random actions, each step: extract_state -> push history ->
  get_window -> extract_z_features -> build_graph;
- asserts: `graph["onehot"].shape == (4, teemo.graph_builder.GRAPH_DIM)`; every
  per-(pair,field) one-hot group sums to 1; all targets in [0, num_classes);
  history valid mask is False for first 5 steps then True.
Run it. Fix any real API mismatch in `extract_state`/`extract_z_features` ONLY
(e.g. if `linear_velocity` attribute differs, find the correct one). Do not touch
the graph schema or label logic.

## Step 2 — Build affordance set (per semantic class)
1. Collect success demos:
   `python -m mani_skill.examples.motionplanning.panda.run -e StackCube-v1 -n 200 --save-traj --only-count-success`
   (control_mode pd_joint_pos is set by the solver). Find output .h5 path.
2. Replay to STATE:
   `python -m mani_skill.trajectory.replay_trajectory --traj-path <that.h5> --use-first-env-state -o state --save-traj -b cpu`
3. Run `python -m teemo.affordance_extract --traj <state.h5> --out-dir teemo/affordances --kmeans-k 12`.
   IMPORTANT: the h5 obs key paths in `affordance_extract.extract_from_episode`
   (`extra/tcp_pose`, `extra/cubeA_pose`, `extra/cubeB_pose`, `agent/qpos`) are a
   best guess. INSPECT the actual replayed .h5 with h5py, print the key tree, and
   fix the key paths to match. This is the one place you must adapt to real data.
   Verify `teemo/affordances/cubeA.npz` and `cubeB.npz` are written with sane
   candidate counts.

## Step 3 — Calibrate thresholds (data-driven, replaces all magic numbers)
`python -m teemo.calibrate_thresholds --traj <state.h5> --affordance-dir teemo/affordances --out teemo/thresholds.json`
- Same h5-key caveat: make calibrate's `iter_eps` reader match the real layout
  (reuse whatever you fixed in Step 2 — ideally factor the reader into one shared
  helper).
- Confirm `teemo/thresholds.json` has populated distance_edges, height_edges,
  align_edges, aperture_edges, deadbands, speed_* edges, lambda_rot.
- Add a quick histogram dump (matplotlib) per quantity with edges overlaid, save
  to `teemo/calib_figs/` — this is an appendix figure and a sanity check that no
  bin is empty.

## Step 4 — Visualization figures (the image output)
`python -m teemo.viz_graph --env-id StackCube-v1 --thresholds teemo/thresholds.json --affordance-dir teemo/affordances --episodes 2 --out-dir viz_out`
- Produces per-step RGB overlay PNGs (nodes + active labeled edges) and node-link
  PNGs. Fix camera-projection key names if `intrinsic_cv`/`extrinsic_cv` differ in
  `obs["sensor_param"]["base_camera"]` (print the dict, adapt `project_points`).
- Stitch each episode's overlays into a gif so edges turning on/off (grasp gained,
  support established) are visible. EYEBALL these before any training — they are
  the fastest correctness check on labels/thresholds.

## Step 5 — Wire into ppo_rgb.py (critic + aux, actor clean)
Fork `examples/baselines/ppo/ppo_rgb.py` to `ppo_rgb_teemo.py`. Mirror the OLD
prototype's integration STYLE (it already solved rollout/GAE/minibatch plumbing
and auto-reset) but swap the graph source to `teemo`:
- Build the graph each env step from `teemo.graph_builder.build_graph(...)` using
  a `GraphHistory`, `AffordanceBank`, loaded `thresholds.json`, a `class_embed`
  `nn.Embedding(len(vocab.SEMANTIC_CLASSES), 8)`, and the `TeemoGraphSpec`.
- Store the per-step `graph["onehot"]` (and `graph["targets"]` for aux) in the
  rollout buffers, shape `(num_steps, num_envs, GRAPH_DIM)` and per-target longs.
- Encoder options (CLI flags):
  - `--use_graph_critic`: encode the graph and concat to the CRITIC input only.
    Use `teemo.encoders.RelationGraphEncoder` (preferred, message-passing over the
    structured graph dict) when `--graph_encoder=gnn`, OR a simple MLP over
    `onehot` when `--graph_encoder=mlp` (default mlp for speed; gnn is the paper
    encoder). The ACTOR never receives the graph (keep policy deployable).
  - `--use_graph_aux`: add `teemo.encoders.GraphAuxiliaryHeads` on the obs latent,
    CE loss vs `graph["targets"]`, scaled by `--graph_aux_coef` (default 0.1).
  - `--use_causal_mask`: instantiate `teemo.encoders.CausalRelationMask`, apply its
    mask inside the GNN encoder, add `--mask_l1_coef * l1` to the loss. (Optional;
    only meaningful with `--graph_encoder=gnn`.)
- When all graph flags are False, behavior == stock `ppo_rgb.py`.
- Keep `obs_mode="rgb+segmentation"` so z_i is live. (If GPU memory is tight at
  high num_envs, allow `--no_z` to skip z extraction; z stays zero in schema.)
- Add the same cheap per-iteration sanity asserts the old readme lists (shapes,
  one-hot sums, finite aux loss).

## Step 6 — Experiments (StackCube-v1)
Run, >=3 seeds each, same hyperparameters except the flags:
- A) baseline:        `ppo_rgb_teemo.py --env_id StackCube-v1 --num_envs 256`
- B) +critic (gnn):   `... --use_graph_critic True --graph_encoder gnn`
- C) +critic+aux:     `... --use_graph_critic True --use_graph_aux True --graph_encoder gnn`
- D) +causal mask:    `... --use_graph_critic True --use_graph_aux True --use_causal_mask True --graph_encoder gnn`
Plot eval success-rate vs env steps for A–D. Also overlay state-PPO (existing
`ppo.py`) as a dotted privileged upper bound for context.
Decision rule: if B/C don't beat A on StackCube, report it — that's a clean
negative result about the oracle graph's value to the critic.

## Constraints / do-not
- Do NOT change any label vocabulary, eligibility table, K=5 logic, or transition
  rule in `teemo/`. They are the paper contract (`teemo/vocab.py`).
- Do NOT feed the graph to the actor.
- Do NOT invent thresholds — they come from `calibrate_thresholds`.
- The ONLY places you adapt to reality are: h5 obs key paths (Steps 2-3), camera
  param key names (Step 4), and any ManiSkill attribute that differs from the
  facts above (Step 1). Everything else should work as written.

Deliverables: working `selftest.py` (passes), `teemo/affordances/*.npz`,
`teemo/thresholds.json`, `viz_out/` figures + gifs, `ppo_rgb_teemo.py`, and the
A–D training curves.
