# TEEMO integration notes

Record of what was done to wire the TEEMO spatio-temporal semantic graph
package into the ManiSkill PPO RGB baseline (StackCube-v1), under the
"code-revision-and-integration, no execution" mode.

Scope cut from the original `CLAUDE_CODE_PROMPT.md`:
- DONE in this session: Steps 0, 1, 5 (PPO wiring) plus a placeholder
  `thresholds_default.json`.
- NOT DONE (deferred until deps are installed + experiments are wanted):
  Step 2 (affordance extraction), Step 3 (threshold calibration),
  Step 4 (visualization figures + gifs), Step 6 (A/B/C/D training sweep).

Nothing in `teemo/` was redesigned. The graph schema, vocabulary,
eligibility table, K=5 logic, transition rule, and encoder interfaces are
preserved exactly.

---

## 1. Files placed or created

| Path | Purpose |
| --- | --- |
| `teemo/__init__.py` | Module marker. Imports `vocab`. Lists runtime + offline-only deps in a docstring. |
| `teemo/thresholds_default.json` | Placeholder bin edges (no calibration). Keys match the schema produced by `calibrate_thresholds.py` so it is a drop-in until you replace it. |
| `teemo/selftest.py` | Static-shape + label-range + history-validity self test against a live env. NOT RUN in this session. |
| `examples/baselines/ppo/ppo_rgb_teemo.py` | Fork of `ppo_rgb.py` with teemo as the graph source. The old prototype (`stackcube_graph.py`, `graph_modules.py`) is no longer imported. |
| `teemo/INTEGRATION_NOTES.md` | This file. |

The old `stackcube_graph.py`, `graph_modules.py`, `readme_ppo_graph.md`
were left in place for reference but are dormant — nothing imports them.

---

## 2. Placeholder thresholds (`thresholds_default.json`)

Values keyed to the verified StackCube facts (`cube_half_size=0.02`,
`max_episode_steps=50`, ~10 cm workspace reach). They follow the schema
produced by `teemo/calibrate_thresholds.py` so the same JSON is consumed
by `graph_builder` either way:

| Key | Placeholder | Rationale |
| --- | --- | --- |
| `k_window` | 5 | Matches `TeemoGraphSpec.temporal_k`. |
| `lambda_rot` | 0.05 | Rough meter/radian scale match in alignment metric. |
| `distance_edges` | [0.025, 0.05, 0.10, 0.20] | very-near ≈ grasp range; rest log-ish spacing across reach. |
| `height_edges` | [-0.10, -0.02, 0.02, 0.10] | Inner band = ±cube_half_size; outer = ±reach. |
| `align_edges` | [0.02, 0.05, 0.10, 0.20] | meters + λ·rad ≈ same scale family as distance. |
| `aperture_edges` | [-0.03, -0.005, 0.005, 0.03] | Tight band around fit (half a gripper closing). |
| `deadband_*` | 0.004 / 0.003 / 0.005 / 0.002 | Small enough to mark "stable" only when truly stationary. |
| `speed_*` | [slow, med] per relation | per-step rate cutoffs (`rate = abs(delta)/K`). |
| `contact_force_eps` | 0.001 N | Floor (matches the calibrate fallback). |

Replace with `teemo/thresholds.json` once
`python -m teemo.calibrate_thresholds` has run on a replayed state .h5.

---

## 3. `teemo/selftest.py`

Behavior:
- `gym.make("StackCube-v1", num_envs=4, obs_mode="rgb+segmentation",
  control_mode="pd_joint_delta_pos")`, wrapped in `ManiSkillVectorEnv`.
- Loads `teemo/thresholds_default.json`. Loads `AffordanceBank` from
  `teemo/affordances/` — if the dir is empty the bank uses sentinel
  values (alignment large, aperture zero) and the corresponding bins
  collapse to constants. That is expected and does not fail the test.
- Seeds the history with the post-reset state. Steps 20 random actions.
- At every step builds the graph and asserts:
  - `graph["onehot"].shape == (num_envs, GRAPH_DIM)` and finite.
  - Every per-(pair,field) one-hot group sums to exactly 1 per env.
  - Every integer target ∈ [0, num_classes for that field).
  - `GraphHistory.valid` matches an independently-mirrored per-env
    push count (`valid_count >= K+1`). The mirror tracks
    `just_reset`, so partial auto-resets within the 20-step window
    do not break the assertion.
  - Each node feature tensor is 2-D with batch dim = num_envs.

Run: `python -m teemo.selftest` (after deps are installed).

---

## 4. `ppo_rgb_teemo.py` — what changed vs. `ppo_rgb.py`

### 4.1 Imports
Replaced the old prototype imports with:
```python
from teemo import vocab
from teemo.graph_builder import (
    GRAPH_DIM, PAIRS, SLOT_ORDER, TeemoGraphSpec,
    build_graph, extract_state, extract_z_features,
)
from teemo.history import GraphHistory
from teemo.affordance_use import AffordanceBank
from teemo.encoders import (
    CausalRelationMask, GraphAuxiliaryHeads, RelationGraphEncoder,
)
```
A `sys.path.insert(0, <repo_root>)` is done at the top so the file can
be launched from `examples/baselines/ppo/` and still resolve
`import teemo`.

### 4.2 New CLI flags (extend `Args`)
| Flag | Default | Purpose |
| --- | --- | --- |
| `--use_graph_critic` | False | Concat graph latent to the critic input. Actor is unchanged. |
| `--use_graph_aux` | False | Per-relation CE heads on the obs latent (`GraphAuxiliaryHeads`). |
| `--use_causal_mask` | False | Gumbel-Sigmoid binary mask over relations + L1 penalty. Only meaningful with `--use_graph_critic --graph_encoder gnn`; the script warns and ignores otherwise. |
| `--graph_encoder` | `mlp` | `mlp` (over `graph["onehot"]`, fast) or `gnn` (paper: `RelationGraphEncoder` message passing over the structured dict). |
| `--graph_aux_coef` | 0.1 | Scale on aux CE loss. |
| `--mask_l1_coef` | 1e-3 | Scale on the mask L1 sparsity penalty. |
| `--graph_hidden` | 128 | Encoder hidden dim. |
| `--graph_layers` | 2 | GNN message-passing layers. |
| `--graph_class_emb_dim` | 8 | Semantic-class embedding dim. |
| `--no_z` | False | Skip masked-RGB evidence; z stays zero in the node schema. |
| `--affordance_dir` | `<repo>/teemo/affordances` | Per-class .npz dir for `AffordanceBank`. |
| `--thresholds_path` | `<repo>/teemo/thresholds_default.json` | JSON edges for `graph_builder`. |

Default value of `env_id` was switched to `StackCube-v1`; default
`num_envs` to 256; wandb group renamed to `PPO-TEEMO`. With all graph
flags off, behavior is otherwise identical to stock `ppo_rgb.py`.

### 4.3 `SidecarSensorWrapper`
`FlattenRGBDObservationWrapper.observation()` pops `sensor_data` and
extracts only `rgb`/`depth`. Segmentation is dropped, so
`extract_z_features` would always see no sensor data and return zeros.
To keep z live, a tiny `gym.ObservationWrapper` is inserted **between**
the base env and `FlattenRGBDObservationWrapper`:

```python
class SidecarSensorWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        if isinstance(observation, dict) and "sensor_data" in observation:
            self.unwrapped._teemo_last_sensor_data = observation["sensor_data"]
        return observation
```

It only stashes a reference; `FlattenRGBDObservationWrapper.observation`
then pops `sensor_data` from the outer dict, but the dict it pointed to
remains alive via the sidecar's reference. The training loop builds a
`{"sensor_data": ...}` view from `base_env._teemo_last_sensor_data` and
hands it to `extract_z_features`. Order of wrappers:

```
gym.make(env_id, obs_mode="rgb+segmentation", ...)
  -> SidecarSensorWrapper       (only if use_graph and not no_z)
  -> FlattenRGBDObservationWrapper
  -> (optional FlattenActionSpaceWrapper)
  -> ManiSkillVectorEnv
```

With `--no_z` or when no graph flag is set, the wrapper is skipped and
`obs_mode` falls back to `"rgb"`.

### 4.4 Graph state plumbing
- `TeemoGraphSpec()` instantiated when `use_graph_critic or use_graph_aux`.
- `class_embed = nn.Embedding(len(vocab.SEMANTIC_CLASSES), graph_class_emb_dim)`
  with `requires_grad_(False)`. Lives OUTSIDE the agent (it is just a
  lookup table; rollout stores already-embedded node features, so a
  gradient through the embedding would not flow anyway).
- `AffordanceBank(affordance_dir, device)` — empty-dir tolerant.
- `thresholds = json.load(thresholds_path)`.
- `GraphHistory(num_envs, spec.temporal_k, device)`.
- Initial seed: `extract_state(envs.unwrapped)` → push with
  `just_reset=all-True`. One dummy `build_graph` call discovers
  `node_feat_dim = next(iter(init_graph["nodes"].values())).shape[-1]`.
- Rollout buffers allocated once `node_feat_dim` is known:
  - `graphs_onehot_buf : (num_steps, num_envs, GRAPH_DIM)`
  - `graph_targets_buf : dict[slot_key -> (num_steps, num_envs) long]` (30 slots)
  - `graph_nodes_buf   : dict[name -> (num_steps, num_envs, node_feat_dim)]` (4 nodes)

`GRAPH_DIM` for StackCube pairs = 138 (= 36+36+24+24+18 across the 5
instantiated pairs); `len(SLOT_ORDER) = 30`.

### 4.5 Per-step rollout (mirrors the old prototype's timing)
1. `cur_state = extract_state(envs.unwrapped)` — matches `next_obs`.
2. `cur_z = extract_z_features(_z_obs_view(envs.unwrapped), envs.unwrapped, spec)`
   (or `{}` when `--no_z`).
3. `window, valid = graph_history.get_window()` — newest = last push.
4. `graph = build_graph(cur_state, cur_z, window, valid, thresholds,
   affordance_bank, spec, class_embed)` under `torch.no_grad()`.
5. Store `graph["onehot"]`, `graph["targets"]`, `graph["nodes"]` into
   the rollout buffers.
6. `agent.get_action_and_value(next_obs, graph_dict=cur_graph_dict)`
   (graph_dict only when `--use_graph_critic`).
7. `envs.step(action)`.
8. `new_state = extract_state(envs.unwrapped)`;
   `graph_history.push(new_state, just_reset=(terms|truncs))`.
9. If `final_info` present, done envs are bootstrapped with
   `make_zero_graph_dict(n_done, spec, node_feat_dim, device)` —
   matches the old prototype's "zero graph on done bootstrap" choice
   (auto-reset already overwrote the privileged state, so a real
   graph for the natural-next obs is not available).

### 4.6 End-of-rollout bootstrap
A real graph is built for the post-rollout state and fed to
`agent.get_value(next_obs, graph_dict=post_graph)` to compute
`next_value`. The `graph_history` already holds the last push from the
final step, so `get_window()`'s newest equals `post_state`.

### 4.7 Agent
- Actor: `self.actor_mean(self.feature_net(obs))`. No graph arg on
  `get_action`. Deployable from RGB+state alone.
- Critic: `self.critic(concat(obs_latent, graph_encoder_out))` when
  `--use_graph_critic`; otherwise `self.critic(obs_latent)`.
- Encoders:
  - `mlp`: `OneHotGraphMLP(GRAPH_DIM -> hidden)` — fast, uses only
    `graph_dict["onehot"]`.
  - `gnn`: `RelationGraphEncoder(spec, node_in_dim=node_feat_dim,
    hidden, layers)` — consumes `graph_dict["nodes"]` and
    `graph_dict["targets"]`; rel-embeds the labels, message-passes
    over the 4-node graph along the eligible edges, mean-pools.
- Causal mask (optional, gnn only):
  `CausalRelationMask(spec, rel_feat_dim=GRAPH_DIM)`. At each forward,
  rel_feat = `graph_dict["onehot"]`; produces a per-slot mask dict and
  an L1 penalty on the mask probabilities. The mask dict is forwarded
  to `RelationGraphEncoder` to gate per-(pair,field) relation
  embeddings. During rollout (`agent.eval()`) the mask is the hard
  threshold; during update (`agent.train()`) it is Gumbel-Sigmoid + STE.
- Aux heads (optional): `GraphAuxiliaryHeads(latent_size)`. Logits
  come purely from the obs latent (no graph input). Loss is
  `GraphAuxiliaryHeads.loss(logits, mb_targets)` (mean CE across the
  30 slots). Gradient flows into the visual feature net.

### 4.8 Update step
For each minibatch:
- `gather_graph_dict(b_graph_oh, b_graph_targets, b_graph_nodes,
  mb_inds, spec)` reconstructs the full graph dict.
- `agent.get_action_value_and_aux(b_obs[mb_inds], b_actions[mb_inds],
  graph_dict=mb_graph_dict)` returns `(action, logprob, entropy,
  value, aux_logits, mask_l1)`.
- Loss:
  `pg_loss + vf_coef * v_loss - ent_coef * entropy`
  `+ graph_aux_coef * GraphAuxiliaryHeads.loss(aux_logits, mb_targets)`
  `+ mask_l1_coef * mask_l1`.

### 4.9 Sanity asserts and telemetry (once per iteration)
- `b_graph_oh.shape == (batch_size, GRAPH_DIM)`.
- Each per-(pair,field) group of `b_graph_oh` sums to 1 per env.
- Every integer target lies in `[0, num_classes_for_that_field)`.
- `aux_loss` finite before backprop.
- Logged: `losses/graph_aux_loss`, `losses/mask_l1`,
  `charts/use_graph_critic`, `charts/use_graph_aux`,
  `charts/use_causal_mask`, plus per-class frequency of the four
  StackCube 7-way continuous temporal heads under
  `graph_class_freq/<pair>:<field>/class_<id>`.

---

## 5. Verification (read-only)

The user asked for "verify by reading, not running". Result:

- All teemo imports resolve to real symbols defined in `teemo/*.py`:
  `vocab`, `graph_builder.GRAPH_DIM`, `SLOT_ORDER`, `PAIRS`,
  `TeemoGraphSpec`, `build_graph`, `extract_state`,
  `extract_z_features`, `history.GraphHistory`,
  `affordance_use.AffordanceBank`, `encoders.RelationGraphEncoder`,
  `encoders.CausalRelationMask`, `encoders.GraphAuxiliaryHeads`.
- All function signatures match the call sites in `ppo_rgb_teemo.py`
  and `selftest.py`. In particular:
  - `build_graph(state, z_feats, prev_window, valid, th, aff, spec,
    class_embed)` — 8 positional, returns dict with keys
    `nodes / abs / temp / onehot / targets`.
  - `GraphHistory(num_envs, k, device)`,
    `.push(state, just_reset)`, `.get_window() -> (window, valid)`.
  - `AffordanceBank(aff_dir, device)`, empty-dir tolerant.
  - `RelationGraphEncoder(spec, node_in_dim, hidden=128, layers=2)`,
    `.forward(graph_dict, mask=None)`, `.out_dim`.
  - `CausalRelationMask(spec, rel_feat_dim, hidden=64)`,
    `.forward(rel_feat, tau=1.0, hard=True) -> (mask_dict, l1)`.
  - `GraphAuxiliaryHeads(obs_latent_dim)`,
    `.forward(obs_latent) -> dict`,
    `.loss(logits, targets)` staticmethod.
- Buffer shape contract is consistent end-to-end:
  - rollout `(num_steps, num_envs, ...)` -> flatten
    `(batch_size, ...)` -> minibatch index `(mb_size, ...)`.
  - `onehot` last-dim = `GRAPH_DIM = 138`.
  - `nodes[name]` last-dim = `node_feat_dim = 3 (type_oh) +
    graph_class_emb_dim (8) + 1 (tau) + spec.z_feat_dim (3) +
    3 (pos) = 18`.
  - `targets[slot_key]` is a long tensor (no last dim).
- Mask key set, aux head key set, encoder edge iteration, and
  rollout buffer slot keys are all the same set: `{f"{p}:{f}" for
  (p,f) in SLOT_ORDER}` (30 keys).

No signature mismatch had to be fixed in `teemo/`; the package was
already self-consistent. The one real-world adapter needed was the
`SidecarSensorWrapper` to keep segmentation visible after
`FlattenRGBDObservationWrapper`.

The three "known adaptation points" from `teemo/README.md` (h5 obs
key paths, camera param key names, ManiSkill attribute differences)
were NOT exercised in this session because Steps 2-4 are deferred.

---

## 6. Dependencies to install before running anything

Already in the repo / part of ManiSkill:
- `torch`, `numpy`, `gymnasium`, `mani_skill`, `transforms3d`,
  `tyro`, `tensorboard`.

Offline-only (Steps 2-4):
- `h5py` — `affordance_extract.py`, `calibrate_thresholds.py`.
- `scikit-learn` — `affordance_extract.py` k-means (falls back to
  random subsample without it).
- `networkx`, `matplotlib`, `opencv-python` — `viz_graph.py` and the
  calibration appendix histograms.

Single install command:
```
pip install h5py networkx scikit-learn matplotlib opencv-python
```

The PPO training fork itself does NOT depend on the offline-only
deps. As long as `teemo/thresholds_default.json` is present, the
training script's import surface is satisfied even before the
offline pipeline is run; affordances are optional (the bank falls
back to sentinel values, and the alignment/aperture relations
collapse to constants).

---

## 7. Order of operations once deps are installed

1. `python -m teemo.selftest` — should print
   `teemo.selftest PASS  graph_dim=138 num_slots=30 num_envs=4`.
2. Step 2 (deferred):
   ```
   python -m mani_skill.examples.motionplanning.panda.run \
     -e StackCube-v1 -n 200 --save-traj --only-count-success
   python -m mani_skill.trajectory.replay_trajectory \
     --traj-path <demos>.h5 --use-first-env-state -o state \
     --save-traj -b cpu
   python -m teemo.affordance_extract --traj <state>.h5 \
     --out-dir teemo/affordances --kmeans-k 12
   ```
   Adapt the h5 key paths in `affordance_extract.extract_from_episode`
   if your replay layout differs from the expected
   `extra/tcp_pose`, `extra/cubeA_pose`, `extra/cubeB_pose`,
   `agent/qpos`.
3. Step 3 (deferred):
   ```
   python -m teemo.calibrate_thresholds --traj <state>.h5 \
     --affordance-dir teemo/affordances --out teemo/thresholds.json
   ```
   Then point the training script at the new file via
   `--thresholds_path teemo/thresholds.json`.
4. Step 4 (deferred): `python -m teemo.viz_graph ...`. Adapt
   `project_points` camera key names if `intrinsic_cv` /
   `extrinsic_cv` are nested differently for the
   `base_camera` sensor params.
5. Step 6 (deferred): variant sweep
   - A: `python ppo_rgb_teemo.py --env_id StackCube-v1 --num_envs 256`
   - B: `... --use_graph_critic True --graph_encoder gnn`
   - C: `... --use_graph_critic True --use_graph_aux True --graph_encoder gnn`
   - D: `... --use_graph_critic True --use_graph_aux True --use_causal_mask True --graph_encoder gnn`
   Each at >=3 seeds. Decision rule (from the prompt): if B/C don't
   beat A on StackCube, report it as a clean negative result about
   the oracle graph's value to the critic.

---

## 8. Things deliberately not done

- Did not redesign anything in `teemo/`. Labels, eligibility, K=5,
  transition rule are paper-contract.
- Did not feed the graph to the actor.
- Did not invent thresholds beyond a placeholder file clearly
  labeled as such.
- Did not run affordance extraction, calibration, viz, training,
  or any self-test (deferred per user instruction; this machine
  has no deps installed).
- Did not delete or rewrite the old `stackcube_graph.py` /
  `graph_modules.py` prototype — `ppo_rgb_teemo.py` simply stops
  importing them.
