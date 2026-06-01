# PPO + Discrete Temporal Affordance-Progress Graph (StackCube-v1, Phase 1)

This document describes the Phase-1 prototype that augments the ManiSkill PPO
RGB baseline with a discrete temporal affordance-progress graph. The graph is
used as

1. an optional **critic-only** input (oracle privileged supervision), and
2. an optional **auxiliary visual-representation target** trained from the
   normal observation latent.

The actor is never given the graph, so the resulting policy remains deployable
from normal RGB + proprio observations.

---

## 1. Files changed / added

| File | Purpose |
| --- | --- |
| `examples/baselines/ppo/stackcube_graph.py` *(new)* | Graph schema, binning helpers, builder, K+1 ring-buffer history, Graph R-CNN-inspired relation-proposal stub, and a `__main__` self-test. |
| `examples/baselines/ppo/graph_modules.py` *(new)* | `DiscreteTemporalGraphEncoder` (`mlp` / `temporal_transformer`) and `GraphAuxiliaryHeads`. |
| `examples/baselines/ppo/ppo_rgb.py` *(modified)* | 10 new graph CLI flags; `Agent` extended with optional graph encoder + aux heads; graph build → critic / aux plumbed through the rollout, GAE bootstrap, and minibatch update; sanity assertions and logging added. |
| `examples/baselines/ppo/README.md` *(modified)* | New section documenting the four variants and the smoke-test command. |

All changes are localized; the no-graph baseline path (both flags `False`) is
behaviorally identical to the original `ppo_rgb.py` aside from logging two
constant `charts/use_graph_*` scalars.

---

## 2. How the graph is computed (Phase 1)

For each control step on each env, `extract_stackcube_state(base_env)` reads
privileged state from `envs.unwrapped`:

- `tcp_pos`, `cubeA_pos`, `cubeB_pos` (from `Actor.pose.p`)
- `is_grasped`, `is_on`, `is_static` (from `base_env.evaluate()`, which is
  read-only and returns consistent labels for *all* envs, including the ones
  that just auto-reset inside `step`).

`build_stackcube_graph` produces:

### Absolute (15 dims)

| head | dims | semantics |
| --- | --- | --- |
| `ee_target_dist` | 3 | close / mid / far |
| `target_goal_xy` | 3 | aligned / near / far |
| `target_goal_z` | 3 | below_goal / at_goal_height / above_goal |
| `gripper_target_grasp` | 2 | false / true |
| `target_support_contact_or_on` | 2 | false / true |
| `target_motion_state` | 2 | moving / static |

`goal_pos = cubeB_pos + [0, 0, 0.04]` (= 2 × `cube_half_size` for StackCube).

### Temporal (21 dims), comparing t vs. t-K

| head | dims | semantics |
| --- | --- | --- |
| `ee_target_distance_change` | 7 | improving / worsening (× slow/mid/fast) + stable |
| `target_goal_distance_change` | 7 | same scheme |
| `target_support_z_offset_change` | 7 | increasing / decreasing (× slow/mid/fast) + stable |

Magnitude bins use `torch.bucketize` against `(slow, mid)` thresholds, with the
"stable" class taken when `|delta| ≤ eps`.

Output of the builder is

- `graph_onehot` of shape `[num_envs, 36]` (concatenated per-head one-hots in
  `HEAD_ORDER`), and
- `graph_targets` dict of integer class labels per head, used for the auxiliary
  cross-entropy losses.

No continuous values are placed into the graph vector.

---

## 3. How K=5 temporal history is handled

`StackCubeGraphHistory` is a per-env ring buffer of size `K+1 = 6` storing
`tcp_pos`, `cubeA_pos`, `cubeB_pos`. Each step we call
`push(state, just_reset)`:

- non-reset push  → `valid_count = min(valid_count + 1, K+1)`
- reset push (`just_reset[i] = True`) → `valid_count[i] = 1`

`get_prev_k_state()` returns the state at slot `(_head - 1 - K) % (K+1)` along
with a `valid_mask = valid_count >= K+1`. Where the mask is `False` (early in
an episode or right after auto-reset), `build_stackcube_graph` forces the
three temporal labels to **stable** so we never compare across episode
boundaries.

The history is seeded with the initial post-reset state with
`just_reset = all-True` right after `envs.reset(seed=...)`.

For the transformer encoder mode, a parallel K+1 rolling window of
*graph one-hots* (`graph_window`) is maintained the same way and stored in
`graph_histories_buf` so each minibatch sample carries its own
`[K+1, 36]` sequence.

---

## 4. Actor vs. critic inputs

- **Actor** — `actor_mean(feature_net(obs))`. `get_action` does not even
  accept graph args, so the policy is deployable from normal observations.
- **Critic** — `_critic_from_latent` is gated on `use_graph_critic`:
  - `True`: `critic(concat(obs_latent, graph_encoder(graph_onehot[, graph_history])))`
  - `False`: identical to the baseline.
- **Eval rollout** — `agent.get_action(eval_obs, deterministic=True)` only;
  no graph path is exercised.

---

## 5. Auxiliary graph loss

`GraphAuxiliaryHeads` is one `Linear(obs_latent_dim, head_dim)` per graph
head, taking **only** the observation latent (no graph input):

```python
aux_loss = sum( F.cross_entropy(logits_h, targets_h) for h in heads )
total_loss = pg_loss + vf_coef * v_loss - ent_coef * entropy + graph_aux_coef * aux_loss
```

Per-head loss and accuracy are averaged across minibatches and logged.
Per-class frequency for the three 7-way temporal heads is logged once per
rollout to detect "stable"-class dominance.

The aux-loss path runs **only** when `--use_graph_aux=True`. When disabled,
`aux_logits` is `None` and no extra term is added.

---

## 6. Graph R-CNN / STTran-inspired interfaces

- **`StackCubeRelationProposal`** (Graph R-CNN-style) — `forward(node_features, graph_context)` returns a fixed list of 4
  task-relevant edges (`ee↔target`, `gripper↔target`, `target↔support`,
  `target↔goal`). Constructed but unused in Phase 1; future phases can drop
  in a learned top-K object-pair scorer with the same interface.
- **`DiscreteTemporalGraphEncoder`** (STTran-style) — two modes:
  - `"mlp"` (default): 2-layer MLP over the current 36-dim graph.
  - `"temporal_transformer"`: `TransformerEncoder` over the K+1 graph window
    with learned positional embeddings, mean-pooled to a single embedding.
  - The transformer mode is purely optional; the per-step
    `graph_histories_buf` is allocated only when both `--use_graph_critic=True`
    and `--graph_encoder_type=temporal_transformer`.

---

## 7. New CLI flags (added to `Args` in `ppo_rgb.py`)

| flag | default | purpose |
| --- | --- | --- |
| `--use_graph_critic` | `False` | concat graph latent to critic input |
| `--use_graph_aux` | `False` | train aux heads on obs latent to predict graph labels |
| `--graph_aux_coef` | `0.1` | scale on the summed aux CE loss |
| `--graph_temporal_k` | `5` | temporal horizon for change labels |
| `--graph_encoder_type` | `mlp` | `mlp` or `temporal_transformer` |
| `--graph_embed_dim` | `128` | hidden dim of the graph encoder |
| `--graph_transformer_layers` | `2` | transformer mode only |
| `--graph_transformer_heads` | `4` | transformer mode only |
| `--graph_transformer_ff_dim` | `256` | transformer mode only |
| `--graph_transformer_dropout` | `0.1` | transformer mode only |

---

## 8. How to run

Run from the `examples/baselines/ppo/` directory so that `stackcube_graph.py`
and `graph_modules.py` are on `sys.path`.

### Variant A — baseline (unchanged behavior)

```bash
python ppo_rgb.py --env_id=StackCube-v1 --num_envs=256
```

### Variant B — oracle graph → critic only

```bash
python ppo_rgb.py --env_id=StackCube-v1 --num_envs=256 \
  --use_graph_critic True
```

### Variant C — oracle graph → critic + auxiliary graph prediction (Phase-1 main experiment)

```bash
python ppo_rgb.py --env_id=StackCube-v1 --num_envs=256 \
  --use_graph_critic True --use_graph_aux True \
  --graph_aux_coef 0.1 --graph_temporal_k 5 --graph_encoder_type mlp
```

### Variant D — auxiliary graph prediction only

```bash
python ppo_rgb.py --env_id=StackCube-v1 --num_envs=256 \
  --use_graph_aux True --graph_aux_coef 0.1 --graph_temporal_k 5
```

### Optional — STTran-style temporal transformer for the critic

```bash
python ppo_rgb.py --env_id=StackCube-v1 --num_envs=256 \
  --use_graph_critic True --use_graph_aux True \
  --graph_encoder_type temporal_transformer \
  --graph_transformer_layers 2 --graph_transformer_heads 4 \
  --graph_transformer_ff_dim 256 --graph_transformer_dropout 0.1
```

### Smoke test (tiny budget, just verify the pipeline + assertions)

```bash
python ppo_rgb.py --env_id=StackCube-v1 \
  --num_envs 8 --num_eval_envs 2 \
  --num_steps 32 --num_eval_steps 32 \
  --total_timesteps 4096 --num_minibatches 4 --update_epochs 2 \
  --eval_freq 1000 --save_model False --capture_video False \
  --use_graph_critic True --use_graph_aux True
```

### Stand-alone graph self-test

```bash
python stackcube_graph.py
```

Runs synthetic tests verifying:

- one-hot groups sum to 1
- distance shrinking → improving class, distance growing → worsening,
  near-zero delta → stable
- z increasing → increasing class, z decreasing → decreasing
- invalid prev (or no prev at all) → all three temporal labels are stable
- all target class indices lie in `[0, head_dim)`

---

## 9. Logged TensorBoard tags

In addition to the existing PPO scalars:

- `losses/graph_aux_loss`
- `losses/graph_aux_<head>` for each of the 9 heads
- `aux_acc/<head>` per-head accuracy
- `aux_class_freq/<temporal_head>/class_<id>` for the three 7-way temporal heads
- `charts/use_graph_critic`, `charts/use_graph_aux`

`losses/explained_variance` (already present) is useful for judging whether
the graph critic improves value-function fit.

---

## 10. Sanity checks built into the script

Runs once per PPO iteration (cheap):

- `graph_onehot.shape == (batch_size, 36)`
- each per-head one-hot group sums to 1
- every integer graph target lies in `[0, head_dim)`
- `aux_loss` is finite before backprop
- assertion that `env_id == "StackCube-v1"` whenever a graph flag is enabled

Structural (compile-time) invariants:

- the actor never receives the graph (`get_action` has no graph parameter)
- the critic only sees the graph when `--use_graph_critic=True`
  (`_critic_from_latent` gates on the flag)
- the aux loss term is added to `total_loss` only when `--use_graph_aux=True`
- eval rollout calls `get_action(..., deterministic=True)` exclusively
- when both graph flags are `False`, the script allocates no graph buffers
  and runs exactly the baseline code path

---

## 11. ManiSkill API assumptions / adjustments made after inspecting the code

- **RGB obs mode does not expose `cubeA_pose` / `cubeB_pose`** in
  `_get_obs_extra` (those are state-mode only). The builder therefore reads
  privileged state directly from `envs.unwrapped` and calls
  `base_env.evaluate()` for the booleans. This is task-specific extraction;
  to extend to another task you write a new `extract_<task>_state(base_env)`
  but reuse the same generic schema and module interfaces.
- **`ManiSkillVectorEnv` auto-resets inside `step`** and overwrites `infos`
  with the reset's infos (preserving the pre-reset info only under
  `infos["final_info"]`). To get consistent labels for all envs (including
  ones that just reset) we call `base_env.evaluate()` once at the top of
  each step, since it is read-only.
- **Bootstrap value with graph critic for done envs:** the natural-next
  observation in `infos["final_observation"]` has no matching privileged
  state available (auto-reset already overwrote env state for those envs).
  For Phase 1 we bootstrap done envs with a *zero* graph; the mismatch is
  bounded by the `next_not_done` mask in the subsequent GAE pass and only
  affects the terminal-step target. Marked in code with an explanatory
  comment so Phase 2 can replace it with a learned predicted-graph latent.
- **`cube_half_size = 0.02`** (verified in `stack_cube.py`), so the derived
  goal node uses `+[0, 0, 0.04]` exactly as specified. If a future task
  variant changes that, update the `z_stack_offset` argument passed to
  `build_stackcube_graph`.
- **`StackCube-v1` has `max_episode_steps = 50`**, so the default
  `--num_steps=50` is roughly one episode per env per iteration; temporal
  labels are "stable" for the first K=5 steps after every reset by design
  (captured by the `valid_count` mask).

---

## 12. What this prototype intentionally does NOT do (deferred to later phases)

- **Phase 2:** predicted graph latent fed into the critic (so the critic no
  longer needs oracle access at training time). The relation-proposal and
  graph-encoder interfaces are structured so the predicted graph can be
  swapped in without changing the actor.
- **Phase 3:** learned object / mask / edge extractors (Graph R-CNN-style
  detection); STTran-style full spatio-temporal scene-graph transformer.
- EMA-smoothed continuous potentials, graph-based reward shaping,
  continuous values inside the graph vector, oracle graph fed to the
  actor — all deliberately excluded in Phase 1.
