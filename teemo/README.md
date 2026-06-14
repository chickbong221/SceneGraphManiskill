# TEEMO graph package — full draft design (StackCube-v1 demo)

This package implements the FULL spatio-temporal semantic graph from the paper
draft (§3 + Tables 1–2), computed from ground-truth simulator state for the demo.
It is the authoritative design; the old `stackcube_graph.py` prototype in the repo
is a reduced placeholder and should be retired.

## Files
| File | Role |
| --- | --- |
| `vocab.py` | SINGLE SOURCE OF TRUTH. Verbatim label sets (Tables 1–2), node types, semantic-class registry, ELIGIBLE_PAIRS, class counts. |
| `graph_builder.py` | Node features `[type, class_emb, tau, z_i]`, all absolute + temporal relations, K=5, eligibility asserts, serialization. StackCube pair instantiation. |
| `affordance_use.py` | Runtime bank: nearest per-class candidate → tcp-alignment + aperture error. |
| `affordance_extract.py` | OFFLINE: motion-planning success episodes → object-frame TCP-pose + aperture candidates per class. |
| `calibrate_thresholds.py` | OFFLINE: data-driven bin edges (landmarks + quantiles) → `thresholds.json`. No magic numbers. |
| `history.py` | K+1 per-env ring buffer with auto-reset-safe validity. |
| `encoders.py` | Relation-aware message-passing encoder, Gumbel-Sigmoid causal relation mask, aux CE heads. |
| `viz_graph.py` | RGB overlay + node-link figures from the SAME build_graph output. |

## Faithfulness to the draft
- **Nodes**: tcp, gripper, object(s). No goal node (this version). Feature is
  `[type_onehot(3), class_embedding, tau, z_i, pos]`.
  - `emb(c_i)`: semantic-class embedding keyed by id from the task object list
    (NOT a fixed one-hot). `vocab.SEMANTIC_CLASSES`.
  - `tau`: steps since visible. Constant 0 under GT (oracle persistence) — the one
    field that's inert in the demo, honest in the paper.
  - `z_i`: LIVE masked-RGB evidence per object from GT segmentation mask.
- **Absolute relations** (all live except containment):
  distance/height (5-way), tcp_align/aperture (5-way via affordance set),
  contact/grasp/support (2-way). containment = const no-containment (no receptacle).
- **Temporal relations**, K=5: distance/height/alignment/aperture change (7-way
  signed-magnitude), contact/grasp/support transition (4-way with draft
  window-stability debounce). containment transition const.
- **Eligibility**: `ELIGIBLE_PAIRS` mirrors the "Valid pairs" column; asserted at
  import. Only the 5 instantiated StackCube pairs are ever built.
- **Method integration**: graph → critic (+ optional GNN encoder, + optional
  causal relation mask with STE Gumbel-Sigmoid + L1), and → aux CE heads on the
  obs latent. Actor stays graph-free / deployable.

## Deferred (schema present, honest in paper)
- `tau` live, slot-matching / T_mem retention → need perception pipeline.
- `containment` (+ temporal) → need a receptacle task.
- goal node → not used this version.

## Pipeline order
1. `affordance_extract.py` (needs success demos + state replay)
2. `calibrate_thresholds.py` (needs same replay + affordances) → `thresholds.json`
3. `viz_graph.py` — eyeball figures BEFORE training
4. `ppo_rgb_teemo.py` (Claude Code builds this) — variants A–D

See `CLAUDE_CODE_PROMPT.md` for the exact build/integration steps and the three
places that must be adapted to real h5/camera key names.

## Known adaptation points (not design changes)
- h5 obs key paths in `affordance_extract`/`calibrate_thresholds` are best guesses;
  inspect the replayed .h5 and fix.
- camera param keys in `viz_graph.project_points` (`intrinsic_cv`/`extrinsic_cv`).
- any ManiSkill attribute that differs from the verified facts (e.g. velocity
  accessor) — fix in `extract_state` only.
