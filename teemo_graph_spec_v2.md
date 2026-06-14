# TEEMO Demo — Graph Spec v2 (faithful to draft, GT oracle)

The demo computes a **subset** of the draft's edges from **GT simulator state**, but
every field implemented matches the draft (Section 3 + Tables 1–2) label-for-label:
same node types, same relation names, same vocabularies, same valid-pair eligibility,
same temporal-transition rule. Deferred fields stay in the tensor schema as sentinels
so the demo graph is a strict restriction of the paper graph.

Framing: this is the **oracle-graph upper bound** — GT relations + GT-mask node
appearance. It answers "does the graph *structure* help RGB-PPO." Predicting
masks/relations from pixels is the extractor, which is entirely future work; nothing
in this demo is a perception claim, so feeding GT is clean (no hybrid-privilege caveat).

Single source of truth for all label sets: `teemo/vocab.py`, copied verbatim from
Tables 1–2, imported by `build_graph()`, the viz annotator, and the affordance labeler.

---

## 1. Nodes (draft §3.1)

Node types: `tcp`, `gripper`, `object`. **No `goal` node in this version** (even for
PickCube — PickCube's goal is future work). Per-task object slots; StackCube = `cubeA`,
`cubeB`.

Draft node feature: `x_i = [emb(c_i), emb(tau_i), z_i]`.

### emb(c_i) — semantic-class embedding  [CORRECTED]
`c_i` is the object's **semantic label**, not a fixed-arity one-hot. Build a small
embedding table keyed by **semantic class id**, where the class vocabulary is derived
from the task's object list at env-build time (`cube`, `peg`, `bin`, ...). New objects
get a new class id without reshaping the policy network. The node **type**
(`tcp`/`gripper`/`object`) is a *separate* small field (3-way), distinct from the
semantic-class embedding. So a node feature carries: `[type_field, class_embedding,
tau_field, z_i]`.

How to get the class list: enumerate the task's manipulable actors at build time
(from the task file's actor handles) and assign each a stable class id. Store the
id↔name map alongside the run so viz and affordance labels agree.

### emb(tau_i) — steps since last visible
GT ⇒ every entity always visible ⇒ `tau_i = 0` every frame. Field is present in the
tensor (schema matches paper) but constant. Goes live only under the perception
pipeline. (This is the ONE field that stays constant under GT — unavoidable, and
honest: oracle persistence.)

### z_i — masked visual evidence  [NOW LIVE, all objects]
Live for every object node, not just articulated ones. Procedure:
- env in `obs_mode="rgb+segmentation"`.
- for object with `per_scene_id = pid`: `mask = (seg == pid)` → mask-pool the RGB (or a
  light CNN feature) over that mask → `z_i`.
- this uses the **GT segmentation mask** (the demo's oracle); in the real paper the
  extractor predicts this mask itself, but that's out of scope here.
- `z_i` therefore captures real per-object appearance/visual state from pixels, exactly
  the draft's intent.

Implementation note: mask-pooling needs the per-env seg tensor; on GPU sim it has a
leading env dim. For training (many envs) keep it cheap — average-pool RGB over the
mask gives a small fixed vector per object. A heavier conv encoder is optional.

### Slot management / T_mem  [DEFER — oracle persistence]
Fixed slot per known task object, always active. Matching / `T_mem` retention only
matters under perception. Paper: "demo assumes oracle persistence."

---

## 2. Absolute relations (draft Table 1) — all LIVE except containment

Use exact label sets; never collapse the 5-way continuous relations. A pair carries
multiple typed fields (e.g. `cubeA–cubeB` holds both `contact` and `support`).

### Spatial `distance` — pairs: tcp–object  (object–goal absent, no goal node)
Labels (5): `very-near, near, medium, far, very-far`
StackCube live pairs: `tcp–cubeA`, `tcp–cubeB`.
Euclidean distance → `bucketize` with 4 fixed thresholds (named constants, reused in viz).

### Spatial `height` — pairs: tcp–object
Labels (5): `far-below, below, level, above, far-above`
Signed z-offset `a.z - b.z` → 4 thresholds symmetric about 0 (`level` straddles zero).

### Alignment `tcp-alignment` — pairs: tcp–object  [NOW LIVE via affordance set]
Labels (5): `very-misaligned, misaligned, partial, near-aligned, aligned`
Compute from the per-semantic-class affordance candidate set (Section 4): pose error
(position + orientation) from current TCP to the nearest candidate (candidate placed
in world frame via current object pose) → discretize into the 5 bins.

### Alignment `aperture-fit` — pairs: gripper–object  [NOW LIVE via affordance set]
Labels (5): `too-narrow, slightly-narrow, fit, slightly-wide, too-wide`
Compute from difference between current gripper width and the nearest candidate's
aperture → 5 bins.

### Physical `contact` — pairs: gripper–object, object–object
Labels (2): `no-contact, contact`
StackCube: `gripper–cubeA`, `gripper–cubeB`, `cubeA–cubeB`.
`scene.get_pairwise_contact_forces(a,b).norm(dim=-1) > eps`.

### Physical `grasp` — pairs: gripper–object
Labels (2): `no-grasp, grasp`
`agent.is_grasping(obj)`.

### Physical `support` — pairs: object–object
Labels (2): `no-support, support`
`cubeA–cubeB`. Draft definition: object–object contact AND vertical ordering AND
relative pose AND stability ⇒
`contact(A,B) AND (A.z - B.z) ≈ cube_h AND xy-aligned within tol AND |vel_A| < v_eps`.

### Physical `containment` — pairs: object–object, object–goal  [DEFER]
Labels (2): `no-containment, containment`. No receptacle in these tasks. Schema
present, constant `no-containment`. Live for a put-in-bin/fridge task later.

---

## 3. Temporal relations (draft Table 2), horizon K=5  [CORRECTED]

True window over `t-5:t`. Wrapper keeps a 5-deep ring buffer of the underlying
continuous values and binary states per relation, per env, reset per-env on done.

### Continuous temporal — full vocab, magnitude-binned
`distance-change`: `approach-{slow,medium,fast}, stable-distance, recede-{slow,medium,fast}`
`height-change`:   `move-up-{slow,medium,fast}, stable-height, move-down-{slow,medium,fast}`
`alignment-change`: `improve-alignment-{slow,medium,fast}, stable-alignment, worsen-alignment-{slow,medium,fast}`  [LIVE now]
`aperture-change`:  `improve-aperture-{slow,medium,fast}, stable-aperture, worsen-aperture-{slow,medium,fast}`      [LIVE now]
Compute: net change of the underlying value over the K-window (value[t] - value[t-K]),
sign → direction, |magnitude|/K → speed bin (slow/medium/fast), deadband → stable.

### Binary temporal — draft transition rule with window-stability debounce
`contact-transition`, `grasp-transition`, `support-transition`; each 4 labels:
`gain-r, lose-r, maintain-r, maintain-no-r`.
Draft rule, now properly realized at K=5:
- look at the binary trace over `t-5:t`.
- require **stable at the start** (first ~2 frames equal) and **stable at the end**
  (last ~2 frames equal); only then read start-state→end-state into gain/lose/maintain.
- traces that are not stable at both ends (one-step pulses, repeated switches) →
  excluded (no temporal edge emitted this step).
- `maintain-no-r` kept internally but **suppressed on export** (not sent to policy/viz).
`containment-transition`: DEFER (matches deferred absolute containment).

---

## 4. Affordance pose set — build IN this phase (per semantic class) [CORRECTED]

Drives `tcp-alignment`, `aperture-fit`, and their temporal edges. Per **semantic class**
(one set for "cube"), in the **object frame**, so it transfers across instances.

### 4a. Collect success episodes (motion planning)
`python -m mani_skill.examples.motionplanning.panda.run -e "StackCube-v1"`
emits only successful trajectories as `.h5` under `demos/motionplanning/StackCube-v1/`.
Run `-h` for count/control-mode flags; collect a few hundred episodes.

### 4b. Replay to recover per-step state
The saved `.h5` is compressed (env-states, minimal obs). Replay with full state via
`mani_skill.trajectory.replay_trajectory --use-first-env-state` into a state/pose-bearing
trajectory so every timestep has TCP pose, object pose, gripper width. Keep control
mode matched.

### 4c. Extract object-frame candidates
For each episode, for each manipulation object of a given class:
- find the timestep where `is_grasping(obj)` first flips True (grasp event).
- record TCP pose in the **object frame**: `T_cand = inv(T_obj) @ T_tcp`, and the
  gripper width at that step as the candidate aperture.
- (StackCube place phase) at the step where `support(A,B)` first becomes True, record
  the same object-frame TCP pose + aperture relative to the *placement reference*
  object — this gives place-phase candidates.
- accumulate all candidates per class into a set; optionally cluster (k-means) to a
  handful of representative candidates per class to keep nearest-candidate lookup cheap.

Store as `teemo/affordances/<class>.npz`: arrays of `(pos[3], quat[4], aperture[1])`.

### 4d. Use at graph-build time
For a live object of class `c` with current world pose `T_obj`:
- transform each class candidate to world: `T_world_cand = T_obj @ T_cand`.
- `tcp-alignment`: min over candidates of weighted (position err + orientation err)
  between current TCP and `T_world_cand` → discretize to the 5 alignment bins.
- `aperture-fit`: from the matched candidate's aperture vs current gripper width → 5 bins.

---

## 5. Eligibility table (draft §3.2)

`ELIGIBLE_PAIRS` in code mirrors the "Valid pairs" column of Tables 1–2 exactly.
Assert every computed relation comes from it. No all-pairs evaluation.

---

## 6. StackCube demo graph, end to end

Nodes: `tcp, gripper, cubeA, cubeB`
  - per node: `[type(3), class_emb, tau(=0), z_i(live masked RGB)]`

LIVE relations (exact draft labels):
- `distance`, `height`         on `tcp–cubeA`, `tcp–cubeB`
- `tcp-alignment`              on `tcp–cubeA`, `tcp–cubeB`   (via affordance set)
- `aperture-fit`               on `gripper–cubeA`, `gripper–cubeB`
- `contact`                    on `gripper–cubeA`, `gripper–cubeB`, `cubeA–cubeB`
- `grasp`                      on `gripper–cubeA`, `gripper–cubeB`
- `support`                    on `cubeA–cubeB`
- temporal `distance-/height-/alignment-/aperture-change` (K=5, full vocab)
- temporal `contact-/grasp-/support-transition` (4-way, window-stability debounce)

SCHEMA-PRESENT, DEFERRED (honest in paper):
- node `tau` (constant 0 under oracle persistence)
- `containment` + `containment-transition` (no receptacle)
- slot matching / `T_mem` (oracle persistence)
- goal node (not used this version)

Every implemented field is label-for-label the paper's field; deferrals are structural
(missing task affordances / perception), not relabelings.
