# TEEMO Demo — Threshold Calibration Spec

Principle: no magic numbers. Every bin edge is set from the empirical distribution of
the underlying quantity, measured over the motion-planning success episodes (the same
episodes collected for the affordance set). Run calibration ONCE, write the constants
into `teemo/vocab.py`, freeze. The viz annotator imports the same constants so figures
match training exactly.

Data source for calibration: replay the success `.h5` (with full state) and, for every
timestep of every episode, log the raw scalars below into arrays. Then compute edges.

---

## 0. What to log during calibration (per timestep, per episode)

For each eligible pair, log the raw continuous value:
- `d`   = ||tcp.p - obj.p||                      (distance, meters)
- `h`   = tcp.p.z - obj.p.z                       (signed height, meters)
- `a`   = alignment error to nearest affordance candidate (see §4)
- `w`   = gripper width  (qpos[-1]+qpos[-2])      (meters)
- `ap`  = aperture error = w - nearest_candidate_aperture
- `fc`  = ||pairwise_contact_force||              (Newtons)
Also log, for the temporal calibration, the **per-step delta** of each continuous value
(value[t] - value[t-1]) and the **K-window net change** (value[t] - value[t-K]), K=5.

Stack across all episodes → one big 1-D array per quantity. These arrays drive everything.

---

## 1. ABSOLUTE continuous bins (5-way): distance, height, alignment, aperture-fit

These need 4 edges to make 5 bins. Use a hybrid of landmarks + quantiles:

### distance: very-near / near / medium / far / very-far
- `very-near` should mean "essentially at the object" → anchor its upper edge at a
  **physical landmark**: the distance at which `is_grasping` becomes true, measured
  empirically. Compute `d_grasp = median of d over all timesteps where grasp just
  turned True`. Set edge_1 (very-near|near) = `d_grasp` (or a small multiple, ~1.0–1.5x).
- The remaining three edges (near|medium, medium|far, far|very-far): place at the
  **33/'66/90th percentiles** of the full `d` distribution above `d_grasp`. Rationale:
  most timesteps in a successful episode are mid-reach; quantiles spread bins so each
  is actually populated (avoids a bin that never fires).
- `very-far` is the tail above the 90th pct.

> Why mix: the bottom edge must be physically meaningful (grasp range), the upper edges
> just need to partition observed space sensibly. Pure equal-width bins would put almost
> everything in one bin because the arm spends most time at medium range.

### height: far-below / below / level / above / far-above
- `level` must straddle 0 (TCP at object height). Set the inner two edges symmetric:
  `[-h_level, +h_level]` where `h_level` = a physical landmark = half the cube extent
  (read from `stack_cube.py`), so "level" ≈ "within a cube-height of the object center".
- outer two edges at the **±90th percentile** of the signed `h` distribution (so
  `far-below`/`far-above` are genuine tails).
- Result: edges = `[-q90, -h_level, +h_level, +q90]`.

### alignment (tcp-alignment): very-misaligned ... aligned
- `a` is a scalar error (§4 for how it's combined). Lower = better aligned, ≥0.
- `aligned` upper edge = a physical landmark: the alignment error at the moment grasp
  succeeds → `a_grasp = median of a at grasp-onset timesteps`. Within `a_grasp` = aligned.
- remaining edges at 33/66/90th percentiles of `a` above `a_grasp`.
- (Same logic as distance: anchor the "good" bin to the grasp event, quantile the rest.)

### aperture-fit: too-narrow / slightly-narrow / fit / slightly-wide / too-wide
- `ap` is signed (current width minus candidate width). `fit` straddles 0.
- inner edges symmetric `[-ap_fit, +ap_fit]`, `ap_fit` = a physical landmark = a small
  fraction of the candidate aperture (e.g. tolerance ~5–10% of grasp width), OR the std
  of `ap` over timesteps where grasp is held successfully (the gripper-width noise band
  during a stable grasp). Prefer the latter — it's measured, not guessed.
- outer edges at ±90th percentile of `ap`.

---

## 2. BINARY thresholds: contact

Only one number: the force magnitude above which `contact = True`.
- Log `fc` over all timesteps. The distribution is bimodal: a spike near 0 (no contact)
  and a mass at higher force (in contact).
- Set `eps_contact` at the **valley between the two modes** (or simply a small floor
  like the 99th percentile of `fc` measured during known no-contact frames — frames
  where the object is far from gripper and not on another object).
- Practically: take frames where `d > medium-edge` AND object resting on table only;
  their `fc` against the gripper is ~0; set `eps_contact = max(that) * a safety factor`.
- Note: `grasp` and `support` do NOT need a force threshold — `grasp` uses
  `is_grasping` directly, `support` uses contact(bool) + geometry, so only the raw
  `contact` predicate consumes `eps_contact`.

---

## 3. TEMPORAL speed bins: approach-{slow,medium,fast} etc.  ← the subtle one

The temporal label has TWO parts: a **sign/direction** and a **speed magnitude**.

### Step 1 — direction from the K-window net change
For a continuous relation value `v`, over horizon K=5:
```
delta_K = v[t] - v[t-K]          # net change across the window
```
- distance: `delta_K < -deadband` → approach ; `> +deadband` → recede ; else stable
- height:   `delta_K > +deadband` → move-up  ; `< -deadband` → move-down ; else stable
- alignment: improvement means error DECREASES, so
            `delta_K < -deadband` → improve-alignment ; `> +deadband` → worsen ; else stable
- aperture:  same as alignment (error toward 0 = improve)

`deadband` per quantity = the **K-window change observed when the arm is essentially
stationary**. Measure it: take windows where the TCP barely moves (e.g. ||tcp.p[t] -
tcp.p[t-K]|| < 1mm) and look at the resulting `|delta_K|` of each relation; set
`deadband = 95th percentile` of that. This makes "stable" mean "indistinguishable from
not moving," which is exactly right and fully data-driven.

### Step 2 — speed = magnitude of change, binned by per-step RATE
Speed should be a RATE (per step), not raw window change, so it's K-independent in
meaning:
```
rate = |delta_K| / K             # average per-step change over the window
```
Bin `rate` into slow / medium / fast with TWO edges. Set those edges from the
distribution of `rate` over **only the moving windows** (exclude stable ones, i.e.
exclude `|delta_K| <= deadband`):
- edge(slow|medium)   = 50th percentile of `rate` among moving windows
- edge(medium|fast)   = 85th percentile of `rate` among moving windows

So: most motion is "medium", clearly slow creep is "slow", the fastest 15% is "fast".
These percentiles come from the SAME logged `rate` arrays, one per relation
(distance-rate, height-rate, alignment-rate, aperture-rate) — each relation gets its
OWN slow/medium/fast edges because their natural scales differ (a fast aperture change
≠ a fast reach).

> Worked example — distance approach:
>   v = distance(tcp, cubeA). At step t, delta_K = d[t] - d[t-5].
>   Say deadband = 0.004 m (4mm net over 5 steps when stationary).
>   delta_K = -0.06 m  → |delta_K|=0.06 > deadband, and negative → APPROACH.
>   rate = 0.06 / 5 = 0.012 m/step.
>   If among moving windows the median rate is 0.008 and the 85th pct is 0.020,
>   then 0.012 is between → MEDIUM → label = "approach-medium".

### Step 3 — window-stability debounce (binary temporal) — no thresholds, pure logic
For contact/grasp/support transitions over `t-5:t`:
- read the binary trace b[t-5..t].
- require start-stable: b[t-5]==b[t-4]  ; end-stable: b[t-1]==b[t].
- if not both stable → emit nothing (excluded; pulses/switches ignored).
- else map (b_start, b_end): (0,1)=gain, (1,0)=lose, (1,1)=maintain, (0,0)=maintain-no.
- suppress maintain-no on export.
No numeric threshold here — it's the draft's stability rule realized over K=5.

---

## 4. Alignment error scalar `a` — how position+orientation combine (one tunable)

`tcp-alignment` compares current TCP to the nearest per-class affordance candidate
(candidate placed in world frame via current object pose). The error must be ONE scalar
before binning:
```
e_pos  = ||tcp.p - cand_world.p||                       # meters
e_rot  = angle between tcp.q and cand_world.q           # radians (geodesic on SO(3))
a      = e_pos + lambda_rot * e_rot
```
`lambda_rot` converts radians to meters-equivalent. Set it by SCALE MATCHING, not guess:
`lambda_rot = (std of e_pos over episodes) / (std of e_rot over episodes)`.
This makes a 1-std rotation error count the same as a 1-std position error — principled,
data-driven, single line. Tune visually only if the M1 figure looks off.

"nearest candidate" = argmin over the per-class candidate set of this same `a`.

---

## 5. Calibration script deliverable

`teemo/calibrate_thresholds.py`:
1. load replayed success `.h5` (state-bearing).
2. loop episodes/timesteps, log all raw scalars + deltas + K-window changes.
3. compute every edge per the rules above.
4. dump a single `teemo/thresholds.json` (or write constants into `vocab.py`).
5. print a histogram per quantity with the chosen edges overlaid (sanity figure) —
   this doubles as a paper appendix figure showing bins are well-populated.

Freeze after one run. If you change task or cube size, re-run calibration; don't
hand-edit edges.

---

## 6. Summary: which edges are landmark vs quantile

| Quantity            | Inner/key edge          | Outer edges        |
|---------------------|-------------------------|--------------------|
| distance            | grasp-range (landmark)  | 33/66/90 pct       |
| height              | ±half-cube (landmark)   | ±90 pct            |
| alignment (abs)     | grasp-onset err (lmk)   | 33/66/90 pct       |
| aperture-fit (abs)  | ±grasp width band (lmk) | ±90 pct            |
| contact eps         | mode valley / no-contact tail | —            |
| temporal deadband   | stationary-window 95pct | —                  |
| temporal speed      | —                       | 50/85 pct of moving rate |
| lambda_rot          | std(e_pos)/std(e_rot)   | —                  |

Everything traces to measured statistics of successful behavior. Nothing is invented.
