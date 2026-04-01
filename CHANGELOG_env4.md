# env4 — Arclength Progress + Local Guidance

## Problem with env2/env3
The previous attempts failed because:
- **CenterlineWaypointProgress** picked the nearest waypoint across *all* vessel branches → noisy branch-flipping at bifurcations
- Sparse waypoint rewards drowned by step penalty → agent learns "don't move"
- **Centerlines2D** observation was 154 dimensions → hard to learn from

## What env4 does differently

### 1. Continuous arclength reward (replaces discrete waypoints)
- Projects the guidewire tip onto the *correct-path polyline only* — no branch flipping possible
- Reward = `0.01 × (forward progress in mm) - 0.001 × (cross-track distance)`
- Dense, continuous signal every step. Reward budget: ~+2.0 for a successful 400mm traversal vs -1.0 for doing nothing

### 2. Compact 8-dim local guidance observation (replaces 154-dim Centerlines2D)
- 8 features, each encoding exactly what the policy needs to navigate:

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 0 | `d_rem_norm` | [0, 1] | Remaining arclength to the target, normalized by total path length. 1.0 = at start, 0.0 = at target. Tells the agent "how far to go." |
| 1 | `cross_track_dist` | [0, 50] mm | Perpendicular distance from the guidewire tip to the nearest point on the correct-path centerline. 0 = perfectly on the path. Tells the agent "how far off-path am I." |
| 2 | `tangent_x_2d` | [-1, 1] | X-component of the unit tangent vector of the path at the projection point, projected to 2D (dropping y). Tells the agent "which direction should I be heading" (horizontal). |
| 3 | `tangent_z_2d` | [-1, 1] | Z-component of the same tangent vector. Together with tangent_x, gives the local path direction in the fluoroscopy plane. |
| 4 | `heading_error` | [-π, π] | Signed angle between the device's current tip direction and the path tangent. 0 = perfectly aligned. Positive/negative indicates which way to rotate. |
| 5 | `curvature_ahead` | [0, 10] | Maximum curvature in the next 20mm of path ahead. High values signal an upcoming sharp turn (e.g., aortic arch bend or bifurcation). Lets the agent anticipate and slow down. |
| 6 | `dist_to_bifurcation` | [0, 200] mm | Arclength along the path to the next branching point ahead. Low values warn "a fork is coming — pay attention to steering." |
| 7 | `on_correct_branch` | {0, 1} | 1 if the tip is currently on a vessel branch that belongs to the correct path, 0 if it has strayed onto a wrong branch. Binary alarm signal. |

### 3. Action-space curriculum (optional, `--curriculum` flag)
- Stage 1 (0–200k steps): Guidewire only, catheter auto-follows at 0.8× translation
- Stage 2 (200k–500k): Catheter enabled but scaled ×0.1
- Stage 3 (500k+): Full 4D control

### 4. Heuristic replay seeding (optional, `--heuristic_seeding N` flag)
- Centerline-following heuristic generates "okay" trajectories before training starts
- Translation proportional to remaining distance, rotation aligns with tangent + corrects cross-track error
- Actions normalized to [-1, 1] before storing in replay buffer (matching what SAC expects with `normalize_actions=True`)
- Avoids the early "collapse to do-nothing" failure mode

## Bugs found & fixed during audit
- **Action normalization mismatch**: Heuristic produced raw mm/s actions but replay buffer expects [-1,1] — fixed with inverse normalization
- **Rotation controller incomplete**: Originally only used heading alignment — added cross-track correction term
- **Unused import**: Removed dead `tracking3d_to_2d` import from localguidance.py

---

## All files changed

### New files (6)

| File | Purpose |
|------|---------|
| `eve/eve/util/polyline.py` | Shared polyline projection utilities (`project_onto_polyline`, `compute_cumulative_arclength`, `compute_segment_tangents`, `compute_curvature`) |
| `eve/eve/reward/arclengthprogress.py` | `ArcLengthProgress` reward — continuous progress along correct-path polyline |
| `eve/eve/observation/localguidance.py` | `LocalGuidance` observation — 8-dim compact guidance features |
| `training_scripts/util/env4.py` | `BenchEnv4` — assembles FixedPathfinder + ArcLengthProgress + LocalGuidance |
| `training_scripts/util/action_curriculum.py` | `ActionCurriculumWrapper` — 3-stage catheter curriculum |
| `training_scripts/util/heuristic_controller.py` | `CenterlineFollowerHeuristic` — replay buffer seeding |

### Modified files (3)

| File | Change |
|------|--------|
| `eve/eve/reward/__init__.py` | Added `from .arclengthprogress import ArcLengthProgress` |
| `eve/eve/observation/__init__.py` | Added `from .localguidance import LocalGuidance` |
| `training_scripts/DualDeviceNav_train_vs.py` | Added `--env_version 4`, `--curriculum`, `--heuristic_seeding` flags + seeding loop with action normalization |

### Not implemented (deferred)
- **Section 3.5 — Training speed**: Batch size increase (32→128), update ratio (1/20→1/5), step-replay + MLP. These are hyperparameter-only changes to try after validating that env4's reward design works.
