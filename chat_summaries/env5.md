---
name: env4 Code Analysis
overview: A comprehensive analysis of the env4 implementation rationale, identifying the design decisions, potential inefficiencies, and improvement opportunities based on the reward design, observation structure, and training aids.
todos:
  - id: implement-projection-cache
    content: Create PathProjectionCache to eliminate redundant polyline projections
    status: pending
  - id: optimize-branch-check
    content: Optimize on_correct_branch feature or replace with projection-based inference
    status: pending
  - id: ablation-study
    content: Run ablation experiment with 5-dim vs 8-dim LocalGuidance features
    status: pending
  - id: conditional-copy
    content: Make ActionCurriculumWrapper conditionally copy action array
    status: pending
---

# Analysis of env4 Implementation

## 1. Design Rationale Summary

The env4 design addresses three core failures from env2/env3:

| Problem | env2/env3 | env4 Solution |

|---------|-----------|---------------|

| Branch-flipping noise | `CenterlineWaypointProgress` picks nearest waypoint globally across all branches | `ArcLengthProgress` projects only onto the correct-path polyline |

| Sparse rewards | Discrete waypoints drowned by step penalty | Continuous progress reward every step (0.01 mm forward progress) |

| Observation complexity | 154-dim `Centerlines2D` | Compact 8-dim `LocalGuidance` |

## 2. Identified Inefficiencies

### 2.1 **Redundant Polyline Projection (Most Significant)**

Both `ArcLengthProgress` and `LocalGuidance` independently call `project_onto_polyline()` with the same tip position and polyline:

```14:17:eve/eve/reward/arclengthprogress.py
    def step(self) -> None:
        tip_vessel_cs = self._get_tip_vessel_cs()
        result = project_onto_polyline(tip_vessel_cs, self._polyline, self._cumlen)
        ...
```
```103:117:eve/eve/observation/localguidance.py
    def step(self) -> None:
        tip_vessel = tracking3d_to_vessel_cs(...)
        proj = project_onto_polyline(tip_vessel, self._polyline, self._cumlen)
        ...
```

**Impact**: 2x the projection work per step (80+ segment iterations each).

**Fix**: Add a shared `PathProjectionCache` or have the pathfinder compute projection once per step.

---

### 2.2 **Expensive `on_correct_branch` Check Every Step**

`LocalGuidance._is_on_path_branch()` calls `find_nearest_branch_to_point()` which iterates over ALL branches with O(B * N) complexity:

```72:84:eve/intervention/vesseltree/vesseltree.py
def find_nearest_branch_to_point(point, vessel_tree):
    nearest_branch = None
    min_dist = np.inf
    for branch in vessel_tree.branches:
        distances = np.linalg.norm(branch.coordinates - point, axis=1)
        dist = np.min(distances)
        if dist < min_dist:
            ...
```

**Impact**: For a tree with 5-10 branches, ~100 points each, this is 500-1000 distance computations per step just for a binary feature.

**Fix Options**:

1. Cache branch membership if cross_track_dist < threshold (e.g., 5mm)
2. Use the projection segment index to infer branch membership directly from path_branches
3. Remove the feature if ablation shows it's not critical (cross_track_dist already encodes "how far off path")

---

### 2.3 **Redundant Coordinate Transforms**

Both reward and observation classes independently transform tip position to vessel CS:

- `ArcLengthProgress._get_tip_vessel_cs()`
- `LocalGuidance.step()` → `tracking3d_to_vessel_cs(tip_3d, ...)`

**Impact**: Minor (2 matrix multiplies per step), but adds up.

---

### 2.4 **Unnecessary Array Copy in ActionCurriculumWrapper**

```57:58:training_scripts/util/action_curriculum.py
def step(self, action):
    action = np.asarray(action, dtype=np.float64).copy()  # Always copies
```

**Fix**: Only copy when modification is needed:

```python
if stage < 3:
    action = np.asarray(action, dtype=np.float64).copy()
    # modify action
```

---

## 3. Observation Feature Analysis

| Feature | Value | Potential Redundancy |

|---------|-------|---------------------|

| `d_rem_norm` | [0,1] | Essential - tells agent "how far to go" |

| `cross_track_dist` | [0,50] mm | Essential - "how off-path" |

| `tangent_x_2d`, `tangent_z_2d` | [-1,1] | Essential - "which way to steer" |

| `heading_error` | [-pi,pi] | Somewhat redundant with tangent (can be inferred from tracking) |

| `curvature_ahead` | [0,10] | Useful for anticipation |

| `dist_to_bifurcation` | [0,200] mm | Useful but correlated with curvature |

| `on_correct_branch` | {0,1} | Possibly redundant with cross_track_dist |

**Suggestion**: Run an ablation experiment with just features 0-4 (5-dim) before committing to all 8.

---

## 4. Numerical Robustness

The code handles edge cases well:

- Zero-length segments: `np.maximum(lengths, 1e-8)`
- Zero device direction: `if d_norm < 1e-8`
- Zero path length: Early returns with zeros
- Division by zero in projection: `np.maximum(seg_len_sq, 1e-16)`

---

## 5. Heuristic Controller Assessment

The `CenterlineFollowerHeuristic` is well-designed with:

- Combined heading + cross-track correction (lines 136-137)
- Noise injection for trajectory diversity (lines 141-142)
- Never-retract safeguard (line 143)

**Minor issue**: The minimum forward push of 5.0 mm/s (line 106) might be too aggressive in tight spaces.

---

## 6. Recommended Optimizations

### Priority 1 (High Impact)

1. **Shared projection cache**: Create a per-step cache for polyline projection results shared between reward and observation.

### Priority 2 (Medium Impact)

2. **Optimize `on_correct_branch`**: Either derive from projection segment or remove feature.
3. **Lazy coordinate transforms**: Compute tip_vessel_cs once and pass to both reward and observation.

### Priority 3 (Low Impact)

4. **Conditional action copy**: Only copy action array when curriculum modifies it.
5. **Pre-compute 2D tangents**: Store 2D tangent projections at reset instead of computing per step.

---

## 7. Architecture Suggestion

Consider refactoring to a shared projection service:

```python
# In env4.py, add a shared projection context
class PathContext:
    def __init__(self, pathfinder, intervention):
        self.pathfinder = pathfinder
        self.intervention = intervention
        self._proj_cache = {}
        self._step_id = -1
    
    def get_projection(self, step_id):
        if step_id != self._step_id:
            tip_vessel = self._get_tip_vessel_cs()
            self._proj_cache = project_onto_polyline(tip_vessel, ...)
            self._step_id = step_id
        return self._proj_cache
```

Then pass this context to both `ArcLengthProgress` and `LocalGuidance`.

---

## 8. Summary

The env4 design is **fundamentally sound** and addresses the key failures of env2/env3. The main inefficiency is redundant computation (projection called twice per step, expensive branch lookup). These optimizations would roughly halve the per-step overhead of the reward+observation computation but are unlikely to be the training bottleneck (SOFA simulation dominates).

**Recommendation**: Proceed with training as-is to validate the reward design works. If env4 training succeeds, consider the optimizations as a later polish step. If it fails, the issues are more likely in reward scaling or action curriculum tuning than in computational efficiency.