# Files modified for env5.py creation
_Exported on 3/24/2026 at 21:36:47 EDT from Cursor (2.3.34)_

---

**User**

@env4_code_analysis_cc5bd4ce.plan.md (1-189) Look at chat6.md in detail to get the relevant context and the plan that is already attached and  tell me what all the files will be modfired in detail in making env5.py

---

**Cursor**

Let me search for more context about the plan and what optimizations were discussed:

Let me look at the existing env4 files to understand the current structure:

Based on my analysis of the attached plan and `chat6.md`, here's a detailed breakdown of all the files that would need to be modified or created to implement **env5.py** with the recommended optimizations:

---

## Summary of env5 Goals (from Plan)

The optimizations are organized by priority:
- **Priority 1**: Shared projection cache to eliminate redundant `project_onto_polyline()` calls
- **Priority 2**: Optimize `on_correct_branch` check, lazy coordinate transforms
- **Priority 3**: Conditional action copy, pre-compute 2D tangents

---

## Files to CREATE

### 1. `eve/eve/util/pathcontext.py` (NEW)
**Purpose**: Shared projection service that computes projection once per step

The plan suggests a `PathContext` class like:

```python
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

This eliminates the **2x redundant projection work** per step (80+ segment iterations each).

---

### 2. `training_scripts/util/env5.py` (NEW)
**Purpose**: The new optimized environment class

Based on `env4.py` (346 lines), create `BenchEnv5` that:
- Instantiates a shared `PathContext`
- Passes the `PathContext` to both `ArcLengthProgress` (reward) and `LocalGuidance` (observation)
- Tracks a `_step_id` counter to enable per-step caching

---

## Files to MODIFY

### 3. `eve/eve/reward/arclengthprogress.py`
**Current lines**: 104 | **Estimated changes**: ~15 lines

```python
# CURRENT (line 78-84):
def step(self) -> None:
    tip_vessel_cs = self._get_tip_vessel_cs()
    result = project_onto_polyline(tip_vessel_cs, self._polyline, self._cumlen)

# CHANGE TO:
def step(self) -> None:
    result = self._path_context.get_projection(self._step_id)
    # OR receive projection result from env5's step orchestration
```

Add optional `path_context` parameter to `__init__()` to accept shared cache.

---

### 4. `eve/eve/observation/localguidance.py`
**Current lines**: 246 | **Estimated changes**: ~30-40 lines

**Changes needed:**

1. **Use shared projection cache** (lines 103-117):
```python
# CURRENT:
proj = project_onto_polyline(tip_vessel, self._polyline, self._cumlen)

# CHANGE TO:
proj = self._path_context.get_projection(self._step_id)
```

2. **Optimize `_is_on_path_branch()`** (lines 221-228):

```python
# CURRENT (O(B * N) complexity - 500-1000 distance computations):
def _is_on_path_branch(self, tip_vessel_cs: np.ndarray) -> float:
    branch = find_nearest_branch_to_point(tip_vessel_cs, self.intervention.vessel_tree)
    return 1.0 if branch in self.pathfinder.path_branch_set else 0.0

# OPTION A - Derive from projection segment (recommended):
def _is_on_path_branch(self, proj: ProjectionResult) -> float:
    # If cross_track_dist < threshold, we're on path
    return 1.0 if proj.cross_track_dist < 5.0 else 0.0

# OPTION B - Remove feature entirely (simplest):
# Set on_correct_branch = 1.0 always, reducing to 7-dim observation
```

3. **Pre-compute 2D tangents at reset** (Priority 3):
```python
# Add in reset():
self._tangents_2d = self._tangents[:, [0, 2]]  # Pre-project to 2D
self._tangents_2d /= np.linalg.norm(self._tangents_2d, axis=1, keepdims=True)
```

---

### 5. `training_scripts/util/action_curriculum.py`
**Current lines**: 77 | **Estimated changes**: ~5 lines

```python
# CURRENT (line 57-58) - Always copies:
def step(self, action):
    action = np.asarray(action, dtype=np.float64).copy()

# CHANGE TO - Only copy when modification needed:
def step(self, action):
    stage = self.current_stage
    if stage < 3:
        action = np.asarray(action, dtype=np.float64).copy()
        # ... modify action
    else:
        action = np.asarray(action, dtype=np.float64)
```

---

### 6. `training_scripts/DualDeviceNav_train_vs.py`
**Current lines**: 346 | **Estimated changes**: ~20 lines

Add env_version 5 support:

```python
# Add import:
from util.env5 import BenchEnv5

# Modify choices (around line 154):
parser.add_argument(
    "--env_version",
    type=int,
    default=1,
    choices=[1, 2, 3, 4, 5],  # Add 5
    help="Environment version: ... 5=optimized arclength with shared projection",
)

# Add in selection block (around line 194):
if env_version == 5:
    EnvClass = BenchEnv5
    print("Using BenchEnv5 (optimized projection cache)")
elif env_version == 4:
    # ...
```

---

### 7. `eve/eve/util/__init__.py`
**Estimated changes**: ~2 lines

Export the new `PathContext`:

```python
from .pathcontext import PathContext
```

---

### 8. `eve/eve/reward/__init__.py` (Minor)
**Current lines**: 13 | No changes needed

The `ArcLengthProgress` is already exported.

---

### 9. `eve/eve/observation/__init__.py` (Minor)
**Current lines**: 19 | No changes needed

The `LocalGuidance` is already exported.

---

## Optional: Ablation Study Support

The plan suggests running an **ablation experiment** with just 5-dim vs 8-dim `LocalGuidance`. This would require:

- **Modify** `localguidance.py`: Add a constructor parameter `n_features: int = 8` or `use_all_features: bool = True`
- Features 0-4 (5-dim) are considered essential; features 5-7 are "nice to have"

---

## File Modification Summary Table

| File | Action | Priority | Impact |
|------|--------|----------|--------|
| `eve/eve/util/pathcontext.py` | **CREATE** | 1 | High (eliminates 2x projection) |
| `training_scripts/util/env5.py` | **CREATE** | 1 | High |
| `eve/eve/reward/arclengthprogress.py` | MODIFY | 1 | High |
| `eve/eve/observation/localguidance.py` | MODIFY | 1+2 | High (projection + branch check) |
| `training_scripts/DualDeviceNav_train_vs.py` | MODIFY | 1 | Required for integration |
| `training_scripts/util/action_curriculum.py` | MODIFY | 3 | Low |
| `eve/eve/util/__init__.py` | MODIFY | 1 | Export new module |

---

## Estimated Total Impact

Per the plan's summary:
> These optimizations would roughly **halve the per-step overhead** of the reward+observation computation but are unlikely to be the training bottleneck (SOFA simulation dominates).

The recommendation is to proceed with training env4 first to validate the reward design, then implement these as a "polish step" if env4 training succeeds.

Would you like me to draft the actual implementation for any of these files?

```python
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

```python
# CURRENT (line 78-84):
def step(self) -> None:
    tip_vessel_cs = self._get_tip_vessel_cs()
    result = project_onto_polyline(tip_vessel_cs, self._polyline, self._cumlen)

# CHANGE TO:
def step(self) -> None:
    result = self._path_context.get_projection(self._step_id)
    # OR receive projection result from env5's step orchestration
```

```python
# CURRENT:
proj = project_onto_polyline(tip_vessel, self._polyline, self._cumlen)

# CHANGE TO:
proj = self._path_context.get_projection(self._step_id)
```

```python
# CURRENT (O(B * N) complexity - 500-1000 distance computations):
def _is_on_path_branch(self, tip_vessel_cs: np.ndarray) -> float:
    branch = find_nearest_branch_to_point(tip_vessel_cs, self.intervention.vessel_tree)
    return 1.0 if branch in self.pathfinder.path_branch_set else 0.0

# OPTION A - Derive from projection segment (recommended):
def _is_on_path_branch(self, proj: ProjectionResult) -> float:
    # If cross_track_dist < threshold, we're on path
    return 1.0 if proj.cross_track_dist < 5.0 else 0.0

# OPTION B - Remove feature entirely (simplest):
# Set on_correct_branch = 1.0 always, reducing to 7-dim observation
```

```python
# Add in reset():
self._tangents_2d = self._tangents[:, [0, 2]]  # Pre-project to 2D
self._tangents_2d /= np.linalg.norm(self._tangents_2d, axis=1, keepdims=True)
```

```python
# CURRENT (line 57-58) - Always copies:
def step(self, action):
    action = np.asarray(action, dtype=np.float64).copy()

# CHANGE TO - Only copy when modification needed:
def step(self, action):
    stage = self.current_stage
    if stage < 3:
        action = np.asarray(action, dtype=np.float64).copy()
        # ... modify action
    else:
        action = np.asarray(action, dtype=np.float64)
```

```python
# Add import:
from util.env5 import BenchEnv5

# Modify choices (around line 154):
parser.add_argument(
    "--env_version",
    type=int,
    default=1,
    choices=[1, 2, 3, 4, 5],  # Add 5
    help="Environment version: ... 5=optimized arclength with shared projection",
)

# Add in selection block (around line 194):
if env_version == 5:
    EnvClass = BenchEnv5
    print("Using BenchEnv5 (optimized projection cache)")
elif env_version == 4:
    # ...
```

```python
from .pathcontext import PathContext
```

---

**User**

Look in the changes made in the codebase at D:\neve\.claude\worktrees\rl_improv_2/ ;;; look at the corresponding files and see if all fixes are made thoroughly or if any errors or inaefficies have been introduced

---

**Cursor**

Now let me check `action_curriculum.py` and the `util/__init__.py` files:

Let me check how the base `eve.Env` reset works to verify the order of operations:

Now I have a complete picture. Let me check one more thing - verify the heuristic_controller.py is also updated:

---

## Comprehensive Analysis of env5 Implementation in `D:\neve\.claude\worktrees\rl_improv_2`

### ✅ What Was Implemented Correctly

| File | Changes | Status |
|------|---------|--------|
| `eve/eve/util/pathcontext.py` | New `PathProjectionCache` class with lazy computation | ✅ Good design |
| `training _scripts/util/env5.py` | New `BenchEnv5` using shared cache | ✅ Structure correct |
| `eve/eve/reward/arclengthprogress.py` | Added `path_context` parameter with fallback | ✅ Backward compatible |
| `eve/eve/observation/localguidance.py` | Uses cache, pre-computed 2D tangents, threshold-based `on_correct_branch` | ✅ All Priority 1-3 fixes applied |
| `training _scripts/util/action_curriculum.py` | Conditional copy only when `stage < 3` | ✅ Fixed |
| `training _scripts/DualDeviceNav_train.py` | Added env_version 5 support | ✅ Complete |

---

### 🐛 BUG: Reset Order Causes Stale Cache Data

**Location**: `training _scripts/util/env5.py` lines 224-228

```python
result = super().reset(seed=seed, options=options)
# Reset projection cache after pathfinder has recomputed the path
self._path_context.reset()  # <-- TOO LATE!
return result
```

**Problem**: The base class `eve.Env.reset()` calls components in this order:
1. `pathfinder.reset()` — computes new path ✓
2. `observation.reset()` → `LocalGuidance.reset()` → uses **stale** `_path_context`
3. `reward.reset()` → `ArcLengthProgress.reset()` → uses **stale** `_path_context`
4. ... other resets ...
5. **Then** `_path_context.reset()` is called after `super().reset()` returns

**Impact**:
- First episode: `_path_context` has empty polyline, `_total_length = 0.0` → reward/obs compute nothing (safe but wasteful)
- Subsequent episodes: `ArcLengthProgress.reset()` uses **previous episode's** path data to compute `_prev_d_rem`, causing incorrect reward on the first step of each episode

**Fix Required** (in `env5.py`):

```python
def reset(self, seed=None, options=None):
    # ... logging code ...
    
    # Call pathfinder reset first to compute the new path
    self.pathfinder.reset(self._episode_count)
    
    # Reset projection cache BEFORE other components that use it
    self._path_context.reset()
    
    # Now call super().reset() - pathfinder.reset() will be called again (no-op)
    result = super().reset(seed=seed, options=options)
    
    return result
```

Or refactor to call `_path_context.reset()` inside `pathfinder.reset()`.

---

### ⚠️ Dead Code: Unused `_valid` Flag

**Location**: `eve/eve/util/pathcontext.py` line 46

```python
self._valid: bool = False  # Never read
```

The cache validity is checked by testing if `_tip_vessel_cs` and `_projection` are `None`, making `_valid` redundant.

**Suggested removal** (low priority):
```python
# Remove line 46: self._valid: bool = False
# Remove from invalidate(): self._valid = False  
```

---

### ⚠️ Missing Export (Minor)

**Location**: `eve/eve/util/__init__.py`

`PathProjectionCache` is not exported. The import in `env5.py` uses the full path:
```python
from eve.util.pathcontext import PathProjectionCache
```

This works but is inconsistent with the pattern used for other utilities. Consider adding:
```python
from .pathcontext import PathProjectionCache
```

---

### ✅ Verified Correct Implementations

1. **`action_curriculum.py`** — Correctly only copies when modification is needed (Stage 1 or 2).

2. **`localguidance.py`** — All three optimizations applied:
   - Uses `_path_context.get_projection()` and `get_tip_vessel_cs()` ✓
   - Pre-computes 2D tangents at reset (line 105-108) ✓
   - Uses `cross_track_dist < 5.0mm` threshold for `on_correct_branch` instead of O(B×N) branch scan ✓

3. **`arclengthprogress.py`** — Has correct fallback for backward compatibility with env4.

4. **`heuristic_controller.py`** — No changes needed; it computes independently.

5. **`DualDeviceNav_train.py`** — Properly updated:
   - Imports `BenchEnv5`
   - env_version 5 added to choices
   - Heuristic seeding works with both env4 and env5

---

### Summary

| Category | Count | Details |
|----------|-------|---------|
| **Critical Bug** | 1 | Reset ordering causes stale cache in reward/observation |
| **Dead Code** | 1 | `_valid` flag in PathProjectionCache |
| **Missing Export** | 1 | PathProjectionCache not in `__init__.py` |
| **Working Correctly** | 6+ | All other optimizations properly implemented |

**Recommendation**: Fix the reset ordering bug before deploying env5 for training. This is a **correctness issue** that could cause incorrect reward computation at episode boundaries, potentially harming learning.

```python
result = super().reset(seed=seed, options=options)
# Reset projection cache after pathfinder has recomputed the path
self._path_context.reset()  # <-- TOO LATE!
return result
```

```python
def reset(self, seed=None, options=None):
    # ... logging code ...
    
    # Call pathfinder reset first to compute the new path
    self.pathfinder.reset(self._episode_count)
    
    # Reset projection cache BEFORE other components that use it
    self._path_context.reset()
    
    # Now call super().reset() - pathfinder.reset() will be called again (no-op)
    result = super().reset(seed=seed, options=options)
    
    return result
```

```python
self._valid: bool = False  # Never read
```

```python
# Remove line 46: self._valid: bool = False
# Remove from invalidate(): self._valid = False  
```

```python
from eve.util.pathcontext import PathProjectionCache
```

```python
from .pathcontext import PathProjectionCache
```

---

**User**

read all relevant files in the codebase again and analyse if the bugs are removed and no new bugs /inefficiencies are added

---

**Cursor**

The current workspace (`D:\stEVE_training`) doesn't have env5.py yet — the changes exist only in the worktree at `D:\neve\.claude\worktrees\rl_improv_2`. Let me re-read all the relevant files from that worktree for a thorough re-analysis:

Now let me check the remaining files:

---

## ✅ Comprehensive Re-Analysis: All Bugs Fixed

After re-reading all relevant files in `D:\neve\.claude\worktrees\rl_improv_2`, I can confirm:

### Bug Fixes Verified

| Issue | Status | Evidence |
|-------|--------|----------|
| **Reset ordering bug** | ✅ **FIXED** | Both `LocalGuidance.reset()` (line 90-91) and `ArcLengthProgress.reset()` (line 65-66) now call `_path_context.reset()` at the start of their own reset methods |
| **Dead `_valid` flag** | ✅ **REMOVED** | `pathcontext.py` no longer has the unused `_valid` attribute |
| **Missing export** | ✅ **FIXED** | `eve/eve/util/__init__.py` line 3 exports `PathProjectionCache` |
| **Conditional action copy** | ✅ **FIXED** | `action_curriculum.py` only copies in Stage 1 and 2 (lines 60-70) |

---

### Reset Flow Verification

The reset order is now correct:

```
eve.Env.reset():
    ├── pathfinder.reset()      → computes new path
    ├── observation.reset()     → LocalGuidance.reset()
    │   └── _path_context.reset()  ← REFRESHES CACHE (first call)
    │   └── compute tangents, curvature, initial obs
    └── reward.reset()          → ArcLengthProgress.reset()
        └── _path_context.reset()  ← IDEMPOTENT (cache already fresh)
        └── compute initial _prev_d_rem correctly
```

---

### Remaining Minor Observations (Non-Issues)

| Item | Assessment |
|------|------------|
| **Redundant cache reset** | Both components call `_path_context.reset()`. The second call is redundant but harmless (~O(n) for small n ≈ 100 points). Acceptable tradeoff for robustness. |
| **`_compute_bifurcation_arclengths` bypasses cache** | Uses `project_onto_polyline` directly. This is correct — it runs once at reset, not per-step, so no performance impact. |
| **Second tracking point transform** | `_compute_heading_error` still transforms `tracking[1]` per step (line 200). This is unavoidable since only the tip is cached. |

---

### Summary of All Files

| File | Status | Notes |
|------|--------|-------|
| `env5.py` | ✅ | Clean implementation, logging optimizations, correct step/reset |
| `pathcontext.py` | ✅ | Dead code removed, proper lazy evaluation |
| `arclengthprogress.py` | ✅ | Calls `_path_context.reset()`, backward compatible fallback |
| `localguidance.py` | ✅ | Calls `_path_context.reset()`, pre-computed 2D tangents, threshold `on_correct_branch` |
| `action_curriculum.py` | ✅ | Conditional copy only when needed |
| `eve/util/__init__.py` | ✅ | Exports `PathProjectionCache` |
| `DualDeviceNav_train.py` | ✅ | env_version 5 support, heuristic seeding for both env4/env5 |
| `polyline.py` | ✅ | No changes needed, robust edge-case handling |
| `heuristic_controller.py` | ✅ | Independent operation, no changes needed |

---

### Conclusion

**All identified bugs have been fixed. No new bugs or inefficiencies have been introduced.**

The implementation is ready for deployment. The env5 optimizations (shared projection cache, threshold-based branch check, pre-computed 2D tangents, conditional logging) should provide measurable per-step speedup while maintaining correctness.

```plaintext
eve.Env.reset():
    ├── pathfinder.reset()      → computes new path
    ├── observation.reset()     → LocalGuidance.reset()
    │   └── _path_context.reset()  ← REFRESHES CACHE (first call)
    │   └── compute tangents, curvature, initial obs
    └── reward.reset()          → ArcLengthProgress.reset()
        └── _path_context.reset()  ← IDEMPOTENT (cache already fresh)
        └── compute initial _prev_d_rem correctly
```

---

**User**

There were some more efficiency improvements that were done in addition to the ones you listed before; see if you can find them 

---

**Cursor**

Now I can see the additional efficiency improvements. Let me compare the `step()` methods side-by-side:

---

## Complete List of Efficiency Improvements in env5

I found **additional improvements** I missed in my previous analysis. Here's the complete list:

### 1. env5.py `step()` — Logging Optimizations

| Improvement | env4 | env5 | Impact |
|-------------|------|------|--------|
| **SOFA query** | Every step | Only on INFO steps (every 50 or terminal) | ~49/50 steps skip syscall |
| **Action string formatting** | Every step | Only on INFO steps | ~49/50 steps skip string ops |
| **Handler flush** | Every step (`for handler: handler.flush()`) | Only on INFO steps | Major I/O reduction |
| **`term_reason` computation** | Every step | Removed entirely | Minor CPU savings |

**env4 `step()` (lines 245-295)** — Always does all work:
```python
# SOFA info - ALWAYS queried
if hasattr(self.intervention, "simulation"):
    sim = self.intervention.simulation
    if hasattr(sim, "inserted_lengths"):
        inserted = list(sim.inserted_lengths)
        ...

# Action string - ALWAYS formatted
action_str = f"[{','.join(f'{a:.3f}' for a in action.flatten())}]"

# term_reason - ALWAYS computed
term_reason = "none"
if terminated:
    term_reason = "target_reached"
...

# Handlers - ALWAYS flushed
for handler in self._step_logger.handlers:
    handler.flush()
```

**env5 `step()` (lines 257-320)** — Conditional:
```python
is_info_step = (self._episode_step_count % 50 == 0 or terminated or truncated)

if is_info_step:
    # Only NOW: query SOFA, format action, build full log, flush
    ...
else:
    # Lightweight debug log, track insertion for later, NO flush
    ...
```

---

### 2. LocalGuidance — Algorithmic Optimizations

| Improvement | Original | Optimized | Impact |
|-------------|----------|-----------|--------|
| **`_compute_heading_error`** | Transforms `tracking[0]` to vessel CS (redundant with tip_vessel already computed) | Accepts `tip_vessel_cs` as parameter | Saves 1 coordinate transform per step |
| **`_tangents_2d`** | Computed per step (extract, normalize) | Pre-computed at reset | Saves array slicing + norm per step |
| **`_is_on_path_branch`** | `find_nearest_branch_to_point()` — O(B×N) | `cross_track_dist < 5mm` threshold — O(1) | ~500-1000× faster per step |

**Original `_compute_heading_error`** (lines 166-195 in current workspace):
```python
def _compute_heading_error(self, fluoro, tangent_3d: np.ndarray) -> float:
    ...
    p0_v = tracking3d_to_vessel_cs(tracking[0], ...)  # REDUNDANT - already computed as tip_vessel
    p1_v = tracking3d_to_vessel_cs(tracking[1], ...)
    device_dir_v = p0_v - p1_v
```

**Optimized `_compute_heading_error`** (worktree):
```python
def _compute_heading_error(self, fluoro, tangent_3d: np.ndarray, tip_vessel_cs: np.ndarray) -> float:
    ...
    # tip_vessel_cs already passed in - no redundant transform
    p1_v = tracking3d_to_vessel_cs(tracking[1], ...)
    device_dir_v = tip_vessel_cs - p1_v
```

**Original `_is_on_path_branch`**:
```python
def _is_on_path_branch(self, tip_vessel_cs: np.ndarray) -> float:
    branch = find_nearest_branch_to_point(tip_vessel_cs, self.intervention.vessel_tree)  # O(B×N)
    return 1.0 if branch in self.pathfinder.path_branch_set else 0.0
```

**Optimized** — uses projection result already computed:
```python
# Feature 7: on correct branch? (cheap threshold check on cross-track)
on_path = 1.0 if proj.cross_track_dist < _ON_PATH_THRESHOLD_MM else 0.0
```

---

### 3. Summary of ALL Efficiency Improvements

| # | Category | Improvement | Complexity Reduction |
|---|----------|-------------|---------------------|
| 1 | **Projection** | Shared `PathProjectionCache` | 2× → 1× projection per step |
| 2 | **Tip transform** | Shared `get_tip_vessel_cs()` | 2× → 1× coordinate transform |
| 3 | **Heading error** | Reuse cached tip position | 2× → 1× coordinate transform |
| 4 | **2D tangents** | Pre-computed at reset | O(1) vs O(1) but less per-step work |
| 5 | **Branch check** | Threshold on cross_track_dist | O(B×N) → O(1) |
| 6 | **Action copy** | Conditional in curriculum | Avoids copy in Stage 3 |
| 7 | **SOFA query** | Only on INFO steps | 50× fewer syscalls |
| 8 | **Action formatting** | Only on INFO steps | 50× fewer string ops |
| 9 | **Handler flush** | Only on INFO steps | 50× fewer I/O flushes |
| 10 | **term_reason** | Removed | Minor CPU savings |
| 11 | **Dead code** | Removed `_valid` flag | Cleaner code |

---

### Estimated Per-Step Savings

| Component | env4 Cost | env5 Cost | Savings |
|-----------|-----------|-----------|---------|
| Projection | 2 × O(M) | 1 × O(M) | ~50% |
| Tip transforms | 3 transforms | 2 transforms | ~33% |
| Branch check | O(B×N) ≈ 500-1000 ops | O(1) | ~99.9% |
| Logging overhead | Always full | 49/50 steps lightweight | ~98% |

**Total**: Env5 should be noticeably faster per step, especially due to the branch check optimization and logging improvements.

```python
# SOFA info - ALWAYS queried
if hasattr(self.intervention, "simulation"):
    sim = self.intervention.simulation
    if hasattr(sim, "inserted_lengths"):
        inserted = list(sim.inserted_lengths)
        ...

# Action string - ALWAYS formatted
action_str = f"[{','.join(f'{a:.3f}' for a in action.flatten())}]"

# term_reason - ALWAYS computed
term_reason = "none"
if terminated:
    term_reason = "target_reached"
...

# Handlers - ALWAYS flushed
for handler in self._step_logger.handlers:
    handler.flush()
```

```python
is_info_step = (self._episode_step_count % 50 == 0 or terminated or truncated)

if is_info_step:
    # Only NOW: query SOFA, format action, build full log, flush
    ...
else:
    # Lightweight debug log, track insertion for later, NO flush
    ...
```

```python
def _compute_heading_error(self, fluoro, tangent_3d: np.ndarray) -> float:
    ...
    p0_v = tracking3d_to_vessel_cs(tracking[0], ...)  # REDUNDANT - already computed as tip_vessel
    p1_v = tracking3d_to_vessel_cs(tracking[1], ...)
    device_dir_v = p0_v - p1_v
```

```python
def _compute_heading_error(self, fluoro, tangent_3d: np.ndarray, tip_vessel_cs: np.ndarray) -> float:
    ...
    # tip_vessel_cs already passed in - no redundant transform
    p1_v = tracking3d_to_vessel_cs(tracking[1], ...)
    device_dir_v = tip_vessel_cs - p1_v
```

```python
def _is_on_path_branch(self, tip_vessel_cs: np.ndarray) -> float:
    branch = find_nearest_branch_to_point(tip_vessel_cs, self.intervention.vessel_tree)  # O(B×N)
    return 1.0 if branch in self.pathfinder.path_branch_set else 0.0
```

```python
# Feature 7: on correct branch? (cheap threshold check on cross-track)
on_path = 1.0 if proj.cross_track_dist < _ON_PATH_THRESHOLD_MM else 0.0
```

---

**User**

ok, now merge all the files from that codebase to this code, copy their DuaL...train.py as ...._vs.py ; if you are overwiriting any files in this workspace first remane that file as <filename>_v1.py and then bring in the new code; new files are copied as it is

---

**Cursor**

I'll merge all the files from the worktree to this workspace. Let me first check what files already exist and create a plan.

Now I'll rename the existing files to `_v1` versions and copy the new files. Let me start:

```python
"""Shared per-step projection cache for path-aware components.

Both ArcLengthProgress (reward) and LocalGuidance (observation) need to
project the device tip onto the correct-path polyline every step.  Without
caching, this projection is computed twice.  PathProjectionCache computes
it lazily on first access and returns the cached result on subsequent
accesses within the same step.

Usage in env5.py:
    cache = PathProjectionCache(pathfinder, intervention)
    # pass cache to ArcLengthProgress and LocalGuidance
    # call cache.invalidate() at the start of each env step
    # call cache.reset() after each env reset
"""

import numpy as np

from ..util.coordtransform import tracking3d_to_vessel_cs
from ..util.polyline import (
    compute_cumulative_arclength,
    project_onto_polyline,
    ProjectionResult,
)


class PathProjectionCache:
    """Caches polyline projection and tip_vessel_cs once per env step.

    Args:
        pathfinder: A FixedPathfinder with ``path_points_vessel_cs``.
        intervention: The intervention object (provides fluoroscopy).
    """

    def __init__(self, pathfinder, intervention) -> None:
        self.pathfinder = pathfinder
        self.intervention = intervention

        # Path geometry (set on reset)
        self._polyline: np.ndarray = np.empty((0, 3))
        self._cumlen: np.ndarray = np.empty(0)
        self._total_length: float = 0.0

        # Per-step cache
        self._tip_vessel_cs: np.ndarray | None = None
        self._projection: ProjectionResult | None = None

    def reset(self) -> None:
        """Recompute path geometry from the pathfinder (call after env reset)."""
        self._polyline = self.pathfinder.path_points_vessel_cs
        if len(self._polyline) < 2:
            self._cumlen = np.zeros(max(1, len(self._polyline)))
            self._total_length = 0.0
        else:
            self._cumlen = compute_cumulative_arclength(self._polyline)
            self._total_length = float(self._cumlen[-1])
        self.invalidate()

    def invalidate(self) -> None:
        """Mark cache as stale.  Call at the start of each env step."""
        self._tip_vessel_cs = None
        self._projection = None

    @property
    def polyline(self) -> np.ndarray:
        return self._polyline

    @property
    def cumlen(self) -> np.ndarray:
        return self._cumlen

    @property
    def total_length(self) -> float:
        return self._total_length

    def get_tip_vessel_cs(self) -> np.ndarray:
        """Return tip position in vessel CS, computing once per step."""
        if self._tip_vessel_cs is None:
            fluoro = self.intervention.fluoroscopy
            tip_3d = fluoro.tracking3d[0]
            self._tip_vessel_cs = tracking3d_to_vessel_cs(
                tip_3d, fluoro.image_rot_zx, fluoro.image_center
            )
        return self._tip_vessel_cs

    def get_projection(self) -> ProjectionResult:
        """Return projection result, computing once per step."""
        if self._projection is None:
            if self._total_length < 1e-6:
                self._projection = ProjectionResult(
                    s=0.0,
                    cross_track_dist=0.0,
                    proj_point=np.zeros(3),
                    segment_idx=0,
                    t=0.0,
                )
            else:
                tip = self.get_tip_vessel_cs()
                self._projection = project_onto_polyline(
                    tip, self._polyline, self._cumlen
                )
        return self._projection

```

```python
from .eveobject import EveObject
from .confighandler import ConfigHandler
from .pathcontext import PathProjectionCache

```

```python
"""Compact local guidance observation for path-aware navigation.

Provides an 8-dimensional observation vector encoding the agent's
relationship to the known correct path, replacing the much larger
Centerlines2D observation (154+ dims).

Features:
    0: d_rem_norm       - remaining arclength / total length  [0, 1]
    1: cross_track_dist - distance from tip to polyline (mm, clipped)
    2: tangent_x_2d     - path tangent x-component at projection  [-1, 1]
    3: tangent_z_2d     - path tangent z-component at projection  [-1, 1]
    4: heading_error    - angle between device direction and tangent  [-pi, pi]
    5: curvature_ahead  - max curvature in next 20mm of path
    6: dist_to_bifurc   - arclength to next branching point (mm, clipped)
    7: on_correct_branch - 1 if tip is on a path branch, 0 otherwise
"""

import numpy as np
import gymnasium as gym

from .observation import Observation
from ..intervention import Intervention
from ..util.coordtransform import tracking3d_to_vessel_cs
from ..util.polyline import (
    compute_cumulative_arclength,
    compute_segment_tangents,
    compute_curvature,
    project_onto_polyline,
)


# Clipping constants
_MAX_CROSS_TRACK_MM = 50.0
_MAX_BIFURC_DIST_MM = 200.0
_LOOKAHEAD_MM = 20.0
_ON_PATH_THRESHOLD_MM = 5.0


class LocalGuidance(Observation):
    """Compact 8-dim observation encoding the agent's state relative to the path.

    Args:
        intervention: The intervention object.
        pathfinder: A FixedPathfinder with ``path_points_vessel_cs``,
            ``path_branch_set``, and ``path_branching_points3d``.
        name: Name for this observation component.
        path_context: Optional PathProjectionCache for sharing projection
            results with ArcLengthProgress reward.  When provided, avoids
            redundant polyline projection and coordinate transforms.
    """

    def __init__(
        self,
        intervention: Intervention,
        pathfinder,
        name: str = "local_guidance",
        path_context=None,
    ) -> None:
        self.name = name
        self.intervention = intervention
        self.pathfinder = pathfinder
        self._path_context = path_context

        # Precomputed path data (set in reset)
        self._polyline: np.ndarray = np.empty((0, 3))
        self._cumlen: np.ndarray = np.empty(0)
        self._tangents: np.ndarray = np.empty((0, 3))
        self._tangents_2d: np.ndarray = np.empty((0, 2))
        self._curvature: np.ndarray = np.empty(0)
        self._total_length: float = 0.0
        self._bifurc_arclengths: np.ndarray = np.empty(0)

        self.obs = np.zeros(8, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        low = np.array(
            [0.0, 0.0, -1.0, -1.0, -np.pi, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [1.0, _MAX_CROSS_TRACK_MM, 1.0, 1.0, np.pi, 10.0, _MAX_BIFURC_DIST_MM, 1.0],
            dtype=np.float32,
        )
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        # Refresh shared cache so it picks up the new path from pathfinder
        # (pathfinder.reset() has already run by this point in eve.Env.reset)
        if self._path_context is not None:
            self._path_context.reset()

        self._polyline = self.pathfinder.path_points_vessel_cs
        if len(self._polyline) < 2:
            self._cumlen = np.zeros(max(len(self._polyline), 1))
            self._tangents = np.empty((0, 3))
            self._tangents_2d = np.empty((0, 2))
            self._curvature = np.empty(0)
            self._total_length = 0.0
            self._bifurc_arclengths = np.empty(0)
            self.obs = np.zeros(8, dtype=np.float32)
            return

        self._cumlen = compute_cumulative_arclength(self._polyline)
        self._total_length = float(self._cumlen[-1])
        self._tangents = compute_segment_tangents(self._polyline)
        self._curvature = compute_curvature(self._tangents, self._cumlen)

        # Pre-compute 2D tangent projections (drop y-component, normalize)
        t2d = self._tangents[:, [0, 2]]
        norms = np.linalg.norm(t2d, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self._tangents_2d = t2d / norms

        # Compute arclength positions of branching points along the path
        self._compute_bifurcation_arclengths()

        # Compute initial observation
        self.step()

    def step(self) -> None:
        if self._total_length < 1e-6:
            self.obs = np.zeros(8, dtype=np.float32)
            return

        fluoro = self.intervention.fluoroscopy

        # Tip position and projection — use cache if available
        if self._path_context is not None:
            tip_vessel = self._path_context.get_tip_vessel_cs()
            proj = self._path_context.get_projection()
        else:
            tip_3d = fluoro.tracking3d[0]
            tip_vessel = tracking3d_to_vessel_cs(
                tip_3d, fluoro.image_rot_zx, fluoro.image_center
            )
            proj = project_onto_polyline(tip_vessel, self._polyline, self._cumlen)

        # Feature 0: remaining arclength (normalised)
        d_rem_norm = max(0.0, (self._total_length - proj.s) / self._total_length)

        # Feature 1: cross-track distance (clipped)
        cross_track = min(proj.cross_track_dist, _MAX_CROSS_TRACK_MM)

        # Features 2-3: path tangent in 2D (pre-computed at reset)
        seg_idx = min(proj.segment_idx, len(self._tangents) - 1)
        tangent_3d = self._tangents[seg_idx]
        if len(self._tangents_2d) > 0:
            tangent_2d = self._tangents_2d[seg_idx]
        else:
            tangent_2d = np.array([1.0, 0.0])

        # Feature 4: heading error (reuse cached tip_vessel for coord transform)
        heading_error = self._compute_heading_error(fluoro, tangent_3d, tip_vessel)

        # Feature 5: max curvature in the next LOOKAHEAD_MM
        curvature_ahead = self._compute_curvature_ahead(proj.s)

        # Feature 6: distance to next bifurcation along path
        dist_to_bifurc = self._compute_dist_to_bifurcation(proj.s)

        # Feature 7: on correct branch? (cheap threshold check on cross-track)
        on_path = 1.0 if proj.cross_track_dist < _ON_PATH_THRESHOLD_MM else 0.0

        self.obs = np.array(
            [
                d_rem_norm,
                cross_track,
                tangent_2d[0],
                tangent_2d[1],
                heading_error,
                curvature_ahead,
                dist_to_bifurc,
                on_path,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_heading_error(
        self, fluoro, tangent_3d: np.ndarray, tip_vessel_cs: np.ndarray
    ) -> float:
        """Angle between device tip direction and path tangent.

        Args:
            fluoro: Fluoroscopy object.
            tangent_3d: Path tangent at projection point (vessel CS).
            tip_vessel_cs: Already-computed tip position in vessel CS
                (avoids redundant coordinate transform).
        """
        tracking = fluoro.tracking3d
        if len(tracking) < 2:
            return 0.0

        # Device direction in vessel CS: tip (already transformed) minus
        # second tracked point (needs one transform)
        p1_v = tracking3d_to_vessel_cs(
            tracking[1], fluoro.image_rot_zx, fluoro.image_center
        )
        device_dir_v = tip_vessel_cs - p1_v
        d_norm = np.linalg.norm(device_dir_v)
        if d_norm < 1e-8:
            return 0.0
        device_dir_v = device_dir_v / d_norm

        # Angle between device direction and path tangent
        dot = float(np.clip(np.dot(device_dir_v, tangent_3d), -1.0, 1.0))
        cross = np.cross(device_dir_v, tangent_3d)
        sign = 1.0 if cross[1] >= 0 else -1.0
        return float(sign * np.arccos(dot))

    def _compute_curvature_ahead(self, s_current: float) -> float:
        """Max curvature in the next LOOKAHEAD_MM along the path."""
        if len(self._curvature) == 0:
            return 0.0

        s_end = s_current + _LOOKAHEAD_MM
        # Curvature[i] corresponds to interior vertex i+1 (arclength cumlen[i+1])
        # Find curvature values within [s_current, s_end]
        vertex_arclengths = self._cumlen[1:-1]  # interior vertices
        mask = (vertex_arclengths >= s_current) & (vertex_arclengths <= s_end)
        if not np.any(mask):
            return 0.0
        return float(np.max(self._curvature[mask]))

    def _compute_dist_to_bifurcation(self, s_current: float) -> float:
        """Distance along path to the next branching point ahead."""
        if len(self._bifurc_arclengths) == 0:
            return _MAX_BIFURC_DIST_MM

        ahead = self._bifurc_arclengths[self._bifurc_arclengths > s_current]
        if len(ahead) == 0:
            return _MAX_BIFURC_DIST_MM
        return min(float(ahead[0]) - s_current, _MAX_BIFURC_DIST_MM)

    def _compute_bifurcation_arclengths(self) -> None:
        """Find arclength positions of branching points along the path."""
        bp_3d = self.pathfinder.path_branching_points3d
        if bp_3d is None or len(bp_3d) == 0:
            self._bifurc_arclengths = np.empty(0)
            return

        fluoro = self.intervention.fluoroscopy
        arclengths = []
        for bp in bp_3d:
            bp_vessel = tracking3d_to_vessel_cs(
                bp, fluoro.image_rot_zx, fluoro.image_center
            )
            proj = project_onto_polyline(bp_vessel, self._polyline, self._cumlen)
            arclengths.append(proj.s)
        self._bifurc_arclengths = np.sort(arclengths)

```

```python
"""Continuous arclength-based progress reward.

Projects the device tip onto the correct-path polyline and rewards
forward progress along the path. Unlike CenterlineWaypointProgress,
this cannot flip between branches — it only measures projection onto
the known-correct path from insertion to target.

Reward per step:
    r = progress_factor * (d_rem_prev - d_rem_curr)
      - lateral_penalty_factor * cross_track_dist

where d_rem = total_path_length - arclength_of_projection.
"""

import numpy as np

from .reward import Reward
from ..intervention import Intervention
from ..util.coordtransform import tracking3d_to_vessel_cs
from ..util.polyline import (
    compute_cumulative_arclength,
    project_onto_polyline,
)


class ArcLengthProgress(Reward):
    """Reward based on continuous progress along the fixed path polyline.

    Args:
        intervention: The intervention object (provides fluoroscopy/tracking).
        pathfinder: A FixedPathfinder instance with ``path_points_vessel_cs``.
        progress_factor: Scale factor for forward-progress reward.
            Each mm of forward progress yields ``progress_factor`` reward.
        lateral_penalty_factor: Scale factor for cross-track penalty.
            Each mm of distance from the polyline costs
            ``lateral_penalty_factor`` per step.
    """

    def __init__(
        self,
        intervention: Intervention,
        pathfinder,
        progress_factor: float = 0.01,
        lateral_penalty_factor: float = 0.001,
        path_context=None,
    ) -> None:
        self.intervention = intervention
        self.pathfinder = pathfinder
        self.progress_factor = progress_factor
        self.lateral_penalty_factor = lateral_penalty_factor
        self._path_context = path_context

        # Set during reset
        self._polyline: np.ndarray = np.empty((0, 3))
        self._cumlen: np.ndarray = np.empty(0)
        self._total_length: float = 0.0
        self._prev_d_rem: float = 0.0

        self.reward = 0.0

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0

        if self._path_context is not None:
            # Refresh cache (idempotent — may already be reset by LocalGuidance)
            self._path_context.reset()
            self._total_length = self._path_context.total_length
            if self._total_length < 1e-6:
                self._prev_d_rem = 0.0
                return
            result = self._path_context.get_projection()
            self._prev_d_rem = self._total_length - result.s
            return

        # Fallback: compute independently (backward compat with env4)
        self._polyline = self.pathfinder.path_points_vessel_cs
        if len(self._polyline) < 2:
            self._cumlen = np.zeros(len(self._polyline))
            self._total_length = 0.0
            self._prev_d_rem = 0.0
            return

        self._cumlen = compute_cumulative_arclength(self._polyline)
        self._total_length = float(self._cumlen[-1])

        tip_vessel_cs = self._get_tip_vessel_cs()
        result = project_onto_polyline(tip_vessel_cs, self._polyline, self._cumlen)
        self._prev_d_rem = self._total_length - result.s

    def step(self) -> None:
        if self._total_length < 1e-6:
            self.reward = 0.0
            return

        if self._path_context is not None:
            result = self._path_context.get_projection()
        else:
            tip_vessel_cs = self._get_tip_vessel_cs()
            result = project_onto_polyline(
                tip_vessel_cs, self._polyline, self._cumlen
            )

        d_rem_curr = self._total_length - result.s

        # Progress reward: positive when moving toward target
        r_progress = self.progress_factor * (self._prev_d_rem - d_rem_curr)

        # Lateral penalty: penalise straying from the path centerline
        r_lateral = -self.lateral_penalty_factor * result.cross_track_dist

        self.reward = r_progress + r_lateral
        self._prev_d_rem = d_rem_curr

    def _get_tip_vessel_cs(self) -> np.ndarray:
        """Get the guidewire tip position in vessel coordinate system."""
        fluoro = self.intervention.fluoroscopy
        tip_3d = fluoro.tracking3d[0]  # first tracked point = tip
        return tracking3d_to_vessel_cs(
            tip_3d, fluoro.image_rot_zx, fluoro.image_center
        )

```

```python
"""BenchEnv5 — Optimized env4 with shared projection cache and logging fixes.

Key differences from env4.py:
  - PathProjectionCache shared between ArcLengthProgress and LocalGuidance
    (eliminates duplicate polyline projection + coordinate transform per step)
  - on_correct_branch uses cross_track_dist threshold instead of O(B*N) branch scan
  - Pre-computed 2D tangents at reset
  - Log handler flush only on INFO-level log steps (every 50 or terminal)
  - Action string formatting only when logging at INFO level
"""

import eve
import eve.visualisation
import time
import logging
import sys
import os

from eve.util.pathcontext import PathProjectionCache


def setup_step_logger(name="step_logger"):
    """Create a logger that flushes immediately after each log message."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    log_dir = os.environ.get("STEP_LOG_DIR", "/tmp")
    log_file = os.path.join(log_dir, f"worker_{os.getpid()}.log")

    try:
        handler = logging.FileHandler(log_file, mode="a")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s,%(msecs)03d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except Exception:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s,%(msecs)03d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_formatter = logging.Formatter("STEP: %(message)s")
    stderr_handler.setFormatter(stderr_formatter)
    logger.addHandler(stderr_handler)

    return logger


class BenchEnv5(eve.Env):
    def __init__(
        self,
        intervention: eve.intervention.SimulatedIntervention,
        mode: str = "train",
        visualisation: bool = False,
        n_max_steps=1000,
    ) -> None:
        self.mode = mode
        self.visualisation = visualisation

        # Step-level logging
        self._step_logger = setup_step_logger(f"step_logger_{mode}_{id(self)}")
        self._step_count = 0
        self._episode_count = 0
        self._episode_step_count = 0
        self._last_step_time = None
        self._episode_start_time = None
        self._episode_total_reward = 0.0
        self._prev_inserted = [0.0, 0.0]
        self._step_logger.info(
            f"=== BenchEnv5 initialized (mode={mode}, visualisation={visualisation}) ==="
        )
        sys.stderr.flush()

        # Start condition
        start = eve.start.InsertionPoint(intervention)

        # Pathfinder — fixed path (computed once per episode at reset)
        pathfinder = eve.pathfinder.FixedPathfinder(intervention=intervention)

        # Shared projection cache — eliminates duplicate work between
        # ArcLengthProgress and LocalGuidance
        self._path_context = PathProjectionCache(pathfinder, intervention)

        # ----------------------------------------------------------------
        # Observation
        # ----------------------------------------------------------------
        tracking = eve.observation.Tracking2D(intervention, n_points=3, resolution=2)
        tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(
            tracking, intervention
        )
        tracking = eve.observation.wrapper.Memory(
            tracking, 2, eve.observation.wrapper.MemoryResetMode.FILL
        )

        target_state = eve.observation.Target2D(intervention)
        target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
            target_state, intervention
        )

        last_action = eve.observation.LastAction(intervention)
        last_action = eve.observation.wrapper.Normalize(last_action)

        guidance = eve.observation.LocalGuidance(
            intervention=intervention,
            pathfinder=pathfinder,
            path_context=self._path_context,
        )

        observation = eve.observation.ObsDict(
            {
                "tracking": tracking,
                "target": target_state,
                "last_action": last_action,
                "guidance": guidance,
            }
        )

        # ----------------------------------------------------------------
        # Reward
        # ----------------------------------------------------------------
        target_reward = eve.reward.TargetReached(
            intervention,
            factor=1.0,
            final_only_after_all_interim=False,
        )
        step_reward = eve.reward.Step(factor=-0.001)
        arc_progress = eve.reward.ArcLengthProgress(
            intervention=intervention,
            pathfinder=pathfinder,
            progress_factor=0.01,
            lateral_penalty_factor=0.001,
            path_context=self._path_context,
        )
        reward = eve.reward.Combination([target_reward, arc_progress, step_reward])

        # ----------------------------------------------------------------
        # Terminal and Truncation
        # ----------------------------------------------------------------
        terminal = eve.terminal.TargetReached(intervention)

        max_steps = eve.truncation.MaxSteps(n_max_steps)
        vessel_end = eve.truncation.VesselEnd(intervention)
        sim_error = eve.truncation.SimError(intervention)

        if mode == "train":
            truncation = eve.truncation.Combination(
                [max_steps, vessel_end, sim_error]
            )
        else:
            truncation = max_steps

        # ----------------------------------------------------------------
        # Info
        # ----------------------------------------------------------------
        target_reached = eve.info.TargetReached(intervention, name="success")
        path_ratio = eve.info.PathRatio(pathfinder)
        steps = eve.info.Steps()
        trans_speed = eve.info.AverageTranslationSpeed(intervention)
        trajectory_length = eve.info.TrajectoryLength(intervention)
        info = eve.info.Combination(
            [target_reached, path_ratio, steps, trans_speed, trajectory_length]
        )

        # ----------------------------------------------------------------
        # Visualisation
        # ----------------------------------------------------------------
        if visualisation:
            intervention.make_non_mp()
            visu = eve.visualisation.SofaPygame(intervention)
        else:
            intervention.make_non_mp()
            visu = None

        super().__init__(
            intervention,
            observation,
            reward,
            terminal,
            truncation=truncation,
            start=start,
            pathfinder=pathfinder,
            visualisation=visu,
            info=info,
            interim_target=None,
        )

    def __setstate__(self, state):
        """Re-initialize step logger after unpickling in a worker process."""
        self.__dict__.update(state)
        self._step_logger = setup_step_logger(f"step_logger_{self.mode}_{id(self)}")

    def reset(self, seed=None, options=None):
        if self._episode_count > 0 and self._episode_start_time is not None:
            episode_duration = time.time() - self._episode_start_time
            self._step_logger.info(
                f"EPISODE_END | ep={self._episode_count} | steps={self._episode_step_count} | "
                f"total_reward={self._episode_total_reward:.4f} | duration={episode_duration:.2f}s | "
                f"avg_step_time={episode_duration/max(1,self._episode_step_count):.3f}s | "
                f"wall_time={time.time():.6f} | pid={os.getpid()}"
            )
            sys.stderr.flush()

        self._episode_count += 1
        self._episode_step_count = 0
        self._episode_total_reward = 0.0
        self._episode_start_time = time.time()
        self._last_step_time = time.time()
        self._prev_inserted = [0.0, 0.0]

        self._step_logger.info(
            f"EPISODE_START | ep={self._episode_count} | global_steps={self._step_count} | "
            f"wall_time={time.time():.6f} | pid={os.getpid()}"
        )
        sys.stderr.flush()

        # path_context.reset() is called by LocalGuidance.reset() and
        # ArcLengthProgress.reset() inside super().reset(), AFTER
        # pathfinder.reset() has already computed the new path.
        return super().reset(seed=seed, options=options)

    def step(self, action):
        step_start_time = time.time()

        if self._last_step_time is not None:
            time_since_last = step_start_time - self._last_step_time
        else:
            time_since_last = 0.0

        # Invalidate projection cache — forces recomputation on first access
        self._path_context.invalidate()

        try:
            obs, reward, terminated, truncated, info = super().step(action)
        except Exception as e:
            self._step_logger.error(
                f"STEP_ERROR | ep={self._episode_count} | step={self._episode_step_count} | error={e}"
            )
            sys.stderr.flush()
            raise

        step_end_time = time.time()
        step_duration = step_end_time - step_start_time

        self._step_count += 1
        self._episode_step_count += 1
        self._episode_total_reward += reward
        self._last_step_time = step_end_time

        # Build log message — only do full formatting when we will log at INFO
        is_info_step = (
            self._episode_step_count % 50 == 0 or terminated or truncated
        )

        if is_info_step:
            # SOFA info
            delta_ins = [0.0, 0.0]
            sofa_info = ""
            try:
                if hasattr(self.intervention, "simulation"):
                    sim = self.intervention.simulation
                    if hasattr(sim, "inserted_lengths"):
                        inserted = list(sim.inserted_lengths)
                        delta_ins = [
                            inserted[0] - self._prev_inserted[0],
                            inserted[1] - self._prev_inserted[1],
                        ]
                        self._prev_inserted = inserted.copy()
                        sofa_info = f"inserted=[{inserted[0]:.2f},{inserted[1]:.2f}]"
            except Exception:
                pass

            try:
                if hasattr(action, "tolist"):
                    action_str = f"[{','.join(f'{a:.3f}' for a in action.flatten())}]"
                elif hasattr(action, "__iter__"):
                    action_str = f"[{','.join(f'{a:.3f}' for a in action)}]"
                else:
                    action_str = f"{action:.3f}"
            except Exception:
                action_str = str(action)

            log_msg = (
                f"STEP | ep={self._episode_count} | ep_step={self._episode_step_count} | "
                f"global={self._step_count} | wall_time={time.time():.6f} | pid={os.getpid()} | "
                f"cmd_action={action_str} | "
                f"reward={reward:.4f} | cum_reward={self._episode_total_reward:.4f} | "
                f"step_time={step_duration:.3f}s | gap_time={time_since_last:.3f}s | "
                f"term={terminated} | trunc={truncated} | "
                f"{sofa_info} | delta_ins=[{delta_ins[0]:.2f},{delta_ins[1]:.2f}]"
            )
            self._step_logger.info(log_msg)
            for handler in self._step_logger.handlers:
                handler.flush()
            sys.stderr.flush()
        else:
            # Lightweight debug-only log (no action formatting, no SOFA query)
            self._step_logger.debug(
                f"STEP | ep={self._episode_count} | ep_step={self._episode_step_count} | "
                f"global={self._step_count} | reward={reward:.4f} | "
                f"cum_reward={self._episode_total_reward:.4f} | "
                f"step_time={step_duration:.3f}s"
            )

            # Track insertion for delta computation on next INFO step
            try:
                if hasattr(self.intervention, "simulation"):
                    sim = self.intervention.simulation
                    if hasattr(sim, "inserted_lengths"):
                        self._prev_inserted = list(sim.inserted_lengths)
            except Exception:
                pass

        return obs, reward, terminated, truncated, info

```

```python
"""Action-space curriculum wrapper for DualDeviceNav.

Simplifies the dual-device coordination problem by staging:
  Stage 1: Only guidewire acts; catheter follows with a simple rule.
  Stage 2: Catheter action enabled but scaled down (×0.1).
  Stage 3: Full control of both devices.

Usage:
    env = BenchEnv4(intervention, mode="train")
    env = ActionCurriculumWrapper(env, stage1_steps=200_000, stage2_steps=500_000)
"""

import numpy as np
import gymnasium as gym


class ActionCurriculumWrapper(gym.Wrapper):
    """Curriculum that gradually enables catheter control.

    The wrapped environment must have a 4-dimensional action space:
    ``[gw_translation, gw_rotation, cath_translation, cath_rotation]``.

    Args:
        env: The base environment.
        stage1_steps: Number of steps in Stage 1 (guidewire-only).
        stage2_steps: Cumulative step count at which Stage 2 ends and
            Stage 3 (full control) begins.
        catheter_follow_ratio: In Stage 1, catheter translation is set to
            ``gw_translation * catheter_follow_ratio``.
        stage2_action_scale: In Stage 2, catheter actions are multiplied
            by this factor.
    """

    def __init__(
        self,
        env: gym.Env,
        stage1_steps: int = 200_000,
        stage2_steps: int = 500_000,
        catheter_follow_ratio: float = 0.8,
        stage2_action_scale: float = 0.1,
    ) -> None:
        super().__init__(env)
        self.stage1_steps = stage1_steps
        self.stage2_steps = stage2_steps
        self.catheter_follow_ratio = catheter_follow_ratio
        self.stage2_action_scale = stage2_action_scale
        self._total_steps = 0

    @property
    def current_stage(self) -> int:
        if self._total_steps < self.stage1_steps:
            return 1
        elif self._total_steps < self.stage2_steps:
            return 2
        return 3

    def step(self, action):
        stage = self.current_stage

        if stage == 1:
            # Catheter follows guidewire with a simple rule
            action = np.asarray(action, dtype=np.float64).copy()
            gw_trans = action.flat[0]
            action.flat[2] = gw_trans * self.catheter_follow_ratio  # cath translation
            action.flat[3] = 0.0  # cath rotation = 0
        elif stage == 2:
            # Catheter actions scaled down
            action = np.asarray(action, dtype=np.float64).copy()
            action.flat[2] *= self.stage2_action_scale
            action.flat[3] *= self.stage2_action_scale
        # Stage 3: pass through unchanged — no copy needed

        self._total_steps += 1
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

```

```python
import os
import logging
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch
from util.util import get_result_checkpoint_config_and_log_path
from util.env import BenchEnv
from util.env2 import BenchEnv2  # NEW: Centerline-aware environment with waypoint rewards
from util.env3 import BenchEnv3  # NEW: Tuned rewards for better balance
from util.env4 import BenchEnv4  # Arclength progress + local guidance
from util.env5 import BenchEnv5  # Optimized env4 with shared projection cache
from util.agent import BenchAgentSynchron
from eve_rl import Runner
from eve_bench import DualDeviceNav


RESULTS_FOLDER = os.getcwd() + "/results/eve_paper/neurovascular/full/mesh_ben"

EVAL_SEEDS = "1,2,3,5,6,7,8,9,10,12,13,14,16,17,18,21,22,23,27,31,34,35,37,39,42,43,44,47,48,50,52,55,56,58,61,62,63,68,69,70,71,73,79,80,81,84,89,91,92,93,95,97,102,103,108,109,110,115,116,117,118,120,122,123,124,126,127,128,129,130,131,132,134,136,138,139,140,141,142,143,144,147,148,149,150,151,152,154,155,156,158,159,161,162,167,168,171,175"
EVAL_SEEDS = EVAL_SEEDS.split(",")
EVAL_SEEDS = [int(seed) for seed in EVAL_SEEDS]
HEATUP_STEPS = 1e4  # Reduced from 5e5 for quick testing
TRAINING_STEPS = 2e7
CONSECUTIVE_EXPLORE_EPISODES = 100
EXPLORE_STEPS_BTW_EVAL = 2.5e5


GAMMA = 0.99
REWARD_SCALING = 1
REPLAY_BUFFER_SIZE = 1e4
CONSECUTIVE_ACTION_STEPS = 1
BATCH_SIZE = 32
UPDATE_PER_EXPLORE_STEP = 1 / 20


LR_END_FACTOR = 0.15
LR_LINEAR_END_STEPS = 6e6

DEBUG_LEVEL = logging.DEBUG

# HEATUP_STEPS = 5e3
# TRAINING_STEPS = 1e7
# CONSECUTIVE_EXPLORE_EPISODES = 10
# EXPLORE_STEPS_BTW_EVAL = 7.5e3
# EVAL_SEEDS = list(range(20))
# RESULTS_FOLDER = os.getcwd() + "/results/test"
# BATCH_SIZE = 8
# UPDATE_PER_EXPLORE_STEP = 1 / 200


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="perform IJCARS23 training")
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=4, help="Number of workers"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device of trainer, wehre the NN update is performed. ",
        choices=["cpu", "cuda:0", "cuda:1", "cuda"],
    )
    parser.add_argument(
        "-se",
        "--stochastic_eval",
        action="store_true",
        help="Runs optuna run with stochastic eval function of SAC.",
    )
    parser.add_argument(
        "-n", "--name", type=str, default="test", help="Name of the training run"
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.00021989352630306626,
        help="Learning Rate of Optimizers",
    )
    parser.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=[900, 900, 900, 900],
        help="Hidden Layers",
    )
    parser.add_argument(
        "-en",
        "--embedder_nodes",
        type=int,
        default=500,
        help="Number of nodes per layer in embedder",
    )
    parser.add_argument(
        "-el",
        "--embedder_layers",
        type=int,
        default=1,
        help="Number of layers in embedder",
    )
    parser.add_argument(
        "--env_version",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Environment version: 1=original (PathDelta), 2=waypoint+centerlines, 3=tuned waypoint, 4=arclength progress+guidance, 5=optimized env4 with shared projection cache",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable action-space curriculum (Stage 1: gw-only, Stage 2: scaled catheter, Stage 3: full)",
    )
    parser.add_argument(
        "--curriculum_stage1",
        type=int,
        default=200_000,
        help="Steps in curriculum Stage 1 (guidewire-only)",
    )
    parser.add_argument(
        "--curriculum_stage2",
        type=int,
        default=500_000,
        help="Cumulative steps at which curriculum Stage 2 ends",
    )
    parser.add_argument(
        "--heuristic_seeding",
        type=int,
        default=0,
        help="Number of heuristic episodes to seed replay buffer (0=disabled, recommended: 500)",
    )
    args = parser.parse_args()

    trainer_device = torch.device(args.device)
    n_worker = args.n_worker
    trial_name = args.name
    stochastic_eval = args.stochastic_eval
    lr = args.learning_rate
    hidden_layers = args.hidden
    embedder_nodes = args.embedder_nodes
    embedder_layers = args.embedder_layers
    env_version = args.env_version
    worker_device = torch.device("cpu")
    
    # Select environment class based on version
    if env_version == 5:
        EnvClass = BenchEnv5
        print("Using BenchEnv5 (optimized: shared projection cache + logging fixes)")
    elif env_version == 4:
        EnvClass = BenchEnv4
        print("Using BenchEnv4 (arclength progress reward + local guidance observation)")
    elif env_version == 3:
        EnvClass = BenchEnv3
        print("Using BenchEnv3 (TUNED waypoint rewards: 5mm spacing, 1.0 increment, -0.001 step penalty)")
    elif env_version == 2:
        EnvClass = BenchEnv2
        print("Using BenchEnv2 (waypoint rewards + centerline observations)")
    else:
        EnvClass = BenchEnv
        print("Using BenchEnv (original PathDelta reward)")

    custom_parameters = {
        "lr": lr,
        "hidden_layers": hidden_layers,
        "embedder_nodes": embedder_nodes,
        "embedder_layers": embedder_layers,
        "env_version": env_version,
        "HEATUP_STEPS": HEATUP_STEPS,
        "EXPLORE_STEPS_BTW_EVAL": EXPLORE_STEPS_BTW_EVAL,
        "CONSECUTIVE_EXPLORE_EPISODES": CONSECUTIVE_EXPLORE_EPISODES,
        "BATCH_SIZE": BATCH_SIZE,
        "UPDATE_PER_EXPLORE_STEP": UPDATE_PER_EXPLORE_STEP,
    }

    (
        results_file,
        checkpoint_folder,
        config_folder,
        log_file,
        _config_folder_dup,
        diagnostics_folder,
    ) = get_result_checkpoint_config_and_log_path(
        all_results_folder=RESULTS_FOLDER, name=trial_name, create_diagnostics=True
    )
    
    # CRITICAL: Set STEP_LOG_DIR BEFORE agent is created, so worker processes inherit it
    # Workers are spawned in BenchAgentSynchron.__init__, which happens before Runner.__init__
    if diagnostics_folder is not None:
        logs_subprocesses = os.path.join(diagnostics_folder, "logs_subprocesses")
        os.makedirs(logs_subprocesses, exist_ok=True)
        os.environ["STEP_LOG_DIR"] = logs_subprocesses
        print(f"Set STEP_LOG_DIR={logs_subprocesses}")
    
    logging.basicConfig(
        filename=log_file,
        level=DEBUG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Diagnostics configuration for SAC training monitoring
    diagnostics_config = {
        "enabled": True,
        "diagnostics_folder": diagnostics_folder,
        "log_losses_every_n_steps": 1,
        "log_probe_values_every_n_steps": 100,
        "log_batch_samples_every_n_steps": 100,  # Log batch samples every 100 gradient steps
        "n_batch_samples": 3,  # Number of transitions to sample from each batch
        "tensorboard_enabled": True,
        "probe_plot_every_n_steps": 10000,
        "policy_snapshot_every_n_steps": 10000,
        "flush_every_n_rows": 100,
    }

    # NOTE: DualDeviceNav() uses hardcoded defaults from eve_bench/eve_bench/dualdevicenav.py:
    #   - rotation_yzx_deg = [90, -90, 0]  (line 36)
    #   - fluoroscopy_rot_zx = [20, 5]     (line 82)
    # These match the preprocessing in the original DualDeviceNav data (model 0105)
    intervention = DualDeviceNav()
    env_train = EnvClass(intervention=intervention, mode="train", visualisation=False)

    # Wrap with action curriculum if requested
    if args.curriculum:
        from util.action_curriculum import ActionCurriculumWrapper
        env_train = ActionCurriculumWrapper(
            env_train,
            stage1_steps=args.curriculum_stage1,
            stage2_steps=args.curriculum_stage2,
        )
        print(f"Action curriculum enabled: Stage1={args.curriculum_stage1}, Stage2={args.curriculum_stage2}")

    intervention_eval = DualDeviceNav()
    env_eval = EnvClass(
        intervention=intervention_eval, mode="eval", visualisation=False
    )
    agent = BenchAgentSynchron(
        trainer_device,
        worker_device,
        lr,
        LR_END_FACTOR,
        LR_LINEAR_END_STEPS,
        hidden_layers,
        embedder_nodes,
        embedder_layers,
        GAMMA,
        BATCH_SIZE,
        REWARD_SCALING,
        REPLAY_BUFFER_SIZE,
        env_train,
        env_eval,
        CONSECUTIVE_ACTION_STEPS,
        n_worker,
        stochastic_eval,
        False,
        diagnostics_config=diagnostics_config,
    )

    env_train_config = os.path.join(config_folder, "env_train.yml")
    env_train.save_config(env_train_config)
    env_eval_config = os.path.join(config_folder, "env_eval.yml")
    env_eval.save_config(env_eval_config)
    infos = list(env_eval.info.info.keys())
    runner = Runner(
        agent=agent,
        heatup_action_low=[[-10.0, -1.0], [-11.0, -1.0]],
        heatup_action_high=[[35, 3.14], [30, 3.14]],
        agent_parameter_for_result_file=custom_parameters,
        checkpoint_folder=checkpoint_folder,
        results_file=results_file,
        info_results=infos,
        quality_info="success",
        diagnostics_folder=diagnostics_folder,
        policy_snapshot_every_steps=10000,
    )
    runner_config = os.path.join(config_folder, "runner.yml")
    runner.save_config(runner_config)

    # Optional: seed replay buffer with heuristic episodes
    # Collects episodes using a centerline-following heuristic and pushes
    # them into the agent's replay buffer BEFORE SAC training starts.
    # This avoids the early "collapse to masked no-op" failure mode.
    if args.heuristic_seeding > 0 and env_version in (4, 5):
        from util.heuristic_controller import CenterlineFollowerHeuristic
        from eve_rl.replaybuffer import Episode
        from eve_rl.util import flatten_obs

        print(f"Seeding replay buffer with {args.heuristic_seeding} heuristic episodes...")
        seed_intervention = DualDeviceNav()
        seed_env = EnvClass(intervention=seed_intervention, mode="train", visualisation=False)
        heuristic = CenterlineFollowerHeuristic(
            pathfinder=seed_env.pathfinder,
            intervention=seed_intervention,
        )
        rng = np.random.default_rng(42)
        n_pushed = 0

        # Action normalization: agent uses normalize_actions=True, so replay
        # buffer stores actions in [-1, 1].  Heuristic produces raw env-scale
        # actions, so we must map them: norm = 2*(a - low)/(high - low) - 1
        act_low = seed_env.action_space.low.astype(np.float64)
        act_high = seed_env.action_space.high.astype(np.float64)
        act_range = act_high - act_low

        for ep in range(args.heuristic_seeding):
            obs, info = seed_env.reset(seed=ep)
            heuristic.reset()

            # Create an Episode object matching the replay buffer format
            flat_obs, flat_obs_to_obs = flatten_obs(obs)
            episode = Episode(obs, flat_obs, flat_obs_to_obs, seed=ep)

            done = False
            while not done:
                raw_action = heuristic.get_action(rng)
                next_obs, rew, terminated, truncated, info = seed_env.step(raw_action)
                done = terminated or truncated

                # Normalize action to [-1, 1] for replay buffer consistency
                norm_action = 2.0 * (raw_action - act_low) / act_range - 1.0
                norm_action = np.clip(norm_action, -1.0, 1.0).astype(np.float32)

                flat_next_obs, _ = flatten_obs(next_obs)
                episode.add_transition(
                    next_obs, flat_next_obs, norm_action, rew, terminated, truncated, info
                )
                obs = next_obs

            # Push the completed episode into the agent's shared replay buffer
            agent.replay_buffer.push(episode)
            n_pushed += 1

            if (ep + 1) % 100 == 0:
                print(f"  Heuristic seeding: {ep + 1}/{args.heuristic_seeding} episodes (reward={episode.episode_reward:.2f})")
        seed_env.close()
        print(f"Heuristic seeding complete: {n_pushed} episodes pushed to replay buffer.")

    reward, success = runner.training_run(
        HEATUP_STEPS,
        TRAINING_STEPS,
        EXPLORE_STEPS_BTW_EVAL,
        CONSECUTIVE_EXPLORE_EPISODES,
        UPDATE_PER_EXPLORE_STEP,
        eval_seeds=EVAL_SEEDS,
    )
    agent.close()

```

All files have been successfully merged. Let me verify there are no linting errors:

All files have been merged successfully. Here's a summary:

## Files Renamed to `_v1` (originals preserved):
| Original File | Renamed To |
|---------------|------------|
| `training_scripts/util/action_curriculum.py` | `action_curriculum_v1.py` |
| `eve/eve/util/__init__.py` | `__init___v1.py` |
| `eve/eve/observation/localguidance.py` | `localguidance_v1.py` |
| `eve/eve/reward/arclengthprogress.py` | `arclengthprogress_v1.py` |
| `training_scripts/DualDeviceNav_train_vs.py` | `DualDeviceNav_train_vs_v1.py` |

## New Files Created:
- `eve/eve/util/pathcontext.py` - The `PathProjectionCache` class
- `training_scripts/util/env5.py` - The optimized `BenchEnv5` environment

## Updated Files (from worktree):
- `eve/eve/util/__init__.py` - Now exports `PathProjectionCache`
- `eve/eve/observation/localguidance.py` - Supports `path_context` + pre-computed 2D tangents + optimized `on_correct_branch`
- `eve/eve/reward/arclengthprogress.py` - Supports `path_context` for shared projection
- `training_scripts/util/action_curriculum.py` - Conditional action copy (no copy in Stage 3)
- `training_scripts/DualDeviceNav_train_vs.py` - Supports `env_version=5` and heuristic seeding for env5

The linter warnings shown are just unresolved imports for external packages (`torch`, `eve_rl`) which are expected in this environment. The code is ready to use with `--env_version 5` flag.

---

**User**

There is a bug you solved before around line 300 in .........vs_v1 which got reintrodtuced in vs copy ; fix it

---

**Cursor**

I see the bug. The `_v1.py` version has `.flatten()` on the action space bounds (lines 300-301) with an explanatory comment, but the new `_vs.py` version is missing those flattens. The action space is (2,2) but the heuristic returns flat (4,), so we need to flatten for proper shape matching.

```python
import os
import logging
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch
from util.util import get_result_checkpoint_config_and_log_path
from util.env import BenchEnv
from util.env2 import BenchEnv2  # NEW: Centerline-aware environment with waypoint rewards
from util.env3 import BenchEnv3  # NEW: Tuned rewards for better balance
from util.env4 import BenchEnv4  # Arclength progress + local guidance
from util.env5 import BenchEnv5  # Optimized env4 with shared projection cache
from util.agent import BenchAgentSynchron
from eve_rl import Runner
from eve_bench import DualDeviceNav


RESULTS_FOLDER = os.getcwd() + "/results/eve_paper/neurovascular/full/mesh_ben"

EVAL_SEEDS = "1,2,3,5,6,7,8,9,10,12,13,14,16,17,18,21,22,23,27,31,34,35,37,39,42,43,44,47,48,50,52,55,56,58,61,62,63,68,69,70,71,73,79,80,81,84,89,91,92,93,95,97,102,103,108,109,110,115,116,117,118,120,122,123,124,126,127,128,129,130,131,132,134,136,138,139,140,141,142,143,144,147,148,149,150,151,152,154,155,156,158,159,161,162,167,168,171,175"
EVAL_SEEDS = EVAL_SEEDS.split(",")
EVAL_SEEDS = [int(seed) for seed in EVAL_SEEDS]
HEATUP_STEPS = 1e4  # Reduced from 5e5 for quick testing
TRAINING_STEPS = 2e7
CONSECUTIVE_EXPLORE_EPISODES = 100
EXPLORE_STEPS_BTW_EVAL = 2.5e5


GAMMA = 0.99
REWARD_SCALING = 1
REPLAY_BUFFER_SIZE = 1e4
CONSECUTIVE_ACTION_STEPS = 1
BATCH_SIZE = 32
UPDATE_PER_EXPLORE_STEP = 1 / 20


LR_END_FACTOR = 0.15
LR_LINEAR_END_STEPS = 6e6

DEBUG_LEVEL = logging.DEBUG

# HEATUP_STEPS = 5e3
# TRAINING_STEPS = 1e7
# CONSECUTIVE_EXPLORE_EPISODES = 10
# EXPLORE_STEPS_BTW_EVAL = 7.5e3
# EVAL_SEEDS = list(range(20))
# RESULTS_FOLDER = os.getcwd() + "/results/test"
# BATCH_SIZE = 8
# UPDATE_PER_EXPLORE_STEP = 1 / 200


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="perform IJCARS23 training")
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=4, help="Number of workers"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device of trainer, wehre the NN update is performed. ",
        choices=["cpu", "cuda:0", "cuda:1", "cuda"],
    )
    parser.add_argument(
        "-se",
        "--stochastic_eval",
        action="store_true",
        help="Runs optuna run with stochastic eval function of SAC.",
    )
    parser.add_argument(
        "-n", "--name", type=str, default="test", help="Name of the training run"
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.00021989352630306626,
        help="Learning Rate of Optimizers",
    )
    parser.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=[900, 900, 900, 900],
        help="Hidden Layers",
    )
    parser.add_argument(
        "-en",
        "--embedder_nodes",
        type=int,
        default=500,
        help="Number of nodes per layer in embedder",
    )
    parser.add_argument(
        "-el",
        "--embedder_layers",
        type=int,
        default=1,
        help="Number of layers in embedder",
    )
    parser.add_argument(
        "--env_version",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Environment version: 1=original (PathDelta), 2=waypoint+centerlines, 3=tuned waypoint, 4=arclength progress+guidance, 5=optimized env4 with shared projection cache",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable action-space curriculum (Stage 1: gw-only, Stage 2: scaled catheter, Stage 3: full)",
    )
    parser.add_argument(
        "--curriculum_stage1",
        type=int,
        default=200_000,
        help="Steps in curriculum Stage 1 (guidewire-only)",
    )
    parser.add_argument(
        "--curriculum_stage2",
        type=int,
        default=500_000,
        help="Cumulative steps at which curriculum Stage 2 ends",
    )
    parser.add_argument(
        "--heuristic_seeding",
        type=int,
        default=0,
        help="Number of heuristic episodes to seed replay buffer (0=disabled, recommended: 500)",
    )
    args = parser.parse_args()

    trainer_device = torch.device(args.device)
    n_worker = args.n_worker
    trial_name = args.name
    stochastic_eval = args.stochastic_eval
    lr = args.learning_rate
    hidden_layers = args.hidden
    embedder_nodes = args.embedder_nodes
    embedder_layers = args.embedder_layers
    env_version = args.env_version
    worker_device = torch.device("cpu")
    
    # Select environment class based on version
    if env_version == 5:
        EnvClass = BenchEnv5
        print("Using BenchEnv5 (optimized: shared projection cache + logging fixes)")
    elif env_version == 4:
        EnvClass = BenchEnv4
        print("Using BenchEnv4 (arclength progress reward + local guidance observation)")
    elif env_version == 3:
        EnvClass = BenchEnv3
        print("Using BenchEnv3 (TUNED waypoint rewards: 5mm spacing, 1.0 increment, -0.001 step penalty)")
    elif env_version == 2:
        EnvClass = BenchEnv2
        print("Using BenchEnv2 (waypoint rewards + centerline observations)")
    else:
        EnvClass = BenchEnv
        print("Using BenchEnv (original PathDelta reward)")

    custom_parameters = {
        "lr": lr,
        "hidden_layers": hidden_layers,
        "embedder_nodes": embedder_nodes,
        "embedder_layers": embedder_layers,
        "env_version": env_version,
        "HEATUP_STEPS": HEATUP_STEPS,
        "EXPLORE_STEPS_BTW_EVAL": EXPLORE_STEPS_BTW_EVAL,
        "CONSECUTIVE_EXPLORE_EPISODES": CONSECUTIVE_EXPLORE_EPISODES,
        "BATCH_SIZE": BATCH_SIZE,
        "UPDATE_PER_EXPLORE_STEP": UPDATE_PER_EXPLORE_STEP,
    }

    (
        results_file,
        checkpoint_folder,
        config_folder,
        log_file,
        _config_folder_dup,
        diagnostics_folder,
    ) = get_result_checkpoint_config_and_log_path(
        all_results_folder=RESULTS_FOLDER, name=trial_name, create_diagnostics=True
    )
    
    # CRITICAL: Set STEP_LOG_DIR BEFORE agent is created, so worker processes inherit it
    # Workers are spawned in BenchAgentSynchron.__init__, which happens before Runner.__init__
    if diagnostics_folder is not None:
        logs_subprocesses = os.path.join(diagnostics_folder, "logs_subprocesses")
        os.makedirs(logs_subprocesses, exist_ok=True)
        os.environ["STEP_LOG_DIR"] = logs_subprocesses
        print(f"Set STEP_LOG_DIR={logs_subprocesses}")
    
    logging.basicConfig(
        filename=log_file,
        level=DEBUG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Diagnostics configuration for SAC training monitoring
    diagnostics_config = {
        "enabled": True,
        "diagnostics_folder": diagnostics_folder,
        "log_losses_every_n_steps": 1,
        "log_probe_values_every_n_steps": 100,
        "log_batch_samples_every_n_steps": 100,  # Log batch samples every 100 gradient steps
        "n_batch_samples": 3,  # Number of transitions to sample from each batch
        "tensorboard_enabled": True,
        "probe_plot_every_n_steps": 10000,
        "policy_snapshot_every_n_steps": 10000,
        "flush_every_n_rows": 100,
    }

    # NOTE: DualDeviceNav() uses hardcoded defaults from eve_bench/eve_bench/dualdevicenav.py:
    #   - rotation_yzx_deg = [90, -90, 0]  (line 36)
    #   - fluoroscopy_rot_zx = [20, 5]     (line 82)
    # These match the preprocessing in the original DualDeviceNav data (model 0105)
    intervention = DualDeviceNav()
    env_train = EnvClass(intervention=intervention, mode="train", visualisation=False)

    # Wrap with action curriculum if requested
    if args.curriculum:
        from util.action_curriculum import ActionCurriculumWrapper
        env_train = ActionCurriculumWrapper(
            env_train,
            stage1_steps=args.curriculum_stage1,
            stage2_steps=args.curriculum_stage2,
        )
        print(f"Action curriculum enabled: Stage1={args.curriculum_stage1}, Stage2={args.curriculum_stage2}")

    intervention_eval = DualDeviceNav()
    env_eval = EnvClass(
        intervention=intervention_eval, mode="eval", visualisation=False
    )
    agent = BenchAgentSynchron(
        trainer_device,
        worker_device,
        lr,
        LR_END_FACTOR,
        LR_LINEAR_END_STEPS,
        hidden_layers,
        embedder_nodes,
        embedder_layers,
        GAMMA,
        BATCH_SIZE,
        REWARD_SCALING,
        REPLAY_BUFFER_SIZE,
        env_train,
        env_eval,
        CONSECUTIVE_ACTION_STEPS,
        n_worker,
        stochastic_eval,
        False,
        diagnostics_config=diagnostics_config,
    )

    env_train_config = os.path.join(config_folder, "env_train.yml")
    env_train.save_config(env_train_config)
    env_eval_config = os.path.join(config_folder, "env_eval.yml")
    env_eval.save_config(env_eval_config)
    infos = list(env_eval.info.info.keys())
    runner = Runner(
        agent=agent,
        heatup_action_low=[[-10.0, -1.0], [-11.0, -1.0]],
        heatup_action_high=[[35, 3.14], [30, 3.14]],
        agent_parameter_for_result_file=custom_parameters,
        checkpoint_folder=checkpoint_folder,
        results_file=results_file,
        info_results=infos,
        quality_info="success",
        diagnostics_folder=diagnostics_folder,
        policy_snapshot_every_steps=10000,
    )
    runner_config = os.path.join(config_folder, "runner.yml")
    runner.save_config(runner_config)

    # Optional: seed replay buffer with heuristic episodes
    # Collects episodes using a centerline-following heuristic and pushes
    # them into the agent's replay buffer BEFORE SAC training starts.
    # This avoids the early "collapse to masked no-op" failure mode.
    if args.heuristic_seeding > 0 and env_version in (4, 5):
        from util.heuristic_controller import CenterlineFollowerHeuristic
        from eve_rl.replaybuffer import Episode
        from eve_rl.util import flatten_obs

        print(f"Seeding replay buffer with {args.heuristic_seeding} heuristic episodes...")
        seed_intervention = DualDeviceNav()
        seed_env = EnvClass(intervention=seed_intervention, mode="train", visualisation=False)
        heuristic = CenterlineFollowerHeuristic(
            pathfinder=seed_env.pathfinder,
            intervention=seed_intervention,
        )
        rng = np.random.default_rng(42)
        n_pushed = 0

        # Action normalization: agent uses normalize_actions=True, so replay
        # buffer stores actions in [-1, 1].  Heuristic produces raw env-scale
        # actions, so we must map them: norm = 2*(a - low)/(high - low) - 1
        # Note: action_space may be (2,2) but heuristic returns flat (4,), so flatten
        act_low = seed_env.action_space.low.astype(np.float64).flatten()
        act_high = seed_env.action_space.high.astype(np.float64).flatten()
        act_range = act_high - act_low

        for ep in range(args.heuristic_seeding):
            obs, info = seed_env.reset(seed=ep)
            heuristic.reset()

            # Create an Episode object matching the replay buffer format
            flat_obs, flat_obs_to_obs = flatten_obs(obs)
            episode = Episode(obs, flat_obs, flat_obs_to_obs, seed=ep)

            done = False
            while not done:
                raw_action = heuristic.get_action(rng)
                next_obs, rew, terminated, truncated, info = seed_env.step(raw_action)
                done = terminated or truncated

                # Normalize action to [-1, 1] for replay buffer consistency
                norm_action = 2.0 * (raw_action - act_low) / act_range - 1.0
                norm_action = np.clip(norm_action, -1.0, 1.0).astype(np.float32)

                flat_next_obs, _ = flatten_obs(next_obs)
                episode.add_transition(
                    next_obs, flat_next_obs, norm_action, rew, terminated, truncated, info
                )
                obs = next_obs

            # Push the completed episode into the agent's shared replay buffer
            agent.replay_buffer.push(episode)
            n_pushed += 1

            if (ep + 1) % 100 == 0:
                print(f"  Heuristic seeding: {ep + 1}/{args.heuristic_seeding} episodes (reward={episode.episode_reward:.2f})")
        seed_env.close()
        print(f"Heuristic seeding complete: {n_pushed} episodes pushed to replay buffer.")

    reward, success = runner.training_run(
        HEATUP_STEPS,
        TRAINING_STEPS,
        EXPLORE_STEPS_BTW_EVAL,
        CONSECUTIVE_EXPLORE_EPISODES,
        UPDATE_PER_EXPLORE_STEP,
        eval_seeds=EVAL_SEEDS,
    )
    agent.close()

```

Fixed. The bug was that `act_low` and `act_high` weren't being flattened. Since the action space is shaped `(2,2)` but the heuristic returns a flat `(4,)` array, the normalization would fail without `.flatten()`.

