# RL Resume Fixes Verification Report

**Codebase Verified**: `D:\neve\.claude\worktrees\rl_resume`  
**Date**: March 2026  
**Status**: ⚠️ **One Bug Found, Otherwise Correct**

---

## Executive Summary

The resume fixes in the `rl_resume` worktree are **well-implemented** with comprehensive documentation. One bug was found in the heuristic seeding action normalization code.

| Fix | Status | Notes |
|-----|--------|-------|
| 1. Resume into same folder | ✅ Correct | Path derivation logic is solid |
| 2. Save runner_state | ✅ Correct | Properly uses `.copy()` for dict |
| 3. Restore runner_state | ✅ Correct | Has backward compatibility fallback |
| 4. Probe states restoration | ✅ Correct | Handles missing file gracefully |
| 5. Curriculum step reset | ✅ Correct | Restores from `exploration` counter |
| 6. Feature parity | ⚠️ Partial | Missing env5, has heuristic bug |

---

## Detailed Verification

### Fix 1: Resume Into Same Experiment Folder ✅

**File**: `DualDeviceNav_resume.py` lines 185-207

```python
if resume_checkpoint:
    checkpoint_folder = os.path.dirname(os.path.abspath(resume_checkpoint))
    results_folder = os.path.dirname(checkpoint_folder)
    results_file = results_folder + ".csv"
    log_file = os.path.join(results_folder, "main.log")
    config_folder = os.path.join(results_folder, "config")
    diagnostics_folder = os.path.join(results_folder, "diagnostics")
```

**Verification**: ✅ **Correct**
- Path derivation assumes standard layout: `.../experiment_folder/checkpoints/checkpoint*.everl`
- Creates missing directories with `os.makedirs(..., exist_ok=True)`
- Sets `STEP_LOG_DIR` before agent creation (critical for worker process inheritance)

---

### Fix 2: Save Runner State in Checkpoints ✅

**File**: `runner.py` lines 214-219

```python
eval_results["runner_state"] = {
    "best_eval": self.best_eval.copy(),  # ✅ Uses .copy() to avoid reference issues
    "episode_summary_counter": self._episode_summary_counter,
    "next_snapshot_step": self._next_snapshot_step,
}
```

**Verification**: ✅ **Correct**
- Uses `.copy()` for `best_eval` dict to avoid reference aliasing
- Saved in both regular checkpoints and `best_checkpoint.everl`
- Stored inside `additional_info` which is properly serialized by torch.save

---

### Fix 3: Restore Runner State on Resume ✅

**File**: `runner.py` lines 101-143

```python
def restore_runner_state(self, checkpoint_path: str, probe_states_dir: Optional[str] = None):
    checkpoint = torch.load(checkpoint_path)
    additional_info = checkpoint.get("additional_info", {})
    runner_state = additional_info.get("runner_state") if additional_info else None

    if runner_state:
        self.best_eval = runner_state["best_eval"]
        self._episode_summary_counter = runner_state["episode_summary_counter"]
        self._next_snapshot_step = runner_state["next_snapshot_step"]
    else:
        # Backward compatibility for old checkpoints
        explore_steps = self.step_counter.exploration
        self._next_snapshot_step = (
            (explore_steps // self.policy_snapshot_every_steps) + 1
        ) * self.policy_snapshot_every_steps
```

**Verification**: ✅ **Correct**
- Handles missing `additional_info` gracefully (`checkpoint.get(...)`)
- Handles missing `runner_state` for old checkpoints
- Computes safe fallback for `_next_snapshot_step` to prevent snapshot burst
- Logs warnings when using fallback values

---

### Fix 4: Restore Probe States ✅

**File**: `runner.py` lines 131-143

```python
if probe_states_dir is not None:
    probe_path = os.path.join(probe_states_dir, "probes", "probe_states.npz")
    if os.path.isfile(probe_path):
        try:
            data = np.load(probe_path)
            probe_states = data["probe_states"]
            if hasattr(self.agent, 'set_probe_states'):
                self.agent.set_probe_states(probe_states.tolist())
                self._probe_states_set = True
        except Exception as e:
            self.logger.warning(f"Failed to restore probe states: {e}")
```

**Verification**: ✅ **Correct**
- Checks file existence before loading
- Wrapped in try/except for robustness
- Sets `_probe_states_set = True` to prevent re-capture
- Logs success/failure appropriately

---

### Fix 5: ActionCurriculumWrapper Step Reset ✅

**File**: `DualDeviceNav_resume.py` lines 296-300

```python
if args.curriculum:
    restored_steps = agent.step_counter.exploration
    env_train._total_steps = restored_steps
    print(f"  Curriculum restored: _total_steps={restored_steps}, Stage={env_train.current_stage}")
```

**Verification**: ✅ **Correct**
- Uses `agent.step_counter.exploration` which IS saved in checkpoints
- Directly sets `env_train._total_steps` (wrapper's internal counter)
- `current_stage` property correctly computes stage from `_total_steps`
- Print statement confirms restoration for debugging

---

### Fix 6: Feature Parity ⚠️ PARTIAL

#### 6a. env4 Support ✅
- Import added: `from util.env4 import BenchEnv4`
- Argument updated: `choices=[1, 2, 3, 4]`
- Selection logic added

#### 6b. env5 Support ❌ **MISSING**
- No `env5.py` in `training _scripts/util/`
- No import for `BenchEnv5`
- `choices` doesn't include `5`

**Impact**: Cannot resume experiments started with env5 in stEVE_training

#### 6c. Curriculum Arguments ✅
- `--curriculum` flag added
- `--curriculum_stage1` added (default 200,000)
- `--curriculum_stage2` added (default 500,000)

#### 6d. Heuristic Seeding ⚠️ **BUG FOUND**

**File**: `DualDeviceNav_resume.py` lines 351-353

```python
# BUG: Missing .flatten() calls
act_low = seed_env.action_space.low.astype(np.float64)
act_high = seed_env.action_space.high.astype(np.float64)
```

**Correct version** (from `stEVE_training/DualDeviceNav_train_vs.py`):

```python
# CORRECT: action_space is (2,2), heuristic returns (4,), must flatten
act_low = seed_env.action_space.low.astype(np.float64).flatten()
act_high = seed_env.action_space.high.astype(np.float64).flatten()
```

**Impact**: 
- Shape mismatch during normalization: `(2,2)` vs `(4,)`
- Will cause numpy broadcasting error or incorrect normalization
- Heuristic seeding will crash or produce wrong normalized actions

---

## Bugs Found

### Bug 1: Missing `.flatten()` in Heuristic Seeding Action Normalization 🔴

**Location**: `DualDeviceNav_resume.py` lines 351-353

**Current (buggy)**:
```python
act_low = seed_env.action_space.low.astype(np.float64)
act_high = seed_env.action_space.high.astype(np.float64)
act_range = act_high - act_low
```

**Should be**:
```python
act_low = seed_env.action_space.low.astype(np.float64).flatten()
act_high = seed_env.action_space.high.astype(np.float64).flatten()
act_range = act_high - act_low
```

**Reason**: The DualDeviceNav action space is shaped `(2, 2)` representing `[[gw_trans, gw_rot], [cath_trans, cath_rot]]`, but the heuristic controller returns a flat `(4,)` array. Without flattening, the normalization calculation:

```python
norm_action = 2.0 * (raw_action - act_low) / act_range - 1.0
```

will either:
1. Crash with a shape mismatch error
2. Incorrectly broadcast and produce wrong normalized values

---

## Missing Features (Compared to stEVE_training)

| Feature | stEVE_training | rl_resume |
|---------|----------------|-----------|
| env5 (BenchEnv5) | ✅ | ❌ |
| env5 import | ✅ | ❌ |
| env_version choice=5 | ✅ | ❌ |

---

## Efficiency Assessment ✅

The implementation is **efficient** with no identified inefficiencies:

1. **No duplicate file I/O**: Checkpoint is loaded once in `restore_runner_state()`
2. **No redundant computation**: Fallback `_next_snapshot_step` computed only when needed
3. **Minimal memory overhead**: Only `probe_states` array loaded, not entire buffer
4. **Proper cleanup**: `seed_env.close()` called after heuristic seeding

---

## Recommendations

### Immediate (Bug Fix)

1. **Fix heuristic seeding normalization** in `DualDeviceNav_resume.py`:

```python
# Line 351-353: Add .flatten() calls
act_low = seed_env.action_space.low.astype(np.float64).flatten()
act_high = seed_env.action_space.high.astype(np.float64).flatten()
```

### Short-term (Feature Parity)

2. **Add env5 support** if experiments use env5:
   - Copy `env5.py` from stEVE_training to `training _scripts/util/`
   - Add import: `from util.env5 import BenchEnv5`
   - Update env_version choices to `[1, 2, 3, 4, 5]`
   - Add selection logic for env5

### Long-term (Best Practices)

3. **Add unit tests** for resume functionality
4. **Add integration test** that:
   - Starts training for 1000 steps
   - Saves checkpoint
   - Resumes and verifies all state is restored correctly

---

## Conclusion

The resume fixes are **well-designed and thoroughly documented**. The implementation correctly addresses:

- ✅ Path continuity (same experiment folder)
- ✅ Runner state persistence (best_eval, counters, snapshot step)
- ✅ Probe state restoration
- ✅ Curriculum step restoration
- ✅ Backward compatibility with old checkpoints

**One bug must be fixed** before production use: the missing `.flatten()` calls in heuristic seeding will cause crashes or incorrect behavior.

**env5 support is missing** but may not be needed if the rl_resume codebase is older than the env5 addition.
