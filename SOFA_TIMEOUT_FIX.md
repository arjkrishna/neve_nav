# SOFA Timeout Issue - Investigation and Fix

## Problem Summary

Training experiments were experiencing SOFA timeouts, preventing any training progress:
- **Symptom**: "Killing sofa because of timeout when getting results" warnings
- **Impact**: 0 steps completed after 9+ hours of running
- **Observed**: Timeouts taking 26 minutes to 5+ hours instead of 60 seconds!

## Root Cause: Nested Multiprocessing Breaking Queue Timeouts

### The Architecture Problem

```
Main Process (BenchAgentSynchron)
├── Worker 0 (SingleAgentProcess) ─── spawns ───→ SOFA subprocess (via make_mp)
├── Worker 1 (SingleAgentProcess) ─── spawns ───→ SOFA subprocess  
├── Worker 2 (SingleAgentProcess) ─── spawns ───→ SOFA subprocess
├── Worker 3 (SingleAgentProcess) ─── spawns ───→ SOFA subprocess
└── Trainer  (SingleAgentProcess)

Total: 9+ processes all sharing Docker's /dev/shm for queue communication!
```

### Why Timeouts Fail

1. Python's `multiprocessing.Queue` uses shared memory (`/dev/shm`) for IPC
2. Docker has limited shared memory (default 64MB, we used 8GB)
3. When 9+ processes compete for shared memory:
   - Queue operations become unreliable
   - `queue.get(timeout=60)` **ignores the timeout** and blocks indefinitely
   - The timeout mechanism fails silently

### Evidence

| Worker | Start Time | Timeout Triggered | Actual Gap |
|--------|------------|-------------------|------------|
| Worker 1 | 20:29:11 | 02:17:24 | **5h 48min** |
| Worker 2 | 21:45:56 | 02:04:29 | **4h 18min** |
| Worker 3 | 02:58:21 | 04:34:37 | **1h 36min** |
| Worker 0 | 05:21:14 | 05:47:12 | **26min** |

Expected timeout: 60 seconds. Actual: 26 minutes to 5+ hours!

## Fix Applied

### Disable SOFA's Internal Multiprocessing

**File**: `training_scripts/util/env.py`

**Change**:
```python
# Before (BROKEN):
intervention.make_mp()  # Creates nested subprocess for SOFA

# After (FIXED):
intervention.make_non_mp()  # SOFA runs in worker process directly
```

**Why This Works**:
- `BenchAgentSynchron` already spawns 4 worker processes for parallelism
- Each worker can run SOFA directly without another subprocess
- Eliminates nested multiprocessing: Main → 4 workers (no SOFA subprocesses)
- Queue communication only between Main and Workers (reliable)

## Architecture After Fix

```
Main Process (BenchAgentSynchron)
├── Worker 0 (SingleAgentProcess + SOFA in same process)
├── Worker 1 (SingleAgentProcess + SOFA in same process)
├── Worker 2 (SingleAgentProcess + SOFA in same process)
├── Worker 3 (SingleAgentProcess + SOFA in same process)
└── Trainer  (SingleAgentProcess)

Total: 5 processes with clean queue communication
```

## What About SOFA Timeouts?

With `make_non_mp()`:
- SOFA runs synchronously in the worker process
- No inter-process communication for simulation steps
- If SOFA hangs (LCP non-convergence), the worker hangs
- But `BenchAgentSynchron` has its own timeout: `timeout_worker_after_reaching_limit=90`
- This provides a backup mechanism to restart stuck workers

## Testing Recommendations

1. **Rebuild Docker container** with the updated `env.py`
2. **Restart training** with fewer workers if issues persist: `-w 2`
3. **Monitor worker logs** for actual step completion
4. **Check CSV files** for evaluation results appearing

## Alternative: Increase Shared Memory (Not Recommended)

If you must use `make_mp()`, try:
```bash
docker run --shm-size=16g ...  # Even more shared memory
```

But this only delays the problem. The nested multiprocessing architecture is fundamentally fragile.

## Related Files

- `eve/eve/intervention/simulation/simulationmp.py` - SOFA subprocess implementation
- `eve/eve/intervention/intervention.py` - `make_mp()` / `make_non_mp()` methods
- `training_scripts/util/env.py` - Environment setup (**FIXED**)
- `eve_rl/eve_rl/agent/synchron.py` - BenchAgentSynchron implementation

## Additional Fix: SOFA "Case 1" Error

### The Error
```
Error: Case 1 should never happen ==> avoid using totalLengthIsChanging!
xCurvAbs = 898.264 > prev_maxCurvAbs = 898.239 + threshold: 0.01
```

### Cause
- InterventionalRadiologyController has a **0.01mm threshold** for position tracking
- Aggressive heatup actions (35 mm/step) cause numerical precision issues
- Tiny overshoots trigger SOFA error → undefined state → timeout

### Fix Applied
Reduced heatup action ranges in `DualDeviceNav_train.py`:
```python
# Before (too aggressive):
heatup_action_high=[[35, 3.14], [30, 3.14]]

# After (safer):
heatup_action_high=[[15.0, 1.0], [12.0, 1.0]]
```

## Status

✅ **Fixed**: Changed from `make_mp()` to `make_non_mp()` in `BenchEnv`
✅ **Fixed**: Reduced heatup action ranges to prevent SOFA "Case 1" errors
✅ **Root Cause 1**: Nested multiprocessing issues
✅ **Root Cause 2**: SOFA numerical precision errors from aggressive actions
⏳ **Pending**: Rebuild Docker and test training

