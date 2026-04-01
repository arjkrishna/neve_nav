# Resume Operations: Issues and Fixes

This document catalogs all identified issues when resuming DualDeviceNav training from a checkpoint, along with their context, impact, and proposed fixes.

---

## Table of Contents

1. [ActionCurriculumWrapper Step Count Reset](#1-actioncurriculumwrapper-step-count-reset)
2. [Environment Logging Counters Reset](#2-environment-logging-counters-reset)
3. [Replay Buffer Empty After Resume](#3-replay-buffer-empty-after-resume)
4. [Resume Script Missing Features](#4-resume-script-missing-features)
5. [Runner Best Eval Tracking Lost](#5-runner-best-eval-tracking-lost)
6. [Probe States Not Captured](#6-probe-states-not-captured)
7. [Diagnostics Logger State Reset](#7-diagnostics-logger-state-reset)
8. [Policy Snapshot Step Misalignment](#8-policy-snapshot-step-misalignment)

---

## 1. ActionCurriculumWrapper Step Count Reset

### Context

The `ActionCurriculumWrapper` in `training_scripts/util/action_curriculum.py` implements a 3-stage curriculum:
- **Stage 1** (steps 0 → 200k): Guidewire-only control
- **Stage 2** (steps 200k → 500k): Catheter actions scaled by 0.1
- **Stage 3** (steps 500k+): Full control of both devices

The wrapper tracks progress via `self._total_steps`:

```python
class ActionCurriculumWrapper(gym.Wrapper):
    def __init__(self, env, stage1_steps=200_000, stage2_steps=500_000, ...):
        self._total_steps = 0  # NOT SAVED IN CHECKPOINT!
    
    @property
    def current_stage(self) -> int:
        if self._total_steps < self.stage1_steps:
            return 1
        elif self._total_steps < self.stage2_steps:
            return 2
        return 3
```

### Problem

When resuming from a checkpoint:
- `_total_steps` resets to 0
- Training that stopped at 600k steps (Stage 3) restarts at Stage 1
- Policy expects full control but catheter actions get overwritten

### Impact: **CRITICAL** 🔴

- Policy outputs catheter actions → wrapper overwrites with follow rule
- Q-values learned for Stage 3 receive Stage 1 rewards → catastrophic unlearning
- Training degrades rapidly after resume

### Fix

**Location**: `DualDeviceNav_resume.py` (after `agent.load_checkpoint()`)

```python
# Restore curriculum step count from checkpoint
if args.curriculum and resume_checkpoint:
    restored_steps = agent.step_counter.exploration
    env_train._total_steps = restored_steps
    print(f"Restored curriculum: _total_steps={restored_steps}, Stage={env_train.current_stage}")
```

**Alternative (wrapper-level fix)**: Modify `ActionCurriculumWrapper` to accept initial step count:

```python
class ActionCurriculumWrapper(gym.Wrapper):
    def __init__(self, env, stage1_steps=200_000, stage2_steps=500_000, 
                 initial_steps=0, ...):  # NEW parameter
        self._total_steps = initial_steps
```

---

## 2. Environment Logging Counters Reset

### Context

All environment classes (`BenchEnv`, `BenchEnv2`, ..., `BenchEnv5`) maintain internal counters for logging:

```python
class BenchEnv(eve.Env):
    def __init__(self, ...):
        self._step_count = 0           # Global step counter
        self._episode_count = 0        # Episode counter
        self._episode_step_count = 0   # Steps in current episode
        self._episode_total_reward = 0.0
        self._episode_start_time = None
        self._prev_inserted = [0.0, 0.0]  # For delta computation
```

These counters are used in step/episode logging:

```python
log_msg = f"STEP | ep={self._episode_count} | ep_step={self._episode_step_count} | global={self._step_count} | ..."
```

### Problem

When resuming:
- All counters reset to 0
- Log files show `ep=1, global=0` immediately after resume
- No continuity with pre-resume logging
- Episode IDs become meaningless for analysis

### Impact: **MEDIUM** 🟡

- Log analysis becomes harder (discontinuous numbering)
- Cannot correlate pre/post resume episodes
- Wall-time calculations break at resume boundary

### Fix

**Option A**: Pass step counters to environment on resume

```python
# In DualDeviceNav_resume.py, after loading checkpoint:
if resume_checkpoint:
    env_train._step_count = agent.step_counter.exploration
    env_train._episode_count = agent.episode_counter.exploration
    print(f"Restored env counters: steps={env_train._step_count}, episodes={env_train._episode_count}")
```

**Option B**: Log offset in environment's `__setstate__`

```python
# In BenchEnv.__setstate__:
def __setstate__(self, state):
    self.__dict__.update(state)
    self._step_logger = setup_step_logger(...)
    
    # Check for step offset from environment variable
    step_offset = int(os.environ.get("RESUME_STEP_OFFSET", 0))
    self._step_count = step_offset
```

**Option C**: Include resume marker in logs (minimal change)

```python
# In DualDeviceNav_resume.py, set env var before agent creation:
if resume_checkpoint:
    os.environ["RESUMED_FROM_STEP"] = str(checkpoint_steps)

# In BenchEnv reset():
resumed_from = os.environ.get("RESUMED_FROM_STEP")
if resumed_from and self._episode_count == 0:
    self._step_logger.info(f"=== RESUMED FROM STEP {resumed_from} ===")
```

---

## 3. Replay Buffer Empty After Resume

### Context

The replay buffer (`VanillaEpisodeShared`) runs in a subprocess with an internal ring buffer:

```python
class VanillaEpisode(ReplayBuffer):
    def __init__(self, capacity: int, batch_size: int):
        self.buffer: List[Episode] = []  # In-memory only!
```

The checkpoint saves only the buffer **configuration**, not contents:

```python
# In agent.save_checkpoint():
checkpoint_dict = {
    "replay_buffer": replay_config,  # Just config, not data!
    ...
}
```

### Problem

When resuming:
- Replay buffer starts completely empty
- No experiences to sample from
- First updates use poor/sparse data
- Risk of early instability or collapse

### Impact: **HIGH** 🟠

- Learning instability in first ~10k steps after resume
- May undo recent learning progress
- Heuristic seeding benefits lost

### Fix

**See**: [REPLAY_BUFFER_CHECKPOINT_ARCHITECTURE.md](./REPLAY_BUFFER_CHECKPOINT_ARCHITECTURE.md) for complete solution.

**Quick mitigation** (without replay buffer checkpointing):

```python
# In DualDeviceNav_resume.py, after loading checkpoint:
if resume_checkpoint:
    print("Warming up replay buffer with current policy...")
    # Collect exploration episodes with learned policy
    warmup_episodes = agent.explore(episodes=200)
    print(f"Collected {len(warmup_episodes)} warmup episodes")
```

---

## 4. Resume Script Missing Features

### Context

`DualDeviceNav_resume.py` was created as a copy of an earlier training script and hasn't been updated with newer features.

### Problem

Comparing `DualDeviceNav_train_vs.py` vs `DualDeviceNav_resume.py`:

| Feature | train_vs.py | resume.py |
|---------|-------------|-----------|
| env4 support | ✅ | ❌ |
| env5 support | ✅ | ❌ |
| `--curriculum` | ✅ | ❌ |
| `--curriculum_stage1` | ✅ | ❌ |
| `--curriculum_stage2` | ✅ | ❌ |
| `--heuristic_seeding` | ✅ | ❌ |
| env_version choices | `[1,2,3,4,5]` | `[1,2,3]` |

### Impact: **HIGH** 🟠

- Cannot resume env4/env5 experiments
- Cannot resume curriculum experiments
- Cannot use heuristic seeding on resume

### Fix

Update `DualDeviceNav_resume.py` with missing features:

```python
# 1. Add imports
from util.env4 import BenchEnv4
from util.env5 import BenchEnv5

# 2. Update env_version argument
parser.add_argument(
    "--env_version",
    type=int,
    default=1,
    choices=[1, 2, 3, 4, 5],  # Add 4 and 5
    help="Environment version: 1=original, 2=waypoint, 3=tuned, 4=arclength, 5=optimized",
)

# 3. Add curriculum arguments
parser.add_argument("--curriculum", action="store_true")
parser.add_argument("--curriculum_stage1", type=int, default=200_000)
parser.add_argument("--curriculum_stage2", type=int, default=500_000)

# 4. Add heuristic seeding argument
parser.add_argument("--heuristic_seeding", type=int, default=0)

# 5. Add env class selection
if env_version == 5:
    EnvClass = BenchEnv5
elif env_version == 4:
    EnvClass = BenchEnv4
# ... etc

# 6. Add curriculum wrapper with step restoration
if args.curriculum:
    from util.action_curriculum import ActionCurriculumWrapper
    env_train = ActionCurriculumWrapper(env_train, ...)
    if resume_checkpoint:
        env_train._total_steps = agent.step_counter.exploration

# 7. Add heuristic seeding (optional on resume)
if args.heuristic_seeding > 0 and env_version in (4, 5):
    # ... heuristic seeding code ...
```

---

## 5. Runner Best Eval Tracking Lost

### Context

The `Runner` class tracks the best evaluation result to save `best_checkpoint.everl`:

```python
class Runner:
    def __init__(self, ...):
        self.best_eval = {"steps": 0, "quality": -inf}
    
    def eval(self, ...):
        if quality > self.best_eval["quality"]:
            save_best = True
            self.best_eval["quality"] = quality
            self.best_eval["steps"] = explore_steps
```

### Problem

When resuming:
- `best_eval` resets to `{"steps": 0, "quality": -inf}`
- Pre-resume best quality is forgotten
- A worse post-resume checkpoint may overwrite the true best

### Impact: **MEDIUM** 🟡

- May lose the actual best checkpoint
- `best_checkpoint.everl` becomes unreliable

### Fix

**Option A**: Store best_eval in checkpoint

```python
# In agent.save_checkpoint():
checkpoint_dict["runner_state"] = {
    "best_quality": runner.best_eval["quality"],
    "best_steps": runner.best_eval["steps"],
}

# In resume script, after creating Runner:
if resume_checkpoint:
    cp = torch.load(resume_checkpoint)
    if "runner_state" in cp:
        runner.best_eval["quality"] = cp["runner_state"]["best_quality"]
        runner.best_eval["steps"] = cp["runner_state"]["best_steps"]
```

**Option B**: Read from results.csv on resume

```python
# In resume script:
if resume_checkpoint and os.path.exists(results_file):
    import pandas as pd
    df = pd.read_csv(results_file, sep=";", skiprows=1)
    if len(df) > 0:
        best_idx = df["quality"].idxmax()
        runner.best_eval["quality"] = df.loc[best_idx, "quality"]
        runner.best_eval["steps"] = df.loc[best_idx, "steps explore"]
```

---

## 6. Probe States Not Captured

### Context

The `Runner` captures probe states after heatup for diagnostics:

```python
def training_run(self, heatup_steps, ...):
    heatup_episodes = self.heatup(heatup_steps)
    self._capture_and_set_probe_states(heatup_episodes)  # Uses heatup episodes
```

Probe states are used for Q-value trajectory visualization.

### Problem

When resuming with `heatup_steps=0`:
- `heatup()` returns empty list
- `_capture_and_set_probe_states([])` captures nothing
- Diagnostics plots are empty/missing

### Impact: **LOW** 🟢

- Only affects diagnostics visualization
- Training still works correctly

### Fix

```python
# In DualDeviceNav_resume.py, after creating Runner:
if resume_checkpoint:
    # Run a few evaluation episodes just to capture probe states
    print("Capturing probe states from evaluation episodes...")
    probe_episodes = agent.evaluate(episodes=runner.n_probe_episodes)
    runner._capture_and_set_probe_states(probe_episodes)
    print(f"Captured probe states from {len(probe_episodes)} episodes")
```

---

## 7. Diagnostics Logger State Reset

### Context

The `DiagnosticsLogger` maintains internal counters:

```python
class DiagnosticsLogger:
    def __init__(self, config, process_name="trainer"):
        # ... logger setup ...
        
class CSVScalarLogger:
    def __init__(self, ...):
        self._row_count = 0  # Resets on init

class Runner:
    def __init__(self, ...):
        self._episode_summary_counter = 0  # Resets on init
```

### Problem

When resuming:
- `_row_count` in loggers resets to 0
- `_episode_summary_counter` resets to 0
- Episode IDs in logs restart from 0
- TensorBoard shows discontinuity

### Impact: **LOW** 🟢

- Log files still append correctly (data not lost)
- But numbering has discontinuity
- Analysis requires accounting for resume point

### Fix

**Option A**: Include offset in log output

```python
# In Runner, track resume offset:
if resume_checkpoint:
    self._episode_id_offset = agent.episode_counter.exploration
else:
    self._episode_id_offset = 0

# In _log_episode_summaries:
episode_id = self._episode_summary_counter + self._episode_id_offset
```

**Option B**: Log resume marker

```python
# At start of resumed training:
if self.diagnostics_folder and resume_checkpoint:
    with open(os.path.join(self.diagnostics_folder, "resume_points.txt"), "a") as f:
        f.write(f"{datetime.now().isoformat()}: Resumed from step {agent.step_counter.exploration}\n")
```

---

## 8. Policy Snapshot Step Misalignment

### Context

The `Runner` saves policy snapshots at regular intervals:

```python
class Runner:
    def __init__(self, ..., policy_snapshot_every_steps=10000):
        self._next_snapshot_step = policy_snapshot_every_steps
    
    def _maybe_save_policy_snapshot(self):
        if self.step_counter.exploration >= self._next_snapshot_step:
            # Save snapshot
            self._next_snapshot_step += self.policy_snapshot_every_steps
```

### Problem

When resuming:
- `_next_snapshot_step` resets to 10000 (initial value)
- If resumed at step 600k, next snapshot triggers immediately
- Snapshots saved at wrong intervals (600k instead of 610k)

### Impact: **LOW** 🟢

- Extra snapshot saved (no data loss)
- Just slightly unexpected timing

### Fix

```python
# In Runner.__init__ or resume logic:
if resume_checkpoint:
    current_step = agent.step_counter.exploration
    # Align to next milestone
    self._next_snapshot_step = (
        (current_step // self.policy_snapshot_every_steps + 1) 
        * self.policy_snapshot_every_steps
    )
    print(f"Next policy snapshot at step {self._next_snapshot_step}")
```

---

## Summary: Priority Order

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| 🔴 **P0** | ActionCurriculumWrapper reset | Critical | Low |
| 🟠 **P1** | Resume script missing features | High | Medium |
| 🟠 **P1** | Replay buffer empty | High | High* |
| 🟡 **P2** | Runner best_eval lost | Medium | Low |
| 🟡 **P2** | Environment counters reset | Medium | Low |
| 🟢 **P3** | Probe states not captured | Low | Low |
| 🟢 **P3** | Diagnostics counters reset | Low | Low |
| 🟢 **P3** | Snapshot step misalignment | Low | Low |

*See [REPLAY_BUFFER_CHECKPOINT_ARCHITECTURE.md](./REPLAY_BUFFER_CHECKPOINT_ARCHITECTURE.md) for full solution.

---

## Quick Fix Checklist

### Minimum Viable Resume Fix

Apply these changes to `DualDeviceNav_resume.py` for a working resume:

```python
# 1. Add missing imports
from util.env4 import BenchEnv4
from util.env5 import BenchEnv5

# 2. Update env_version choices to [1,2,3,4,5]

# 3. Add curriculum arguments

# 4. Add env selection for env4/env5

# 5. After agent.load_checkpoint():
if resume_checkpoint:
    # Restore curriculum
    if args.curriculum:
        env_train._total_steps = agent.step_counter.exploration
        print(f"Curriculum restored to Stage {env_train.current_stage}")
    
    # Warm up replay buffer
    print("Collecting warmup episodes...")
    agent.explore(episodes=100)
    
    # Capture probe states
    probe_episodes = agent.evaluate(episodes=3)
    runner._capture_and_set_probe_states(probe_episodes)
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `training_scripts/DualDeviceNav_resume.py` | Add env4/5, curriculum, heuristic seeding, all fixes |
| `training_scripts/util/action_curriculum.py` | (Optional) Add `initial_steps` parameter |
| `eve_rl/eve_rl/runner/runner.py` | (Optional) Store `best_eval` in checkpoint |
| `training_scripts/util/env.py` (and env2-5) | (Optional) Add step counter restoration |

---

## Testing Resume

After applying fixes, test with:

```bash
# 1. Start training with curriculum
python DualDeviceNav_train_vs.py -n test_resume --env_version 4 --curriculum \
    --curriculum_stage1 1000 --curriculum_stage2 2000

# 2. Wait for at least one checkpoint (or Ctrl+C after ~5k steps)

# 3. Resume and verify curriculum stage is correct
python DualDeviceNav_resume.py -n test_resume_continued --env_version 4 --curriculum \
    --resume results/.../checkpoints/checkpoint5000.everl

# Expected output:
# "Curriculum restored to Stage X" (where X matches the checkpoint step)
```
