# Understanding RL code and training resumption
_Exported on 3/24/2026 at 21:41:46 EDT from Cursor (2.3.34)_

---

**User**

Understand the RL code thorough w.r.t DualDeviceNav_train.py ; ; understand all the changes made w.r.t all the logging and diagnostics and enviroments like env1/2/4.py and then tell me what's the file te resume training and analyse what are the variables and state/data that won't be restored perfectly that may affect logging or training in it's resumption

---

**Cursor**

I'll analyze the RL code thoroughly, starting with the main training script and then exploring the environments and related files.

Now let me look at the eve_rl library to understand what's saved in checkpoints and how the Runner handles training resumption:

Now I have a complete picture. Let me provide a comprehensive analysis.

---

# Comprehensive Analysis: RL Code for DualDeviceNav Training & Resume

## 1. Resume File

**The file for resuming training is: `DualDeviceNav_resume.py`**

Usage:
```bash
# Fresh training:
python DualDeviceNav_resume.py -n my_experiment -d cuda:0 -nw 16

# Resume from checkpoint:
python DualDeviceNav_resume.py -n my_experiment_resumed -d cuda:0 -nw 16 \
  --resume /path/to/checkpoint.everl
```

---

## 2. Overview of Environment Versions

| Version | Class | Key Features | Reward Strategy |
|---------|-------|--------------|-----------------|
| **env.py (v1)** | `BenchEnv` | Original, PathDelta reward | `PathLengthDelta(0.001)` + `Step(-0.005)` + `TargetReached(1.0)` |
| **env2.py (v2)** | `BenchEnv2` | Centerline-aware + waypoints | `CenterlineWaypointProgress(10mm, 0.1 increment)` + `Step(-0.005)` |
| **env3.py (v3)** | `BenchEnv3` | Tuned waypoints | `CenterlineWaypointProgress(5mm, 1.0 increment)` + `Step(-0.001)` |
| **env4.py (v4)** | `BenchEnv4` | Arclength progress + local guidance | `ArcLengthProgress(0.01 progress, 0.001 lateral)` + `Step(-0.001)` |
| **env5.py (v5)** | `BenchEnv5` | Optimized env4 with shared cache | Same as env4, with `PathProjectionCache` |

---

## 3. What IS Saved in Checkpoints (✅ Restored Correctly)

The checkpoint (`.everl` file) saves:

```python
checkpoint_dict = {
    # Neural Network Weights
    "network_state_dicts": {
        "q1", "q2", "target_q1", "target_q2", "policy", 
        "log_alpha"  # SAC temperature parameter
    },
    
    # Optimizer States
    "optimizer_state_dicts": {
        "q1_optim", "q2_optim", "policy_optim", "alpha_optim"
    },
    
    # LR Scheduler States
    "scheduler_state_dicts": {
        "q1_scheduler", "q2_scheduler", "policy_scheduler"
    },
    
    # Counters
    "steps": {
        "heatup": int,        # ✅ Restored
        "exploration": int,   # ✅ Restored
        "update": int,        # ✅ Restored
        "evaluation": int     # ✅ Restored
    },
    "episodes": {
        "heatup": int,        # ✅ Restored
        "exploration": int,   # ✅ Restored
        "evaluation": int     # ✅ Restored
    },
}
```

---

## 4. What is NOT Saved — Variables That Won't Restore Correctly ⚠️

### **A. Environment-Level State (All env versions)**

Each environment has internal counters that are **NOT persisted**:

```python
# In BenchEnv/BenchEnv2/BenchEnv3/BenchEnv4/BenchEnv5:
self._step_count = 0           # ❌ Resets to 0 on resume
self._episode_count = 0        # ❌ Resets to 0 on resume
self._episode_step_count = 0   # ❌ Resets to 0 on resume
self._episode_total_reward = 0.0  # ❌ Resets
self._episode_start_time = None   # ❌ No timing continuity
self._prev_inserted = [0.0, 0.0]  # ❌ For delta computation
```

**Impact on Logging:**
- The per-worker log files will show `ep=1` immediately after resume
- `global_steps` in logs starts from 0, NOT from the checkpoint's exploration step
- Episode numbering in logs won't match total episodes completed

### **B. ActionCurriculumWrapper State**

```python
# In util/action_curriculum.py:
self._total_steps = 0  # ❌ NOT SAVED!
```

**This is CRITICAL!** If you resume with curriculum enabled:
- `_total_steps` resets to 0
- You restart from **Stage 1 (guidewire-only)** regardless of where you were
- Example: Resumed at 600k steps → should be Stage 3 (full control) but will be Stage 1

**Fix Required:** The curriculum wrapper needs to be initialized with the checkpoint's exploration steps:
```python
if args.curriculum:
    env_train = ActionCurriculumWrapper(env_train, ...)
    if resume_checkpoint:
        # Need to manually set the step count!
        env_train._total_steps = agent.step_counter.exploration
```

### **C. Replay Buffer Contents**

```python
# VanillaEpisodeShared runs in a subprocess
# The internal buffer is NOT checkpointed
internal_replay_buffer = VanillaEpisode(capacity, batch_size)
```

**Impact:**
- Replay buffer is **EMPTY** after resume
- Agent must collect new experiences before meaningful updates can happen
- Early updates after resume will have less diverse samples

### **D. Diagnostics Logger State**

```python
# In DiagnosticsLogger:
self._row_count = 0          # ❌ Counter resets
self._episode_summary_counter = 0  # ❌ Episode ID numbering resets
```

**Impact on Logging:**
- CSV/JSONL files will continue appending but row counts restart
- TensorBoard will have a discontinuity (new events start from resume point)
- Episode summary IDs will restart from 0 instead of continuing

### **E. Runner Internal State**

```python
# In Runner:
self._results = {...}        # ❌ Resets
self.best_eval = {"steps": 0, "quality": -inf}  # ❌ Loses best checkpoint tracking
self._next_snapshot_step = policy_snapshot_every_steps  # ❌ May skip snapshots
self._probe_states_set = False  # ❌ Probe states must be recaptured
```

**Impact:**
- `best_checkpoint.everl` decision is based only on post-resume performance
- The "best quality" tracking doesn't know about pre-resume performance
- Policy snapshots may be saved at unexpected intervals

### **F. Probe States (Diagnostics Feature)**

```python
# In Runner.training_run():
heatup_episodes = self.heatup(heatup_steps)
self._capture_and_set_probe_states(heatup_episodes)  # Only after heatup!
```

**Impact:**
- When resuming with `heatup_steps=0`, `_capture_and_set_probe_states()` receives an empty list
- Probe state diagnostics will be missing or empty
- Q-value trajectory plots won't have meaningful reference states

---

## 5. Summary Table: Resume Impact by Component

| Component | State Location | Saved? | Impact on Resume |
|-----------|---------------|--------|------------------|
| Neural Networks | `SACModel.state_dicts_network()` | ✅ | Networks continue learning correctly |
| Optimizers | `SACModel.state_dicts_optimizer()` | ✅ | Adam momentum preserved |
| LR Schedulers | `SACModel.state_dicts_scheduler()` | ✅ | LR continues decay from correct position |
| `log_alpha` (SAC temperature) | `SACModel` | ✅ | Entropy tuning continues correctly |
| Step/Episode Counters | `agent.step_counter` | ✅ | Training loop starts at correct position |
| **Replay Buffer** | Subprocess memory | ❌ | Empty; needs re-collection |
| **ActionCurriculum step count** | `wrapper._total_steps` | ❌ | **Curriculum resets to Stage 1!** |
| **Env logging counters** | `env._step_count` etc. | ❌ | Log numbering discontinuity |
| **Runner best_eval** | `Runner.best_eval` | ❌ | Best checkpoint tracking lost |
| **Probe states** | `DiagnosticsLogger.probe_states` | ❌ | Q-trajectory diagnostics incomplete |

---

## 6. Recommendations for Clean Resume

### Fix 1: ActionCurriculumWrapper
Add to `DualDeviceNav_resume.py` after `agent.load_checkpoint()`:

```python
if args.curriculum and resume_checkpoint:
    # Restore curriculum stage from checkpoint
    env_train._total_steps = agent.step_counter.exploration
    print(f"Restored curriculum step count to {env_train._total_steps} (Stage {env_train.current_stage})")
```

### Fix 2: Probe States on Resume
The heatup is skipped but probe states need collection. Consider running a few "dummy" episodes for probe state capture:

```python
if resume_checkpoint:
    # Collect a few episodes just for probe state capture
    probe_episodes = agent.evaluate(episodes=runner.n_probe_episodes)
    runner._capture_and_set_probe_states(probe_episodes)
```

### Fix 3: Logging Continuity
Consider adding the checkpoint's step counters to log prefixes:

```python
# In env.__setstate__ or reset()
self._global_step_offset = step_counter.exploration  # Pass from main script
```

### Fix 4: Replay Buffer
Since the buffer is empty after resume, consider:
- Running extra heatup-style exploration (random actions) initially
- OR accepting that first few updates will use less diverse data

---

## 7. Key Files Summary

| File | Purpose |
|------|---------|
| `DualDeviceNav_train_vs.py` | Main training script (latest version with env5 support) |
| `DualDeviceNav_resume.py` | Resume-capable training script |
| `util/env.py` → `util/env5.py` | Environment variants (v1-v5) |
| `util/action_curriculum.py` | Curriculum wrapper (Stage 1→2→3) |
| `util/agent.py` | `BenchAgentSynchron` wrapper |
| `eve_rl/agent/agent.py` | `save_checkpoint` / `load_checkpoint` |
| `eve_rl/runner/runner.py` | Training loop orchestration |

```shellscript
# Fresh training:
python DualDeviceNav_resume.py -n my_experiment -d cuda:0 -nw 16

# Resume from checkpoint:
python DualDeviceNav_resume.py -n my_experiment_resumed -d cuda:0 -nw 16 \
  --resume /path/to/checkpoint.everl
```

```python
checkpoint_dict = {
    # Neural Network Weights
    "network_state_dicts": {
        "q1", "q2", "target_q1", "target_q2", "policy", 
        "log_alpha"  # SAC temperature parameter
    },
    
    # Optimizer States
    "optimizer_state_dicts": {
        "q1_optim", "q2_optim", "policy_optim", "alpha_optim"
    },
    
    # LR Scheduler States
    "scheduler_state_dicts": {
        "q1_scheduler", "q2_scheduler", "policy_scheduler"
    },
    
    # Counters
    "steps": {
        "heatup": int,        # ✅ Restored
        "exploration": int,   # ✅ Restored
        "update": int,        # ✅ Restored
        "evaluation": int     # ✅ Restored
    },
    "episodes": {
        "heatup": int,        # ✅ Restored
        "exploration": int,   # ✅ Restored
        "evaluation": int     # ✅ Restored
    },
}
```

```python
# In BenchEnv/BenchEnv2/BenchEnv3/BenchEnv4/BenchEnv5:
self._step_count = 0           # ❌ Resets to 0 on resume
self._episode_count = 0        # ❌ Resets to 0 on resume
self._episode_step_count = 0   # ❌ Resets to 0 on resume
self._episode_total_reward = 0.0  # ❌ Resets
self._episode_start_time = None   # ❌ No timing continuity
self._prev_inserted = [0.0, 0.0]  # ❌ For delta computation
```

```python
# In util/action_curriculum.py:
self._total_steps = 0  # ❌ NOT SAVED!
```

```python
if args.curriculum:
    env_train = ActionCurriculumWrapper(env_train, ...)
    if resume_checkpoint:
        # Need to manually set the step count!
        env_train._total_steps = agent.step_counter.exploration
```

```python
# VanillaEpisodeShared runs in a subprocess
# The internal buffer is NOT checkpointed
internal_replay_buffer = VanillaEpisode(capacity, batch_size)
```

```python
# In DiagnosticsLogger:
self._row_count = 0          # ❌ Counter resets
self._episode_summary_counter = 0  # ❌ Episode ID numbering resets
```

```python
# In Runner:
self._results = {...}        # ❌ Resets
self.best_eval = {"steps": 0, "quality": -inf}  # ❌ Loses best checkpoint tracking
self._next_snapshot_step = policy_snapshot_every_steps  # ❌ May skip snapshots
self._probe_states_set = False  # ❌ Probe states must be recaptured
```

```python
# In Runner.training_run():
heatup_episodes = self.heatup(heatup_steps)
self._capture_and_set_probe_states(heatup_episodes)  # Only after heatup!
```

```python
if args.curriculum and resume_checkpoint:
    # Restore curriculum stage from checkpoint
    env_train._total_steps = agent.step_counter.exploration
    print(f"Restored curriculum step count to {env_train._total_steps} (Stage {env_train.current_stage})")
```

```python
if resume_checkpoint:
    # Collect a few episodes just for probe state capture
    probe_episodes = agent.evaluate(episodes=runner.n_probe_episodes)
    runner._capture_and_set_probe_states(probe_episodes)
```

```python
# In env.__setstate__ or reset()
self._global_step_offset = step_counter.exploration  # Pass from main script
```

---

**User**

what about heuristic seeding and action curriculum 

---

**Cursor**

Great question! Let me analyze **heuristic seeding** and **action curriculum** specifically in the context of resume.

---

## Heuristic Seeding on Resume

### What It Does (in `DualDeviceNav_train_vs.py`)

```python
281:350:d:\stEVE_training\training_scripts\DualDeviceNav_train_vs.py
# Optional: seed replay buffer with heuristic episodes
# Collects episodes using a centerline-following heuristic and pushes
# them into the agent's replay buffer BEFORE SAC training starts.
# This avoids the early "collapse to masked no-op" failure mode.
if args.heuristic_seeding > 0 and env_version in (4, 5):
    from util.heuristic_controller import CenterlineFollowerHeuristic
    # ... collects heuristic episodes and pushes to replay buffer
```

### Problem on Resume ⚠️

**In `DualDeviceNav_resume.py`: Heuristic seeding is NOT implemented!**

Looking at the resume script:

```240:280:d:\stEVE_training\training_scripts\DualDeviceNav_resume.py
    env_train_config = os.path.join(config_folder, "env_train.yml")
    env_train.save_config(env_train_config)
    # ...
    # When resuming, use heatup_steps=0 to skip heatup phase
    effective_heatup = 0 if resume_checkpoint else HEATUP_STEPS
    
    reward, success = runner.training_run(
        effective_heatup,
        # ...
    )
```

**Issues:**
1. **No `--heuristic_seeding` argument** in `DualDeviceNav_resume.py`
2. **No heuristic seeding code block** at all in the resume script
3. **Replay buffer is EMPTY** after resume (it's not checkpointed)

### Impact

When you resume:
- Replay buffer starts **completely empty**
- No heuristic episodes to bootstrap
- First ~10,000 steps have very sparse/poor data for updates
- Risk of early instability or "collapse" that heuristic seeding was meant to prevent

### Recommended Fix

Add heuristic seeding option to `DualDeviceNav_resume.py`:

```python
parser.add_argument(
    "--heuristic_seeding",
    type=int,
    default=0,
    help="Number of heuristic episodes to seed replay buffer on resume (0=disabled)",
)

# After agent creation, before training_run:
if args.heuristic_seeding > 0 and env_version in (4, 5):
    # Same heuristic seeding code as DualDeviceNav_train_vs.py
    ...
```

Or alternatively, run a few random exploration episodes before resuming updates:

```python
if resume_checkpoint:
    # Collect some exploration episodes to refill buffer
    print("Refilling replay buffer with exploration episodes...")
    agent.explore(episodes=100)  # Collect ~100 episodes with current policy
```

---

## Action Curriculum on Resume

### What It Does

```python
17:56:d:\stEVE_training\training_scripts\util\action_curriculum.py
class ActionCurriculumWrapper(gym.Wrapper):
    """Curriculum that gradually enables catheter control.
    
    Stage 1: Only guidewire acts; catheter follows with a simple rule.
    Stage 2: Catheter action enabled but scaled down (×0.1).
    Stage 3: Full control of both devices.
    """
    
    def __init__(
        self,
        env: gym.Env,
        stage1_steps: int = 200_000,
        stage2_steps: int = 500_000,
        # ...
    ) -> None:
        # ...
        self._total_steps = 0  # ❌ NOT SAVED IN CHECKPOINT!
    
    @property
    def current_stage(self) -> int:
        if self._total_steps < self.stage1_steps:
            return 1
        elif self._total_steps < self.stage2_steps:
            return 2
        return 3
```

### Problem on Resume ⚠️

**In `DualDeviceNav_resume.py`: No `--curriculum` argument at all!**

The resume script doesn't have:
```python
# MISSING from DualDeviceNav_resume.py:
parser.add_argument("--curriculum", ...)
parser.add_argument("--curriculum_stage1", ...)
parser.add_argument("--curriculum_stage2", ...)
```

And even if it did, there's a critical bug:

### The `_total_steps` Reset Problem

```python
self._total_steps = 0  # Always starts at 0!
```

**Example scenario:**
- Original training stopped at 600,000 steps (should be Stage 3 - full control)
- Resume with curriculum enabled
- `_total_steps = 0` → Stage 1 (guidewire-only)!
- The agent's policy expects to control both devices but catheter actions are overwritten

**This is a TRAINING DISASTER:**
- Policy outputs catheter actions
- Wrapper overwrites them with `gw_trans * 0.8` (follow rule)
- Q-values learned for full control now get mismatched rewards
- Catastrophic policy degradation

### Recommended Fix for `DualDeviceNav_resume.py`

```python
# Add curriculum arguments
parser.add_argument(
    "--curriculum",
    action="store_true",
    help="Enable action-space curriculum",
)
parser.add_argument(
    "--curriculum_stage1",
    type=int,
    default=200_000,
)
parser.add_argument(
    "--curriculum_stage2",
    type=int,
    default=500_000,
)

# After wrapping with curriculum:
if args.curriculum:
    from util.action_curriculum import ActionCurriculumWrapper
    env_train = ActionCurriculumWrapper(
        env_train,
        stage1_steps=args.curriculum_stage1,
        stage2_steps=args.curriculum_stage2,
    )
    
    # CRITICAL: Restore step count from checkpoint!
    if resume_checkpoint:
        env_train._total_steps = agent.step_counter.exploration
        print(f"Curriculum restored: _total_steps={env_train._total_steps}, Stage={env_train.current_stage}")
```

---

## Summary: Resume Gaps for These Features

| Feature | In `train_vs.py` | In `resume.py` | Fix Needed |
|---------|------------------|----------------|------------|
| **Heuristic Seeding** | ✅ Implemented | ❌ Missing entirely | Add argument + code block |
| **Action Curriculum** | ✅ Implemented | ❌ Missing entirely | Add arguments + restore `_total_steps` |
| **env_version 4/5** | ✅ Supported | ⚠️ Only 1/2/3 | Add choices=[1,2,3,4,5] |

### Quick Check of `DualDeviceNav_resume.py`

```python
103:109:d:\stEVE_training\training_scripts\DualDeviceNav_resume.py
parser.add_argument(
    "--env_version",
    type=int,
    default=1,
    choices=[1, 2, 3],  # ❌ Missing 4 and 5!
    help="Environment version: 1=original, 2=waypoint+centerlines, 3=tuned waypoint",
)
```

**`DualDeviceNav_resume.py` is outdated** — it doesn't support:
- env4/env5
- Action curriculum  
- Heuristic seeding

---

## Recommended: Update `DualDeviceNav_resume.py`

The simplest fix is to **copy the missing features from `DualDeviceNav_train_vs.py`** to the resume script:

1. Add `env4`/`env5` imports and choices
2. Add `--curriculum`, `--curriculum_stage1`, `--curriculum_stage2` arguments
3. Add `--heuristic_seeding` argument
4. Add curriculum wrapper code with **step restoration**
5. Add heuristic seeding code block

Would you like me to update `DualDeviceNav_resume.py` with these fixes?

```python
281:350:d:\stEVE_training\training_scripts\DualDeviceNav_train_vs.py
# Optional: seed replay buffer with heuristic episodes
# Collects episodes using a centerline-following heuristic and pushes
# them into the agent's replay buffer BEFORE SAC training starts.
# This avoids the early "collapse to masked no-op" failure mode.
if args.heuristic_seeding > 0 and env_version in (4, 5):
    from util.heuristic_controller import CenterlineFollowerHeuristic
    # ... collects heuristic episodes and pushes to replay buffer
```

```plaintext
    env_train_config = os.path.join(config_folder, "env_train.yml")
    env_train.save_config(env_train_config)
    # ...
    # When resuming, use heatup_steps=0 to skip heatup phase
    effective_heatup = 0 if resume_checkpoint else HEATUP_STEPS
    
    reward, success = runner.training_run(
        effective_heatup,
        # ...
    )
```

```python
parser.add_argument(
    "--heuristic_seeding",
    type=int,
    default=0,
    help="Number of heuristic episodes to seed replay buffer on resume (0=disabled)",
)

# After agent creation, before training_run:
if args.heuristic_seeding > 0 and env_version in (4, 5):
    # Same heuristic seeding code as DualDeviceNav_train_vs.py
    ...
```

```python
if resume_checkpoint:
    # Collect some exploration episodes to refill buffer
    print("Refilling replay buffer with exploration episodes...")
    agent.explore(episodes=100)  # Collect ~100 episodes with current policy
```

```python
17:56:d:\stEVE_training\training_scripts\util\action_curriculum.py
class ActionCurriculumWrapper(gym.Wrapper):
    """Curriculum that gradually enables catheter control.
    
    Stage 1: Only guidewire acts; catheter follows with a simple rule.
    Stage 2: Catheter action enabled but scaled down (×0.1).
    Stage 3: Full control of both devices.
    """
    
    def __init__(
        self,
        env: gym.Env,
        stage1_steps: int = 200_000,
        stage2_steps: int = 500_000,
        # ...
    ) -> None:
        # ...
        self._total_steps = 0  # ❌ NOT SAVED IN CHECKPOINT!
    
    @property
    def current_stage(self) -> int:
        if self._total_steps < self.stage1_steps:
            return 1
        elif self._total_steps < self.stage2_steps:
            return 2
        return 3
```

```python
# MISSING from DualDeviceNav_resume.py:
parser.add_argument("--curriculum", ...)
parser.add_argument("--curriculum_stage1", ...)
parser.add_argument("--curriculum_stage2", ...)
```

```python
self._total_steps = 0  # Always starts at 0!
```

```python
# Add curriculum arguments
parser.add_argument(
    "--curriculum",
    action="store_true",
    help="Enable action-space curriculum",
)
parser.add_argument(
    "--curriculum_stage1",
    type=int,
    default=200_000,
)
parser.add_argument(
    "--curriculum_stage2",
    type=int,
    default=500_000,
)

# After wrapping with curriculum:
if args.curriculum:
    from util.action_curriculum import ActionCurriculumWrapper
    env_train = ActionCurriculumWrapper(
        env_train,
        stage1_steps=args.curriculum_stage1,
        stage2_steps=args.curriculum_stage2,
    )
    
    # CRITICAL: Restore step count from checkpoint!
    if resume_checkpoint:
        env_train._total_steps = agent.step_counter.exploration
        print(f"Curriculum restored: _total_steps={env_train._total_steps}, Stage={env_train.current_stage}")
```

```python
103:109:d:\stEVE_training\training_scripts\DualDeviceNav_resume.py
parser.add_argument(
    "--env_version",
    type=int,
    default=1,
    choices=[1, 2, 3],  # ❌ Missing 4 and 5!
    help="Environment version: 1=original, 2=waypoint+centerlines, 3=tuned waypoint",
)
```

---

**User**

what's the total memory cost of saving replay buffer

---

**Cursor**

Let me analyze the replay buffer memory cost by examining the data structures.

Now let me check the observation space dimensions for DualDeviceNav:

# Replay Buffer Memory Cost Analysis

## Configuration Parameters

From `DualDeviceNav_train_vs.py`:
```python
REPLAY_BUFFER_SIZE = 1e4    # 10,000 episodes
n_max_steps = 1000          # max steps per episode
```

---

## Per-Step Data Size by Environment Version

### env1 (BenchEnv) - Original
| Component | Dimensions | Size |
|-----------|------------|------|
| `Tracking2D` (3 pts × 2D × memory=2) | 3 × 2 × 3 = 18 | 18 floats |
| `Target2D` | 2 | 2 floats |
| `LastAction` | 4 | 4 floats |
| **Total observation** | | **24 floats** |
| Action | 4 | 4 floats |
| Reward | 1 | 1 float |
| Terminal | 1 | 1 byte |

**Per step: 29 floats + 1 byte ≈ 117 bytes**

### env4/env5 (BenchEnv4/5) - With LocalGuidance
| Component | Dimensions | Size |
|-----------|------------|------|
| `Tracking2D` (3 pts × 2D × memory=2) | 18 | 18 floats |
| `Target2D` | 2 | 2 floats |
| `LastAction` | 4 | 4 floats |
| `LocalGuidance` | 8 | 8 floats |
| **Total observation** | | **32 floats** |
| Action | 4 | 4 floats |
| Reward | 1 | 1 float |
| Terminal | 1 | 1 byte |

**Per step: 37 floats + 1 byte ≈ 149 bytes**

### env2/env3 (BenchEnv2/3) - With Centerlines2D
| Component | Dimensions | Size |
|-----------|------------|------|
| `Tracking2D` | 18 | 18 floats |
| `Target2D` | 2 | 2 floats |
| `LastAction` | 4 | 4 floats |
| `Centerlines2D` (~7 branches × 10 pts × 2D + mask) | ~154 | ~154 floats |
| **Total observation** | | **~178 floats** |
| Action | 4 | 4 floats |
| Reward | 1 | 1 float |
| Terminal | 1 | 1 byte |

**Per step: 183 floats + 1 byte ≈ 733 bytes**

---

## Total Memory Calculation

### Replay Buffer Storage Format

From `vanillaepisode.py`:
```python
episode_np = (
    np.array(episode.flat_obs),    # shape: (steps+1, obs_dim), float32
    np.array(episode.actions),     # shape: (steps, act_dim), float32
    np.array(episode.rewards),     # shape: (steps,), float32
    np.array(episode.terminals),   # shape: (steps,), bool
)
```

### Memory Per Episode (assuming average ~500 steps)

| Env | Obs Size | Episode Memory |
|-----|----------|----------------|
| **env1** | 24 × 4 bytes × 501 = 48 KB obs<br>+ 4 × 4 × 500 = 8 KB act<br>+ 4 × 500 = 2 KB rew<br>+ 500 bytes term | **~59 KB** |
| **env4/5** | 32 × 4 × 501 = 64 KB obs<br>+ 8 KB act + 2 KB rew + 500 B | **~75 KB** |
| **env2/3** | 178 × 4 × 501 = 357 KB obs<br>+ 8 KB act + 2 KB rew + 500 B | **~368 KB** |

### Total Buffer Memory (10,000 episodes)

| Environment | Memory per Episode | Total Buffer Size |
|-------------|-------------------|-------------------|
| **env1 (BenchEnv)** | ~59 KB | **~590 MB** |
| **env4/env5** | ~75 KB | **~750 MB** |
| **env2/env3 (Centerlines)** | ~368 KB | **~3.7 GB** |

---

## Checkpoint File Size (if buffer were saved)

Adding Python/pickle overhead (~20-30%):

| Environment | Estimated Checkpoint Size |
|-------------|--------------------------|
| **env1** | **~700 MB - 770 MB** |
| **env4/5** | **~900 MB - 1 GB** |
| **env2/3** | **~4.5 - 5 GB** |

---

## Current Checkpoint Size (WITHOUT replay buffer)

The checkpoint currently saves only:
- Network weights: ~15-20 MB (4 × [900×4] MLPs + LSTM)
- Optimizer states: ~30-40 MB (Adam momentum/variance)
- Counters: negligible

**Current checkpoint: ~50-60 MB**

---

## Summary

| What | env1 | env4/5 | env2/3 |
|------|------|--------|--------|
| **Current checkpoint** | ~50 MB | ~50 MB | ~50 MB |
| **If buffer included** | ~700 MB | ~1 GB | ~5 GB |
| **Increase factor** | **14×** | **20×** | **100×** |

### Trade-off Decision

**Pros of saving replay buffer:**
- Resume with existing experience diversity
- No "cold start" learning instability
- More consistent training continuation

**Cons of saving replay buffer:**
- **14-100× larger checkpoint files**
- Slower checkpoint save/load
- More disk usage over long training

### Alternative: Warm-start Exploration

Instead of saving the entire buffer, on resume you could:
```python
# Quick buffer refill with current policy (~5 min)
if resume_checkpoint:
    print("Warming up replay buffer...")
    agent.explore(episodes=500)  # Collect ~500 new episodes
```

This gives ~30% buffer fill with relevant on-policy data, costing minutes rather than gigabytes.

```python
REPLAY_BUFFER_SIZE = 1e4    # 10,000 episodes
n_max_steps = 1000          # max steps per episode
```

```python
episode_np = (
    np.array(episode.flat_obs),    # shape: (steps+1, obs_dim), float32
    np.array(episode.actions),     # shape: (steps, act_dim), float32
    np.array(episode.rewards),     # shape: (steps,), float32
    np.array(episode.terminals),   # shape: (steps,), bool
)
```

```python
# Quick buffer refill with current policy (~5 min)
if resume_checkpoint:
    print("Warming up replay buffer...")
    agent.explore(episodes=500)  # Collect ~500 new episodes
```

---

**User**

what will be cost of saving the heuristic seeeding then (only 100 episodes )

---

**Cursor**

# Memory Cost: Heuristic Seeding (100 Episodes)

## Quick Calculation

Heuristic seeding only works with **env4/env5**, so we use those dimensions.

### Per Episode (env4/5)
- Observation: 32 floats × 4 bytes × ~501 steps = **64 KB**
- Actions: 4 floats × 4 bytes × ~500 steps = **8 KB**
- Rewards: 4 bytes × 500 steps = **2 KB**
- Terminals: 500 bytes ≈ **0.5 KB**

**Total per episode: ~75 KB**

### For 100 Heuristic Episodes

| Metric | Value |
|--------|-------|
| Raw data | 100 × 75 KB = **7.5 MB** |
| With pickle overhead (~25%) | **~9-10 MB** |

---

## Comparison

| What to Save | Size | Save Time |
|--------------|------|-----------|
| Current checkpoint (no buffer) | ~50 MB | <1 sec |
| + 100 heuristic episodes | ~60 MB | ~1 sec |
| + Full buffer (10K episodes) | ~1 GB | ~10-20 sec |

---

## Verdict

**Saving 100 heuristic episodes adds only ~10 MB** — that's essentially free!

The typical `--heuristic_seeding 500` recommended value would be:
- 500 × 75 KB = **~37.5 MB** → **~47 MB** with overhead

Still very reasonable for resume capability.

---

## Implementation Suggestion

You could save heuristic episodes separately or as part of the checkpoint:

```python
# In save_checkpoint:
checkpoint_dict["heuristic_episodes"] = [
    (ep.flat_obs, ep.actions, ep.rewards, ep.terminals)
    for ep in heuristic_episodes
]

# On resume, push them back:
for ep_data in checkpoint.get("heuristic_episodes", []):
    agent.replay_buffer.push(ep_data)
```

Or simpler: just re-run heuristic seeding on resume (takes ~2-5 minutes for 500 episodes, which is often acceptable).

```python
# In save_checkpoint:
checkpoint_dict["heuristic_episodes"] = [
    (ep.flat_obs, ep.actions, ep.rewards, ep.terminals)
    for ep in heuristic_episodes
]

# On resume, push them back:
for ep_data in checkpoint.get("heuristic_episodes", []):
    agent.replay_buffer.push(ep_data)
```

---

**User**

make a plan in detail  (write an md file ) for architectural changes to 1. store saving replay buffer every 4 checkpoints (or 1M steps) as a separate file which when used for resuming that same experiment from any checkpoint ; the latest one gets used ; hence overwrite replay_buffer everytime when saving ; also save heuristic seeding (100 episodes ) and heatup (10k steps ) as separate file that can be used even by nre experiments hence we don't have to heuristic seed every new experiment (saves time)  Analyse code in detail and make a detailed plabn for this and then write an md file in docs

---

**Cursor**

Let me analyze the code in detail first to understand all the touch points needed for this architecture.

Now let me write the comprehensive architectural plan document:

```markdown
# Replay Buffer Checkpoint Architecture

## Overview

This document outlines the architectural changes needed to implement persistent replay buffer checkpointing with reusable experience caches for the DualDeviceNav training system.

### Goals

1. **Replay Buffer Persistence**: Save replay buffer every 4 checkpoints (~1M steps) as a separate file
2. **Reusable Heuristic Seeding**: Save heuristic episodes (100+) to a shareable file for new experiments
3. **Reusable Heatup Cache**: Save heatup episodes (10k steps) to a shareable file for new experiments
4. **Seamless Resume**: Automatically load latest replay buffer when resuming an experiment

---

## Current Architecture Analysis

### File Locations

| Component | Location |
|-----------|----------|
| Agent checkpoint logic | `eve_rl/eve_rl/agent/agent.py` (lines 321-380) |
| Runner save/eval logic | `eve_rl/eve_rl/runner/runner.py` (lines 115-185) |
| Replay buffer (shared) | `eve_rl/eve_rl/replaybuffer/vanillashared.py` |
| Replay buffer (episode) | `eve_rl/eve_rl/replaybuffer/vanillaepisode.py` |
| Training script | `training_scripts/DualDeviceNav_train_vs.py` |
| Resume script | `training_scripts/DualDeviceNav_resume.py` |

### Current Checkpoint Structure

```
results/eve_paper/neurovascular/full/mesh_ben/{experiment}/
├── checkpoints/
│   ├── checkpoint250000.everl      # ~50 MB (no replay buffer)
│   ├── checkpoint500000.everl
│   ├── checkpoint750000.everl
│   ├── checkpoint1000000.everl
│   └── best_checkpoint.everl
├── config/
│   ├── env_train.yml
│   ├── env_eval.yml
│   └── runner.yml
├── diagnostics/
│   ├── csv/
│   ├── tensorboard/
│   └── logs_subprocesses/
└── results.csv
```

### Current Checkpoint Contents (agent.py:337-357)

```python
checkpoint_dict = {
    "algo": algo_config,
    "replay_buffer": replay_config,  # Config only, NOT data
    "env_train": env_train_config,
    "env_eval": env_eval_config,
    "steps": {...},
    "episodes": {...},
    "network_state_dicts": {...},
    "optimizer_state_dicts": {...},
    "scheduler_state_dicts": {...},
}
```

### Replay Buffer Internals

The `VanillaEpisodeShared` runs in a subprocess with an internal `VanillaEpisode` buffer:

```python
# vanillaepisode.py
class VanillaEpisode(ReplayBuffer):
    def __init__(self, capacity: int, batch_size: int):
        self.buffer: List[Episode] = []  # Ring buffer of episodes
        self.position = 0
    
    def push(self, episode: Episode):
        # Stores as numpy arrays:
        episode_np = (
            np.array(episode.flat_obs),    # (steps+1, obs_dim), float32
            np.array(episode.actions),     # (steps, 4), float32
            np.array(episode.rewards),     # (steps,), float32
            np.array(episode.terminals),   # (steps,), bool
        )
```

---

## Proposed Architecture

### New File Structure

```
results/eve_paper/neurovascular/full/mesh_ben/{experiment}/
├── checkpoints/
│   ├── checkpoint250000.everl
│   ├── checkpoint500000.everl
│   ├── checkpoint750000.everl
│   ├── checkpoint1000000.everl
│   ├── best_checkpoint.everl
│   └── replay_buffer.npz          # NEW: Single file, overwritten periodically
├── config/
├── diagnostics/
└── results.csv

# Shared experience caches (reusable across experiments)
shared_experience/                  # NEW: Top-level folder
├── heuristic_env4_100ep.npz       # Heuristic episodes for env4/5
├── heuristic_env4_500ep.npz       # Larger heuristic cache
├── heatup_env4_10k.npz            # 10k heatup steps for env4
├── heatup_env5_10k.npz            # 10k heatup steps for env5
└── manifest.json                   # Metadata about each cache
```

### Memory Estimates

| Cache Type | Episodes | Avg Steps | Obs Dim | Size |
|------------|----------|-----------|---------|------|
| Heuristic (100 ep) | 100 | ~200 | 32 | ~3 MB |
| Heuristic (500 ep) | 500 | ~200 | 32 | ~15 MB |
| Heatup (10k steps) | ~20 | ~500 | 32 | ~3 MB |
| Full buffer (10k ep) | 10,000 | ~500 | 32 | ~750 MB |

---

## Implementation Plan

### Phase 1: Experience Cache Format Definition

#### 1.1 Create `eve_rl/eve_rl/util/experience_cache.py`

```python
"""
experience_cache.py - Serialization utilities for replay buffer data

Formats:
- EpisodeCache: Compressed numpy archive of episode tuples
- Manifest: JSON metadata for cache discovery
"""

import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime


@dataclass
class CacheMetadata:
    """Metadata for an experience cache file."""
    cache_type: str           # "heuristic", "heatup", "replay_buffer"
    env_version: int          # 1, 2, 3, 4, or 5
    n_episodes: int
    total_steps: int
    obs_dim: int
    action_dim: int
    created_at: str           # ISO timestamp
    source_experiment: str    # Experiment name or "shared"
    description: str


EpisodeTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# (flat_obs, actions, rewards, terminals)


def save_experience_cache(
    filepath: str,
    episodes: List[EpisodeTuple],
    metadata: CacheMetadata,
) -> None:
    """Save episodes to compressed numpy archive with metadata."""
    n_episodes = len(episodes)
    
    # Stack episodes with variable lengths using object arrays
    flat_obs_list = [ep[0] for ep in episodes]
    actions_list = [ep[1] for ep in episodes]
    rewards_list = [ep[2] for ep in episodes]
    terminals_list = [ep[3] for ep in episodes]
    
    # Save as compressed .npz
    np.savez_compressed(
        filepath,
        # Episode data as object arrays (variable length)
        flat_obs=np.array(flat_obs_list, dtype=object),
        actions=np.array(actions_list, dtype=object),
        rewards=np.array(rewards_list, dtype=object),
        terminals=np.array(terminals_list, dtype=object),
        # Metadata as JSON string
        metadata=json.dumps(asdict(metadata)),
    )


def load_experience_cache(
    filepath: str,
) -> Tuple[List[EpisodeTuple], CacheMetadata]:
    """Load episodes from compressed numpy archive."""
    data = np.load(filepath, allow_pickle=True)
    
    flat_obs = data["flat_obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    terminals = data["terminals"]
    metadata_dict = json.loads(str(data["metadata"]))
    metadata = CacheMetadata(**metadata_dict)
    
    episodes = [
        (flat_obs[i], actions[i], rewards[i], terminals[i])
        for i in range(len(flat_obs))
    ]
    
    return episodes, metadata


def get_shared_cache_path(
    cache_type: str,
    env_version: int,
    n_episodes: Optional[int] = None,
    n_steps: Optional[int] = None,
) -> str:
    """Get path to shared experience cache."""
    base_dir = os.path.join(os.getcwd(), "shared_experience")
    os.makedirs(base_dir, exist_ok=True)
    
    if cache_type == "heuristic":
        return os.path.join(base_dir, f"heuristic_env{env_version}_{n_episodes}ep.npz")
    elif cache_type == "heatup":
        return os.path.join(base_dir, f"heatup_env{env_version}_{n_steps}steps.npz")
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
```

### Phase 2: Replay Buffer Export/Import

#### 2.1 Modify `eve_rl/eve_rl/replaybuffer/vanillashared.py`

Add task handlers for export/import in the subprocess loop:

```python
# In VanillaStepShared.loop(), add to task handling:

elif task[0] == "export_buffer":
    export_path = task[1]
    logger.info(f"EXPORT: saving buffer to {export_path}")
    episodes_data = []
    for ep in internal_replay_buffer.buffer:
        if ep is not None:
            episodes_data.append(ep)
    self._result_queue.put(("export_done", episodes_data, len(episodes_data)))
    
elif task[0] == "import_buffer":
    episodes_to_import = task[1]
    logger.info(f"IMPORT: loading {len(episodes_to_import)} episodes")
    for ep in episodes_to_import:
        internal_replay_buffer.push(ep)
    self._result_queue.put(("import_done", len(episodes_to_import)))
```

#### 2.2 Add methods to `VanillaSharedBase`

```python
def export_episodes(self) -> List[EpisodeTuple]:
    """Export all episodes from the internal buffer."""
    with self._request_lock:
        self._task_queue.put(["export_buffer", None])
        result = self._result_queue.get()
        if result[0] == "export_done":
            return result[1]
        raise RuntimeError(f"Export failed: {result}")

def import_episodes(self, episodes: List[EpisodeTuple]) -> int:
    """Import episodes into the internal buffer."""
    with self._request_lock:
        self._task_queue.put(["import_buffer", episodes])
        result = self._result_queue.get()
        if result[0] == "import_done":
            return result[1]
        raise RuntimeError(f"Import failed: {result}")
```

### Phase 3: Agent Checkpoint Extensions

#### 3.1 Modify `eve_rl/eve_rl/agent/agent.py`

Add new methods for replay buffer checkpointing:

```python
def save_replay_buffer(self, filepath: str) -> None:
    """Save replay buffer contents to file."""
    from ..util.experience_cache import save_experience_cache, CacheMetadata
    
    episodes = self.replay_buffer.export_episodes()
    
    total_steps = sum(len(ep[1]) for ep in episodes)  # Sum of action lengths
    obs_dim = episodes[0][0].shape[1] if episodes else 0
    action_dim = episodes[0][1].shape[1] if episodes else 0
    
    metadata = CacheMetadata(
        cache_type="replay_buffer",
        env_version=getattr(self, "_env_version", 0),
        n_episodes=len(episodes),
        total_steps=total_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        created_at=datetime.now().isoformat(),
        source_experiment=os.path.basename(os.path.dirname(filepath)),
        description=f"Replay buffer at step {self.step_counter.exploration}",
    )
    
    save_experience_cache(filepath, episodes, metadata)
    logging.getLogger(__name__).info(
        f"Saved replay buffer: {len(episodes)} episodes, {total_steps} steps"
    )


def load_replay_buffer(self, filepath: str) -> int:
    """Load replay buffer contents from file."""
    from ..util.experience_cache import load_experience_cache
    
    episodes, metadata = load_experience_cache(filepath)
    n_loaded = self.replay_buffer.import_episodes(episodes)
    
    logging.getLogger(__name__).info(
        f"Loaded replay buffer: {n_loaded} episodes from {filepath}"
    )
    return n_loaded
```

### Phase 4: Runner Integration

#### 4.1 Modify `eve_rl/eve_rl/runner/runner.py`

Add replay buffer save logic to `Runner`:

```python
class Runner(EveRLObject):
    def __init__(
        self,
        # ... existing params ...
        replay_buffer_save_every_n_evals: int = 4,  # NEW
        replay_buffer_save_every_n_steps: int = 1_000_000,  # NEW
    ) -> None:
        # ... existing init ...
        self.replay_buffer_save_every_n_evals = replay_buffer_save_every_n_evals
        self.replay_buffer_save_every_n_steps = replay_buffer_save_every_n_steps
        self._eval_count = 0
        self._last_replay_save_step = 0


    def _maybe_save_replay_buffer(self):
        """Save replay buffer if conditions are met."""
        current_step = self.step_counter.exploration
        
        # Check step-based trigger
        steps_since_last = current_step - self._last_replay_save_step
        eval_trigger = (self._eval_count % self.replay_buffer_save_every_n_evals == 0)
        step_trigger = (steps_since_last >= self.replay_buffer_save_every_n_steps)
        
        if eval_trigger or step_trigger:
            replay_path = os.path.join(self.checkpoint_folder, "replay_buffer.npz")
            self.agent.save_replay_buffer(replay_path)
            self._last_replay_save_step = current_step
            self.logger.info(f"Replay buffer saved at step {current_step}")


    def eval(self, *, episodes=None, seeds=None):
        # ... existing eval logic ...
        
        self._eval_count += 1
        self._maybe_save_replay_buffer()  # NEW: Save replay buffer
        
        return quality, reward
```

### Phase 5: Training Script Changes

#### 5.1 Create `training_scripts/util/experience_manager.py`

```python
"""
experience_manager.py - High-level experience cache management

Handles:
- Heuristic seeding with caching
- Heatup with caching
- Replay buffer resume
"""

import os
import numpy as np
from typing import List, Optional, Tuple
from eve_rl.util.experience_cache import (
    save_experience_cache, load_experience_cache,
    get_shared_cache_path, CacheMetadata
)
from eve_rl.replaybuffer.replaybuffer import Episode, EpisodeReplay
from eve_rl.util import flatten_obs
from datetime import datetime


class ExperienceManager:
    """Manages experience caches for training."""
    
    def __init__(
        self,
        env_version: int,
        checkpoint_folder: str,
        shared_cache_dir: str = "shared_experience",
    ):
        self.env_version = env_version
        self.checkpoint_folder = checkpoint_folder
        self.shared_cache_dir = shared_cache_dir
        os.makedirs(shared_cache_dir, exist_ok=True)
    
    # =========== Heuristic Seeding ===========
    
    def get_heuristic_cache_path(self, n_episodes: int) -> str:
        """Get path to heuristic cache file."""
        return os.path.join(
            self.shared_cache_dir,
            f"heuristic_env{self.env_version}_{n_episodes}ep.npz"
        )
    
    def heuristic_cache_exists(self, n_episodes: int) -> bool:
        """Check if heuristic cache exists."""
        return os.path.exists(self.get_heuristic_cache_path(n_episodes))
    
    def save_heuristic_cache(
        self,
        episodes: List[Episode],
        n_episodes: int,
    ) -> str:
        """Save heuristic episodes to shared cache."""
        cache_path = self.get_heuristic_cache_path(n_episodes)
        
        # Convert Episode objects to tuples
        episode_tuples = [
            (
                np.array(ep.flat_obs, dtype=np.float32),
                np.array(ep.actions, dtype=np.float32),
                np.array(ep.rewards, dtype=np.float32),
                np.array(ep.terminals, dtype=bool),
            )
            for ep in episodes
        ]
        
        total_steps = sum(len(ep[1]) for ep in episode_tuples)
        obs_dim = episode_tuples[0][0].shape[1] if episode_tuples else 0
        
        metadata = CacheMetadata(
            cache_type="heuristic",
            env_version=self.env_version,
            n_episodes=len(episode_tuples),
            total_steps=total_steps,
            obs_dim=obs_dim,
            action_dim=4,  # DualDeviceNav
            created_at=datetime.now().isoformat(),
            source_experiment="shared",
            description=f"Heuristic centerline-following episodes",
        )
        
        save_experience_cache(cache_path, episode_tuples, metadata)
        return cache_path
    
    def load_heuristic_cache(
        self,
        n_episodes: int,
    ) -> Tuple[List, CacheMetadata]:
        """Load heuristic episodes from shared cache."""
        cache_path = self.get_heuristic_cache_path(n_episodes)
        return load_experience_cache(cache_path)
    
    # =========== Heatup ===========
    
    def get_heatup_cache_path(self, n_steps: int) -> str:
        """Get path to heatup cache file."""
        return os.path.join(
            self.shared_cache_dir,
            f"heatup_env{self.env_version}_{n_steps}steps.npz"
        )
    
    def heatup_cache_exists(self, n_steps: int) -> bool:
        """Check if heatup cache exists."""
        return os.path.exists(self.get_heatup_cache_path(n_steps))
    
    def save_heatup_cache(
        self,
        episodes: List[Episode],
        n_steps: int,
    ) -> str:
        """Save heatup episodes to shared cache."""
        cache_path = self.get_heatup_cache_path(n_steps)
        
        episode_tuples = [
            (
                np.array(ep.flat_obs, dtype=np.float32),
                np.array(ep.actions, dtype=np.float32),
                np.array(ep.rewards, dtype=np.float32),
                np.array(ep.terminals, dtype=bool),
            )
            for ep in episodes
        ]
        
        total_steps = sum(len(ep[1]) for ep in episode_tuples)
        obs_dim = episode_tuples[0][0].shape[1] if episode_tuples else 0
        
        metadata = CacheMetadata(
            cache_type="heatup",
            env_version=self.env_version,
            n_episodes=len(episode_tuples),
            total_steps=total_steps,
            obs_dim=obs_dim,
            action_dim=4,
            created_at=datetime.now().isoformat(),
            source_experiment="shared",
            description=f"Random heatup episodes ({n_steps} steps)",
        )
        
        save_experience_cache(cache_path, episode_tuples, metadata)
        return cache_path
    
    def load_heatup_cache(
        self,
        n_steps: int,
    ) -> Tuple[List, CacheMetadata]:
        """Load heatup episodes from shared cache."""
        cache_path = self.get_heatup_cache_path(n_steps)
        return load_experience_cache(cache_path)
    
    # =========== Replay Buffer ===========
    
    def get_replay_buffer_path(self) -> str:
        """Get path to experiment's replay buffer file."""
        return os.path.join(self.checkpoint_folder, "replay_buffer.npz")
    
    def replay_buffer_exists(self) -> bool:
        """Check if replay buffer checkpoint exists."""
        return os.path.exists(self.get_replay_buffer_path())
```

#### 5.2 Modify `DualDeviceNav_train_vs.py`

```python
# Add imports
from util.experience_manager import ExperienceManager

# After checkpoint folders are created:
experience_mgr = ExperienceManager(
    env_version=env_version,
    checkpoint_folder=checkpoint_folder,
    shared_cache_dir=os.path.join(RESULTS_FOLDER, "..", "shared_experience"),
)

# Modify heuristic seeding section:
if args.heuristic_seeding > 0 and env_version in (4, 5):
    # Check for cached heuristic episodes
    if experience_mgr.heuristic_cache_exists(args.heuristic_seeding):
        print(f"Loading cached heuristic episodes...")
        episodes, metadata = experience_mgr.load_heuristic_cache(args.heuristic_seeding)
        for ep in episodes:
            # Convert to EpisodeReplay and push
            agent.replay_buffer.push(EpisodeReplay(*ep))
        print(f"Loaded {len(episodes)} cached heuristic episodes")
    else:
        print(f"Generating {args.heuristic_seeding} heuristic episodes...")
        # ... existing heuristic generation code ...
        
        # Save to cache for future experiments
        experience_mgr.save_heuristic_cache(collected_episodes, args.heuristic_seeding)
        print(f"Saved heuristic cache for future experiments")
```

#### 5.3 Update `DualDeviceNav_resume.py`

Add complete support for:
- env4/env5
- Action curriculum with step restoration
- Heuristic seeding
- Replay buffer loading

```python
# After agent.load_checkpoint():

# Load replay buffer if available
experience_mgr = ExperienceManager(env_version, checkpoint_folder)
if experience_mgr.replay_buffer_exists():
    print(f"Loading replay buffer checkpoint...")
    n_loaded = agent.load_replay_buffer(experience_mgr.get_replay_buffer_path())
    print(f"Loaded {n_loaded} episodes from replay buffer checkpoint")
else:
    print("No replay buffer checkpoint found, starting with empty buffer")
    
    # Optionally load heuristic cache to bootstrap
    if args.heuristic_seeding > 0 and experience_mgr.heuristic_cache_exists(args.heuristic_seeding):
        print(f"Loading cached heuristic episodes to bootstrap...")
        episodes, _ = experience_mgr.load_heuristic_cache(args.heuristic_seeding)
        for ep in episodes:
            agent.replay_buffer.push(EpisodeReplay(*ep))

# Restore curriculum step count
if args.curriculum and resume_checkpoint:
    env_train._total_steps = agent.step_counter.exploration
    print(f"Restored curriculum to step {env_train._total_steps} (Stage {env_train.current_stage})")
```

---

## Implementation Checklist

### Phase 1: Experience Cache Format (Priority: HIGH)
- [ ] Create `eve_rl/eve_rl/util/experience_cache.py`
- [ ] Define `CacheMetadata` dataclass
- [ ] Implement `save_experience_cache()` function
- [ ] Implement `load_experience_cache()` function
- [ ] Add unit tests for serialization/deserialization

### Phase 2: Replay Buffer Export/Import (Priority: HIGH)
- [ ] Add `export_buffer` task handler to `VanillaStepShared.loop()`
- [ ] Add `import_buffer` task handler to `VanillaStepShared.loop()`
- [ ] Add `export_episodes()` method to `VanillaSharedBase`
- [ ] Add `import_episodes()` method to `VanillaSharedBase`
- [ ] Test with multiprocessing

### Phase 3: Agent Checkpoint Extensions (Priority: HIGH)
- [ ] Add `save_replay_buffer()` method to `Agent`
- [ ] Add `load_replay_buffer()` method to `Agent`
- [ ] Update `Synchron.save_replay_buffer()` if needed
- [ ] Add logging for save/load operations

### Phase 4: Runner Integration (Priority: MEDIUM)
- [ ] Add `replay_buffer_save_every_n_evals` parameter
- [ ] Add `replay_buffer_save_every_n_steps` parameter
- [ ] Add `_maybe_save_replay_buffer()` method
- [ ] Call in `eval()` method
- [ ] Add to `save_config()` for reproducibility

### Phase 5: Training Scripts (Priority: MEDIUM)
- [ ] Create `training_scripts/util/experience_manager.py`
- [ ] Update `DualDeviceNav_train_vs.py` with heuristic caching
- [ ] Update `DualDeviceNav_resume.py` with full support:
  - [ ] env4/env5 support
  - [ ] Action curriculum restoration
  - [ ] Heuristic seeding
  - [ ] Replay buffer loading
- [ ] Update argument parsers in both scripts

### Phase 6: Documentation & Testing (Priority: LOW)
- [ ] Add docstrings to all new functions
- [ ] Update README with new features
- [ ] Create example usage scripts
- [ ] Performance benchmarks for save/load times

---

## Usage Examples

### Fresh Training with Heuristic Caching

```bash
# First experiment: generates and caches heuristic episodes
python DualDeviceNav_train_vs.py -n exp1 --env_version 4 --heuristic_seeding 500

# Second experiment: reuses cached heuristic episodes (instant!)
python DualDeviceNav_train_vs.py -n exp2 --env_version 4 --heuristic_seeding 500
```

### Resuming with Replay Buffer

```bash
# Training stopped at 1M steps
# Checkpoint folder has: checkpoint1000000.everl + replay_buffer.npz

python DualDeviceNav_resume.py -n exp1_resumed --env_version 4 \
    --resume results/.../checkpoints/checkpoint1000000.everl

# Automatically loads replay_buffer.npz from same folder
```

### Shared Cache Structure

```
shared_experience/
├── heuristic_env4_100ep.npz    # 100 heuristic episodes for env4
├── heuristic_env4_500ep.npz    # 500 heuristic episodes for env4
├── heuristic_env5_500ep.npz    # 500 heuristic episodes for env5
├── heatup_env4_10000steps.npz  # 10k heatup steps for env4
└── heatup_env5_10000steps.npz  # 10k heatup steps for env5
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Version mismatch between cached data and new env | Training failure | Include env_version in metadata, validate on load |
| Large replay buffer files slow down checkpointing | Training delay | Save async, save less frequently (every 1M steps) |
| Corrupted cache file | Training failure | Validate checksums, fall back to regeneration |
| Memory spike during export | OOM | Stream export in batches if buffer >5GB |
| Cross-platform compatibility | Pickle failures | Use numpy native types only, no custom objects |

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1 | 2-3 hours | None |
| Phase 2 | 3-4 hours | Phase 1 |
| Phase 3 | 2-3 hours | Phase 2 |
| Phase 4 | 2-3 hours | Phase 3 |
| Phase 5 | 3-4 hours | Phase 4 |
| Phase 6 | 2-3 hours | Phase 5 |
| **Total** | **14-20 hours** | |

---

## Appendix: Data Structure Reference

### EpisodeTuple Format

```python
EpisodeTuple = Tuple[
    np.ndarray,  # flat_obs: shape (T+1, obs_dim), dtype float32
    np.ndarray,  # actions:  shape (T, 4), dtype float32
    np.ndarray,  # rewards:  shape (T,), dtype float32
    np.ndarray,  # terminals: shape (T,), dtype bool
]
# where T = number of steps in episode
```

### CacheMetadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `cache_type` | str | "heuristic", "heatup", or "replay_buffer" |
| `env_version` | int | 1-5 |
| `n_episodes` | int | Number of episodes in cache |
| `total_steps` | int | Sum of steps across all episodes |
| `obs_dim` | int | Observation dimension |
| `action_dim` | int | Action dimension (always 4 for DualDeviceNav) |
| `created_at` | str | ISO timestamp |
| `source_experiment` | str | Experiment name or "shared" |
| `description` | str | Human-readable description |

```

I've created a comprehensive architectural plan at `d:\stEVE_training\docs\REPLAY_BUFFER_CHECKPOINT_ARCHITECTURE.md`.

## Summary of the Plan

### Key Features

| Feature | Description | Storage |
|---------|-------------|---------|
| **Replay Buffer Checkpoint** | Saves every 4 evals OR 1M steps | `{experiment}/checkpoints/replay_buffer.npz` (~750MB, overwritten) |
| **Heuristic Cache** | Reusable across experiments | `shared_experience/heuristic_env{v}_{n}ep.npz` (~15MB for 500 ep) |
| **Heatup Cache** | Reusable across experiments | `shared_experience/heatup_env{v}_{steps}steps.npz` (~3MB) |

### Implementation Phases

```
Phase 1: Experience Cache Format     [2-3 hrs]  ─┐
Phase 2: Replay Buffer Export/Import [3-4 hrs]  ─┼─ Core Infrastructure
Phase 3: Agent Checkpoint Extensions [2-3 hrs]  ─┘
Phase 4: Runner Integration          [2-3 hrs]  ─┬─ Integration
Phase 5: Training Scripts            [3-4 hrs]  ─┘
Phase 6: Documentation & Testing     [2-3 hrs]
                                     ──────────
Total:                               14-20 hrs
```

### New Files to Create

```
eve_rl/eve_rl/util/experience_cache.py      # Serialization utilities
training_scripts/util/experience_manager.py  # High-level cache management
shared_experience/                           # Reusable caches folder
```

### Benefits

1. **Time Savings**: Heuristic seeding goes from ~5 min → instant (cached)
2. **Resume Reliability**: Replay buffer restored = no "cold start" instability
3. **Experiment Reproducibility**: Same heuristic/heatup data across experiments
4. **Storage Efficient**: Single ~750MB replay file (overwritten, not accumulated)

Would you like me to start implementing any of these phases?

```plaintext
Phase 1: Experience Cache Format     [2-3 hrs]  ─┐
Phase 2: Replay Buffer Export/Import [3-4 hrs]  ─┼─ Core Infrastructure
Phase 3: Agent Checkpoint Extensions [2-3 hrs]  ─┘
Phase 4: Runner Integration          [2-3 hrs]  ─┬─ Integration
Phase 5: Training Scripts            [3-4 hrs]  ─┘
Phase 6: Documentation & Testing     [2-3 hrs]
                                     ──────────
Total:                               14-20 hrs
```

```plaintext
eve_rl/eve_rl/util/experience_cache.py      # Serialization utilities
training_scripts/util/experience_manager.py  # High-level cache management
shared_experience/                           # Reusable caches folder
```

---

**User**

ALso write an md file for all the proposed fixes that you proposed for resume operations (context , reason and fix ) that you mentioned before our chat of saving seedings and replay_buffer .

---

**Cursor**

```markdown
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

```

I've created the document at `d:\stEVE_training\docs\RESUME_OPERATIONS_FIXES.md`.

## Summary of Documents Created

| Document | Location | Contents |
|----------|----------|----------|
| **RESUME_OPERATIONS_FIXES.md** | `docs/` | 8 issues with context, impact, and fixes for resume operations |
| **REPLAY_BUFFER_CHECKPOINT_ARCHITECTURE.md** | `docs/` | Full architecture plan for replay buffer & experience caching |

### Issues Documented in RESUME_OPERATIONS_FIXES.md

| # | Issue | Priority | Impact |
|---|-------|----------|--------|
| 1 | ActionCurriculumWrapper step reset | 🔴 P0 | Catastrophic unlearning |
| 2 | Environment logging counters reset | 🟡 P2 | Log discontinuity |
| 3 | Replay buffer empty | 🟠 P1 | Learning instability |
| 4 | Resume script missing features | 🟠 P1 | Cannot resume env4/5/curriculum |
| 5 | Runner best_eval tracking lost | 🟡 P2 | May overwrite best checkpoint |
| 6 | Probe states not captured | 🟢 P3 | Empty diagnostics plots |
| 7 | Diagnostics logger counters reset | 🟢 P3 | Episode ID discontinuity |
| 8 | Policy snapshot step misalignment | 🟢 P3 | Extra snapshot saved |

Each issue includes:
- **Context**: Where the code lives and what it does
- **Problem**: What breaks on resume
- **Impact**: Severity level and consequences
- **Fix**: Concrete code changes with examples

---

**User**

All these fixes (RESUME_OPERATIONS_FIXES) are done in D:\neve\.claude\worktrees\rl_resume codebase . Verify if they are done Correctly and efficiently wothout introducing any new bugs or inefficicies  .

---

**Cursor**

Let me examine the `rl_resume` worktree to verify the fixes have been implemented correctly.

Now let me examine the actual implementation files:

Now let me check the action curriculum and verify no issues exist there:

Now I have enough information to provide a comprehensive verification report. Let me write it:

```markdown
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

```

I've completed a thorough verification of the `rl_resume` worktree. Here's a summary:

## Verification Summary

### ✅ Fixes Implemented Correctly

| Fix | Verification |
|-----|-------------|
| **1. Same folder resume** | Path derivation from checkpoint location is correct |
| **2. Save runner_state** | Uses `.copy()` for `best_eval` dict, saves in `additional_info` |
| **3. Restore runner_state** | Backward-compatible fallback for old checkpoints |
| **4. Probe states** | Graceful file handling with try/except |
| **5. Curriculum step** | Restores `_total_steps` from `agent.step_counter.exploration` |

### 🔴 Bug Found

**Location**: `DualDeviceNav_resume.py` lines 351-353

```python
# CURRENT (buggy):
act_low = seed_env.action_space.low.astype(np.float64)
act_high = seed_env.action_space.high.astype(np.float64)

# SHOULD BE:
act_low = seed_env.action_space.low.astype(np.float64).flatten()
act_high = seed_env.action_space.high.astype(np.float64).flatten()
```

The action space is shaped `(2, 2)` but the heuristic returns a flat `(4,)` array. Without `.flatten()`, the normalization will cause a **shape mismatch error** or produce incorrect normalized actions.

### ⚠️ Missing Feature

**env5 support** is not present in `rl_resume` - there's no `env5.py` or `BenchEnv5` import. If experiments in `stEVE_training` use env5, they cannot be resumed with this codebase.

---

I've written a detailed report to `d:\stEVE_training\docs\RL_RESUME_VERIFICATION_REPORT.md`.

Would you like me to fix the bug in the `rl_resume` codebase?

```python
# CURRENT (buggy):
act_low = seed_env.action_space.low.astype(np.float64)
act_high = seed_env.action_space.high.astype(np.float64)

# SHOULD BE:
act_low = seed_env.action_space.low.astype(np.float64).flatten()
act_high = seed_env.action_space.high.astype(np.float64).flatten()
```

---

**User**

which files were changed for these fixes

---

**Cursor**

Based on the `RESUME_TRAINING_FIXES.md` documentation in the `rl_resume` codebase, here are the files that were changed:

## Files Modified

| File | Role |
|------|------|
| `training _scripts/DualDeviceNav_resume.py` | Resume entry point script |
| `eve_rl/eve_rl/runner/runner.py` | Training orchestrator (Runner class) |

### Specific Changes:

**1. `DualDeviceNav_resume.py`:**
- Path derivation logic to resume in same experiment folder (lines 185-207)
- Curriculum step restoration (lines 296-300)
- Runner state restoration call (line 326)
- Added env4 support, curriculum args, heuristic seeding

**2. `runner.py`:**
- Added `restore_runner_state()` method (lines 101-143)
- Saves `runner_state` in checkpoints (lines 214-219)

---

## Files NOT Modified (already correct)

| File | Why |
|------|-----|
| `eve_rl/eve_rl/agent/agent.py` | Checkpoint save/load for agent state was already correct |
| `eve_rl/eve_rl/util/diagnostics_logger.py` | Already uses append mode and skips headers on existing files |
| `eve_rl/eve_rl/agent/singelagentprocess.py` | Subprocess logging derives paths from `main.log` location |
| `training _scripts/util/env*.py` | Worker step logs use `STEP_LOG_DIR` env var and append mode |
| `training _scripts/util/action_curriculum.py` | Not modified - fix is in resume script instead |

---

**User**

Did you read all the changes in that file thoroughly?

---

**Cursor**

Let me read both modified files in their entirety to do a thorough verification:

Now I've read both files thoroughly. Let me provide a comprehensive comparison of what was planned vs. what was implemented:

## Detailed Fix-by-Fix Verification

| Issue | Planned Fix | Implementation Status |
|-------|-------------|----------------------|
| **1. ActionCurriculumWrapper Reset** (P0) | Restore `_total_steps` from checkpoint | ✅ **DONE** - Lines 296-300 |
| **2. Environment Logging Counters** (P2) | Pass step counters to env | ❌ **NOT DONE** |
| **3. Replay Buffer Empty** (P1) | Warmup or checkpoint buffer | ⚠️ **PARTIAL** - Heuristic seeding optional, no auto-warmup |
| **4. Missing Features** (P1) | Add env4/5, curriculum, heuristic | ⚠️ **PARTIAL** - env5 missing |
| **5. Runner best_eval Lost** (P2) | Store in checkpoint | ✅ **DONE** - Lines 214-219 (save), 112-114 (restore) |
| **6. Probe States Not Captured** (P3) | Restore from saved file | ✅ **DONE** - Lines 131-143 |
| **7. Diagnostics Counters Reset** (P3) | Include in runner_state | ✅ **DONE** - Line 114 |
| **8. Snapshot Step Misalignment** (P3) | Compute aligned next step | ✅ **DONE** - Lines 115, 122-129 |

---

## Issues Found in Implementation

### 🔴 Bug: Missing `.flatten()` in Heuristic Seeding
**Location**: `DualDeviceNav_resume.py` lines 351-353

```python
# Current (BUGGY):
act_low = seed_env.action_space.low.astype(np.float64)
act_high = seed_env.action_space.high.astype(np.float64)

# Should be (per stEVE_training/DualDeviceNav_train_vs.py lines 304-305):
act_low = seed_env.action_space.low.astype(np.float64).flatten()
act_high = seed_env.action_space.high.astype(np.float64).flatten()
```

### ⚠️ Missing: env5 Support
- No `from util.env5 import BenchEnv5` import
- No `env5.py` file exists in `training _scripts/util/`
- `choices=[1, 2, 3, 4]` missing `5`

### ⚠️ Not Implemented: Environment Counter Restoration  
The document suggested restoring env counters:
```python
env_train._step_count = agent.step_counter.exploration
env_train._episode_count = agent.episode_counter.exploration
```
This was NOT implemented (marked as optional in the plan).

### ⚠️ Not Implemented: Automatic Replay Buffer Warmup
The document suggested auto-warmup on resume:
```python
if resume_checkpoint:
    agent.explore(episodes=100)  # NOT DONE - only optional --heuristic_seeding
```

---

## Summary

**5 of 8 fixes fully implemented**, 2 partial, 1 not done (but marked optional).

**1 bug introduced** that needs fixing before production use.

```python
# Current (BUGGY):
act_low = seed_env.action_space.low.astype(np.float64)
act_high = seed_env.action_space.high.astype(np.float64)

# Should be (per stEVE_training/DualDeviceNav_train_vs.py lines 304-305):
act_low = seed_env.action_space.low.astype(np.float64).flatten()
act_high = seed_env.action_space.high.astype(np.float64).flatten()
```

```python
env_train._step_count = agent.step_counter.exploration
env_train._episode_count = agent.episode_counter.exploration
```

```python
if resume_checkpoint:
    agent.explore(episodes=100)  # NOT DONE - only optional --heuristic_seeding
```

---

**User**

ok so merge all the code here and fix the two bugs identified

---

**Cursor**

I'll fix both issues in the `rl_resume` codebase:

1. **Bug fix**: Add missing `.flatten()` in heuristic seeding
2. **Missing feature**: Add env5 support

---

**User**

no I mean merge code in this codebase ; don't change that codebase

---

**Cursor**

Got it! I'll merge the fixes into the `stEVE_training` codebase (current workspace) and fix the bugs. Let me first check what exists here:

Now I'll update both files. First, let me add the `restore_runner_state` method to `runner.py` and save `runner_state` in checkpoints:

```python
from typing import List, Optional
from math import inf
import csv
import json
import logging
import os
import time
import numpy as np
from ..util import EveRLObject
from ..agent.agent import Agent, StepCounter, EpisodeCounter


class Runner(EveRLObject):
    def __init__(
        self,
        agent: Agent,
        heatup_action_low: List[float],
        heatup_action_high: List[float],
        agent_parameter_for_result_file: dict,
        checkpoint_folder: str,
        results_file: str,
        quality_info: Optional[str] = None,
        info_results: Optional[List[str]] = None,
        diagnostics_folder: Optional[str] = None,
        policy_snapshot_every_steps: int = 10000,
        n_probe_episodes: int = 3,
        n_probe_near_start_steps: int = 10,
    ) -> None:
        self.agent = agent
        self.heatup_action_low = heatup_action_low
        self.heatup_action_high = heatup_action_high
        self.agent_parameter_for_result_file = agent_parameter_for_result_file
        self.checkpoint_folder = checkpoint_folder
        self.results_file = results_file
        self.quality_info = quality_info
        self.info_results = info_results or []
        self.logger = logging.getLogger(self.__module__)
        
        # Diagnostics settings
        self.diagnostics_folder = diagnostics_folder
        # Ensure worker step logs directory exists.
        # NOTE: STEP_LOG_DIR must be set BEFORE agent creation (which spawns workers).
        # The training script should set os.environ["STEP_LOG_DIR"] before creating the agent.
        # This setdefault is a fallback but will NOT help if workers are already spawned.
        if self.diagnostics_folder is not None:
            logs_subprocesses = os.path.join(self.diagnostics_folder, "logs_subprocesses")
            os.makedirs(logs_subprocesses, exist_ok=True)
            os.environ.setdefault("STEP_LOG_DIR", logs_subprocesses)
        self.policy_snapshot_every_steps = policy_snapshot_every_steps
        self.n_probe_episodes = n_probe_episodes
        self.n_probe_near_start_steps = n_probe_near_start_steps
        self._next_snapshot_step = policy_snapshot_every_steps
        self._probe_states_set = False
        
        # Episode summary logger
        self._episode_summary_file = None
        self._episode_summary_logger = None
        self._episode_summary_counter = 0  # Fix #5: monotonic episode_id
        if self.diagnostics_folder is not None:
            csv_folder = os.path.join(self.diagnostics_folder, "csv")
            os.makedirs(csv_folder, exist_ok=True)
            self._episode_summary_file = os.path.join(csv_folder, "episode_summary.jsonl")
            self.logger.info(f"Episode summary will be logged to {self._episode_summary_file}")

        self._results = {
            "episodes explore": 0,
            "steps explore": 0,
        }
        self._results["quality"] = 0.0
        for info_result in self.info_results:
            self._results[info_result] = 0.0
        self._results["reward"] = 0.0
        self._results["best quality"] = 0.0
        self._results["best explore steps"] = 0.0

        file_existed = os.path.isfile(results_file)

        with open(results_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_existed:
                writer.writerow(
                    list(self._results.keys())
                    + [" "]
                    + list(agent_parameter_for_result_file.keys())
                )
                writer.writerow(
                    [" "] * (len(self._results.values()) + 1)
                    + list(agent_parameter_for_result_file.values())
                )

        self.best_eval = {"steps": 0, "quality": -inf}

    @property
    def step_counter(self) -> StepCounter:
        return self.agent.step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self.agent.episode_counter

    def heatup(self, steps: int):
        episodes = self.agent.heatup(
            steps=steps,
            custom_action_low=self.heatup_action_low,
            custom_action_high=self.heatup_action_high,
        )
        return episodes

    def explore(self, n_episodes: int):
        self.agent.explore(episodes=n_episodes)

    def update(self, n_steps: int):
        self.agent.update(n_steps)

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[List[int]] = None
    ):
        explore_steps = self.step_counter.exploration
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{explore_steps}.everl"
        )
        result_episodes = self.agent.evaluate(episodes=episodes, seeds=seeds)
        qualities, rewards = [], []
        results_for_info = {
            info_result_name: [] for info_result_name in self.info_results
        }
        eval_results = {"episodes": []}
        for episode in result_episodes:
            reward = episode.episode_reward
            quality = (
                episode.infos[-1][self.quality_info]
                if self.quality_info is not None
                else reward
            )
            qualities.append(quality)
            rewards.append(reward)
            for info_result_name in self.info_results:
                info = episode.infos[-1][info_result_name]
                results_for_info[info_result_name].append(info)
            eval_results["episodes"].append(
                {
                    "seed": episode.seed,
                    "options": episode.options,
                    "quality": episode.infos[-1][self.quality_info],
                }
            )

        reward = sum(rewards) / len(rewards)
        quality = sum(qualities) / len(qualities)
        for info_result_name, results in results_for_info.items():
            result = sum(results) / len(results)
            self._results[info_result_name] = round(result, 3)
        save_best = False
        if quality > self.best_eval["quality"]:
            save_best = True
            self.best_eval["quality"] = quality
            self.best_eval["steps"] = explore_steps

        self._results["episodes explore"] = self.episode_counter.exploration
        self._results["steps explore"] = explore_steps
        self._results["reward"] = round(reward, 3)
        self._results["quality"] = round(quality, 3)
        self._results["best quality"] = self.best_eval["quality"]
        self._results["best explore steps"] = self.best_eval["steps"]

        eval_results.update(self._results)
        eval_results.pop("best quality")
        eval_results.pop("best explore steps")

        # Include runner state for resume support
        eval_results["runner_state"] = {
            "best_eval": self.best_eval.copy(),
            "episode_summary_counter": self._episode_summary_counter,
            "next_snapshot_step": self._next_snapshot_step,
        }

        self.agent.save_checkpoint(checkpoint_file, eval_results)
        if save_best:
            checkpoint_file = os.path.join(
                self.checkpoint_folder, "best_checkpoint.everl"
            )
            self.agent.save_checkpoint(checkpoint_file, eval_results)

        log_info = (
            f"Quality: {quality}, Reward: {reward}, Exploration steps: {explore_steps}"
        )
        self.logger.info(log_info)
        with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(self._results.values())

        return quality, reward

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: int = None,
        explore_steps_limit: int = None,
    ):
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        while self.step_counter.exploration < explore_steps_limit:
            # Total experience = heatup + exploration (both contribute to replay buffer)
            total_experience_steps = (
                self.step_counter.heatup + self.step_counter.exploration
            )
            update_steps = (
                total_experience_steps * update_steps_per_explore_step
                - self.step_counter.update
            )
            result = self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )
            
            # Log episode summaries if we got results
            if result is not None:
                episodes, _ = result if isinstance(result, tuple) else (result, None)
                if episodes:
                    self._log_episode_summaries(episodes)
            
            # Check for policy snapshots after each exploration cycle
            # This ensures snapshots are saved at the actual milestone with correct weights
            self._maybe_save_policy_snapshot()

    def training_run(
        self,
        heatup_steps: int,
        training_steps: int,
        explore_steps_between_eval: int,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
    ):
        # TODO: Log Training Run Infos
        heatup_episodes = self.heatup(heatup_steps)
        
        # Capture and set probe states after heatup
        self._capture_and_set_probe_states(heatup_episodes)
        
        next_eval_step_limt = (
            self.agent.step_counter.exploration + explore_steps_between_eval
        )
        while self.agent.step_counter.exploration < training_steps:
            self.explore_and_update(
                explore_episodes_between_updates,
                update_steps_per_explore_step,
                explore_steps_limit=next_eval_step_limt,
            )
            
            # Note: policy snapshots are now saved inside explore_and_update()
            # at the actual milestone steps with correct weights
            
            quality, reward = self.eval(episodes=eval_episodes, seeds=eval_seeds)
            next_eval_step_limt += explore_steps_between_eval

        return quality, reward

    def _capture_and_set_probe_states(self, episodes: List):
        """Capture probe states from heatup episodes and set them in trainer."""
        if not episodes or self._probe_states_set:
            return
            
        try:
            probe_states = []
            n_episodes = min(len(episodes), self.n_probe_episodes)
            
            for ep_idx in range(n_episodes):
                episode = episodes[ep_idx]
                if hasattr(episode, 'flat_obs') and len(episode.flat_obs) > 0:
                    # Collect start state
                    probe_states.append(episode.flat_obs[0])
                    
                    # Collect near-start states
                    n_states = min(self.n_probe_near_start_steps, len(episode.flat_obs) - 1)
                    for step_idx in range(1, n_states + 1):
                        probe_states.append(episode.flat_obs[step_idx])
                        
            if probe_states:
                probe_states_array = np.array(probe_states)
                
                # Set probe states in agent (if it supports it)
                if hasattr(self.agent, 'set_probe_states'):
                    self.agent.set_probe_states(probe_states_array.tolist())
                    self.logger.info(f"Set {len(probe_states)} probe states for diagnostics")
                    
                # Save probe states to disk
                if self.diagnostics_folder is not None:
                    probes_dir = os.path.join(self.diagnostics_folder, "probes")
                    os.makedirs(probes_dir, exist_ok=True)
                    probe_path = os.path.join(probes_dir, "probe_states.npz")
                    np.savez(probe_path, probe_states=probe_states_array)
                    self.logger.info(f"Saved probe states to {probe_path}")
                    
                self._probe_states_set = True
                
        except Exception as e:
            self.logger.warning(f"Failed to capture probe states: {e}")

    def _maybe_save_policy_snapshot(self):
        """
        Save policy snapshot if we've reached the next snapshot step.
        
        FIXED: Uses a while loop to catch up on all missed snapshots when the
        eval cycle spans multiple snapshot intervals.
        """
        if self.diagnostics_folder is None:
            return
            
        current_step = self.step_counter.exploration
        
        # Save all missed snapshots (in case eval cycle spans multiple snapshot intervals)
        while current_step >= self._next_snapshot_step:
            try:
                snapshots_dir = os.path.join(self.diagnostics_folder, "policy_snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)
                
                # Use the scheduled snapshot step, not current step, for consistent naming
                snapshot_step = self._next_snapshot_step
                snapshot_path = os.path.join(snapshots_dir, f"policy_{snapshot_step}.pt")
                
                if hasattr(self.agent, 'save_policy_snapshot'):
                    self.agent.save_policy_snapshot(snapshot_path)
                    self.logger.info(f"Saved policy snapshot at scheduled step {snapshot_step} (current={current_step})")
                    
            except Exception as e:
                self.logger.warning(f"Failed to save policy snapshot at step {self._next_snapshot_step}: {e}")
                
            self._next_snapshot_step += self.policy_snapshot_every_steps
        
        # Flush diagnostics logs after all snapshots are saved
        if hasattr(self.agent, 'flush_diagnostics'):
            try:
                self.agent.flush_diagnostics()
            except Exception as e:
                self.logger.warning(f"Failed to flush diagnostics: {e}")

    def _log_episode_summaries(self, episodes: List):
        """
        Log episode summaries to episode_summary.jsonl with explore_step and update_step.
        
        This directly solves the timestamp correlation problem by providing exact
        step counters at episode completion time.
        """
        if self._episode_summary_file is None:
            return
            
        try:
            # Fix #4: Use per-episode step counters stamped by the worker at
            # completion time. Fall back to current shared counter for backward
            # compatibility with Episode objects that don't carry step snapshots.
            fallback_explore = self.step_counter.exploration
            fallback_update = self.step_counter.update

            with open(self._episode_summary_file, "a", encoding="utf-8") as f:
                for i, episode in enumerate(episodes):
                    try:
                        wall_time = time.time()  # Per-episode timestamp
                        # Use per-episode step snapshot if available, else fallback
                        explore_step = getattr(episode, 'explore_step_at_completion', None) or fallback_explore
                        update_step = getattr(episode, 'update_step_at_completion', None) or fallback_update
                        # Extract episode data
                        total_reward = getattr(episode, 'episode_reward', 0.0)
                        steps = len(getattr(episode, 'rewards', []))
                        terminated = getattr(episode, 'terminals', [False])[-1] if getattr(episode, 'terminals', []) else False
                        truncated = getattr(episode, 'truncations', [False])[-1] if getattr(episode, 'truncations', []) else False

                        # Try to get max_insertion from episode info if available
                        max_insertion = None
                        infos = getattr(episode, 'infos', [])
                        if infos:
                            last_info = infos[-1] if infos else {}
                            # Look for insertion depth in common info keys
                            for key in ['max_insertion', 'inserted_length', 'insertion_depth']:
                                if key in last_info:
                                    val = last_info[key]
                                    if hasattr(val, '__iter__') and not isinstance(val, str):
                                        max_insertion = float(max(val))
                                    else:
                                        max_insertion = float(val)
                                    break

                        # Fix #5: Monotonically increasing counter for episode_id.
                        # Replaces retroactive calculation that assumed sequential,
                        # non-interleaved episodes.
                        episode_id = self._episode_summary_counter
                        self._episode_summary_counter += 1

                        summary = {
                            "wall_time": wall_time,
                            "explore_step": explore_step,
                            "update_step": update_step,
                            "episode_id": episode_id,
                            "total_reward": float(total_reward),
                            "steps": steps,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                        if max_insertion is not None:
                            summary["max_insertion"] = max_insertion
                        
                        f.write(json.dumps(summary) + "\n")
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to log episode {i} summary: {e}")
                        
                f.flush()
                
        except Exception as e:
            self.logger.warning(f"Failed to log episode summaries: {e}")

```

```python
from typing import List, Optional
from math import inf
import csv
import json
import logging
import os
import time
import numpy as np
from ..util import EveRLObject
from ..agent.agent import Agent, StepCounter, EpisodeCounter


class Runner(EveRLObject):
    def __init__(
        self,
        agent: Agent,
        heatup_action_low: List[float],
        heatup_action_high: List[float],
        agent_parameter_for_result_file: dict,
        checkpoint_folder: str,
        results_file: str,
        quality_info: Optional[str] = None,
        info_results: Optional[List[str]] = None,
        diagnostics_folder: Optional[str] = None,
        policy_snapshot_every_steps: int = 10000,
        n_probe_episodes: int = 3,
        n_probe_near_start_steps: int = 10,
    ) -> None:
        self.agent = agent
        self.heatup_action_low = heatup_action_low
        self.heatup_action_high = heatup_action_high
        self.agent_parameter_for_result_file = agent_parameter_for_result_file
        self.checkpoint_folder = checkpoint_folder
        self.results_file = results_file
        self.quality_info = quality_info
        self.info_results = info_results or []
        self.logger = logging.getLogger(self.__module__)
        
        # Diagnostics settings
        self.diagnostics_folder = diagnostics_folder
        # Ensure worker step logs directory exists.
        # NOTE: STEP_LOG_DIR must be set BEFORE agent creation (which spawns workers).
        # The training script should set os.environ["STEP_LOG_DIR"] before creating the agent.
        # This setdefault is a fallback but will NOT help if workers are already spawned.
        if self.diagnostics_folder is not None:
            logs_subprocesses = os.path.join(self.diagnostics_folder, "logs_subprocesses")
            os.makedirs(logs_subprocesses, exist_ok=True)
            os.environ.setdefault("STEP_LOG_DIR", logs_subprocesses)
        self.policy_snapshot_every_steps = policy_snapshot_every_steps
        self.n_probe_episodes = n_probe_episodes
        self.n_probe_near_start_steps = n_probe_near_start_steps
        self._next_snapshot_step = policy_snapshot_every_steps
        self._probe_states_set = False
        
        # Episode summary logger
        self._episode_summary_file = None
        self._episode_summary_logger = None
        self._episode_summary_counter = 0  # Fix #5: monotonic episode_id
        if self.diagnostics_folder is not None:
            csv_folder = os.path.join(self.diagnostics_folder, "csv")
            os.makedirs(csv_folder, exist_ok=True)
            self._episode_summary_file = os.path.join(csv_folder, "episode_summary.jsonl")
            self.logger.info(f"Episode summary will be logged to {self._episode_summary_file}")

        self._results = {
            "episodes explore": 0,
            "steps explore": 0,
        }
        self._results["quality"] = 0.0
        for info_result in self.info_results:
            self._results[info_result] = 0.0
        self._results["reward"] = 0.0
        self._results["best quality"] = 0.0
        self._results["best explore steps"] = 0.0

        file_existed = os.path.isfile(results_file)

        with open(results_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_existed:
                writer.writerow(
                    list(self._results.keys())
                    + [" "]
                    + list(agent_parameter_for_result_file.keys())
                )
                writer.writerow(
                    [" "] * (len(self._results.values()) + 1)
                    + list(agent_parameter_for_result_file.values())
                )

        self.best_eval = {"steps": 0, "quality": -inf}

    @property
    def step_counter(self) -> StepCounter:
        return self.agent.step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self.agent.episode_counter

    def restore_runner_state(self, checkpoint_path: str, probe_states_dir: Optional[str] = None):
        """Restore Runner-level state from a checkpoint for resume support.

        Restores: best_eval, _episode_summary_counter, _next_snapshot_step,
        and optionally probe states from a saved .npz file.
        """
        import torch
        checkpoint = torch.load(checkpoint_path)
        additional_info = checkpoint.get("additional_info", {})
        runner_state = additional_info.get("runner_state") if additional_info else None

        if runner_state:
            self.best_eval = runner_state["best_eval"]
            self._episode_summary_counter = runner_state["episode_summary_counter"]
            self._next_snapshot_step = runner_state["next_snapshot_step"]
            self.logger.info(
                f"Restored runner state: best_eval={self.best_eval}, "
                f"episode_counter={self._episode_summary_counter}, "
                f"next_snapshot_step={self._next_snapshot_step}"
            )
        else:
            # Checkpoint predates runner_state support — compute safe defaults
            explore_steps = self.step_counter.exploration
            self._next_snapshot_step = (
                (explore_steps // self.policy_snapshot_every_steps) + 1
            ) * self.policy_snapshot_every_steps
            self.logger.warning(
                f"No runner_state in checkpoint. Setting _next_snapshot_step={self._next_snapshot_step}"
            )

        # Restore probe states from saved .npz if available
        if probe_states_dir is not None:
            probe_path = os.path.join(probe_states_dir, "probes", "probe_states.npz")
            if os.path.isfile(probe_path):
                try:
                    data = np.load(probe_path)
                    probe_states = data["probe_states"]
                    if hasattr(self.agent, 'set_probe_states'):
                        self.agent.set_probe_states(probe_states.tolist())
                        self._probe_states_set = True
                        self.logger.info(f"Restored {len(probe_states)} probe states from {probe_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to restore probe states: {e}")

    def heatup(self, steps: int):
        episodes = self.agent.heatup(
            steps=steps,
            custom_action_low=self.heatup_action_low,
            custom_action_high=self.heatup_action_high,
        )
        return episodes

    def explore(self, n_episodes: int):
        self.agent.explore(episodes=n_episodes)

    def update(self, n_steps: int):
        self.agent.update(n_steps)

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[List[int]] = None
    ):
        explore_steps = self.step_counter.exploration
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{explore_steps}.everl"
        )
        result_episodes = self.agent.evaluate(episodes=episodes, seeds=seeds)
        qualities, rewards = [], []
        results_for_info = {
            info_result_name: [] for info_result_name in self.info_results
        }
        eval_results = {"episodes": []}
        for episode in result_episodes:
            reward = episode.episode_reward
            quality = (
                episode.infos[-1][self.quality_info]
                if self.quality_info is not None
                else reward
            )
            qualities.append(quality)
            rewards.append(reward)
            for info_result_name in self.info_results:
                info = episode.infos[-1][info_result_name]
                results_for_info[info_result_name].append(info)
            eval_results["episodes"].append(
                {
                    "seed": episode.seed,
                    "options": episode.options,
                    "quality": episode.infos[-1][self.quality_info],
                }
            )

        reward = sum(rewards) / len(rewards)
        quality = sum(qualities) / len(qualities)
        for info_result_name, results in results_for_info.items():
            result = sum(results) / len(results)
            self._results[info_result_name] = round(result, 3)
        save_best = False
        if quality > self.best_eval["quality"]:
            save_best = True
            self.best_eval["quality"] = quality
            self.best_eval["steps"] = explore_steps

        self._results["episodes explore"] = self.episode_counter.exploration
        self._results["steps explore"] = explore_steps
        self._results["reward"] = round(reward, 3)
        self._results["quality"] = round(quality, 3)
        self._results["best quality"] = self.best_eval["quality"]
        self._results["best explore steps"] = self.best_eval["steps"]

        eval_results.update(self._results)
        eval_results.pop("best quality")
        eval_results.pop("best explore steps")

        # Include runner state for resume support
        eval_results["runner_state"] = {
            "best_eval": self.best_eval.copy(),
            "episode_summary_counter": self._episode_summary_counter,
            "next_snapshot_step": self._next_snapshot_step,
        }

        self.agent.save_checkpoint(checkpoint_file, eval_results)
        if save_best:
            checkpoint_file = os.path.join(
                self.checkpoint_folder, "best_checkpoint.everl"
            )
            self.agent.save_checkpoint(checkpoint_file, eval_results)

        log_info = (
            f"Quality: {quality}, Reward: {reward}, Exploration steps: {explore_steps}"
        )
        self.logger.info(log_info)
        with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(self._results.values())

        return quality, reward

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: int = None,
        explore_steps_limit: int = None,
    ):
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        while self.step_counter.exploration < explore_steps_limit:
            # Total experience = heatup + exploration (both contribute to replay buffer)
            total_experience_steps = (
                self.step_counter.heatup + self.step_counter.exploration
            )
            update_steps = (
                total_experience_steps * update_steps_per_explore_step
                - self.step_counter.update
            )
            result = self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )
            
            # Log episode summaries if we got results
            if result is not None:
                episodes, _ = result if isinstance(result, tuple) else (result, None)
                if episodes:
                    self._log_episode_summaries(episodes)
            
            # Check for policy snapshots after each exploration cycle
            # This ensures snapshots are saved at the actual milestone with correct weights
            self._maybe_save_policy_snapshot()

    def training_run(
        self,
        heatup_steps: int,
        training_steps: int,
        explore_steps_between_eval: int,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
    ):
        # TODO: Log Training Run Infos
        heatup_episodes = self.heatup(heatup_steps)
        
        # Capture and set probe states after heatup
        self._capture_and_set_probe_states(heatup_episodes)
        
        next_eval_step_limt = (
            self.agent.step_counter.exploration + explore_steps_between_eval
        )
        while self.agent.step_counter.exploration < training_steps:
            self.explore_and_update(
                explore_episodes_between_updates,
                update_steps_per_explore_step,
                explore_steps_limit=next_eval_step_limt,
            )
            
            # Note: policy snapshots are now saved inside explore_and_update()
            # at the actual milestone steps with correct weights
            
            quality, reward = self.eval(episodes=eval_episodes, seeds=eval_seeds)
            next_eval_step_limt += explore_steps_between_eval

        return quality, reward

    def _capture_and_set_probe_states(self, episodes: List):
        """Capture probe states from heatup episodes and set them in trainer."""
        if not episodes or self._probe_states_set:
            return
            
        try:
            probe_states = []
            n_episodes = min(len(episodes), self.n_probe_episodes)
            
            for ep_idx in range(n_episodes):
                episode = episodes[ep_idx]
                if hasattr(episode, 'flat_obs') and len(episode.flat_obs) > 0:
                    # Collect start state
                    probe_states.append(episode.flat_obs[0])
                    
                    # Collect near-start states
                    n_states = min(self.n_probe_near_start_steps, len(episode.flat_obs) - 1)
                    for step_idx in range(1, n_states + 1):
                        probe_states.append(episode.flat_obs[step_idx])
                        
            if probe_states:
                probe_states_array = np.array(probe_states)
                
                # Set probe states in agent (if it supports it)
                if hasattr(self.agent, 'set_probe_states'):
                    self.agent.set_probe_states(probe_states_array.tolist())
                    self.logger.info(f"Set {len(probe_states)} probe states for diagnostics")
                    
                # Save probe states to disk
                if self.diagnostics_folder is not None:
                    probes_dir = os.path.join(self.diagnostics_folder, "probes")
                    os.makedirs(probes_dir, exist_ok=True)
                    probe_path = os.path.join(probes_dir, "probe_states.npz")
                    np.savez(probe_path, probe_states=probe_states_array)
                    self.logger.info(f"Saved probe states to {probe_path}")
                    
                self._probe_states_set = True
                
        except Exception as e:
            self.logger.warning(f"Failed to capture probe states: {e}")

    def _maybe_save_policy_snapshot(self):
        """
        Save policy snapshot if we've reached the next snapshot step.
        
        FIXED: Uses a while loop to catch up on all missed snapshots when the
        eval cycle spans multiple snapshot intervals.
        """
        if self.diagnostics_folder is None:
            return
            
        current_step = self.step_counter.exploration
        
        # Save all missed snapshots (in case eval cycle spans multiple snapshot intervals)
        while current_step >= self._next_snapshot_step:
            try:
                snapshots_dir = os.path.join(self.diagnostics_folder, "policy_snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)
                
                # Use the scheduled snapshot step, not current step, for consistent naming
                snapshot_step = self._next_snapshot_step
                snapshot_path = os.path.join(snapshots_dir, f"policy_{snapshot_step}.pt")
                
                if hasattr(self.agent, 'save_policy_snapshot'):
                    self.agent.save_policy_snapshot(snapshot_path)
                    self.logger.info(f"Saved policy snapshot at scheduled step {snapshot_step} (current={current_step})")
                    
            except Exception as e:
                self.logger.warning(f"Failed to save policy snapshot at step {self._next_snapshot_step}: {e}")
                
            self._next_snapshot_step += self.policy_snapshot_every_steps
        
        # Flush diagnostics logs after all snapshots are saved
        if hasattr(self.agent, 'flush_diagnostics'):
            try:
                self.agent.flush_diagnostics()
            except Exception as e:
                self.logger.warning(f"Failed to flush diagnostics: {e}")

    def _log_episode_summaries(self, episodes: List):
        """
        Log episode summaries to episode_summary.jsonl with explore_step and update_step.
        
        This directly solves the timestamp correlation problem by providing exact
        step counters at episode completion time.
        """
        if self._episode_summary_file is None:
            return
            
        try:
            # Fix #4: Use per-episode step counters stamped by the worker at
            # completion time. Fall back to current shared counter for backward
            # compatibility with Episode objects that don't carry step snapshots.
            fallback_explore = self.step_counter.exploration
            fallback_update = self.step_counter.update

            with open(self._episode_summary_file, "a", encoding="utf-8") as f:
                for i, episode in enumerate(episodes):
                    try:
                        wall_time = time.time()  # Per-episode timestamp
                        # Use per-episode step snapshot if available, else fallback
                        explore_step = getattr(episode, 'explore_step_at_completion', None) or fallback_explore
                        update_step = getattr(episode, 'update_step_at_completion', None) or fallback_update
                        # Extract episode data
                        total_reward = getattr(episode, 'episode_reward', 0.0)
                        steps = len(getattr(episode, 'rewards', []))
                        terminated = getattr(episode, 'terminals', [False])[-1] if getattr(episode, 'terminals', []) else False
                        truncated = getattr(episode, 'truncations', [False])[-1] if getattr(episode, 'truncations', []) else False

                        # Try to get max_insertion from episode info if available
                        max_insertion = None
                        infos = getattr(episode, 'infos', [])
                        if infos:
                            last_info = infos[-1] if infos else {}
                            # Look for insertion depth in common info keys
                            for key in ['max_insertion', 'inserted_length', 'insertion_depth']:
                                if key in last_info:
                                    val = last_info[key]
                                    if hasattr(val, '__iter__') and not isinstance(val, str):
                                        max_insertion = float(max(val))
                                    else:
                                        max_insertion = float(val)
                                    break

                        # Fix #5: Monotonically increasing counter for episode_id.
                        # Replaces retroactive calculation that assumed sequential,
                        # non-interleaved episodes.
                        episode_id = self._episode_summary_counter
                        self._episode_summary_counter += 1

                        summary = {
                            "wall_time": wall_time,
                            "explore_step": explore_step,
                            "update_step": update_step,
                            "episode_id": episode_id,
                            "total_reward": float(total_reward),
                            "steps": steps,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                        if max_insertion is not None:
                            summary["max_insertion"] = max_insertion
                        
                        f.write(json.dumps(summary) + "\n")
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to log episode {i} summary: {e}")
                        
                f.flush()
                
        except Exception as e:
            self.logger.warning(f"Failed to log episode summaries: {e}")

```

Now let me completely update `DualDeviceNav_resume.py` with all the fixes:

```python
"""
DualDeviceNav training script with resume capability.

Usage:
  # Fresh training:
  python DualDeviceNav_resume.py -n my_experiment -d cuda:0 -nw 16

  # Resume from checkpoint (continues in the SAME experiment folder):
  python DualDeviceNav_resume.py -d cuda:0 -nw 16 \
    --resume /path/to/experiment_folder/checkpoints/checkpoint250000.everl
"""
import os
import logging
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch
from util.util import get_result_checkpoint_config_and_log_path
from util.env import BenchEnv
from util.env2 import BenchEnv2
from util.env3 import BenchEnv3
from util.env4 import BenchEnv4
from util.env5 import BenchEnv5
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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="DualDeviceNav training with resume capability")
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=4, help="Number of workers"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device of trainer, where the NN update is performed.",
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
        help="Environment version: 1=original (PathDelta), 2=waypoint+centerlines, 3=tuned waypoint, 4=arclength progress+guidance, 5=optimized env4",
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
        help="Number of heuristic episodes to seed replay buffer (0=disabled, recommended: 500). Only for env4/5.",
    )
    # Resume from checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file (.everl) to resume training from. "
             "Continues in the same experiment folder. Skips heatup when provided."
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
    resume_checkpoint = args.resume
    worker_device = torch.device("cpu")
    
    # Select environment class based on version
    if env_version == 5:
        EnvClass = BenchEnv5
        print("Using BenchEnv5 (optimized env4 with shared projection cache)")
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
        "resumed_from": resume_checkpoint if resume_checkpoint else "fresh",
    }

    if resume_checkpoint:
        # Derive all paths from the checkpoint location to resume in the SAME folder.
        # Expected layout: .../experiment_folder/checkpoints/checkpoint*.everl
        checkpoint_folder = os.path.dirname(os.path.abspath(resume_checkpoint))
        results_folder = os.path.dirname(checkpoint_folder)
        results_file = results_folder + ".csv"
        log_file = os.path.join(results_folder, "main.log")
        config_folder = os.path.join(results_folder, "config")
        diagnostics_folder = os.path.join(results_folder, "diagnostics")
        os.makedirs(config_folder, exist_ok=True)
        os.makedirs(diagnostics_folder, exist_ok=True)
        print(f"Resuming in existing experiment folder: {results_folder}")
    else:
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

    # Load checkpoint if resuming
    if resume_checkpoint:
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT: {resume_checkpoint}")
        print(f"{'='*60}")
        agent.load_checkpoint(resume_checkpoint)
        print(f"  Loaded network weights, optimizer states, and schedulers")
        print(f"  Heatup steps:      {agent.step_counter.heatup:,}")
        print(f"  Exploration steps: {agent.step_counter.exploration:,}")
        print(f"  Update steps:      {agent.step_counter.update:,}")
        print(f"  Evaluation steps:  {agent.step_counter.evaluation:,}")
        print(f"  Heatup episodes:   {agent.episode_counter.heatup:,}")
        print(f"  Explore episodes:  {agent.episode_counter.exploration:,}")
        print(f"{'='*60}")

        # CRITICAL: Restore curriculum step count to avoid Stage regression
        if args.curriculum:
            restored_steps = agent.step_counter.exploration
            env_train._total_steps = restored_steps
            print(f"  Curriculum restored: _total_steps={restored_steps}, Stage={env_train.current_stage}")

        print()

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

    # When resuming, restore Runner-level state (best_eval, snapshot step, episode counter, probes)
    if resume_checkpoint:
        runner.restore_runner_state(resume_checkpoint, probe_states_dir=diagnostics_folder)

        # NOTE: Replay buffer is NOT saved in checkpoints (too large).
        # The agent resumes with an empty buffer and refills via exploration.
        # Gradient updates will be paused until enough new experience accumulates
        # to satisfy: (heatup + explore) * update_ratio > update_steps.
        # This means a "dead zone" of ~(prior_total_steps * update_ratio) updates
        # where exploration runs but no learning happens.

    # Optional: seed replay buffer with heuristic episodes (fresh training or resume)
    if args.heuristic_seeding > 0 and env_version in (4, 5):
        from util.heuristic_controller import CenterlineFollowerHeuristic
        from eve_rl.replaybuffer import Episode
        from eve_rl.util import flatten_obs

        print(f"Seeding replay buffer with {args.heuristic_seeding} heuristic episodes...")
        seed_intervention = DualDeviceNav()
        # Use the appropriate env class for seeding
        SeedEnvClass = BenchEnv5 if env_version == 5 else BenchEnv4
        seed_env = SeedEnvClass(intervention=seed_intervention, mode="train", visualisation=False)
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

    # When resuming, use heatup_steps=0 to skip heatup phase
    # The checkpoint already contains learned weights and the step counters
    # will pick up from where training left off
    effective_heatup = 0 if resume_checkpoint else HEATUP_STEPS

    if resume_checkpoint:
        print(f"Skipping heatup (checkpoint already has {agent.step_counter.heatup:,} heatup steps)")
        print(f"Continuing training from exploration step {agent.step_counter.exploration:,}")
    
    reward, success = runner.training_run(
        effective_heatup,
        TRAINING_STEPS,
        EXPLORE_STEPS_BTW_EVAL,
        CONSECUTIVE_EXPLORE_EPISODES,
        UPDATE_PER_EXPLORE_STEP,
        eval_seeds=EVAL_SEEDS,
    )
    agent.close()

```

Both files are updated with no linting errors. Here's a summary of the changes made:

## Changes Applied to `stEVE_training`

### 1. `eve_rl/eve_rl/runner/runner.py`

**Added `restore_runner_state()` method** (lines 101-143):
- Restores `best_eval`, `_episode_summary_counter`, `_next_snapshot_step` from checkpoint
- Backward-compatible fallback for old checkpoints without `runner_state`
- Restores probe states from saved `.npz` file

**Added `runner_state` saving in `eval()` method** (lines 169-173):
```python
eval_results["runner_state"] = {
    "best_eval": self.best_eval.copy(),
    "episode_summary_counter": self._episode_summary_counter,
    "next_snapshot_step": self._next_snapshot_step,
}
```

### 2. `training_scripts/DualDeviceNav_resume.py`

**Complete rewrite with all fixes:**

| Feature | Status |
|---------|--------|
| env4 support | ✅ Added |
| env5 support | ✅ Added |
| `--curriculum` flag | ✅ Added |
| `--curriculum_stage1/2` | ✅ Added |
| `--heuristic_seeding` | ✅ Added with `.flatten()` fix |
| Same-folder resume | ✅ Path derivation from checkpoint |
| Curriculum step restore | ✅ `env_train._total_steps = exploration` |
| Runner state restore | ✅ Calls `runner.restore_runner_state()` |

**Bug fix applied**: Action normalization now correctly uses `.flatten()`:
```python
act_low = seed_env.action_space.low.astype(np.float64).flatten()
act_high = seed_env.action_space.high.astype(np.float64).flatten()
```

```python
eval_results["runner_state"] = {
    "best_eval": self.best_eval.copy(),
    "episode_summary_counter": self._episode_summary_counter,
    "next_snapshot_step": self._next_snapshot_step,
}
```

```python
act_low = seed_env.action_space.low.astype(np.float64).flatten()
act_high = seed_env.action_space.high.astype(np.float64).flatten()
```

---

**User**

are you sure the chnages in runner havn't inroduced any new bugs in logging framework with step_counter/episode counters etc.

---

**Cursor**

Good question. Let me re-read the updated `runner.py` to verify the changes don't introduce any bugs:

Good catch to double-check! Let me analyze the changes carefully:

## Analysis

### Correct Separation of Counters

The code correctly distinguishes between:

1. **`self.step_counter`** (property, line 95) → `agent.step_counter` (shared, saved in agent checkpoint)
   - Used for: `exploration`, `heatup`, `update`, `evaluation` counts

2. **`self.episode_counter`** (property, line 99) → `agent.episode_counter` (shared, saved in agent checkpoint)
   - Used for: `results.csv` (line 203)

3. **`self._episode_summary_counter`** (Runner-local, line 58)
   - Used for: monotonic `episode_id` in `episode_summary.jsonl` (line 443)
   - This is what's saved/restored in `runner_state`

### Issue Found: Confusing Log Message

Line 117-118 has a misleading label:

```python
self.logger.info(
    f"Restored runner state: best_eval={self.best_eval}, "
    f"episode_counter={self._episode_summary_counter}, "  # <-- Should say "episode_summary_counter"
```

This could cause confusion during debugging. Let me fix it:

```python
self.logger.info(
    f"Restored runner state: best_eval={self.best_eval}, "
    f"episode_counter={self._episode_summary_counter}, "  # <-- Should say "episode_summary_counter"
```

```python
from typing import List, Optional
from math import inf
import csv
import json
import logging
import os
import time
import numpy as np
from ..util import EveRLObject
from ..agent.agent import Agent, StepCounter, EpisodeCounter


class Runner(EveRLObject):
    def __init__(
        self,
        agent: Agent,
        heatup_action_low: List[float],
        heatup_action_high: List[float],
        agent_parameter_for_result_file: dict,
        checkpoint_folder: str,
        results_file: str,
        quality_info: Optional[str] = None,
        info_results: Optional[List[str]] = None,
        diagnostics_folder: Optional[str] = None,
        policy_snapshot_every_steps: int = 10000,
        n_probe_episodes: int = 3,
        n_probe_near_start_steps: int = 10,
    ) -> None:
        self.agent = agent
        self.heatup_action_low = heatup_action_low
        self.heatup_action_high = heatup_action_high
        self.agent_parameter_for_result_file = agent_parameter_for_result_file
        self.checkpoint_folder = checkpoint_folder
        self.results_file = results_file
        self.quality_info = quality_info
        self.info_results = info_results or []
        self.logger = logging.getLogger(self.__module__)
        
        # Diagnostics settings
        self.diagnostics_folder = diagnostics_folder
        # Ensure worker step logs directory exists.
        # NOTE: STEP_LOG_DIR must be set BEFORE agent creation (which spawns workers).
        # The training script should set os.environ["STEP_LOG_DIR"] before creating the agent.
        # This setdefault is a fallback but will NOT help if workers are already spawned.
        if self.diagnostics_folder is not None:
            logs_subprocesses = os.path.join(self.diagnostics_folder, "logs_subprocesses")
            os.makedirs(logs_subprocesses, exist_ok=True)
            os.environ.setdefault("STEP_LOG_DIR", logs_subprocesses)
        self.policy_snapshot_every_steps = policy_snapshot_every_steps
        self.n_probe_episodes = n_probe_episodes
        self.n_probe_near_start_steps = n_probe_near_start_steps
        self._next_snapshot_step = policy_snapshot_every_steps
        self._probe_states_set = False
        
        # Episode summary logger
        self._episode_summary_file = None
        self._episode_summary_logger = None
        self._episode_summary_counter = 0  # Fix #5: monotonic episode_id
        if self.diagnostics_folder is not None:
            csv_folder = os.path.join(self.diagnostics_folder, "csv")
            os.makedirs(csv_folder, exist_ok=True)
            self._episode_summary_file = os.path.join(csv_folder, "episode_summary.jsonl")
            self.logger.info(f"Episode summary will be logged to {self._episode_summary_file}")

        self._results = {
            "episodes explore": 0,
            "steps explore": 0,
        }
        self._results["quality"] = 0.0
        for info_result in self.info_results:
            self._results[info_result] = 0.0
        self._results["reward"] = 0.0
        self._results["best quality"] = 0.0
        self._results["best explore steps"] = 0.0

        file_existed = os.path.isfile(results_file)

        with open(results_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_existed:
                writer.writerow(
                    list(self._results.keys())
                    + [" "]
                    + list(agent_parameter_for_result_file.keys())
                )
                writer.writerow(
                    [" "] * (len(self._results.values()) + 1)
                    + list(agent_parameter_for_result_file.values())
                )

        self.best_eval = {"steps": 0, "quality": -inf}

    @property
    def step_counter(self) -> StepCounter:
        return self.agent.step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self.agent.episode_counter

    def restore_runner_state(self, checkpoint_path: str, probe_states_dir: Optional[str] = None):
        """Restore Runner-level state from a checkpoint for resume support.

        Restores: best_eval, _episode_summary_counter, _next_snapshot_step,
        and optionally probe states from a saved .npz file.
        """
        import torch
        checkpoint = torch.load(checkpoint_path)
        additional_info = checkpoint.get("additional_info", {})
        runner_state = additional_info.get("runner_state") if additional_info else None

        if runner_state:
            self.best_eval = runner_state["best_eval"]
            self._episode_summary_counter = runner_state["episode_summary_counter"]
            self._next_snapshot_step = runner_state["next_snapshot_step"]
            self.logger.info(
                f"Restored runner state: best_eval={self.best_eval}, "
                f"_episode_summary_counter={self._episode_summary_counter}, "
                f"_next_snapshot_step={self._next_snapshot_step}"
            )
        else:
            # Checkpoint predates runner_state support — compute safe defaults
            explore_steps = self.step_counter.exploration
            self._next_snapshot_step = (
                (explore_steps // self.policy_snapshot_every_steps) + 1
            ) * self.policy_snapshot_every_steps
            self.logger.warning(
                f"No runner_state in checkpoint. Setting _next_snapshot_step={self._next_snapshot_step}"
            )

        # Restore probe states from saved .npz if available
        if probe_states_dir is not None:
            probe_path = os.path.join(probe_states_dir, "probes", "probe_states.npz")
            if os.path.isfile(probe_path):
                try:
                    data = np.load(probe_path)
                    probe_states = data["probe_states"]
                    if hasattr(self.agent, 'set_probe_states'):
                        self.agent.set_probe_states(probe_states.tolist())
                        self._probe_states_set = True
                        self.logger.info(f"Restored {len(probe_states)} probe states from {probe_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to restore probe states: {e}")

    def heatup(self, steps: int):
        episodes = self.agent.heatup(
            steps=steps,
            custom_action_low=self.heatup_action_low,
            custom_action_high=self.heatup_action_high,
        )
        return episodes

    def explore(self, n_episodes: int):
        self.agent.explore(episodes=n_episodes)

    def update(self, n_steps: int):
        self.agent.update(n_steps)

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[List[int]] = None
    ):
        explore_steps = self.step_counter.exploration
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{explore_steps}.everl"
        )
        result_episodes = self.agent.evaluate(episodes=episodes, seeds=seeds)
        qualities, rewards = [], []
        results_for_info = {
            info_result_name: [] for info_result_name in self.info_results
        }
        eval_results = {"episodes": []}
        for episode in result_episodes:
            reward = episode.episode_reward
            quality = (
                episode.infos[-1][self.quality_info]
                if self.quality_info is not None
                else reward
            )
            qualities.append(quality)
            rewards.append(reward)
            for info_result_name in self.info_results:
                info = episode.infos[-1][info_result_name]
                results_for_info[info_result_name].append(info)
            eval_results["episodes"].append(
                {
                    "seed": episode.seed,
                    "options": episode.options,
                    "quality": episode.infos[-1][self.quality_info],
                }
            )

        reward = sum(rewards) / len(rewards)
        quality = sum(qualities) / len(qualities)
        for info_result_name, results in results_for_info.items():
            result = sum(results) / len(results)
            self._results[info_result_name] = round(result, 3)
        save_best = False
        if quality > self.best_eval["quality"]:
            save_best = True
            self.best_eval["quality"] = quality
            self.best_eval["steps"] = explore_steps

        self._results["episodes explore"] = self.episode_counter.exploration
        self._results["steps explore"] = explore_steps
        self._results["reward"] = round(reward, 3)
        self._results["quality"] = round(quality, 3)
        self._results["best quality"] = self.best_eval["quality"]
        self._results["best explore steps"] = self.best_eval["steps"]

        eval_results.update(self._results)
        eval_results.pop("best quality")
        eval_results.pop("best explore steps")

        # Include runner state for resume support
        eval_results["runner_state"] = {
            "best_eval": self.best_eval.copy(),
            "episode_summary_counter": self._episode_summary_counter,
            "next_snapshot_step": self._next_snapshot_step,
        }

        self.agent.save_checkpoint(checkpoint_file, eval_results)
        if save_best:
            checkpoint_file = os.path.join(
                self.checkpoint_folder, "best_checkpoint.everl"
            )
            self.agent.save_checkpoint(checkpoint_file, eval_results)

        log_info = (
            f"Quality: {quality}, Reward: {reward}, Exploration steps: {explore_steps}"
        )
        self.logger.info(log_info)
        with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(self._results.values())

        return quality, reward

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: int = None,
        explore_steps_limit: int = None,
    ):
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        while self.step_counter.exploration < explore_steps_limit:
            # Total experience = heatup + exploration (both contribute to replay buffer)
            total_experience_steps = (
                self.step_counter.heatup + self.step_counter.exploration
            )
            update_steps = (
                total_experience_steps * update_steps_per_explore_step
                - self.step_counter.update
            )
            result = self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )
            
            # Log episode summaries if we got results
            if result is not None:
                episodes, _ = result if isinstance(result, tuple) else (result, None)
                if episodes:
                    self._log_episode_summaries(episodes)
            
            # Check for policy snapshots after each exploration cycle
            # This ensures snapshots are saved at the actual milestone with correct weights
            self._maybe_save_policy_snapshot()

    def training_run(
        self,
        heatup_steps: int,
        training_steps: int,
        explore_steps_between_eval: int,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
    ):
        # TODO: Log Training Run Infos
        heatup_episodes = self.heatup(heatup_steps)
        
        # Capture and set probe states after heatup
        self._capture_and_set_probe_states(heatup_episodes)
        
        next_eval_step_limt = (
            self.agent.step_counter.exploration + explore_steps_between_eval
        )
        while self.agent.step_counter.exploration < training_steps:
            self.explore_and_update(
                explore_episodes_between_updates,
                update_steps_per_explore_step,
                explore_steps_limit=next_eval_step_limt,
            )
            
            # Note: policy snapshots are now saved inside explore_and_update()
            # at the actual milestone steps with correct weights
            
            quality, reward = self.eval(episodes=eval_episodes, seeds=eval_seeds)
            next_eval_step_limt += explore_steps_between_eval

        return quality, reward

    def _capture_and_set_probe_states(self, episodes: List):
        """Capture probe states from heatup episodes and set them in trainer."""
        if not episodes or self._probe_states_set:
            return
            
        try:
            probe_states = []
            n_episodes = min(len(episodes), self.n_probe_episodes)
            
            for ep_idx in range(n_episodes):
                episode = episodes[ep_idx]
                if hasattr(episode, 'flat_obs') and len(episode.flat_obs) > 0:
                    # Collect start state
                    probe_states.append(episode.flat_obs[0])
                    
                    # Collect near-start states
                    n_states = min(self.n_probe_near_start_steps, len(episode.flat_obs) - 1)
                    for step_idx in range(1, n_states + 1):
                        probe_states.append(episode.flat_obs[step_idx])
                        
            if probe_states:
                probe_states_array = np.array(probe_states)
                
                # Set probe states in agent (if it supports it)
                if hasattr(self.agent, 'set_probe_states'):
                    self.agent.set_probe_states(probe_states_array.tolist())
                    self.logger.info(f"Set {len(probe_states)} probe states for diagnostics")
                    
                # Save probe states to disk
                if self.diagnostics_folder is not None:
                    probes_dir = os.path.join(self.diagnostics_folder, "probes")
                    os.makedirs(probes_dir, exist_ok=True)
                    probe_path = os.path.join(probes_dir, "probe_states.npz")
                    np.savez(probe_path, probe_states=probe_states_array)
                    self.logger.info(f"Saved probe states to {probe_path}")
                    
                self._probe_states_set = True
                
        except Exception as e:
            self.logger.warning(f"Failed to capture probe states: {e}")

    def _maybe_save_policy_snapshot(self):
        """
        Save policy snapshot if we've reached the next snapshot step.
        
        FIXED: Uses a while loop to catch up on all missed snapshots when the
        eval cycle spans multiple snapshot intervals.
        """
        if self.diagnostics_folder is None:
            return
            
        current_step = self.step_counter.exploration
        
        # Save all missed snapshots (in case eval cycle spans multiple snapshot intervals)
        while current_step >= self._next_snapshot_step:
            try:
                snapshots_dir = os.path.join(self.diagnostics_folder, "policy_snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)
                
                # Use the scheduled snapshot step, not current step, for consistent naming
                snapshot_step = self._next_snapshot_step
                snapshot_path = os.path.join(snapshots_dir, f"policy_{snapshot_step}.pt")
                
                if hasattr(self.agent, 'save_policy_snapshot'):
                    self.agent.save_policy_snapshot(snapshot_path)
                    self.logger.info(f"Saved policy snapshot at scheduled step {snapshot_step} (current={current_step})")
                    
            except Exception as e:
                self.logger.warning(f"Failed to save policy snapshot at step {self._next_snapshot_step}: {e}")
                
            self._next_snapshot_step += self.policy_snapshot_every_steps
        
        # Flush diagnostics logs after all snapshots are saved
        if hasattr(self.agent, 'flush_diagnostics'):
            try:
                self.agent.flush_diagnostics()
            except Exception as e:
                self.logger.warning(f"Failed to flush diagnostics: {e}")

    def _log_episode_summaries(self, episodes: List):
        """
        Log episode summaries to episode_summary.jsonl with explore_step and update_step.
        
        This directly solves the timestamp correlation problem by providing exact
        step counters at episode completion time.
        """
        if self._episode_summary_file is None:
            return
            
        try:
            # Fix #4: Use per-episode step counters stamped by the worker at
            # completion time. Fall back to current shared counter for backward
            # compatibility with Episode objects that don't carry step snapshots.
            fallback_explore = self.step_counter.exploration
            fallback_update = self.step_counter.update

            with open(self._episode_summary_file, "a", encoding="utf-8") as f:
                for i, episode in enumerate(episodes):
                    try:
                        wall_time = time.time()  # Per-episode timestamp
                        # Use per-episode step snapshot if available, else fallback
                        explore_step = getattr(episode, 'explore_step_at_completion', None) or fallback_explore
                        update_step = getattr(episode, 'update_step_at_completion', None) or fallback_update
                        # Extract episode data
                        total_reward = getattr(episode, 'episode_reward', 0.0)
                        steps = len(getattr(episode, 'rewards', []))
                        terminated = getattr(episode, 'terminals', [False])[-1] if getattr(episode, 'terminals', []) else False
                        truncated = getattr(episode, 'truncations', [False])[-1] if getattr(episode, 'truncations', []) else False

                        # Try to get max_insertion from episode info if available
                        max_insertion = None
                        infos = getattr(episode, 'infos', [])
                        if infos:
                            last_info = infos[-1] if infos else {}
                            # Look for insertion depth in common info keys
                            for key in ['max_insertion', 'inserted_length', 'insertion_depth']:
                                if key in last_info:
                                    val = last_info[key]
                                    if hasattr(val, '__iter__') and not isinstance(val, str):
                                        max_insertion = float(max(val))
                                    else:
                                        max_insertion = float(val)
                                    break

                        # Fix #5: Monotonically increasing counter for episode_id.
                        # Replaces retroactive calculation that assumed sequential,
                        # non-interleaved episodes.
                        episode_id = self._episode_summary_counter
                        self._episode_summary_counter += 1

                        summary = {
                            "wall_time": wall_time,
                            "explore_step": explore_step,
                            "update_step": update_step,
                            "episode_id": episode_id,
                            "total_reward": float(total_reward),
                            "steps": steps,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                        if max_insertion is not None:
                            summary["max_insertion"] = max_insertion
                        
                        f.write(json.dumps(summary) + "\n")
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to log episode {i} summary: {e}")
                        
                f.flush()
                
        except Exception as e:
            self.logger.warning(f"Failed to log episode summaries: {e}")

```

Let me also verify the agent's `save_checkpoint` correctly stores `additional_info`:

The save/load chain is correct:

**Save path** (line 356 in `agent.py`):
```python
checkpoint_dict = {
    ...
    "additional_info": additional_info,  # This receives eval_results containing runner_state
}
```

**Load path** (line 109-110 in `runner.py`):
```python
additional_info = checkpoint.get("additional_info", {})
runner_state = additional_info.get("runner_state") if additional_info else None
```

## Verification Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| `step_counter` vs `_episode_summary_counter` | ✅ Correct | Different purposes, correctly used |
| `episode_counter` vs `_episode_summary_counter` | ✅ Correct | Different purposes, correctly used |
| `.copy()` on `best_eval` | ✅ Correct | Prevents reference aliasing |
| `additional_info` save/load chain | ✅ Correct | Properly passed through agent checkpoint |
| Fallback for old checkpoints | ✅ Correct | Computes safe defaults |
| Log message naming | ✅ Fixed | Changed to `_episode_summary_counter` |

The changes are **safe** and don't introduce any bugs with the logging framework or step/episode counters. The key counters (`step_counter`, `episode_counter`) come from the agent and are already properly saved/restored by `agent.load_checkpoint()`. The Runner-local counters (`_episode_summary_counter`, `_next_snapshot_step`, `best_eval`) are now correctly saved and restored separately via `runner_state`.

```python
checkpoint_dict = {
    ...
    "additional_info": additional_info,  # This receives eval_results containing runner_state
}
```

```python
additional_info = checkpoint.get("additional_info", {})
runner_state = additional_info.get("runner_state") if additional_info else None
```

---

**User**

Can you verify wr.t Diagnostics_collapse4/5/6 

---

**Cursor**

Let me verify the complete compatibility between runner output and diagnose tools:

## Field Compatibility Check

| Runner Output | Type | Diagnose Expected | Parser | Status |
|---------------|------|-------------------|--------|--------|
| `wall_time` | `time.time()` (float) | `datetime` | `_parse_wall_time` (float→datetime) | ✅ |
| `explore_step` | int | int | `_to_int` | ✅ |
| `update_step` | int | int | `_to_int` | ✅ |
| `episode_id` | int | int | `_to_int` | ✅ |
| `total_reward` | float | float | `_to_float` | ✅ |
| `steps` | int | int | `_to_int` | ✅ |
| `terminated` | bool | bool | `bool()` | ✅ |
| `truncated` | bool | bool | `bool()` | ✅ |
| `max_insertion` | Optional[float] | Optional[float] | `_to_float` | ✅ |

Now let me verify the `_log_episode_summaries` method wasn't modified by my changes:

## Verification Complete ✅

The runner changes are **fully compatible** with `diagnose_collapse4/5/6`:

### 1. JSONL Format Unchanged
The `_log_episode_summaries()` method was **NOT modified** by my changes. Output format remains:
```json
{"wall_time": 1711234567.89, "explore_step": 50000, "update_step": 2500, "episode_id": 123, ...}
```

### 2. Field Parsing Compatibility
| Field | Runner Type | Diagnose Parser | Status |
|-------|-------------|-----------------|--------|
| `wall_time` | `time.time()` float | `_parse_wall_time()` → datetime | ✅ |
| `explore_step` | int | `_to_int()` | ✅ |
| `update_step` | int | `_to_int()` | ✅ |
| `episode_id` | int | `_to_int()` | ✅ |

### 3. Sorting Strategy Compatible
Diagnose tools sort by:
- `(explore_step, episode_id)` - line 1358
- `(end_timestamp, episode, worker)` - line 1080

The tools use `episode_id` only as a **tiebreaker**, not as a primary key. They don't assume sequential IDs.

### 4. Resume Fix Actually IMPROVES Compatibility
**Before fix**: `episode_id` restarted at 0 on resume → duplicate IDs in JSONL
**After fix**: `_episode_summary_counter` restored → monotonic, unique IDs

This prevents potential issues with:
- Duplicate episode IDs confusing sorting
- Wall-time matching becoming ambiguous

### 5. No Changes to Critical Methods
| Method | Modified? |
|--------|-----------|
| `_log_episode_summaries()` | ❌ No |
| `_capture_and_set_probe_states()` | ❌ No |
| `_maybe_save_policy_snapshot()` | ❌ No |
| `explore_and_update()` | ❌ No |

**Conclusion**: The diagnose tools will work correctly with both pre-resume and post-resume training data.

```json
{"wall_time": 1711234567.89, "explore_step": 50000, "update_step": 2500, "episode_id": 123, ...}
```

---

**User**

Training runs take hours/days. When interrupted, the replay buffer (10k episodes, ~750MB) is lost — the agent resumes with an empty buffer, stalling gradient updates. Additionally, heuristic seeding (100+ episodes via CenterlineFollowerHeuristic) and heatup (10k random steps) are regenerated for every new experiment, wasting time.

Goals:

Save replay buffer periodically (every 4 evals / ~1M steps) as a single overwritten file
Auto-load replay buffer on resume from the same experiment folder
Save heuristic/heatup episodes as standalone reusable files shareable across experiments  ## CLI Reference

### Fresh Training

```bash
# Generate and save caches for reuse
python DualDeviceNav_train.py -n exp1 --env_version 4 -d cuda:0 -nw 16 \
    --heuristic_seeding 100 \
    --save_heuristic_cache shared_experience/heuristic_env4_100ep.npz \
    --save_heatup_cache shared_experience/heatup_env4_10k.npz

# Load existing caches (skips generation, saves 10-30 min)
python DualDeviceNav_train.py -n exp2 --env_version 4 -d cuda:0 -nw 16 \
    --heuristic_seeding 100 \
    --heuristic_cache_file shared_experience/heuristic_env4_100ep.npz \
    --heatup_cache_file shared_experience/heatup_env4_10k.npz
```

### Resume

```bash
# Replay buffer auto-loaded from experiment folder, no extra args needed
python DualDeviceNav_resume.py -d cuda:0 -nw 16 --env_version 4 \
    --resume results/.../checkpoints/checkpoint1000000.everl

# With optional heuristic seeding if replay buffer checkpoint doesn't exist
python DualDeviceNav_resume.py -d cuda:0 -nw 16 --env_version 4 \
    --heuristic_seeding 100 \
    --heuristic_cache_file shared_experience/heuristic_env4_100ep.npz \
    --resume results/.../checkpoints/checkpoint1000000.everl
```   In that codebase (rl_resume) following files have been chnaged replaybuffer.py — EpisodeReplay.to_replay() for push compatibility
vanillaepisode.py — export_all() / import_all() on the internal ring buffer
vanillashared.py — save_buffer / load_buffer IPC tasks (subprocess does file I/O directly) + save_buffer_to_file() / load_buffer_from_file() convenience methods
init.py — Exported EpisodeReplay
runner.py — Periodic replay save every 4 evals, load_replay_buffer(), _eval_count in runner_state, heatup_cache_save_path param in training_run()
DualDeviceNav_train.py (vs for this codebase) — --heuristic_cache_file, --save_heuristic_cache, --heatup_cache_file, --save_heatup_cache
DualDeviceNav_resume.py — Auto-loads replay_buffer.npz from checkpoint folder on resume;;;;;;;;;;;;;;;; SEE IF ALL THE CHANGES are done correctly and for intended purposes in a most efficient manner without introducing any extra bugs . Check thoroughly.

---

**Cursor**

I'll thoroughly verify the replay buffer checkpoint implementation in the `rl_resume` codebase. Let me examine each modified file:

## Comprehensive Verification Report

I've thoroughly analyzed all the modified files. Here's my assessment:

### ✅ Correctly Implemented

| File | Component | Status |
|------|-----------|--------|
| `replaybuffer.py` | `EpisodeReplay.to_replay()` | ✅ Correct - enables push compatibility |
| `vanillaepisode.py` | `export_all()` / `import_all()` | ✅ Correct - proper ring buffer state |
| `vanillashared.py` | IPC tasks `save_buffer`/`load_buffer` | ✅ Correct - thread-safe with lock |
| `experience_cache.py` | NPZ serialization | ✅ Excellent - efficient concatenated format |
| `runner.py` | `_eval_count` + periodic save | ✅ Correct - saves every 4 evals |
| `runner.py` | `load_replay_buffer()` | ✅ Correct - auto-load from checkpoint folder |
| `__init__.py` | `EpisodeReplay` export | ✅ Correct |

### 🔴 Bugs Found

#### Bug 1: Missing `.flatten()` in `DualDeviceNav_train.py` (lines 324-326)

```python
# CURRENT (BUGGY):
act_low = seed_env.action_space.low.astype(np.float64)
act_high = seed_env.action_space.high.astype(np.float64)

# SHOULD BE:
act_low = seed_env.action_space.low.astype(np.float64).flatten()
act_high = seed_env.action_space.high.astype(np.float64).flatten()
```

#### Bug 2: Same missing `.flatten()` in `DualDeviceNav_resume.py` (lines 352-354)

Same fix needed.

### ⚠️ Missing Features

#### 1. env5 support missing in `DualDeviceNav_resume.py`
- No `from util.env5 import BenchEnv5`
- No `choices=[1, 2, 3, 4, 5]`
- No env5 selection logic

#### 2. `--heuristic_cache_file` arg missing in resume script
Per CLI reference, resume should support loading cached heuristic:
```bash
python DualDeviceNav_resume.py --heuristic_cache_file shared_experience/heuristic_env4_100ep.npz ...
```

Currently, resume only generates heuristic episodes inline, doesn't load from cache.

### 🔍 Architecture Review

#### Data Flow (Correct):
```
Episode → to_replay() → EpisodeReplay → push_queue → subprocess → VanillaEpisode.push()
                                                              ↓
                     save_buffer_to_file() ← task_queue ← export_all()
                             ↓
                     experience_cache.save_episodes_npz()
                             ↓
                     .npz file (concatenated arrays + lengths index)
```

#### NPZ Format (Efficient):
```
n_episodes: int
lengths: (N,) int array
flat_obs: (sum(lengths+1), obs_dim) concatenated
actions: (sum(lengths), action_dim) concatenated
rewards: (sum(lengths),) concatenated
terminals: (sum(lengths),) concatenated
position: int (ring buffer state)
```

This avoids object arrays which break `np.savez_compressed` - **excellent design**.

### 📊 Thread Safety (Verified)

```python
def save_buffer_to_file(self, path):
    with self._request_lock:  # Mutex ensures single task
        self._task_queue.put(["save_buffer", path])
        return self._result_queue.get()  # Blocking wait inside lock
```

No race conditions - lock held during entire request/response cycle.

---

## Summary

| Category | Count |
|----------|-------|
| ✅ Correct implementations | 7 |
| 🔴 Bugs to fix | 2 (same `.flatten()` bug in 2 files) |
| ⚠️ Missing features | 2 (env5, heuristic cache in resume) |

**Overall**: The architecture is well-designed and efficient. Two bugs need fixing (both are the same `.flatten()` issue), and two features need to be added to achieve full CLI parity.

```python
# CURRENT (BUGGY):
act_low = seed_env.action_space.low.astype(np.float64)
act_high = seed_env.action_space.high.astype(np.float64)

# SHOULD BE:
act_low = seed_env.action_space.low.astype(np.float64).flatten()
act_high = seed_env.action_space.high.astype(np.float64).flatten()
```

```shellscript
python DualDeviceNav_resume.py --heuristic_cache_file shared_experience/heuristic_env4_100ep.npz ...
```

```plaintext
Episode → to_replay() → EpisodeReplay → push_queue → subprocess → VanillaEpisode.push()
                                                              ↓
                     save_buffer_to_file() ← task_queue ← export_all()
                             ↓
                     experience_cache.save_episodes_npz()
                             ↓
                     .npz file (concatenated arrays + lengths index)
```

```plaintext
n_episodes: int
lengths: (N,) int array
flat_obs: (sum(lengths+1), obs_dim) concatenated
actions: (sum(lengths), action_dim) concatenated
rewards: (sum(lengths),) concatenated
terminals: (sum(lengths),) concatenated
position: int (ring buffer state)
```

```python
def save_buffer_to_file(self, path):
    with self._request_lock:  # Mutex ensures single task
        self._task_queue.put(["save_buffer", path])
        return self._result_queue.get()  # Blocking wait inside lock
```

---

**User**

what about this" There's also verbose logging in the subprocess that could impact performance, though it's not technically broken."

---

**Cursor**

Let me look at the subprocess logging in `vanillashared.py` to examine this potential performance concern:

I see the issue now. Let me compare with the original to see what was added:

Now I see the full picture. The verbose logging exists in **both** codebases and is a **pre-existing concern**, not introduced by the resume fixes. Here's the performance impact:

## Verbose Logging Analysis

### Hot Path Issue: `push()` method (lines 55-68)

```python
def push(self, episode: Episode):
    logger = logging.getLogger("ReplayBufferSubprocess")  # Logger lookup every call
    ...
    logger.info(f"PUSH CALLED: episode length={len(episode)}")  # String formatting
    ...
    logger.info(f"PUSH COMPLETE: episode put in queue successfully")  # More I/O
```

**Impact**: Every single episode push triggers:
- 2x `logger.info()` calls with f-string formatting
- Logger lookup on every call (minor but unnecessary)
- Stdout I/O in the main process

During training:
- Heuristic seeding: 100 episodes → 200 log lines
- Heatup (~10k steps, ~100-200 episodes): 200-400 log lines  
- Continuous training: 10,000+ episodes → 20,000+ log lines

### Subprocess Logging (lines 133-201)
The subprocess logging is **reasonably throttled**:
- Status every 10 seconds ✓
- First 5 + every 100th batch ✓
- First 10 + every 50th episode pushed ✓

### Recommendation

If you want to reduce overhead, the `push()` method should be made less verbose:

```python
def push(self, episode: Episode):
    if not self._shutdown_event.is_set():
        try:
            replay_data = episode.to_replay()
            explore_step = getattr(episode, 'explore_step_at_completion', None)
            self._push_queue.put((replay_data, explore_step))
        except Exception as e:
            logging.getLogger("ReplayBufferSubprocess").error(f"PUSH ERROR: {e}", exc_info=True)
            raise
```

This removes:
1. Logger lookup on every call
2. Two INFO-level log statements per episode
3. String formatting overhead

The subprocess already logs when episodes arrive (throttled), so the main process logging is redundant diagnostic info that was likely added during debugging.

**This is NOT a bug** - just debug logging left in production code. For a multi-day training run with ~50k+ episodes, this adds ~100k+ unnecessary log lines and measurable (though not catastrophic) overhead. The actual training computations dwarf this, but it's still wasteful.

Would you like me to clean this up in the current codebase?

```python
def push(self, episode: Episode):
    logger = logging.getLogger("ReplayBufferSubprocess")  # Logger lookup every call
    ...
    logger.info(f"PUSH CALLED: episode length={len(episode)}")  # String formatting
    ...
    logger.info(f"PUSH COMPLETE: episode put in queue successfully")  # More I/O
```

```python
def push(self, episode: Episode):
    if not self._shutdown_event.is_set():
        try:
            replay_data = episode.to_replay()
            explore_step = getattr(episode, 'explore_step_at_completion', None)
            self._push_queue.put((replay_data, explore_step))
        except Exception as e:
            logging.getLogger("ReplayBufferSubprocess").error(f"PUSH ERROR: {e}", exc_info=True)
            raise
```

---

**User**

yeah clean this up in current codebase

---

**Cursor**

```python
from time import sleep, time
from multiprocessing.synchronize import Lock as mp_lock
from multiprocessing.synchronize import Event as mp_event
import queue
import torch
import torch.multiprocessing as mp
import logging
import sys

from .replaybuffer import ReplayBuffer, Episode, Batch
from .vanillaepisode import VanillaEpisode
from .vanillastep import VanillaStep

# Configure logging for the replay buffer subprocess
def _setup_subprocess_logging():
    """Setup logging for the replay buffer subprocess."""
    logger = logging.getLogger("ReplayBufferSubprocess")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class VanillaSharedBase(ReplayBuffer):
    def __init__(
        self,
        push_queue: mp.SimpleQueue,
        sample_queue: mp.SimpleQueue,
        task_queue: mp.SimpleQueue,
        result_queue: mp.SimpleQueue,
        request_lock: mp_lock,
        shutdown_event: mp_event,
        batch_size: int,
        episode_arrival_queue: mp.Queue = None,
    ):
        self._push_queue = push_queue
        self._task_queue = task_queue
        self._sample_queue = sample_queue
        self._result_queue = result_queue
        self._request_lock = request_lock
        self._shutdown_event = shutdown_event
        self._batch_size = batch_size
        self._episode_arrival_queue = episode_arrival_queue

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def push(self, episode: Episode):
        if not self._shutdown_event.is_set():
            try:
                replay_data = episode.to_replay()
                explore_step = getattr(episode, 'explore_step_at_completion', None)
                self._push_queue.put((replay_data, explore_step))
            except Exception as e:
                logging.getLogger("ReplayBufferSubprocess").error(f"PUSH ERROR: {e}", exc_info=True)
                raise

    def sample(self) -> Batch:

        if self._shutdown_event.is_set():
            return Batch([], [], [], [], [])

        return self._sample_queue.get()

    def __len__(
        self,
    ):
        if self._shutdown_event.is_set():  #
            return 0

        with self._request_lock:
            self._task_queue.put(["length"])
            length = self._result_queue.get()
        return length

    def set_step_counter(self, step_counter):
        """Send step_counter to the replay buffer subprocess for accurate update_step stamping."""
        self._task_queue.put(["set_step_counter", step_counter])

    def drain_episode_arrivals(self):
        """Drain all episode arrival metadata. Returns dict {explore_step: update_step}."""
        arrival_map = {}
        if self._episode_arrival_queue is None:
            return arrival_map
        while True:
            try:
                explore_step, update_step = self._episode_arrival_queue.get_nowait()
                arrival_map[explore_step] = update_step
            except queue.Empty:
                break
        return arrival_map

    def copy(self):
        return self

    def close(self) -> None:
        ...


class VanillaStepShared(VanillaSharedBase):
    def __init__(self, capacity, batch_size, sample_device: torch.device):
        super().__init__(
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.Lock(),
            mp.Event(),
            batch_size,
            episode_arrival_queue=mp.Queue(),
        )
        self.capacity = capacity
        self.sample_device = sample_device
        self._process = mp.Process(target=self.run)
        self._process.start()

    def run(self):
        internal_replay_buffer = VanillaStep(self.capacity, self._batch_size)
        self.loop(internal_replay_buffer)

    def loop(self, internal_replay_buffer: ReplayBuffer):
        logger = _setup_subprocess_logging()
        logger.info(f"ReplayBuffer subprocess started. batch_size={self.batch_size}, capacity={self.capacity}")

        loop_count = 0
        episodes_received = 0
        batches_produced = 0
        last_status_time = time()
        step_counter = None  # Set later via set_step_counter task

        try:
            while not self._shutdown_event.is_set():
                loop_count += 1
                current_len = len(internal_replay_buffer)
                sample_queue_empty = self._sample_queue.empty()
                can_sample = sample_queue_empty and current_len > self.batch_size

                # Log status every 10 seconds
                if time() - last_status_time > 10:
                    logger.info(
                        f"STATUS: buffer_len={current_len}, batch_size={self.batch_size}, "
                        f"can_sample={can_sample}, sample_queue_empty={sample_queue_empty}, "
                        f"episodes_received={episodes_received}, batches_produced={batches_produced}, "
                        f"loops={loop_count}"
                    )
                    last_status_time = time()

                if can_sample:
                    batch = internal_replay_buffer.sample()
                    # IMPORTANT: Keep batch on CPU for safe inter-process transfer
                    # The trainer will move to CUDA after receiving from the queue
                    self._sample_queue.put(batch)
                    batches_produced += 1
                    if batches_produced <= 5 or batches_produced % 100 == 0:
                        logger.info(f"SAMPLED: batch #{batches_produced}, buffer_len={current_len}")
                elif not self._task_queue.empty():
                    task = self._task_queue.get()
                    if task[0] == "length":
                        self._result_queue.put(current_len)
                        logger.debug(f"LENGTH_QUERY: returned {current_len}")
                    elif task[0] == "set_step_counter":
                        step_counter = task[1]
                        logger.info("SET_STEP_COUNTER: received shared step counter for update_step stamping")
                    elif task[0] == "shutdown":
                        logger.info("SHUTDOWN: received shutdown signal")
                        break
                elif not self._push_queue.empty():
                    item = self._push_queue.get()
                    # Unpack (replay_data, explore_step) tuple
                    if isinstance(item, tuple) and len(item) == 2:
                        batch, explore_step = item
                    else:
                        batch, explore_step = item, None
                    internal_replay_buffer.push(batch)
                    episodes_received += 1
                    # Record (explore_step, update_step) for accurate stamping
                    if step_counter is not None and explore_step is not None and self._episode_arrival_queue is not None:
                        self._episode_arrival_queue.put((explore_step, step_counter.update))
                    if episodes_received <= 10 or episodes_received % 50 == 0:
                        logger.info(f"PUSHED: episode #{episodes_received}, new_buffer_len={len(internal_replay_buffer)}")
                else:
                    sleep(0.0001)

        except Exception as e:
            logger.error(f"EXCEPTION in replay buffer subprocess: {e}", exc_info=True)
            raise
        finally:
            logger.info(f"EXITING: episodes_received={episodes_received}, batches_produced={batches_produced}")
            internal_replay_buffer.close()

    def copy(self):
        return VanillaSharedBase(
            self._push_queue,
            self._sample_queue,
            self._task_queue,
            self._result_queue,
            self._request_lock,
            self._shutdown_event,
            self.batch_size,
            episode_arrival_queue=self._episode_arrival_queue,
        )

    def close(self):
        self._shutdown_event.set()
        self._process.join()
        self._process.close()


class VanillaEpisodeShared(VanillaStepShared):
    def run(self):
        # os.nice(15)
        logger = _setup_subprocess_logging()
        logger.info(f"VanillaEpisodeShared.run() starting with capacity={self.capacity}, batch_size={self._batch_size}")
        internal_replay_buffer = VanillaEpisode(self.capacity, self._batch_size)
        logger.info(f"Created VanillaEpisode buffer, starting loop...")
        self.loop(internal_replay_buffer)

```

Done. The `push()` method is now clean:

```55:63:D:\stEVE_training\eve_rl\eve_rl\replaybuffer\vanillashared.py
    def push(self, episode: Episode):
        if not self._shutdown_event.is_set():
            try:
                replay_data = episode.to_replay()
                explore_step = getattr(episode, 'explore_step_at_completion', None)
                self._push_queue.put((replay_data, explore_step))
            except Exception as e:
                logging.getLogger("ReplayBufferSubprocess").error(f"PUSH ERROR: {e}", exc_info=True)
                raise
```

**Changes made:**
- Removed logger lookup on every call
- Removed 2x INFO-level log statements per episode ("PUSH CALLED" and "PUSH COMPLETE")
- Removed the "PUSH SKIPPED" warning (unnecessary - shutdown is expected behavior)
- Kept error logging (only triggers on actual failures)

The subprocess still logs episode arrivals (throttled: first 10 + every 50th), so you retain visibility into buffer activity without main-process spam.

```plaintext
    def push(self, episode: Episode):
        if not self._shutdown_event.is_set():
            try:
                replay_data = episode.to_replay()
                explore_step = getattr(episode, 'explore_step_at_completion', None)
                self._push_queue.put((replay_data, explore_step))
            except Exception as e:
                logging.getLogger("ReplayBufferSubprocess").error(f"PUSH ERROR: {e}", exc_info=True)
                raise
```

---

**User**

ok , now merge all the files from that codebase but before merging first save all the changed files as <filenames>_v(1/2) and then merge ; remeber for us it's DualDeviceNav_train_vs ; before making any edits, copy give me a list of what all to save and change

---

**Cursor**

Here's the list of files that need to be backed up and then merged:

## Files to Backup → Then Merge

| # | Current File (stEVE_training) | Backup As | Source (rl_resume) |
|---|-------------------------------|-----------|---------------------|
| 1 | `eve_rl\eve_rl\replaybuffer\replaybuffer.py` | `replaybuffer_v1.py` | Add `EpisodeReplay.to_replay()` |
| 2 | `eve_rl\eve_rl\replaybuffer\vanillaepisode.py` | `vanillaepisode_v1.py` | Add `export_all()` / `import_all()` |
| 3 | `eve_rl\eve_rl\replaybuffer\vanillashared.py` | `vanillashared_v1.py` | Add `save_buffer_to_file()` / `load_buffer_from_file()` + IPC tasks |
| 4 | `eve_rl\eve_rl\replaybuffer\__init__.py` | `__init__v1.py` | Export `EpisodeReplay` |
| 5 | `eve_rl\eve_rl\runner\runner.py` | `runner_v1.py` | Add `load_replay_buffer()`, periodic save, `_eval_count`, `heatup_cache_save_path` |
| 6 | `training_scripts\DualDeviceNav_train_vs.py` | `DualDeviceNav_train_vs_v2.py` | Add cache CLI args (`--heuristic_cache_file`, `--save_heuristic_cache`, etc.) |
| 7 | `training_scripts\DualDeviceNav_resume.py` | `DualDeviceNav_resume_v1.py` | Add auto-load `replay_buffer.npz`, fix `.flatten()` bug |

## New File to Create (no backup needed)

| # | New File | Purpose |
|---|----------|---------|
| 8 | `eve_rl\eve_rl\util\experience_cache.py` | `save_episodes_npz()` / `load_episodes_npz()` functions |

---

**Note**: `DualDeviceNav_train_vs_v1.py` already exists, so the new backup will be `_v2`.

Does this look correct? Say "go" and I'll proceed with the backups and merges.

---

**User**

ok ; also take of the bugs (🔴 Bugs to fix	2 (same .flatten() bug in 2 files)
⚠️ Missing features	2 (env5, heuristic cache in resume)
) when merging 

---

**Cursor**

I'll proceed with backups first, then merge with bug fixes. Let me start:

Now let me read the rl_resume versions to understand the exact changes:

Let me also check the training scripts in rl_resume:

Now I'll create all the backups and then merge the changes. Let me start:

Backups created. Now let me merge the changes:

**1. Add `EpisodeReplay.to_replay()` method to replaybuffer.py:**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Union
import numpy as np

import torch

from ..util import EveRLObject


class Episode:
    def __init__(
        self,
        reset_obs: Dict[str, np.ndarray],
        reset_flat_obs: np.ndarray,
        flat_obs_to_obs: Optional[Union[List, Dict]] = None,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.obs: List[Dict[str, np.ndarray]] = [reset_obs]
        self.flat_obs: List[np.ndarray] = [reset_flat_obs]
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.terminals: List[bool] = []
        self.truncations: List[bool] = []
        self.infos: List[Dict[str, np.ndarray]] = []
        self.episode_reward: float = 0.0
        self.flat_state_to_state = flat_obs_to_obs
        self.seed = seed
        self.options = options
        # Step counters at episode completion (stamped by worker)
        self.explore_step_at_completion: Optional[int] = None
        self.update_step_at_completion: Optional[int] = None

    def add_transition(
        self,
        obs: Dict[str, np.ndarray],
        flat_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: bool,
        truncation: bool,
        info: Dict[str, np.ndarray],
    ):
        self.obs.append(obs)
        self.flat_obs.append(flat_obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        self.truncations.append(truncation)
        self.infos.append(info)
        self.episode_reward += reward

    def to_replay(self):
        return EpisodeReplay(self.flat_obs, self.actions, self.rewards, self.terminals)

    def __len__(self):
        return len(self.actions)


@dataclass
class EpisodeReplay:
    flat_obs: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    terminals: List[bool]

    def to_replay(self):
        return self

    def __len__(self):
        return len(self.actions)


class Batch(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    padding_mask: torch.Tensor = None

    def to(self, device: torch.device, non_blocking=False):
        obs = self.obs.to(
            device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ).share_memory_()
        actions = self.actions.to(
            device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ).share_memory_()
        rewards = self.rewards.to(
            device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ).share_memory_()
        terminals = self.terminals.to(
            device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ).share_memory_()
        if self.padding_mask is not None:
            padding_mask = self.padding_mask.to(
                device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ).share_memory_()
        else:
            padding_mask = None
        return Batch(obs, actions, rewards, terminals, padding_mask)


class ReplayBuffer(EveRLObject, ABC):
    @property
    @abstractmethod
    def batch_size(self) -> int:
        ...

    @abstractmethod
    def push(self, episode: Union[Episode, EpisodeReplay]) -> None:
        ...

    @abstractmethod
    def sample(self) -> Batch:
        ...

    @abstractmethod
    def copy(self):
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

```

**2. Add `export_all()` and `import_all()` to vanillaepisode.py:**

```python
from math import inf
import random
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence
from .replaybuffer import ReplayBuffer, Episode, Batch
import numpy as np


class VanillaEpisode(ReplayBuffer):
    def __init__(self, capacity: int, batch_size: int):
        self.capacity = capacity
        self._batch_size = batch_size
        self.buffer: List[Episode] = []
        self.position = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def push(self, episode: Episode):
        if len(episode) < 1:
            return
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        episode_np = (
            np.array(episode.flat_obs),
            np.array(episode.actions),
            np.array(episode.rewards),
            np.array(episode.terminals),
        )
        self.buffer[self.position] = episode_np
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self) -> Batch:
        episodes = random.sample(self.buffer, self.batch_size)

        state_batch = [torch.from_numpy(episode[0]) for episode in episodes]
        action_batch = [torch.from_numpy(episode[1]) for episode in episodes]
        reward_batch = [
            torch.from_numpy(episode[2]).unsqueeze(1) for episode in episodes
        ]
        done_batch = [torch.from_numpy(episode[3]).unsqueeze(1) for episode in episodes]

        state_batch = pad_sequence(state_batch, batch_first=True)
        action_batch = pad_sequence(action_batch, batch_first=True)
        reward_batch = pad_sequence(reward_batch, batch_first=True, padding_value=inf)
        done_batch = pad_sequence(done_batch, batch_first=True)

        padding_mask = torch.ones_like(reward_batch)
        padding_mask[reward_batch == inf] = 0
        reward_batch[reward_batch == inf] = 0
        return Batch(state_batch, action_batch, reward_batch, done_batch, padding_mask)

    def __len__(self):
        return len(self.buffer)

    def copy(self):
        copy = self.__class__(self.capacity, self.batch_size)
        for i in range(len(self.buffer)):
            copy.buffer.append(self.buffer[i])
        copy.position = self.position
        return copy

    def export_all(self):
        """Return (buffer_entries, position) for serialization."""
        return (list(self.buffer), self.position)

    def import_all(self, episodes, position):
        """Replace buffer contents with imported episodes."""
        self.buffer = list(episodes)
        self.position = position % self.capacity if episodes else 0

    def close(self):
        del self.buffer

```

**3. Add buffer save/load methods to vanillashared.py:**

```python
from time import sleep, time
from multiprocessing.synchronize import Lock as mp_lock
from multiprocessing.synchronize import Event as mp_event
import queue
import torch
import torch.multiprocessing as mp
import logging
import sys

from .replaybuffer import ReplayBuffer, Episode, Batch
from .vanillaepisode import VanillaEpisode
from .vanillastep import VanillaStep

# Configure logging for the replay buffer subprocess
def _setup_subprocess_logging():
    """Setup logging for the replay buffer subprocess."""
    logger = logging.getLogger("ReplayBufferSubprocess")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class VanillaSharedBase(ReplayBuffer):
    def __init__(
        self,
        push_queue: mp.SimpleQueue,
        sample_queue: mp.SimpleQueue,
        task_queue: mp.SimpleQueue,
        result_queue: mp.SimpleQueue,
        request_lock: mp_lock,
        shutdown_event: mp_event,
        batch_size: int,
        episode_arrival_queue: mp.Queue = None,
    ):
        self._push_queue = push_queue
        self._task_queue = task_queue
        self._sample_queue = sample_queue
        self._result_queue = result_queue
        self._request_lock = request_lock
        self._shutdown_event = shutdown_event
        self._batch_size = batch_size
        self._episode_arrival_queue = episode_arrival_queue

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def push(self, episode: Episode):
        if not self._shutdown_event.is_set():
            try:
                replay_data = episode.to_replay()
                explore_step = getattr(episode, 'explore_step_at_completion', None)
                self._push_queue.put((replay_data, explore_step))
            except Exception as e:
                logging.getLogger("ReplayBufferSubprocess").error(f"PUSH ERROR: {e}", exc_info=True)
                raise

    def sample(self) -> Batch:

        if self._shutdown_event.is_set():
            return Batch([], [], [], [], [])

        return self._sample_queue.get()

    def __len__(
        self,
    ):
        if self._shutdown_event.is_set():  #
            return 0

        with self._request_lock:
            self._task_queue.put(["length"])
            length = self._result_queue.get()
        return length

    def set_step_counter(self, step_counter):
        """Send step_counter to the replay buffer subprocess for accurate update_step stamping."""
        self._task_queue.put(["set_step_counter", step_counter])

    def drain_episode_arrivals(self):
        """Drain all episode arrival metadata. Returns dict {explore_step: update_step}."""
        arrival_map = {}
        if self._episode_arrival_queue is None:
            return arrival_map
        while True:
            try:
                explore_step, update_step = self._episode_arrival_queue.get_nowait()
                arrival_map[explore_step] = update_step
            except queue.Empty:
                break
        return arrival_map

    def save_buffer_to_file(self, path):
        """Have subprocess save buffer directly to file. Returns episode count."""
        with self._request_lock:
            self._task_queue.put(["save_buffer", path])
            return self._result_queue.get()

    def load_buffer_from_file(self, path):
        """Have subprocess load buffer directly from file. Returns episode count."""
        with self._request_lock:
            self._task_queue.put(["load_buffer", path])
            return self._result_queue.get()

    def copy(self):
        return self

    def close(self) -> None:
        ...


class VanillaStepShared(VanillaSharedBase):
    def __init__(self, capacity, batch_size, sample_device: torch.device):
        super().__init__(
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.Lock(),
            mp.Event(),
            batch_size,
            episode_arrival_queue=mp.Queue(),
        )
        self.capacity = capacity
        self.sample_device = sample_device
        self._process = mp.Process(target=self.run)
        self._process.start()

    def run(self):
        internal_replay_buffer = VanillaStep(self.capacity, self._batch_size)
        self.loop(internal_replay_buffer)

    def loop(self, internal_replay_buffer: ReplayBuffer):
        logger = _setup_subprocess_logging()
        logger.info(f"ReplayBuffer subprocess started. batch_size={self.batch_size}, capacity={self.capacity}")

        loop_count = 0
        episodes_received = 0
        batches_produced = 0
        last_status_time = time()
        step_counter = None  # Set later via set_step_counter task

        try:
            while not self._shutdown_event.is_set():
                loop_count += 1
                current_len = len(internal_replay_buffer)
                sample_queue_empty = self._sample_queue.empty()
                can_sample = sample_queue_empty and current_len > self.batch_size

                # Log status every 10 seconds
                if time() - last_status_time > 10:
                    logger.info(
                        f"STATUS: buffer_len={current_len}, batch_size={self.batch_size}, "
                        f"can_sample={can_sample}, sample_queue_empty={sample_queue_empty}, "
                        f"episodes_received={episodes_received}, batches_produced={batches_produced}, "
                        f"loops={loop_count}"
                    )
                    last_status_time = time()

                if can_sample:
                    batch = internal_replay_buffer.sample()
                    # IMPORTANT: Keep batch on CPU for safe inter-process transfer
                    # The trainer will move to CUDA after receiving from the queue
                    self._sample_queue.put(batch)
                    batches_produced += 1
                    if batches_produced <= 5 or batches_produced % 100 == 0:
                        logger.info(f"SAMPLED: batch #{batches_produced}, buffer_len={current_len}")
                elif not self._task_queue.empty():
                    task = self._task_queue.get()
                    if task[0] == "length":
                        self._result_queue.put(current_len)
                        logger.debug(f"LENGTH_QUERY: returned {current_len}")
                    elif task[0] == "set_step_counter":
                        step_counter = task[1]
                        logger.info("SET_STEP_COUNTER: received shared step counter for update_step stamping")
                    elif task[0] == "shutdown":
                        logger.info("SHUTDOWN: received shutdown signal")
                        break
                elif not self._push_queue.empty():
                    item = self._push_queue.get()
                    # Unpack (replay_data, explore_step) tuple
                    if isinstance(item, tuple) and len(item) == 2:
                        batch, explore_step = item
                    else:
                        batch, explore_step = item, None
                    internal_replay_buffer.push(batch)
                    episodes_received += 1
                    # Record (explore_step, update_step) for accurate stamping
                    if step_counter is not None and explore_step is not None and self._episode_arrival_queue is not None:
                        self._episode_arrival_queue.put((explore_step, step_counter.update))
                    if episodes_received <= 10 or episodes_received % 50 == 0:
                        logger.info(f"PUSHED: episode #{episodes_received}, new_buffer_len={len(internal_replay_buffer)}")
                else:
                    sleep(0.0001)

        except Exception as e:
            logger.error(f"EXCEPTION in replay buffer subprocess: {e}", exc_info=True)
            raise
        finally:
            logger.info(f"EXITING: episodes_received={episodes_received}, batches_produced={batches_produced}")
            internal_replay_buffer.close()

    def copy(self):
        return VanillaSharedBase(
            self._push_queue,
            self._sample_queue,
            self._task_queue,
            self._result_queue,
            self._request_lock,
            self._shutdown_event,
            self.batch_size,
            episode_arrival_queue=self._episode_arrival_queue,
        )

    def close(self):
        self._shutdown_event.set()
        self._process.join()
        self._process.close()


class VanillaEpisodeShared(VanillaStepShared):
    def run(self):
        # os.nice(15)
        logger = _setup_subprocess_logging()
        logger.info(f"VanillaEpisodeShared.run() starting with capacity={self.capacity}, batch_size={self._batch_size}")
        internal_replay_buffer = VanillaEpisode(self.capacity, self._batch_size)
        logger.info(f"Created VanillaEpisode buffer, starting loop...")
        self.loop(internal_replay_buffer)

```

Now add the IPC task handlers in the subprocess loop:

```python
from time import sleep, time
from multiprocessing.synchronize import Lock as mp_lock
from multiprocessing.synchronize import Event as mp_event
import queue
import torch
import torch.multiprocessing as mp
import logging
import sys

from .replaybuffer import ReplayBuffer, Episode, Batch
from .vanillaepisode import VanillaEpisode
from .vanillastep import VanillaStep

# Configure logging for the replay buffer subprocess
def _setup_subprocess_logging():
    """Setup logging for the replay buffer subprocess."""
    logger = logging.getLogger("ReplayBufferSubprocess")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class VanillaSharedBase(ReplayBuffer):
    def __init__(
        self,
        push_queue: mp.SimpleQueue,
        sample_queue: mp.SimpleQueue,
        task_queue: mp.SimpleQueue,
        result_queue: mp.SimpleQueue,
        request_lock: mp_lock,
        shutdown_event: mp_event,
        batch_size: int,
        episode_arrival_queue: mp.Queue = None,
    ):
        self._push_queue = push_queue
        self._task_queue = task_queue
        self._sample_queue = sample_queue
        self._result_queue = result_queue
        self._request_lock = request_lock
        self._shutdown_event = shutdown_event
        self._batch_size = batch_size
        self._episode_arrival_queue = episode_arrival_queue

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def push(self, episode: Episode):
        if not self._shutdown_event.is_set():
            try:
                replay_data = episode.to_replay()
                explore_step = getattr(episode, 'explore_step_at_completion', None)
                self._push_queue.put((replay_data, explore_step))
            except Exception as e:
                logging.getLogger("ReplayBufferSubprocess").error(f"PUSH ERROR: {e}", exc_info=True)
                raise

    def sample(self) -> Batch:

        if self._shutdown_event.is_set():
            return Batch([], [], [], [], [])

        return self._sample_queue.get()

    def __len__(
        self,
    ):
        if self._shutdown_event.is_set():  #
            return 0

        with self._request_lock:
            self._task_queue.put(["length"])
            length = self._result_queue.get()
        return length

    def set_step_counter(self, step_counter):
        """Send step_counter to the replay buffer subprocess for accurate update_step stamping."""
        self._task_queue.put(["set_step_counter", step_counter])

    def drain_episode_arrivals(self):
        """Drain all episode arrival metadata. Returns dict {explore_step: update_step}."""
        arrival_map = {}
        if self._episode_arrival_queue is None:
            return arrival_map
        while True:
            try:
                explore_step, update_step = self._episode_arrival_queue.get_nowait()
                arrival_map[explore_step] = update_step
            except queue.Empty:
                break
        return arrival_map

    def save_buffer_to_file(self, path):
        """Have subprocess save buffer directly to file. Returns episode count."""
        with self._request_lock:
            self._task_queue.put(["save_buffer", path])
            return self._result_queue.get()

    def load_buffer_from_file(self, path):
        """Have subprocess load buffer directly from file. Returns episode count."""
        with self._request_lock:
            self._task_queue.put(["load_buffer", path])
            return self._result_queue.get()

    def copy(self):
        return self

    def close(self) -> None:
        ...


class VanillaStepShared(VanillaSharedBase):
    def __init__(self, capacity, batch_size, sample_device: torch.device):
        super().__init__(
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.Lock(),
            mp.Event(),
            batch_size,
            episode_arrival_queue=mp.Queue(),
        )
        self.capacity = capacity
        self.sample_device = sample_device
        self._process = mp.Process(target=self.run)
        self._process.start()

    def run(self):
        internal_replay_buffer = VanillaStep(self.capacity, self._batch_size)
        self.loop(internal_replay_buffer)

    def loop(self, internal_replay_buffer: ReplayBuffer):
        logger = _setup_subprocess_logging()
        logger.info(f"ReplayBuffer subprocess started. batch_size={self.batch_size}, capacity={self.capacity}")

        loop_count = 0
        episodes_received = 0
        batches_produced = 0
        last_status_time = time()
        step_counter = None  # Set later via set_step_counter task

        try:
            while not self._shutdown_event.is_set():
                loop_count += 1
                current_len = len(internal_replay_buffer)
                sample_queue_empty = self._sample_queue.empty()
                can_sample = sample_queue_empty and current_len > self.batch_size

                # Log status every 10 seconds
                if time() - last_status_time > 10:
                    logger.info(
                        f"STATUS: buffer_len={current_len}, batch_size={self.batch_size}, "
                        f"can_sample={can_sample}, sample_queue_empty={sample_queue_empty}, "
                        f"episodes_received={episodes_received}, batches_produced={batches_produced}, "
                        f"loops={loop_count}"
                    )
                    last_status_time = time()

                if can_sample:
                    batch = internal_replay_buffer.sample()
                    # IMPORTANT: Keep batch on CPU for safe inter-process transfer
                    # The trainer will move to CUDA after receiving from the queue
                    self._sample_queue.put(batch)
                    batches_produced += 1
                    if batches_produced <= 5 or batches_produced % 100 == 0:
                        logger.info(f"SAMPLED: batch #{batches_produced}, buffer_len={current_len}")
                elif not self._task_queue.empty():
                    task = self._task_queue.get()
                    if task[0] == "length":
                        self._result_queue.put(current_len)
                        logger.debug(f"LENGTH_QUERY: returned {current_len}")
                    elif task[0] == "set_step_counter":
                        step_counter = task[1]
                        logger.info("SET_STEP_COUNTER: received shared step counter for update_step stamping")
                    elif task[0] == "shutdown":
                        logger.info("SHUTDOWN: received shutdown signal")
                        break
                    elif task[0] == "save_buffer":
                        path = task[1]
                        from eve_rl.util.experience_cache import save_episodes_npz
                        episodes_list, pos = internal_replay_buffer.export_all()
                        save_episodes_npz(path, episodes_list, pos)
                        self._result_queue.put(len(episodes_list))
                        logger.info(f"SAVE_BUFFER: saved {len(episodes_list)} episodes to {path}")
                    elif task[0] == "load_buffer":
                        path = task[1]
                        from eve_rl.util.experience_cache import load_episodes_npz
                        episodes_list, pos = load_episodes_npz(path)
                        internal_replay_buffer.import_all(episodes_list, pos)
                        self._result_queue.put(len(episodes_list))
                        logger.info(f"LOAD_BUFFER: loaded {len(episodes_list)} episodes from {path}")
                elif not self._push_queue.empty():
                    item = self._push_queue.get()
                    # Unpack (replay_data, explore_step) tuple
                    if isinstance(item, tuple) and len(item) == 2:
                        batch, explore_step = item
                    else:
                        batch, explore_step = item, None
                    internal_replay_buffer.push(batch)
                    episodes_received += 1
                    # Record (explore_step, update_step) for accurate stamping
                    if step_counter is not None and explore_step is not None and self._episode_arrival_queue is not None:
                        self._episode_arrival_queue.put((explore_step, step_counter.update))
                    if episodes_received <= 10 or episodes_received % 50 == 0:
                        logger.info(f"PUSHED: episode #{episodes_received}, new_buffer_len={len(internal_replay_buffer)}")
                else:
                    sleep(0.0001)

        except Exception as e:
            logger.error(f"EXCEPTION in replay buffer subprocess: {e}", exc_info=True)
            raise
        finally:
            logger.info(f"EXITING: episodes_received={episodes_received}, batches_produced={batches_produced}")
            internal_replay_buffer.close()

    def copy(self):
        return VanillaSharedBase(
            self._push_queue,
            self._sample_queue,
            self._task_queue,
            self._result_queue,
            self._request_lock,
            self._shutdown_event,
            self.batch_size,
            episode_arrival_queue=self._episode_arrival_queue,
        )

    def close(self):
        self._shutdown_event.set()
        self._process.join()
        self._process.close()


class VanillaEpisodeShared(VanillaStepShared):
    def run(self):
        # os.nice(15)
        logger = _setup_subprocess_logging()
        logger.info(f"VanillaEpisodeShared.run() starting with capacity={self.capacity}, batch_size={self._batch_size}")
        internal_replay_buffer = VanillaEpisode(self.capacity, self._batch_size)
        logger.info(f"Created VanillaEpisode buffer, starting loop...")
        self.loop(internal_replay_buffer)

```

**4. Export EpisodeReplay in __init__.py:**

```python
from .replaybuffer import ReplayBuffer, Batch, Episode, EpisodeReplay
from .vanillastep import VanillaStep
from .vanillaepisode import VanillaEpisode
from .vanillashared import VanillaStepShared, VanillaEpisodeShared
from .vanillashared import VanillaSharedBase as ReplayBufferShared

```

**5. Create experience_cache.py:**

```python
"""Serialization utilities for replay buffer episode data.

Format: compressed numpy archive (.npz) with flat layout + length index.
Variable-length episodes are concatenated along axis 0 with a lengths array
for reconstruction. This avoids object arrays which break np.savez_compressed.

File contents:
    n_episodes: int scalar
    lengths: int array (n_episodes,) — number of actions (steps) per episode
    flat_obs: float array (sum(lengths+1), obs_dim) — concatenated observations
    actions: float array (sum(lengths), action_dim) — concatenated actions
    rewards: float array (sum(lengths),) — concatenated rewards
    terminals: bool array (sum(lengths),) — concatenated terminal flags
    position: int scalar — ring buffer write position (for replay buffer restore)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def save_episodes_npz(path, episodes, position=0):
    """Save a list of episode numpy tuples to a compressed .npz file.

    Args:
        path: Output file path (should end in .npz).
        episodes: List of (flat_obs, actions, rewards, terminals) numpy tuples.
            flat_obs shape: (T+1, obs_dim)
            actions shape: (T, action_dim)
            rewards shape: (T,)
            terminals shape: (T,)
        position: Ring buffer write position (for replay buffer state).
    """
    if not episodes:
        logger.warning("save_episodes_npz: empty episode list, nothing to save")
        return

    lengths = np.array([ep[1].shape[0] for ep in episodes], dtype=np.int32)

    all_obs = np.concatenate([ep[0] for ep in episodes], axis=0)
    all_actions = np.concatenate([ep[1] for ep in episodes], axis=0)
    all_rewards = np.concatenate([ep[2] for ep in episodes], axis=0)
    all_terminals = np.concatenate([ep[3] for ep in episodes], axis=0)

    np.savez_compressed(
        path,
        n_episodes=np.int32(len(episodes)),
        lengths=lengths,
        flat_obs=all_obs,
        actions=all_actions,
        rewards=all_rewards,
        terminals=all_terminals,
        position=np.int32(position),
    )
    total_steps = int(lengths.sum())
    logger.info(f"Saved {len(episodes)} episodes ({total_steps} steps) to {path}")


def load_episodes_npz(path):
    """Load episodes from a compressed .npz file.

    Args:
        path: Input file path.

    Returns:
        (episodes, position) where episodes is a list of
        (flat_obs, actions, rewards, terminals) numpy tuples
        and position is the ring buffer write position.
    """
    data = np.load(path)

    n_episodes = int(data["n_episodes"])
    lengths = data["lengths"]
    position = int(data["position"])

    all_obs = data["flat_obs"]
    all_actions = data["actions"]
    all_rewards = data["rewards"]
    all_terminals = data["terminals"]

    # Split observations: each episode has (length + 1) obs entries
    obs_lengths = lengths + 1
    obs_splits = np.cumsum(obs_lengths)[:-1]
    obs_list = np.split(all_obs, obs_splits, axis=0)

    # Split actions/rewards/terminals: each episode has (length) entries
    step_splits = np.cumsum(lengths)[:-1]
    actions_list = np.split(all_actions, step_splits, axis=0)
    rewards_list = np.split(all_rewards, step_splits, axis=0)
    terminals_list = np.split(all_terminals, step_splits, axis=0)

    episodes = [
        (obs_list[i], actions_list[i], rewards_list[i], terminals_list[i])
        for i in range(n_episodes)
    ]

    total_steps = int(lengths.sum())
    logger.info(f"Loaded {n_episodes} episodes ({total_steps} steps) from {path}")
    return episodes, position

```

**6. Update runner.py with replay buffer features:**

```python
from typing import List, Optional
from math import inf
import csv
import json
import logging
import os
import time
import numpy as np
from ..util import EveRLObject
from ..agent.agent import Agent, StepCounter, EpisodeCounter


class Runner(EveRLObject):
    def __init__(
        self,
        agent: Agent,
        heatup_action_low: List[float],
        heatup_action_high: List[float],
        agent_parameter_for_result_file: dict,
        checkpoint_folder: str,
        results_file: str,
        quality_info: Optional[str] = None,
        info_results: Optional[List[str]] = None,
        diagnostics_folder: Optional[str] = None,
        policy_snapshot_every_steps: int = 10000,
        n_probe_episodes: int = 3,
        n_probe_near_start_steps: int = 10,
    ) -> None:
        self.agent = agent
        self.heatup_action_low = heatup_action_low
        self.heatup_action_high = heatup_action_high
        self.agent_parameter_for_result_file = agent_parameter_for_result_file
        self.checkpoint_folder = checkpoint_folder
        self.results_file = results_file
        self.quality_info = quality_info
        self.info_results = info_results or []
        self.logger = logging.getLogger(self.__module__)
        
        # Diagnostics settings
        self.diagnostics_folder = diagnostics_folder
        # Ensure worker step logs directory exists.
        # NOTE: STEP_LOG_DIR must be set BEFORE agent creation (which spawns workers).
        # The training script should set os.environ["STEP_LOG_DIR"] before creating the agent.
        # This setdefault is a fallback but will NOT help if workers are already spawned.
        if self.diagnostics_folder is not None:
            logs_subprocesses = os.path.join(self.diagnostics_folder, "logs_subprocesses")
            os.makedirs(logs_subprocesses, exist_ok=True)
            os.environ.setdefault("STEP_LOG_DIR", logs_subprocesses)
        self.policy_snapshot_every_steps = policy_snapshot_every_steps
        self.n_probe_episodes = n_probe_episodes
        self.n_probe_near_start_steps = n_probe_near_start_steps
        self._next_snapshot_step = policy_snapshot_every_steps
        self._probe_states_set = False
        self._eval_count = 0
        self._replay_save_interval = 4  # save replay buffer every N evals
        
        # Episode summary logger
        self._episode_summary_file = None
        self._episode_summary_logger = None
        self._episode_summary_counter = 0  # Fix #5: monotonic episode_id
        if self.diagnostics_folder is not None:
            csv_folder = os.path.join(self.diagnostics_folder, "csv")
            os.makedirs(csv_folder, exist_ok=True)
            self._episode_summary_file = os.path.join(csv_folder, "episode_summary.jsonl")
            self.logger.info(f"Episode summary will be logged to {self._episode_summary_file}")

        self._results = {
            "episodes explore": 0,
            "steps explore": 0,
        }
        self._results["quality"] = 0.0
        for info_result in self.info_results:
            self._results[info_result] = 0.0
        self._results["reward"] = 0.0
        self._results["best quality"] = 0.0
        self._results["best explore steps"] = 0.0

        file_existed = os.path.isfile(results_file)

        with open(results_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_existed:
                writer.writerow(
                    list(self._results.keys())
                    + [" "]
                    + list(agent_parameter_for_result_file.keys())
                )
                writer.writerow(
                    [" "] * (len(self._results.values()) + 1)
                    + list(agent_parameter_for_result_file.values())
                )

        self.best_eval = {"steps": 0, "quality": -inf}

    @property
    def step_counter(self) -> StepCounter:
        return self.agent.step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self.agent.episode_counter

    def restore_runner_state(self, checkpoint_path: str, probe_states_dir: Optional[str] = None):
        """Restore Runner-level state from a checkpoint for resume support.

        Restores: best_eval, _episode_summary_counter, _next_snapshot_step,
        and optionally probe states from a saved .npz file.
        """
        import torch
        checkpoint = torch.load(checkpoint_path)
        additional_info = checkpoint.get("additional_info", {})
        runner_state = additional_info.get("runner_state") if additional_info else None

        if runner_state:
            self.best_eval = runner_state["best_eval"]
            self._episode_summary_counter = runner_state["episode_summary_counter"]
            self._next_snapshot_step = runner_state["next_snapshot_step"]
            self.logger.info(
                f"Restored runner state: best_eval={self.best_eval}, "
                f"_episode_summary_counter={self._episode_summary_counter}, "
                f"_next_snapshot_step={self._next_snapshot_step}"
            )
        else:
            # Checkpoint predates runner_state support — compute safe defaults
            explore_steps = self.step_counter.exploration
            self._next_snapshot_step = (
                (explore_steps // self.policy_snapshot_every_steps) + 1
            ) * self.policy_snapshot_every_steps
            self.logger.warning(
                f"No runner_state in checkpoint. Setting _next_snapshot_step={self._next_snapshot_step}"
            )

        # Restore probe states from saved .npz if available
        if probe_states_dir is not None:
            probe_path = os.path.join(probe_states_dir, "probes", "probe_states.npz")
            if os.path.isfile(probe_path):
                try:
                    data = np.load(probe_path)
                    probe_states = data["probe_states"]
                    if hasattr(self.agent, 'set_probe_states'):
                        self.agent.set_probe_states(probe_states.tolist())
                        self._probe_states_set = True
                        self.logger.info(f"Restored {len(probe_states)} probe states from {probe_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to restore probe states: {e}")

    def heatup(self, steps: int):
        episodes = self.agent.heatup(
            steps=steps,
            custom_action_low=self.heatup_action_low,
            custom_action_high=self.heatup_action_high,
        )
        return episodes

    def explore(self, n_episodes: int):
        self.agent.explore(episodes=n_episodes)

    def update(self, n_steps: int):
        self.agent.update(n_steps)

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[List[int]] = None
    ):
        explore_steps = self.step_counter.exploration
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{explore_steps}.everl"
        )
        result_episodes = self.agent.evaluate(episodes=episodes, seeds=seeds)
        qualities, rewards = [], []
        results_for_info = {
            info_result_name: [] for info_result_name in self.info_results
        }
        eval_results = {"episodes": []}
        for episode in result_episodes:
            reward = episode.episode_reward
            quality = (
                episode.infos[-1][self.quality_info]
                if self.quality_info is not None
                else reward
            )
            qualities.append(quality)
            rewards.append(reward)
            for info_result_name in self.info_results:
                info = episode.infos[-1][info_result_name]
                results_for_info[info_result_name].append(info)
            eval_results["episodes"].append(
                {
                    "seed": episode.seed,
                    "options": episode.options,
                    "quality": episode.infos[-1][self.quality_info],
                }
            )

        reward = sum(rewards) / len(rewards)
        quality = sum(qualities) / len(qualities)
        for info_result_name, results in results_for_info.items():
            result = sum(results) / len(results)
            self._results[info_result_name] = round(result, 3)
        save_best = False
        if quality > self.best_eval["quality"]:
            save_best = True
            self.best_eval["quality"] = quality
            self.best_eval["steps"] = explore_steps

        self._results["episodes explore"] = self.episode_counter.exploration
        self._results["steps explore"] = explore_steps
        self._results["reward"] = round(reward, 3)
        self._results["quality"] = round(quality, 3)
        self._results["best quality"] = self.best_eval["quality"]
        self._results["best explore steps"] = self.best_eval["steps"]

        eval_results.update(self._results)
        eval_results.pop("best quality")
        eval_results.pop("best explore steps")

        # Include runner state for resume support
        eval_results["runner_state"] = {
            "best_eval": self.best_eval.copy(),
            "episode_summary_counter": self._episode_summary_counter,
            "next_snapshot_step": self._next_snapshot_step,
        }

        self.agent.save_checkpoint(checkpoint_file, eval_results)
        if save_best:
            checkpoint_file = os.path.join(
                self.checkpoint_folder, "best_checkpoint.everl"
            )
            self.agent.save_checkpoint(checkpoint_file, eval_results)

        log_info = (
            f"Quality: {quality}, Reward: {reward}, Exploration steps: {explore_steps}"
        )
        self.logger.info(log_info)
        with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(self._results.values())

        return quality, reward

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: int = None,
        explore_steps_limit: int = None,
    ):
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        while self.step_counter.exploration < explore_steps_limit:
            # Total experience = heatup + exploration (both contribute to replay buffer)
            total_experience_steps = (
                self.step_counter.heatup + self.step_counter.exploration
            )
            update_steps = (
                total_experience_steps * update_steps_per_explore_step
                - self.step_counter.update
            )
            result = self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )
            
            # Log episode summaries if we got results
            if result is not None:
                episodes, _ = result if isinstance(result, tuple) else (result, None)
                if episodes:
                    self._log_episode_summaries(episodes)
            
            # Check for policy snapshots after each exploration cycle
            # This ensures snapshots are saved at the actual milestone with correct weights
            self._maybe_save_policy_snapshot()

    def training_run(
        self,
        heatup_steps: int,
        training_steps: int,
        explore_steps_between_eval: int,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
    ):
        # TODO: Log Training Run Infos
        heatup_episodes = self.heatup(heatup_steps)
        
        # Capture and set probe states after heatup
        self._capture_and_set_probe_states(heatup_episodes)
        
        next_eval_step_limt = (
            self.agent.step_counter.exploration + explore_steps_between_eval
        )
        while self.agent.step_counter.exploration < training_steps:
            self.explore_and_update(
                explore_episodes_between_updates,
                update_steps_per_explore_step,
                explore_steps_limit=next_eval_step_limt,
            )
            
            # Note: policy snapshots are now saved inside explore_and_update()
            # at the actual milestone steps with correct weights
            
            quality, reward = self.eval(episodes=eval_episodes, seeds=eval_seeds)
            next_eval_step_limt += explore_steps_between_eval

        return quality, reward

    def _capture_and_set_probe_states(self, episodes: List):
        """Capture probe states from heatup episodes and set them in trainer."""
        if not episodes or self._probe_states_set:
            return
            
        try:
            probe_states = []
            n_episodes = min(len(episodes), self.n_probe_episodes)
            
            for ep_idx in range(n_episodes):
                episode = episodes[ep_idx]
                if hasattr(episode, 'flat_obs') and len(episode.flat_obs) > 0:
                    # Collect start state
                    probe_states.append(episode.flat_obs[0])
                    
                    # Collect near-start states
                    n_states = min(self.n_probe_near_start_steps, len(episode.flat_obs) - 1)
                    for step_idx in range(1, n_states + 1):
                        probe_states.append(episode.flat_obs[step_idx])
                        
            if probe_states:
                probe_states_array = np.array(probe_states)
                
                # Set probe states in agent (if it supports it)
                if hasattr(self.agent, 'set_probe_states'):
                    self.agent.set_probe_states(probe_states_array.tolist())
                    self.logger.info(f"Set {len(probe_states)} probe states for diagnostics")
                    
                # Save probe states to disk
                if self.diagnostics_folder is not None:
                    probes_dir = os.path.join(self.diagnostics_folder, "probes")
                    os.makedirs(probes_dir, exist_ok=True)
                    probe_path = os.path.join(probes_dir, "probe_states.npz")
                    np.savez(probe_path, probe_states=probe_states_array)
                    self.logger.info(f"Saved probe states to {probe_path}")
                    
                self._probe_states_set = True
                
        except Exception as e:
            self.logger.warning(f"Failed to capture probe states: {e}")

    def _maybe_save_policy_snapshot(self):
        """
        Save policy snapshot if we've reached the next snapshot step.
        
        FIXED: Uses a while loop to catch up on all missed snapshots when the
        eval cycle spans multiple snapshot intervals.
        """
        if self.diagnostics_folder is None:
            return
            
        current_step = self.step_counter.exploration
        
        # Save all missed snapshots (in case eval cycle spans multiple snapshot intervals)
        while current_step >= self._next_snapshot_step:
            try:
                snapshots_dir = os.path.join(self.diagnostics_folder, "policy_snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)
                
                # Use the scheduled snapshot step, not current step, for consistent naming
                snapshot_step = self._next_snapshot_step
                snapshot_path = os.path.join(snapshots_dir, f"policy_{snapshot_step}.pt")
                
                if hasattr(self.agent, 'save_policy_snapshot'):
                    self.agent.save_policy_snapshot(snapshot_path)
                    self.logger.info(f"Saved policy snapshot at scheduled step {snapshot_step} (current={current_step})")
                    
            except Exception as e:
                self.logger.warning(f"Failed to save policy snapshot at step {self._next_snapshot_step}: {e}")
                
            self._next_snapshot_step += self.policy_snapshot_every_steps
        
        # Flush diagnostics logs after all snapshots are saved
        if hasattr(self.agent, 'flush_diagnostics'):
            try:
                self.agent.flush_diagnostics()
            except Exception as e:
                self.logger.warning(f"Failed to flush diagnostics: {e}")

    def _log_episode_summaries(self, episodes: List):
        """
        Log episode summaries to episode_summary.jsonl with explore_step and update_step.
        
        This directly solves the timestamp correlation problem by providing exact
        step counters at episode completion time.
        """
        if self._episode_summary_file is None:
            return
            
        try:
            # Fix #4: Use per-episode step counters stamped by the worker at
            # completion time. Fall back to current shared counter for backward
            # compatibility with Episode objects that don't carry step snapshots.
            fallback_explore = self.step_counter.exploration
            fallback_update = self.step_counter.update

            with open(self._episode_summary_file, "a", encoding="utf-8") as f:
                for i, episode in enumerate(episodes):
                    try:
                        wall_time = time.time()  # Per-episode timestamp
                        # Use per-episode step snapshot if available, else fallback
                        explore_step = getattr(episode, 'explore_step_at_completion', None) or fallback_explore
                        update_step = getattr(episode, 'update_step_at_completion', None) or fallback_update
                        # Extract episode data
                        total_reward = getattr(episode, 'episode_reward', 0.0)
                        steps = len(getattr(episode, 'rewards', []))
                        terminated = getattr(episode, 'terminals', [False])[-1] if getattr(episode, 'terminals', []) else False
                        truncated = getattr(episode, 'truncations', [False])[-1] if getattr(episode, 'truncations', []) else False

                        # Try to get max_insertion from episode info if available
                        max_insertion = None
                        infos = getattr(episode, 'infos', [])
                        if infos:
                            last_info = infos[-1] if infos else {}
                            # Look for insertion depth in common info keys
                            for key in ['max_insertion', 'inserted_length', 'insertion_depth']:
                                if key in last_info:
                                    val = last_info[key]
                                    if hasattr(val, '__iter__') and not isinstance(val, str):
                                        max_insertion = float(max(val))
                                    else:
                                        max_insertion = float(val)
                                    break

                        # Fix #5: Monotonically increasing counter for episode_id.
                        # Replaces retroactive calculation that assumed sequential,
                        # non-interleaved episodes.
                        episode_id = self._episode_summary_counter
                        self._episode_summary_counter += 1

                        summary = {
                            "wall_time": wall_time,
                            "explore_step": explore_step,
                            "update_step": update_step,
                            "episode_id": episode_id,
                            "total_reward": float(total_reward),
                            "steps": steps,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                        if max_insertion is not None:
                            summary["max_insertion"] = max_insertion
                        
                        f.write(json.dumps(summary) + "\n")
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to log episode {i} summary: {e}")
                        
                f.flush()
                
        except Exception as e:
            self.logger.warning(f"Failed to log episode summaries: {e}")

```

```python
from typing import List, Optional
from math import inf
import csv
import json
import logging
import os
import time
import numpy as np
from ..util import EveRLObject
from ..agent.agent import Agent, StepCounter, EpisodeCounter


class Runner(EveRLObject):
    def __init__(
        self,
        agent: Agent,
        heatup_action_low: List[float],
        heatup_action_high: List[float],
        agent_parameter_for_result_file: dict,
        checkpoint_folder: str,
        results_file: str,
        quality_info: Optional[str] = None,
        info_results: Optional[List[str]] = None,
        diagnostics_folder: Optional[str] = None,
        policy_snapshot_every_steps: int = 10000,
        n_probe_episodes: int = 3,
        n_probe_near_start_steps: int = 10,
    ) -> None:
        self.agent = agent
        self.heatup_action_low = heatup_action_low
        self.heatup_action_high = heatup_action_high
        self.agent_parameter_for_result_file = agent_parameter_for_result_file
        self.checkpoint_folder = checkpoint_folder
        self.results_file = results_file
        self.quality_info = quality_info
        self.info_results = info_results or []
        self.logger = logging.getLogger(self.__module__)
        
        # Diagnostics settings
        self.diagnostics_folder = diagnostics_folder
        # Ensure worker step logs directory exists.
        # NOTE: STEP_LOG_DIR must be set BEFORE agent creation (which spawns workers).
        # The training script should set os.environ["STEP_LOG_DIR"] before creating the agent.
        # This setdefault is a fallback but will NOT help if workers are already spawned.
        if self.diagnostics_folder is not None:
            logs_subprocesses = os.path.join(self.diagnostics_folder, "logs_subprocesses")
            os.makedirs(logs_subprocesses, exist_ok=True)
            os.environ.setdefault("STEP_LOG_DIR", logs_subprocesses)
        self.policy_snapshot_every_steps = policy_snapshot_every_steps
        self.n_probe_episodes = n_probe_episodes
        self.n_probe_near_start_steps = n_probe_near_start_steps
        self._next_snapshot_step = policy_snapshot_every_steps
        self._probe_states_set = False
        self._eval_count = 0
        self._replay_save_interval = 4  # save replay buffer every N evals
        
        # Episode summary logger
        self._episode_summary_file = None
        self._episode_summary_logger = None
        self._episode_summary_counter = 0  # Fix #5: monotonic episode_id
        if self.diagnostics_folder is not None:
            csv_folder = os.path.join(self.diagnostics_folder, "csv")
            os.makedirs(csv_folder, exist_ok=True)
            self._episode_summary_file = os.path.join(csv_folder, "episode_summary.jsonl")
            self.logger.info(f"Episode summary will be logged to {self._episode_summary_file}")

        self._results = {
            "episodes explore": 0,
            "steps explore": 0,
        }
        self._results["quality"] = 0.0
        for info_result in self.info_results:
            self._results[info_result] = 0.0
        self._results["reward"] = 0.0
        self._results["best quality"] = 0.0
        self._results["best explore steps"] = 0.0

        file_existed = os.path.isfile(results_file)

        with open(results_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_existed:
                writer.writerow(
                    list(self._results.keys())
                    + [" "]
                    + list(agent_parameter_for_result_file.keys())
                )
                writer.writerow(
                    [" "] * (len(self._results.values()) + 1)
                    + list(agent_parameter_for_result_file.values())
                )

        self.best_eval = {"steps": 0, "quality": -inf}

    @property
    def step_counter(self) -> StepCounter:
        return self.agent.step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self.agent.episode_counter

    def restore_runner_state(self, checkpoint_path: str, probe_states_dir: Optional[str] = None):
        """Restore Runner-level state from a checkpoint for resume support.

        Restores: best_eval, _episode_summary_counter, _next_snapshot_step,
        and optionally probe states from a saved .npz file.
        """
        import torch
        checkpoint = torch.load(checkpoint_path)
        additional_info = checkpoint.get("additional_info", {})
        runner_state = additional_info.get("runner_state") if additional_info else None

        if runner_state:
            self.best_eval = runner_state["best_eval"]
            self._episode_summary_counter = runner_state["episode_summary_counter"]
            self._next_snapshot_step = runner_state["next_snapshot_step"]
            self._eval_count = runner_state.get("eval_count", 0)
            self.logger.info(
                f"Restored runner state: best_eval={self.best_eval}, "
                f"episode_summary_counter={self._episode_summary_counter}, "
                f"next_snapshot_step={self._next_snapshot_step}, "
                f"eval_count={self._eval_count}"
            )
        else:
            # Checkpoint predates runner_state support — compute safe defaults
            explore_steps = self.step_counter.exploration
            self._next_snapshot_step = (
                (explore_steps // self.policy_snapshot_every_steps) + 1
            ) * self.policy_snapshot_every_steps
            self.logger.warning(
                f"No runner_state in checkpoint. Setting _next_snapshot_step={self._next_snapshot_step}"
            )

        # Restore probe states from saved .npz if available
        if probe_states_dir is not None:
            probe_path = os.path.join(probe_states_dir, "probes", "probe_states.npz")
            if os.path.isfile(probe_path):
                try:
                    data = np.load(probe_path)
                    probe_states = data["probe_states"]
                    if hasattr(self.agent, 'set_probe_states'):
                        self.agent.set_probe_states(probe_states.tolist())
                        self._probe_states_set = True
                        self.logger.info(f"Restored {len(probe_states)} probe states from {probe_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to restore probe states: {e}")

    def heatup(self, steps: int):
        episodes = self.agent.heatup(
            steps=steps,
            custom_action_low=self.heatup_action_low,
            custom_action_high=self.heatup_action_high,
        )
        return episodes

    def explore(self, n_episodes: int):
        self.agent.explore(episodes=n_episodes)

    def update(self, n_steps: int):
        self.agent.update(n_steps)

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[List[int]] = None
    ):
        explore_steps = self.step_counter.exploration
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{explore_steps}.everl"
        )
        result_episodes = self.agent.evaluate(episodes=episodes, seeds=seeds)
        qualities, rewards = [], []
        results_for_info = {
            info_result_name: [] for info_result_name in self.info_results
        }
        eval_results = {"episodes": []}
        for episode in result_episodes:
            reward = episode.episode_reward
            quality = (
                episode.infos[-1][self.quality_info]
                if self.quality_info is not None
                else reward
            )
            qualities.append(quality)
            rewards.append(reward)
            for info_result_name in self.info_results:
                info = episode.infos[-1][info_result_name]
                results_for_info[info_result_name].append(info)
            eval_results["episodes"].append(
                {
                    "seed": episode.seed,
                    "options": episode.options,
                    "quality": episode.infos[-1][self.quality_info],
                }
            )

        reward = sum(rewards) / len(rewards)
        quality = sum(qualities) / len(qualities)
        for info_result_name, results in results_for_info.items():
            result = sum(results) / len(results)
            self._results[info_result_name] = round(result, 3)
        save_best = False
        if quality > self.best_eval["quality"]:
            save_best = True
            self.best_eval["quality"] = quality
            self.best_eval["steps"] = explore_steps

        self._results["episodes explore"] = self.episode_counter.exploration
        self._results["steps explore"] = explore_steps
        self._results["reward"] = round(reward, 3)
        self._results["quality"] = round(quality, 3)
        self._results["best quality"] = self.best_eval["quality"]
        self._results["best explore steps"] = self.best_eval["steps"]

        eval_results.update(self._results)
        eval_results.pop("best quality")
        eval_results.pop("best explore steps")

        # Include runner state for resume support
        eval_results["runner_state"] = {
            "best_eval": self.best_eval.copy(),
            "episode_summary_counter": self._episode_summary_counter,
            "next_snapshot_step": self._next_snapshot_step,
        }

        self.agent.save_checkpoint(checkpoint_file, eval_results)
        if save_best:
            checkpoint_file = os.path.join(
                self.checkpoint_folder, "best_checkpoint.everl"
            )
            self.agent.save_checkpoint(checkpoint_file, eval_results)

        log_info = (
            f"Quality: {quality}, Reward: {reward}, Exploration steps: {explore_steps}"
        )
        self.logger.info(log_info)
        with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(self._results.values())

        return quality, reward

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: int = None,
        explore_steps_limit: int = None,
    ):
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        while self.step_counter.exploration < explore_steps_limit:
            # Total experience = heatup + exploration (both contribute to replay buffer)
            total_experience_steps = (
                self.step_counter.heatup + self.step_counter.exploration
            )
            update_steps = (
                total_experience_steps * update_steps_per_explore_step
                - self.step_counter.update
            )
            result = self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )
            
            # Log episode summaries if we got results
            if result is not None:
                episodes, _ = result if isinstance(result, tuple) else (result, None)
                if episodes:
                    self._log_episode_summaries(episodes)
            
            # Check for policy snapshots after each exploration cycle
            # This ensures snapshots are saved at the actual milestone with correct weights
            self._maybe_save_policy_snapshot()

    def training_run(
        self,
        heatup_steps: int,
        training_steps: int,
        explore_steps_between_eval: int,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
    ):
        # TODO: Log Training Run Infos
        heatup_episodes = self.heatup(heatup_steps)
        
        # Capture and set probe states after heatup
        self._capture_and_set_probe_states(heatup_episodes)
        
        next_eval_step_limt = (
            self.agent.step_counter.exploration + explore_steps_between_eval
        )
        while self.agent.step_counter.exploration < training_steps:
            self.explore_and_update(
                explore_episodes_between_updates,
                update_steps_per_explore_step,
                explore_steps_limit=next_eval_step_limt,
            )
            
            # Note: policy snapshots are now saved inside explore_and_update()
            # at the actual milestone steps with correct weights
            
            quality, reward = self.eval(episodes=eval_episodes, seeds=eval_seeds)
            next_eval_step_limt += explore_steps_between_eval

        return quality, reward

    def _capture_and_set_probe_states(self, episodes: List):
        """Capture probe states from heatup episodes and set them in trainer."""
        if not episodes or self._probe_states_set:
            return
            
        try:
            probe_states = []
            n_episodes = min(len(episodes), self.n_probe_episodes)
            
            for ep_idx in range(n_episodes):
                episode = episodes[ep_idx]
                if hasattr(episode, 'flat_obs') and len(episode.flat_obs) > 0:
                    # Collect start state
                    probe_states.append(episode.flat_obs[0])
                    
                    # Collect near-start states
                    n_states = min(self.n_probe_near_start_steps, len(episode.flat_obs) - 1)
                    for step_idx in range(1, n_states + 1):
                        probe_states.append(episode.flat_obs[step_idx])
                        
            if probe_states:
                probe_states_array = np.array(probe_states)
                
                # Set probe states in agent (if it supports it)
                if hasattr(self.agent, 'set_probe_states'):
                    self.agent.set_probe_states(probe_states_array.tolist())
                    self.logger.info(f"Set {len(probe_states)} probe states for diagnostics")
                    
                # Save probe states to disk
                if self.diagnostics_folder is not None:
                    probes_dir = os.path.join(self.diagnostics_folder, "probes")
                    os.makedirs(probes_dir, exist_ok=True)
                    probe_path = os.path.join(probes_dir, "probe_states.npz")
                    np.savez(probe_path, probe_states=probe_states_array)
                    self.logger.info(f"Saved probe states to {probe_path}")
                    
                self._probe_states_set = True
                
        except Exception as e:
            self.logger.warning(f"Failed to capture probe states: {e}")

    def _maybe_save_policy_snapshot(self):
        """
        Save policy snapshot if we've reached the next snapshot step.
        
        FIXED: Uses a while loop to catch up on all missed snapshots when the
        eval cycle spans multiple snapshot intervals.
        """
        if self.diagnostics_folder is None:
            return
            
        current_step = self.step_counter.exploration
        
        # Save all missed snapshots (in case eval cycle spans multiple snapshot intervals)
        while current_step >= self._next_snapshot_step:
            try:
                snapshots_dir = os.path.join(self.diagnostics_folder, "policy_snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)
                
                # Use the scheduled snapshot step, not current step, for consistent naming
                snapshot_step = self._next_snapshot_step
                snapshot_path = os.path.join(snapshots_dir, f"policy_{snapshot_step}.pt")
                
                if hasattr(self.agent, 'save_policy_snapshot'):
                    self.agent.save_policy_snapshot(snapshot_path)
                    self.logger.info(f"Saved policy snapshot at scheduled step {snapshot_step} (current={current_step})")
                    
            except Exception as e:
                self.logger.warning(f"Failed to save policy snapshot at step {self._next_snapshot_step}: {e}")
                
            self._next_snapshot_step += self.policy_snapshot_every_steps
        
        # Flush diagnostics logs after all snapshots are saved
        if hasattr(self.agent, 'flush_diagnostics'):
            try:
                self.agent.flush_diagnostics()
            except Exception as e:
                self.logger.warning(f"Failed to flush diagnostics: {e}")

    def _log_episode_summaries(self, episodes: List):
        """
        Log episode summaries to episode_summary.jsonl with explore_step and update_step.
        
        This directly solves the timestamp correlation problem by providing exact
        step counters at episode completion time.
        """
        if self._episode_summary_file is None:
            return
            
        try:
            # Fix #4: Use per-episode step counters stamped by the worker at
            # completion time. Fall back to current shared counter for backward
            # compatibility with Episode objects that don't carry step snapshots.
            fallback_explore = self.step_counter.exploration
            fallback_update = self.step_counter.update

            with open(self._episode_summary_file, "a", encoding="utf-8") as f:
                for i, episode in enumerate(episodes):
                    try:
                        wall_time = time.time()  # Per-episode timestamp
                        # Use per-episode step snapshot if available, else fallback
                        explore_step = getattr(episode, 'explore_step_at_completion', None) or fallback_explore
                        update_step = getattr(episode, 'update_step_at_completion', None) or fallback_update
                        # Extract episode data
                        total_reward = getattr(episode, 'episode_reward', 0.0)
                        steps = len(getattr(episode, 'rewards', []))
                        terminated = getattr(episode, 'terminals', [False])[-1] if getattr(episode, 'terminals', []) else False
                        truncated = getattr(episode, 'truncations', [False])[-1] if getattr(episode, 'truncations', []) else False

                        # Try to get max_insertion from episode info if available
                        max_insertion = None
                        infos = getattr(episode, 'infos', [])
                        if infos:
                            last_info = infos[-1] if infos else {}
                            # Look for insertion depth in common info keys
                            for key in ['max_insertion', 'inserted_length', 'insertion_depth']:
                                if key in last_info:
                                    val = last_info[key]
                                    if hasattr(val, '__iter__') and not isinstance(val, str):
                                        max_insertion = float(max(val))
                                    else:
                                        max_insertion = float(val)
                                    break

                        # Fix #5: Monotonically increasing counter for episode_id.
                        # Replaces retroactive calculation that assumed sequential,
                        # non-interleaved episodes.
                        episode_id = self._episode_summary_counter
                        self._episode_summary_counter += 1

                        summary = {
                            "wall_time": wall_time,
                            "explore_step": explore_step,
                            "update_step": update_step,
                            "episode_id": episode_id,
                            "total_reward": float(total_reward),
                            "steps": steps,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                        if max_insertion is not None:
                            summary["max_insertion"] = max_insertion
                        
                        f.write(json.dumps(summary) + "\n")
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to log episode {i} summary: {e}")
                        
                f.flush()
                
        except Exception as e:
            self.logger.warning(f"Failed to log episode summaries: {e}")

```

```python
from typing import List, Optional
from math import inf
import csv
import json
import logging
import os
import time
import numpy as np
from ..util import EveRLObject
from ..agent.agent import Agent, StepCounter, EpisodeCounter


class Runner(EveRLObject):
    def __init__(
        self,
        agent: Agent,
        heatup_action_low: List[float],
        heatup_action_high: List[float],
        agent_parameter_for_result_file: dict,
        checkpoint_folder: str,
        results_file: str,
        quality_info: Optional[str] = None,
        info_results: Optional[List[str]] = None,
        diagnostics_folder: Optional[str] = None,
        policy_snapshot_every_steps: int = 10000,
        n_probe_episodes: int = 3,
        n_probe_near_start_steps: int = 10,
    ) -> None:
        self.agent = agent
        self.heatup_action_low = heatup_action_low
        self.heatup_action_high = heatup_action_high
        self.agent_parameter_for_result_file = agent_parameter_for_result_file
        self.checkpoint_folder = checkpoint_folder
        self.results_file = results_file
        self.quality_info = quality_info
        self.info_results = info_results or []
        self.logger = logging.getLogger(self.__module__)
        
        # Diagnostics settings
        self.diagnostics_folder = diagnostics_folder
        # Ensure worker step logs directory exists.
        # NOTE: STEP_LOG_DIR must be set BEFORE agent creation (which spawns workers).
        # The training script should set os.environ["STEP_LOG_DIR"] before creating the agent.
        # This setdefault is a fallback but will NOT help if workers are already spawned.
        if self.diagnostics_folder is not None:
            logs_subprocesses = os.path.join(self.diagnostics_folder, "logs_subprocesses")
            os.makedirs(logs_subprocesses, exist_ok=True)
            os.environ.setdefault("STEP_LOG_DIR", logs_subprocesses)
        self.policy_snapshot_every_steps = policy_snapshot_every_steps
        self.n_probe_episodes = n_probe_episodes
        self.n_probe_near_start_steps = n_probe_near_start_steps
        self._next_snapshot_step = policy_snapshot_every_steps
        self._probe_states_set = False
        self._eval_count = 0
        self._replay_save_interval = 4  # save replay buffer every N evals
        
        # Episode summary logger
        self._episode_summary_file = None
        self._episode_summary_logger = None
        self._episode_summary_counter = 0  # Fix #5: monotonic episode_id
        if self.diagnostics_folder is not None:
            csv_folder = os.path.join(self.diagnostics_folder, "csv")
            os.makedirs(csv_folder, exist_ok=True)
            self._episode_summary_file = os.path.join(csv_folder, "episode_summary.jsonl")
            self.logger.info(f"Episode summary will be logged to {self._episode_summary_file}")

        self._results = {
            "episodes explore": 0,
            "steps explore": 0,
        }
        self._results["quality"] = 0.0
        for info_result in self.info_results:
            self._results[info_result] = 0.0
        self._results["reward"] = 0.0
        self._results["best quality"] = 0.0
        self._results["best explore steps"] = 0.0

        file_existed = os.path.isfile(results_file)

        with open(results_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_existed:
                writer.writerow(
                    list(self._results.keys())
                    + [" "]
                    + list(agent_parameter_for_result_file.keys())
                )
                writer.writerow(
                    [" "] * (len(self._results.values()) + 1)
                    + list(agent_parameter_for_result_file.values())
                )

        self.best_eval = {"steps": 0, "quality": -inf}

    @property
    def step_counter(self) -> StepCounter:
        return self.agent.step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self.agent.episode_counter

    def restore_runner_state(self, checkpoint_path: str, probe_states_dir: Optional[str] = None):
        """Restore Runner-level state from a checkpoint for resume support.

        Restores: best_eval, _episode_summary_counter, _next_snapshot_step,
        and optionally probe states from a saved .npz file.
        """
        import torch
        checkpoint = torch.load(checkpoint_path)
        additional_info = checkpoint.get("additional_info", {})
        runner_state = additional_info.get("runner_state") if additional_info else None

        if runner_state:
            self.best_eval = runner_state["best_eval"]
            self._episode_summary_counter = runner_state["episode_summary_counter"]
            self._next_snapshot_step = runner_state["next_snapshot_step"]
            self._eval_count = runner_state.get("eval_count", 0)
            self.logger.info(
                f"Restored runner state: best_eval={self.best_eval}, "
                f"episode_summary_counter={self._episode_summary_counter}, "
                f"next_snapshot_step={self._next_snapshot_step}, "
                f"eval_count={self._eval_count}"
            )
        else:
            # Checkpoint predates runner_state support — compute safe defaults
            explore_steps = self.step_counter.exploration
            self._next_snapshot_step = (
                (explore_steps // self.policy_snapshot_every_steps) + 1
            ) * self.policy_snapshot_every_steps
            self.logger.warning(
                f"No runner_state in checkpoint. Setting _next_snapshot_step={self._next_snapshot_step}"
            )

        # Restore probe states from saved .npz if available
        if probe_states_dir is not None:
            probe_path = os.path.join(probe_states_dir, "probes", "probe_states.npz")
            if os.path.isfile(probe_path):
                try:
                    data = np.load(probe_path)
                    probe_states = data["probe_states"]
                    if hasattr(self.agent, 'set_probe_states'):
                        self.agent.set_probe_states(probe_states.tolist())
                        self._probe_states_set = True
                        self.logger.info(f"Restored {len(probe_states)} probe states from {probe_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to restore probe states: {e}")

    def load_replay_buffer(self):
        """Auto-load replay buffer from checkpoint folder if file exists."""
        replay_path = os.path.join(self.checkpoint_folder, "replay_buffer.npz")
        if os.path.isfile(replay_path):
            try:
                n = self.agent.replay_buffer.load_buffer_from_file(replay_path)
                self.logger.info(f"Loaded replay buffer ({n} episodes) from {replay_path}")
                return n
            except Exception as e:
                self.logger.warning(f"Failed to load replay buffer: {e}")
        return 0

    def heatup(self, steps: int):
        episodes = self.agent.heatup(
            steps=steps,
            custom_action_low=self.heatup_action_low,
            custom_action_high=self.heatup_action_high,
        )
        return episodes

    def explore(self, n_episodes: int):
        self.agent.explore(episodes=n_episodes)

    def update(self, n_steps: int):
        self.agent.update(n_steps)

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[List[int]] = None
    ):
        explore_steps = self.step_counter.exploration
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{explore_steps}.everl"
        )
        result_episodes = self.agent.evaluate(episodes=episodes, seeds=seeds)
        qualities, rewards = [], []
        results_for_info = {
            info_result_name: [] for info_result_name in self.info_results
        }
        eval_results = {"episodes": []}
        for episode in result_episodes:
            reward = episode.episode_reward
            quality = (
                episode.infos[-1][self.quality_info]
                if self.quality_info is not None
                else reward
            )
            qualities.append(quality)
            rewards.append(reward)
            for info_result_name in self.info_results:
                info = episode.infos[-1][info_result_name]
                results_for_info[info_result_name].append(info)
            eval_results["episodes"].append(
                {
                    "seed": episode.seed,
                    "options": episode.options,
                    "quality": episode.infos[-1][self.quality_info],
                }
            )

        reward = sum(rewards) / len(rewards)
        quality = sum(qualities) / len(qualities)
        for info_result_name, results in results_for_info.items():
            result = sum(results) / len(results)
            self._results[info_result_name] = round(result, 3)
        save_best = False
        if quality > self.best_eval["quality"]:
            save_best = True
            self.best_eval["quality"] = quality
            self.best_eval["steps"] = explore_steps

        self._results["episodes explore"] = self.episode_counter.exploration
        self._results["steps explore"] = explore_steps
        self._results["reward"] = round(reward, 3)
        self._results["quality"] = round(quality, 3)
        self._results["best quality"] = self.best_eval["quality"]
        self._results["best explore steps"] = self.best_eval["steps"]

        eval_results.update(self._results)
        eval_results.pop("best quality")
        eval_results.pop("best explore steps")

        # Include runner state for resume support
        eval_results["runner_state"] = {
            "best_eval": self.best_eval.copy(),
            "episode_summary_counter": self._episode_summary_counter,
            "next_snapshot_step": self._next_snapshot_step,
        }

        self.agent.save_checkpoint(checkpoint_file, eval_results)
        if save_best:
            checkpoint_file = os.path.join(
                self.checkpoint_folder, "best_checkpoint.everl"
            )
            self.agent.save_checkpoint(checkpoint_file, eval_results)

        log_info = (
            f"Quality: {quality}, Reward: {reward}, Exploration steps: {explore_steps}"
        )
        self.logger.info(log_info)
        with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(self._results.values())

        return quality, reward

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: int = None,
        explore_steps_limit: int = None,
    ):
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        while self.step_counter.exploration < explore_steps_limit:
            # Total experience = heatup + exploration (both contribute to replay buffer)
            total_experience_steps = (
                self.step_counter.heatup + self.step_counter.exploration
            )
            update_steps = (
                total_experience_steps * update_steps_per_explore_step
                - self.step_counter.update
            )
            result = self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )
            
            # Log episode summaries if we got results
            if result is not None:
                episodes, _ = result if isinstance(result, tuple) else (result, None)
                if episodes:
                    self._log_episode_summaries(episodes)
            
            # Check for policy snapshots after each exploration cycle
            # This ensures snapshots are saved at the actual milestone with correct weights
            self._maybe_save_policy_snapshot()

    def training_run(
        self,
        heatup_steps: int,
        training_steps: int,
        explore_steps_between_eval: int,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
    ):
        # TODO: Log Training Run Infos
        heatup_episodes = self.heatup(heatup_steps)
        
        # Capture and set probe states after heatup
        self._capture_and_set_probe_states(heatup_episodes)
        
        next_eval_step_limt = (
            self.agent.step_counter.exploration + explore_steps_between_eval
        )
        while self.agent.step_counter.exploration < training_steps:
            self.explore_and_update(
                explore_episodes_between_updates,
                update_steps_per_explore_step,
                explore_steps_limit=next_eval_step_limt,
            )
            
            # Note: policy snapshots are now saved inside explore_and_update()
            # at the actual milestone steps with correct weights
            
            quality, reward = self.eval(episodes=eval_episodes, seeds=eval_seeds)
            next_eval_step_limt += explore_steps_between_eval

        return quality, reward

    def _capture_and_set_probe_states(self, episodes: List):
        """Capture probe states from heatup episodes and set them in trainer."""
        if not episodes or self._probe_states_set:
            return
            
        try:
            probe_states = []
            n_episodes = min(len(episodes), self.n_probe_episodes)
            
            for ep_idx in range(n_episodes):
                episode = episodes[ep_idx]
                if hasattr(episode, 'flat_obs') and len(episode.flat_obs) > 0:
                    # Collect start state
                    probe_states.append(episode.flat_obs[0])
                    
                    # Collect near-start states
                    n_states = min(self.n_probe_near_start_steps, len(episode.flat_obs) - 1)
                    for step_idx in range(1, n_states + 1):
                        probe_states.append(episode.flat_obs[step_idx])
                        
            if probe_states:
                probe_states_array = np.array(probe_states)
                
                # Set probe states in agent (if it supports it)
                if hasattr(self.agent, 'set_probe_states'):
                    self.agent.set_probe_states(probe_states_array.tolist())
                    self.logger.info(f"Set {len(probe_states)} probe states for diagnostics")
                    
                # Save probe states to disk
                if self.diagnostics_folder is not None:
                    probes_dir = os.path.join(self.diagnostics_folder, "probes")
                    os.makedirs(probes_dir, exist_ok=True)
                    probe_path = os.path.join(probes_dir, "probe_states.npz")
                    np.savez(probe_path, probe_states=probe_states_array)
                    self.logger.info(f"Saved probe states to {probe_path}")
                    
                self._probe_states_set = True
                
        except Exception as e:
            self.logger.warning(f"Failed to capture probe states: {e}")

    def _maybe_save_policy_snapshot(self):
        """
        Save policy snapshot if we've reached the next snapshot step.
        
        FIXED: Uses a while loop to catch up on all missed snapshots when the
        eval cycle spans multiple snapshot intervals.
        """
        if self.diagnostics_folder is None:
            return
            
        current_step = self.step_counter.exploration
        
        # Save all missed snapshots (in case eval cycle spans multiple snapshot intervals)
        while current_step >= self._next_snapshot_step:
            try:
                snapshots_dir = os.path.join(self.diagnostics_folder, "policy_snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)
                
                # Use the scheduled snapshot step, not current step, for consistent naming
                snapshot_step = self._next_snapshot_step
                snapshot_path = os.path.join(snapshots_dir, f"policy_{snapshot_step}.pt")
                
                if hasattr(self.agent, 'save_policy_snapshot'):
                    self.agent.save_policy_snapshot(snapshot_path)
                    self.logger.info(f"Saved policy snapshot at scheduled step {snapshot_step} (current={current_step})")
                    
            except Exception as e:
                self.logger.warning(f"Failed to save policy snapshot at step {self._next_snapshot_step}: {e}")
                
            self._next_snapshot_step += self.policy_snapshot_every_steps
        
        # Flush diagnostics logs after all snapshots are saved
        if hasattr(self.agent, 'flush_diagnostics'):
            try:
                self.agent.flush_diagnostics()
            except Exception as e:
                self.logger.warning(f"Failed to flush diagnostics: {e}")

    def _log_episode_summaries(self, episodes: List):
        """
        Log episode summaries to episode_summary.jsonl with explore_step and update_step.
        
        This directly solves the timestamp correlation problem by providing exact
        step counters at episode completion time.
        """
        if self._episode_summary_file is None:
            return
            
        try:
            # Fix #4: Use per-episode step counters stamped by the worker at
            # completion time. Fall back to current shared counter for backward
            # compatibility with Episode objects that don't carry step snapshots.
            fallback_explore = self.step_counter.exploration
            fallback_update = self.step_counter.update

            with open(self._episode_summary_file, "a", encoding="utf-8") as f:
                for i, episode in enumerate(episodes):
                    try:
                        wall_time = time.time()  # Per-episode timestamp
                        # Use per-episode step snapshot if available, else fallback
                        explore_step = getattr(episode, 'explore_step_at_completion', None) or fallback_explore
                        update_step = getattr(episode, 'update_step_at_completion', None) or fallback_update
                        # Extract episode data
                        total_reward = getattr(episode, 'episode_reward', 0.0)
                        steps = len(getattr(episode, 'rewards', []))
                        terminated = getattr(episode, 'terminals', [False])[-1] if getattr(episode, 'terminals', []) else False
                        truncated = getattr(episode, 'truncations', [False])[-1] if getattr(episode, 'truncations', []) else False

                        # Try to get max_insertion from episode info if available
                        max_insertion = None
                        infos = getattr(episode, 'infos', [])
                        if infos:
                            last_info = infos[-1] if infos else {}
                            # Look for insertion depth in common info keys
                            for key in ['max_insertion', 'inserted_length', 'insertion_depth']:
                                if key in last_info:
                                    val = last_info[key]
                                    if hasattr(val, '__iter__') and not isinstance(val, str):
                                        max_insertion = float(max(val))
                                    else:
                                        max_insertion = float(val)
                                    break

                        # Fix #5: Monotonically increasing counter for episode_id.
                        # Replaces retroactive calculation that assumed sequential,
                        # non-interleaved episodes.
                        episode_id = self._episode_summary_counter
                        self._episode_summary_counter += 1

                        summary = {
                            "wall_time": wall_time,
                            "explore_step": explore_step,
                            "update_step": update_step,
                            "episode_id": episode_id,
                            "total_reward": float(total_reward),
                            "steps": steps,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                        if max_insertion is not None:
                            summary["max_insertion"] = max_insertion
                        
                        f.write(json.dumps(summary) + "\n")
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to log episode {i} summary: {e}")
                        
                f.flush()
                
        except Exception as e:
            self.logger.warning(f"Failed to log episode summaries: {e}")

```

```python
from typing import List, Optional
from math import inf
import csv
import json
import logging
import os
import time
import numpy as np
from ..util import EveRLObject
from ..agent.agent import Agent, StepCounter, EpisodeCounter


class Runner(EveRLObject):
    def __init__(
        self,
        agent: Agent,
        heatup_action_low: List[float],
        heatup_action_high: List[float],
        agent_parameter_for_result_file: dict,
        checkpoint_folder: str,
        results_file: str,
        quality_info: Optional[str] = None,
        info_results: Optional[List[str]] = None,
        diagnostics_folder: Optional[str] = None,
        policy_snapshot_every_steps: int = 10000,
        n_probe_episodes: int = 3,
        n_probe_near_start_steps: int = 10,
    ) -> None:
        self.agent = agent
        self.heatup_action_low = heatup_action_low
        self.heatup_action_high = heatup_action_high
        self.agent_parameter_for_result_file = agent_parameter_for_result_file
        self.checkpoint_folder = checkpoint_folder
        self.results_file = results_file
        self.quality_info = quality_info
        self.info_results = info_results or []
        self.logger = logging.getLogger(self.__module__)
        
        # Diagnostics settings
        self.diagnostics_folder = diagnostics_folder
        # Ensure worker step logs directory exists.
        # NOTE: STEP_LOG_DIR must be set BEFORE agent creation (which spawns workers).
        # The training script should set os.environ["STEP_LOG_DIR"] before creating the agent.
        # This setdefault is a fallback but will NOT help if workers are already spawned.
        if self.diagnostics_folder is not None:
            logs_subprocesses = os.path.join(self.diagnostics_folder, "logs_subprocesses")
            os.makedirs(logs_subprocesses, exist_ok=True)
            os.environ.setdefault("STEP_LOG_DIR", logs_subprocesses)
        self.policy_snapshot_every_steps = policy_snapshot_every_steps
        self.n_probe_episodes = n_probe_episodes
        self.n_probe_near_start_steps = n_probe_near_start_steps
        self._next_snapshot_step = policy_snapshot_every_steps
        self._probe_states_set = False
        self._eval_count = 0
        self._replay_save_interval = 4  # save replay buffer every N evals
        
        # Episode summary logger
        self._episode_summary_file = None
        self._episode_summary_logger = None
        self._episode_summary_counter = 0  # Fix #5: monotonic episode_id
        if self.diagnostics_folder is not None:
            csv_folder = os.path.join(self.diagnostics_folder, "csv")
            os.makedirs(csv_folder, exist_ok=True)
            self._episode_summary_file = os.path.join(csv_folder, "episode_summary.jsonl")
            self.logger.info(f"Episode summary will be logged to {self._episode_summary_file}")

        self._results = {
            "episodes explore": 0,
            "steps explore": 0,
        }
        self._results["quality"] = 0.0
        for info_result in self.info_results:
            self._results[info_result] = 0.0
        self._results["reward"] = 0.0
        self._results["best quality"] = 0.0
        self._results["best explore steps"] = 0.0

        file_existed = os.path.isfile(results_file)

        with open(results_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_existed:
                writer.writerow(
                    list(self._results.keys())
                    + [" "]
                    + list(agent_parameter_for_result_file.keys())
                )
                writer.writerow(
                    [" "] * (len(self._results.values()) + 1)
                    + list(agent_parameter_for_result_file.values())
                )

        self.best_eval = {"steps": 0, "quality": -inf}

    @property
    def step_counter(self) -> StepCounter:
        return self.agent.step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self.agent.episode_counter

    def restore_runner_state(self, checkpoint_path: str, probe_states_dir: Optional[str] = None):
        """Restore Runner-level state from a checkpoint for resume support.

        Restores: best_eval, _episode_summary_counter, _next_snapshot_step,
        and optionally probe states from a saved .npz file.
        """
        import torch
        checkpoint = torch.load(checkpoint_path)
        additional_info = checkpoint.get("additional_info", {})
        runner_state = additional_info.get("runner_state") if additional_info else None

        if runner_state:
            self.best_eval = runner_state["best_eval"]
            self._episode_summary_counter = runner_state["episode_summary_counter"]
            self._next_snapshot_step = runner_state["next_snapshot_step"]
            self._eval_count = runner_state.get("eval_count", 0)
            self.logger.info(
                f"Restored runner state: best_eval={self.best_eval}, "
                f"episode_summary_counter={self._episode_summary_counter}, "
                f"next_snapshot_step={self._next_snapshot_step}, "
                f"eval_count={self._eval_count}"
            )
        else:
            # Checkpoint predates runner_state support — compute safe defaults
            explore_steps = self.step_counter.exploration
            self._next_snapshot_step = (
                (explore_steps // self.policy_snapshot_every_steps) + 1
            ) * self.policy_snapshot_every_steps
            self.logger.warning(
                f"No runner_state in checkpoint. Setting _next_snapshot_step={self._next_snapshot_step}"
            )

        # Restore probe states from saved .npz if available
        if probe_states_dir is not None:
            probe_path = os.path.join(probe_states_dir, "probes", "probe_states.npz")
            if os.path.isfile(probe_path):
                try:
                    data = np.load(probe_path)
                    probe_states = data["probe_states"]
                    if hasattr(self.agent, 'set_probe_states'):
                        self.agent.set_probe_states(probe_states.tolist())
                        self._probe_states_set = True
                        self.logger.info(f"Restored {len(probe_states)} probe states from {probe_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to restore probe states: {e}")

    def load_replay_buffer(self):
        """Auto-load replay buffer from checkpoint folder if file exists."""
        replay_path = os.path.join(self.checkpoint_folder, "replay_buffer.npz")
        if os.path.isfile(replay_path):
            try:
                n = self.agent.replay_buffer.load_buffer_from_file(replay_path)
                self.logger.info(f"Loaded replay buffer ({n} episodes) from {replay_path}")
                return n
            except Exception as e:
                self.logger.warning(f"Failed to load replay buffer: {e}")
        return 0

    def heatup(self, steps: int):
        episodes = self.agent.heatup(
            steps=steps,
            custom_action_low=self.heatup_action_low,
            custom_action_high=self.heatup_action_high,
        )
        return episodes

    def explore(self, n_episodes: int):
        self.agent.explore(episodes=n_episodes)

    def update(self, n_steps: int):
        self.agent.update(n_steps)

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[List[int]] = None
    ):
        explore_steps = self.step_counter.exploration
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{explore_steps}.everl"
        )
        result_episodes = self.agent.evaluate(episodes=episodes, seeds=seeds)
        qualities, rewards = [], []
        results_for_info = {
            info_result_name: [] for info_result_name in self.info_results
        }
        eval_results = {"episodes": []}
        for episode in result_episodes:
            reward = episode.episode_reward
            quality = (
                episode.infos[-1][self.quality_info]
                if self.quality_info is not None
                else reward
            )
            qualities.append(quality)
            rewards.append(reward)
            for info_result_name in self.info_results:
                info = episode.infos[-1][info_result_name]
                results_for_info[info_result_name].append(info)
            eval_results["episodes"].append(
                {
                    "seed": episode.seed,
                    "options": episode.options,
                    "quality": episode.infos[-1][self.quality_info],
                }
            )

        reward = sum(rewards) / len(rewards)
        quality = sum(qualities) / len(qualities)
        for info_result_name, results in results_for_info.items():
            result = sum(results) / len(results)
            self._results[info_result_name] = round(result, 3)
        save_best = False
        if quality > self.best_eval["quality"]:
            save_best = True
            self.best_eval["quality"] = quality
            self.best_eval["steps"] = explore_steps

        self._results["episodes explore"] = self.episode_counter.exploration
        self._results["steps explore"] = explore_steps
        self._results["reward"] = round(reward, 3)
        self._results["quality"] = round(quality, 3)
        self._results["best quality"] = self.best_eval["quality"]
        self._results["best explore steps"] = self.best_eval["steps"]

        eval_results.update(self._results)
        eval_results.pop("best quality")
        eval_results.pop("best explore steps")

        self._eval_count += 1

        # Include runner state for resume support
        eval_results["runner_state"] = {
            "best_eval": self.best_eval.copy(),
            "episode_summary_counter": self._episode_summary_counter,
            "next_snapshot_step": self._next_snapshot_step,
            "eval_count": self._eval_count,
        }

        self.agent.save_checkpoint(checkpoint_file, eval_results)
        if save_best:
            checkpoint_file = os.path.join(
                self.checkpoint_folder, "best_checkpoint.everl"
            )
            self.agent.save_checkpoint(checkpoint_file, eval_results)

        log_info = (
            f"Quality: {quality}, Reward: {reward}, Exploration steps: {explore_steps}"
        )
        self.logger.info(log_info)
        with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(self._results.values())

        return quality, reward

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: int = None,
        explore_steps_limit: int = None,
    ):
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        while self.step_counter.exploration < explore_steps_limit:
            # Total experience = heatup + exploration (both contribute to replay buffer)
            total_experience_steps = (
                self.step_counter.heatup + self.step_counter.exploration
            )
            update_steps = (
                total_experience_steps * update_steps_per_explore_step
                - self.step_counter.update
            )
            result = self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )
            
            # Log episode summaries if we got results
            if result is not None:
                episodes, _ = result if isinstance(result, tuple) else (result, None)
                if episodes:
                    self._log_episode_summaries(episodes)
            
            # Check for policy snapshots after each exploration cycle
            # This ensures snapshots are saved at the actual milestone with correct weights
            self._maybe_save_policy_snapshot()

    def training_run(
        self,
        heatup_steps: int,
        training_steps: int,
        explore_steps_between_eval: int,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
    ):
        # TODO: Log Training Run Infos
        heatup_episodes = self.heatup(heatup_steps)
        
        # Capture and set probe states after heatup
        self._capture_and_set_probe_states(heatup_episodes)
        
        next_eval_step_limt = (
            self.agent.step_counter.exploration + explore_steps_between_eval
        )
        while self.agent.step_counter.exploration < training_steps:
            self.explore_and_update(
                explore_episodes_between_updates,
                update_steps_per_explore_step,
                explore_steps_limit=next_eval_step_limt,
            )
            
            # Note: policy snapshots are now saved inside explore_and_update()
            # at the actual milestone steps with correct weights
            
            quality, reward = self.eval(episodes=eval_episodes, seeds=eval_seeds)
            next_eval_step_limt += explore_steps_between_eval

        return quality, reward

    def _capture_and_set_probe_states(self, episodes: List):
        """Capture probe states from heatup episodes and set them in trainer."""
        if not episodes or self._probe_states_set:
            return
            
        try:
            probe_states = []
            n_episodes = min(len(episodes), self.n_probe_episodes)
            
            for ep_idx in range(n_episodes):
                episode = episodes[ep_idx]
                if hasattr(episode, 'flat_obs') and len(episode.flat_obs) > 0:
                    # Collect start state
                    probe_states.append(episode.flat_obs[0])
                    
                    # Collect near-start states
                    n_states = min(self.n_probe_near_start_steps, len(episode.flat_obs) - 1)
                    for step_idx in range(1, n_states + 1):
                        probe_states.append(episode.flat_obs[step_idx])
                        
            if probe_states:
                probe_states_array = np.array(probe_states)
                
                # Set probe states in agent (if it supports it)
                if hasattr(self.agent, 'set_probe_states'):
                    self.agent.set_probe_states(probe_states_array.tolist())
                    self.logger.info(f"Set {len(probe_states)} probe states for diagnostics")
                    
                # Save probe states to disk
                if self.diagnostics_folder is not None:
                    probes_dir = os.path.join(self.diagnostics_folder, "probes")
                    os.makedirs(probes_dir, exist_ok=True)
                    probe_path = os.path.join(probes_dir, "probe_states.npz")
                    np.savez(probe_path, probe_states=probe_states_array)
                    self.logger.info(f"Saved probe states to {probe_path}")
                    
                self._probe_states_set = True
                
        except Exception as e:
            self.logger.warning(f"Failed to capture probe states: {e}")

    def _maybe_save_policy_snapshot(self):
        """
        Save policy snapshot if we've reached the next snapshot step.
        
        FIXED: Uses a while loop to catch up on all missed snapshots when the
        eval cycle spans multiple snapshot intervals.
        """
        if self.diagnostics_folder is None:
            return
            
        current_step = self.step_counter.exploration
        
        # Save all missed snapshots (in case eval cycle spans multiple snapshot intervals)
        while current_step >= self._next_snapshot_step:
            try:
                snapshots_dir = os.path.join(self.diagnostics_folder, "policy_snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)
                
                # Use the scheduled snapshot step, not current step, for consistent naming
                snapshot_step = self._next_snapshot_step
                snapshot_path = os.path.join(snapshots_dir, f"policy_{snapshot_step}.pt")
                
                if hasattr(self.agent, 'save_policy_snapshot'):
                    self.agent.save_policy_snapshot(snapshot_path)
                    self.logger.info(f"Saved policy snapshot at scheduled step {snapshot_step} (current={current_step})")
                    
            except Exception as e:
                self.logger.warning(f"Failed to save policy snapshot at step {self._next_snapshot_step}: {e}")
                
            self._next_snapshot_step += self.policy_snapshot_every_steps
        
        # Flush diagnostics logs after all snapshots are saved
        if hasattr(self.agent, 'flush_diagnostics'):
            try:
                self.agent.flush_diagnostics()
            except Exception as e:
                self.logger.warning(f"Failed to flush diagnostics: {e}")

    def _log_episode_summaries(self, episodes: List):
        """
        Log episode summaries to episode_summary.jsonl with explore_step and update_step.
        
        This directly solves the timestamp correlation problem by providing exact
        step counters at episode completion time.
        """
        if self._episode_summary_file is None:
            return
            
        try:
            # Fix #4: Use per-episode step counters stamped by the worker at
            # completion time. Fall back to current shared counter for backward
            # compatibility with Episode objects that don't carry step snapshots.
            fallback_explore = self.step_counter.exploration
            fallback_update = self.step_counter.update

            with open(self._episode_summary_file, "a", encoding="utf-8") as f:
                for i, episode in enumerate(episodes):
                    try:
                        wall_time = time.time()  # Per-episode timestamp
                        # Use per-episode step snapshot if available, else fallback
                        explore_step = getattr(episode, 'explore_step_at_completion', None) or fallback_explore
                        update_step = getattr(episode, 'update_step_at_completion', None) or fallback_update
                        # Extract episode data
                        total_reward = getattr(episode, 'episode_reward', 0.0)
                        steps = len(getattr(episode, 'rewards', []))
                        terminated = getattr(episode, 'terminals', [False])[-1] if getattr(episode, 'terminals', []) else False
                        truncated = getattr(episode, 'truncations', [False])[-1] if getattr(episode, 'truncations', []) else False

                        # Try to get max_insertion from episode info if available
                        max_insertion = None
                        infos = getattr(episode, 'infos', [])
                        if infos:
                            last_info = infos[-1] if infos else {}
                            # Look for insertion depth in common info keys
                            for key in ['max_insertion', 'inserted_length', 'insertion_depth']:
                                if key in last_info:
                                    val = last_info[key]
                                    if hasattr(val, '__iter__') and not isinstance(val, str):
                                        max_insertion = float(max(val))
                                    else:
                                        max_insertion = float(val)
                                    break

                        # Fix #5: Monotonically increasing counter for episode_id.
                        # Replaces retroactive calculation that assumed sequential,
                        # non-interleaved episodes.
                        episode_id = self._episode_summary_counter
                        self._episode_summary_counter += 1

                        summary = {
                            "wall_time": wall_time,
                            "explore_step": explore_step,
                            "update_step": update_step,
                            "episode_id": episode_id,
                            "total_reward": float(total_reward),
                            "steps": steps,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                        if max_insertion is not None:
                            summary["max_insertion"] = max_insertion
                        
                        f.write(json.dumps(summary) + "\n")
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to log episode {i} summary: {e}")
                        
                f.flush()
                
        except Exception as e:
            self.logger.warning(f"Failed to log episode summaries: {e}")

```

```python
from typing import List, Optional
from math import inf
import csv
import json
import logging
import os
import time
import numpy as np
from ..util import EveRLObject
from ..agent.agent import Agent, StepCounter, EpisodeCounter


class Runner(EveRLObject):
    def __init__(
        self,
        agent: Agent,
        heatup_action_low: List[float],
        heatup_action_high: List[float],
        agent_parameter_for_result_file: dict,
        checkpoint_folder: str,
        results_file: str,
        quality_info: Optional[str] = None,
        info_results: Optional[List[str]] = None,
        diagnostics_folder: Optional[str] = None,
        policy_snapshot_every_steps: int = 10000,
        n_probe_episodes: int = 3,
        n_probe_near_start_steps: int = 10,
    ) -> None:
        self.agent = agent
        self.heatup_action_low = heatup_action_low
        self.heatup_action_high = heatup_action_high
        self.agent_parameter_for_result_file = agent_parameter_for_result_file
        self.checkpoint_folder = checkpoint_folder
        self.results_file = results_file
        self.quality_info = quality_info
        self.info_results = info_results or []
        self.logger = logging.getLogger(self.__module__)
        
        # Diagnostics settings
        self.diagnostics_folder = diagnostics_folder
        # Ensure worker step logs directory exists.
        # NOTE: STEP_LOG_DIR must be set BEFORE agent creation (which spawns workers).
        # The training script should set os.environ["STEP_LOG_DIR"] before creating the agent.
        # This setdefault is a fallback but will NOT help if workers are already spawned.
        if self.diagnostics_folder is not None:
            logs_subprocesses = os.path.join(self.diagnostics_folder, "logs_subprocesses")
            os.makedirs(logs_subprocesses, exist_ok=True)
            os.environ.setdefault("STEP_LOG_DIR", logs_subprocesses)
        self.policy_snapshot_every_steps = policy_snapshot_every_steps
        self.n_probe_episodes = n_probe_episodes
        self.n_probe_near_start_steps = n_probe_near_start_steps
        self._next_snapshot_step = policy_snapshot_every_steps
        self._probe_states_set = False
        self._eval_count = 0
        self._replay_save_interval = 4  # save replay buffer every N evals
        
        # Episode summary logger
        self._episode_summary_file = None
        self._episode_summary_logger = None
        self._episode_summary_counter = 0  # Fix #5: monotonic episode_id
        if self.diagnostics_folder is not None:
            csv_folder = os.path.join(self.diagnostics_folder, "csv")
            os.makedirs(csv_folder, exist_ok=True)
            self._episode_summary_file = os.path.join(csv_folder, "episode_summary.jsonl")
            self.logger.info(f"Episode summary will be logged to {self._episode_summary_file}")

        self._results = {
            "episodes explore": 0,
            "steps explore": 0,
        }
        self._results["quality"] = 0.0
        for info_result in self.info_results:
            self._results[info_result] = 0.0
        self._results["reward"] = 0.0
        self._results["best quality"] = 0.0
        self._results["best explore steps"] = 0.0

        file_existed = os.path.isfile(results_file)

        with open(results_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_existed:
                writer.writerow(
                    list(self._results.keys())
                    + [" "]
                    + list(agent_parameter_for_result_file.keys())
                )
                writer.writerow(
                    [" "] * (len(self._results.values()) + 1)
                    + list(agent_parameter_for_result_file.values())
                )

        self.best_eval = {"steps": 0, "quality": -inf}

    @property
    def step_counter(self) -> StepCounter:
        return self.agent.step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self.agent.episode_counter

    def restore_runner_state(self, checkpoint_path: str, probe_states_dir: Optional[str] = None):
        """Restore Runner-level state from a checkpoint for resume support.

        Restores: best_eval, _episode_summary_counter, _next_snapshot_step,
        and optionally probe states from a saved .npz file.
        """
        import torch
        checkpoint = torch.load(checkpoint_path)
        additional_info = checkpoint.get("additional_info", {})
        runner_state = additional_info.get("runner_state") if additional_info else None

        if runner_state:
            self.best_eval = runner_state["best_eval"]
            self._episode_summary_counter = runner_state["episode_summary_counter"]
            self._next_snapshot_step = runner_state["next_snapshot_step"]
            self._eval_count = runner_state.get("eval_count", 0)
            self.logger.info(
                f"Restored runner state: best_eval={self.best_eval}, "
                f"episode_summary_counter={self._episode_summary_counter}, "
                f"next_snapshot_step={self._next_snapshot_step}, "
                f"eval_count={self._eval_count}"
            )
        else:
            # Checkpoint predates runner_state support — compute safe defaults
            explore_steps = self.step_counter.exploration
            self._next_snapshot_step = (
                (explore_steps // self.policy_snapshot_every_steps) + 1
            ) * self.policy_snapshot_every_steps
            self.logger.warning(
                f"No runner_state in checkpoint. Setting _next_snapshot_step={self._next_snapshot_step}"
            )

        # Restore probe states from saved .npz if available
        if probe_states_dir is not None:
            probe_path = os.path.join(probe_states_dir, "probes", "probe_states.npz")
            if os.path.isfile(probe_path):
                try:
                    data = np.load(probe_path)
                    probe_states = data["probe_states"]
                    if hasattr(self.agent, 'set_probe_states'):
                        self.agent.set_probe_states(probe_states.tolist())
                        self._probe_states_set = True
                        self.logger.info(f"Restored {len(probe_states)} probe states from {probe_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to restore probe states: {e}")

    def load_replay_buffer(self):
        """Auto-load replay buffer from checkpoint folder if file exists."""
        replay_path = os.path.join(self.checkpoint_folder, "replay_buffer.npz")
        if os.path.isfile(replay_path):
            try:
                n = self.agent.replay_buffer.load_buffer_from_file(replay_path)
                self.logger.info(f"Loaded replay buffer ({n} episodes) from {replay_path}")
                return n
            except Exception as e:
                self.logger.warning(f"Failed to load replay buffer: {e}")
        return 0

    def heatup(self, steps: int):
        episodes = self.agent.heatup(
            steps=steps,
            custom_action_low=self.heatup_action_low,
            custom_action_high=self.heatup_action_high,
        )
        return episodes

    def explore(self, n_episodes: int):
        self.agent.explore(episodes=n_episodes)

    def update(self, n_steps: int):
        self.agent.update(n_steps)

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[List[int]] = None
    ):
        explore_steps = self.step_counter.exploration
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{explore_steps}.everl"
        )
        result_episodes = self.agent.evaluate(episodes=episodes, seeds=seeds)
        qualities, rewards = [], []
        results_for_info = {
            info_result_name: [] for info_result_name in self.info_results
        }
        eval_results = {"episodes": []}
        for episode in result_episodes:
            reward = episode.episode_reward
            quality = (
                episode.infos[-1][self.quality_info]
                if self.quality_info is not None
                else reward
            )
            qualities.append(quality)
            rewards.append(reward)
            for info_result_name in self.info_results:
                info = episode.infos[-1][info_result_name]
                results_for_info[info_result_name].append(info)
            eval_results["episodes"].append(
                {
                    "seed": episode.seed,
                    "options": episode.options,
                    "quality": episode.infos[-1][self.quality_info],
                }
            )

        reward = sum(rewards) / len(rewards)
        quality = sum(qualities) / len(qualities)
        for info_result_name, results in results_for_info.items():
            result = sum(results) / len(results)
            self._results[info_result_name] = round(result, 3)
        save_best = False
        if quality > self.best_eval["quality"]:
            save_best = True
            self.best_eval["quality"] = quality
            self.best_eval["steps"] = explore_steps

        self._results["episodes explore"] = self.episode_counter.exploration
        self._results["steps explore"] = explore_steps
        self._results["reward"] = round(reward, 3)
        self._results["quality"] = round(quality, 3)
        self._results["best quality"] = self.best_eval["quality"]
        self._results["best explore steps"] = self.best_eval["steps"]

        eval_results.update(self._results)
        eval_results.pop("best quality")
        eval_results.pop("best explore steps")

        self._eval_count += 1

        # Include runner state for resume support
        eval_results["runner_state"] = {
            "best_eval": self.best_eval.copy(),
            "episode_summary_counter": self._episode_summary_counter,
            "next_snapshot_step": self._next_snapshot_step,
            "eval_count": self._eval_count,
        }

        self.agent.save_checkpoint(checkpoint_file, eval_results)
        if save_best:
            checkpoint_file = os.path.join(
                self.checkpoint_folder, "best_checkpoint.everl"
            )
            self.agent.save_checkpoint(checkpoint_file, eval_results)

        log_info = (
            f"Quality: {quality}, Reward: {reward}, Exploration steps: {explore_steps}"
        )
        self.logger.info(log_info)
        with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(self._results.values())

        # Periodic replay buffer save
        if self._eval_count % self._replay_save_interval == 0:
            replay_path = os.path.join(self.checkpoint_folder, "replay_buffer.npz")
            try:
                n = self.agent.replay_buffer.save_buffer_to_file(replay_path)
                self.logger.info(f"Saved replay buffer ({n} episodes) at eval #{self._eval_count}")
            except Exception as e:
                self.logger.warning(f"Failed to save replay buffer: {e}")

        return quality, reward

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: int = None,
        explore_steps_limit: int = None,
    ):
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        while self.step_counter.exploration < explore_steps_limit:
            # Total experience = heatup + exploration (both contribute to replay buffer)
            total_experience_steps = (
                self.step_counter.heatup + self.step_counter.exploration
            )
            update_steps = (
                total_experience_steps * update_steps_per_explore_step
                - self.step_counter.update
            )
            result = self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )
            
            # Log episode summaries if we got results
            if result is not None:
                episodes, _ = result if isinstance(result, tuple) else (result, None)
                if episodes:
                    self._log_episode_summaries(episodes)
            
            # Check for policy snapshots after each exploration cycle
            # This ensures snapshots are saved at the actual milestone with correct weights
            self._maybe_save_policy_snapshot()

    def training_run(
        self,
        heatup_steps: int,
        training_steps: int,
        explore_steps_between_eval: int,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
    ):
        # TODO: Log Training Run Infos
        heatup_episodes = self.heatup(heatup_steps)
        
        # Capture and set probe states after heatup
        self._capture_and_set_probe_states(heatup_episodes)
        
        next_eval_step_limt = (
            self.agent.step_counter.exploration + explore_steps_between_eval
        )
        while self.agent.step_counter.exploration < training_steps:
            self.explore_and_update(
                explore_episodes_between_updates,
                update_steps_per_explore_step,
                explore_steps_limit=next_eval_step_limt,
            )
            
            # Note: policy snapshots are now saved inside explore_and_update()
            # at the actual milestone steps with correct weights
            
            quality, reward = self.eval(episodes=eval_episodes, seeds=eval_seeds)
            next_eval_step_limt += explore_steps_between_eval

        return quality, reward

    def _capture_and_set_probe_states(self, episodes: List):
        """Capture probe states from heatup episodes and set them in trainer."""
        if not episodes or self._probe_states_set:
            return
            
        try:
            probe_states = []
            n_episodes = min(len(episodes), self.n_probe_episodes)
            
            for ep_idx in range(n_episodes):
                episode = episodes[ep_idx]
                if hasattr(episode, 'flat_obs') and len(episode.flat_obs) > 0:
                    # Collect start state
                    probe_states.append(episode.flat_obs[0])
                    
                    # Collect near-start states
                    n_states = min(self.n_probe_near_start_steps, len(episode.flat_obs) - 1)
                    for step_idx in range(1, n_states + 1):
                        probe_states.append(episode.flat_obs[step_idx])
                        
            if probe_states:
                probe_states_array = np.array(probe_states)
                
                # Set probe states in agent (if it supports it)
                if hasattr(self.agent, 'set_probe_states'):
                    self.agent.set_probe_states(probe_states_array.tolist())
                    self.logger.info(f"Set {len(probe_states)} probe states for diagnostics")
                    
                # Save probe states to disk
                if self.diagnostics_folder is not None:
                    probes_dir = os.path.join(self.diagnostics_folder, "probes")
                    os.makedirs(probes_dir, exist_ok=True)
                    probe_path = os.path.join(probes_dir, "probe_states.npz")
                    np.savez(probe_path, probe_states=probe_states_array)
                    self.logger.info(f"Saved probe states to {probe_path}")
                    
                self._probe_states_set = True
                
        except Exception as e:
            self.logger.warning(f"Failed to capture probe states: {e}")

    def _maybe_save_policy_snapshot(self):
        """
        Save policy snapshot if we've reached the next snapshot step.
        
        FIXED: Uses a while loop to catch up on all missed snapshots when the
        eval cycle spans multiple snapshot intervals.
        """
        if self.diagnostics_folder is None:
            return
            
        current_step = self.step_counter.exploration
        
        # Save all missed snapshots (in case eval cycle spans multiple snapshot intervals)
        while current_step >= self._next_snapshot_step:
            try:
                snapshots_dir = os.path.join(self.diagnostics_folder, "policy_snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)
                
                # Use the scheduled snapshot step, not current step, for consistent naming
                snapshot_step = self._next_snapshot_step
                snapshot_path = os.path.join(snapshots_dir, f"policy_{snapshot_step}.pt")
                
                if hasattr(self.agent, 'save_policy_snapshot'):
                    self.agent.save_policy_snapshot(snapshot_path)
                    self.logger.info(f"Saved policy snapshot at scheduled step {snapshot_step} (current={current_step})")
                    
            except Exception as e:
                self.logger.warning(f"Failed to save policy snapshot at step {self._next_snapshot_step}: {e}")
                
            self._next_snapshot_step += self.policy_snapshot_every_steps
        
        # Flush diagnostics logs after all snapshots are saved
        if hasattr(self.agent, 'flush_diagnostics'):
            try:
                self.agent.flush_diagnostics()
            except Exception as e:
                self.logger.warning(f"Failed to flush diagnostics: {e}")

    def _log_episode_summaries(self, episodes: List):
        """
        Log episode summaries to episode_summary.jsonl with explore_step and update_step.
        
        This directly solves the timestamp correlation problem by providing exact
        step counters at episode completion time.
        """
        if self._episode_summary_file is None:
            return
            
        try:
            # Fix #4: Use per-episode step counters stamped by the worker at
            # completion time. Fall back to current shared counter for backward
            # compatibility with Episode objects that don't carry step snapshots.
            fallback_explore = self.step_counter.exploration
            fallback_update = self.step_counter.update

            with open(self._episode_summary_file, "a", encoding="utf-8") as f:
                for i, episode in enumerate(episodes):
                    try:
                        wall_time = time.time()  # Per-episode timestamp
                        # Use per-episode step snapshot if available, else fallback
                        explore_step = getattr(episode, 'explore_step_at_completion', None) or fallback_explore
                        update_step = getattr(episode, 'update_step_at_completion', None) or fallback_update
                        # Extract episode data
                        total_reward = getattr(episode, 'episode_reward', 0.0)
                        steps = len(getattr(episode, 'rewards', []))
                        terminated = getattr(episode, 'terminals', [False])[-1] if getattr(episode, 'terminals', []) else False
                        truncated = getattr(episode, 'truncations', [False])[-1] if getattr(episode, 'truncations', []) else False

                        # Try to get max_insertion from episode info if available
                        max_insertion = None
                        infos = getattr(episode, 'infos', [])
                        if infos:
                            last_info = infos[-1] if infos else {}
                            # Look for insertion depth in common info keys
                            for key in ['max_insertion', 'inserted_length', 'insertion_depth']:
                                if key in last_info:
                                    val = last_info[key]
                                    if hasattr(val, '__iter__') and not isinstance(val, str):
                                        max_insertion = float(max(val))
                                    else:
                                        max_insertion = float(val)
                                    break

                        # Fix #5: Monotonically increasing counter for episode_id.
                        # Replaces retroactive calculation that assumed sequential,
                        # non-interleaved episodes.
                        episode_id = self._episode_summary_counter
                        self._episode_summary_counter += 1

                        summary = {
                            "wall_time": wall_time,
                            "explore_step": explore_step,
                            "update_step": update_step,
                            "episode_id": episode_id,
                            "total_reward": float(total_reward),
                            "steps": steps,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                        if max_insertion is not None:
                            summary["max_insertion"] = max_insertion
                        
                        f.write(json.dumps(summary) + "\n")
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to log episode {i} summary: {e}")
                        
                f.flush()
                
        except Exception as e:
            self.logger.warning(f"Failed to log episode summaries: {e}")

```

```python
from typing import List, Optional
from math import inf
import csv
import json
import logging
import os
import time
import numpy as np
from ..util import EveRLObject
from ..agent.agent import Agent, StepCounter, EpisodeCounter


class Runner(EveRLObject):
    def __init__(
        self,
        agent: Agent,
        heatup_action_low: List[float],
        heatup_action_high: List[float],
        agent_parameter_for_result_file: dict,
        checkpoint_folder: str,
        results_file: str,
        quality_info: Optional[str] = None,
        info_results: Optional[List[str]] = None,
        diagnostics_folder: Optional[str] = None,
        policy_snapshot_every_steps: int = 10000,
        n_probe_episodes: int = 3,
        n_probe_near_start_steps: int = 10,
    ) -> None:
        self.agent = agent
        self.heatup_action_low = heatup_action_low
        self.heatup_action_high = heatup_action_high
        self.agent_parameter_for_result_file = agent_parameter_for_result_file
        self.checkpoint_folder = checkpoint_folder
        self.results_file = results_file
        self.quality_info = quality_info
        self.info_results = info_results or []
        self.logger = logging.getLogger(self.__module__)
        
        # Diagnostics settings
        self.diagnostics_folder = diagnostics_folder
        # Ensure worker step logs directory exists.
        # NOTE: STEP_LOG_DIR must be set BEFORE agent creation (which spawns workers).
        # The training script should set os.environ["STEP_LOG_DIR"] before creating the agent.
        # This setdefault is a fallback but will NOT help if workers are already spawned.
        if self.diagnostics_folder is not None:
            logs_subprocesses = os.path.join(self.diagnostics_folder, "logs_subprocesses")
            os.makedirs(logs_subprocesses, exist_ok=True)
            os.environ.setdefault("STEP_LOG_DIR", logs_subprocesses)
        self.policy_snapshot_every_steps = policy_snapshot_every_steps
        self.n_probe_episodes = n_probe_episodes
        self.n_probe_near_start_steps = n_probe_near_start_steps
        self._next_snapshot_step = policy_snapshot_every_steps
        self._probe_states_set = False
        self._eval_count = 0
        self._replay_save_interval = 4  # save replay buffer every N evals
        
        # Episode summary logger
        self._episode_summary_file = None
        self._episode_summary_logger = None
        self._episode_summary_counter = 0  # Fix #5: monotonic episode_id
        if self.diagnostics_folder is not None:
            csv_folder = os.path.join(self.diagnostics_folder, "csv")
            os.makedirs(csv_folder, exist_ok=True)
            self._episode_summary_file = os.path.join(csv_folder, "episode_summary.jsonl")
            self.logger.info(f"Episode summary will be logged to {self._episode_summary_file}")

        self._results = {
            "episodes explore": 0,
            "steps explore": 0,
        }
        self._results["quality"] = 0.0
        for info_result in self.info_results:
            self._results[info_result] = 0.0
        self._results["reward"] = 0.0
        self._results["best quality"] = 0.0
        self._results["best explore steps"] = 0.0

        file_existed = os.path.isfile(results_file)

        with open(results_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_existed:
                writer.writerow(
                    list(self._results.keys())
                    + [" "]
                    + list(agent_parameter_for_result_file.keys())
                )
                writer.writerow(
                    [" "] * (len(self._results.values()) + 1)
                    + list(agent_parameter_for_result_file.values())
                )

        self.best_eval = {"steps": 0, "quality": -inf}

    @property
    def step_counter(self) -> StepCounter:
        return self.agent.step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self.agent.episode_counter

    def restore_runner_state(self, checkpoint_path: str, probe_states_dir: Optional[str] = None):
        """Restore Runner-level state from a checkpoint for resume support.

        Restores: best_eval, _episode_summary_counter, _next_snapshot_step,
        and optionally probe states from a saved .npz file.
        """
        import torch
        checkpoint = torch.load(checkpoint_path)
        additional_info = checkpoint.get("additional_info", {})
        runner_state = additional_info.get("runner_state") if additional_info else None

        if runner_state:
            self.best_eval = runner_state["best_eval"]
            self._episode_summary_counter = runner_state["episode_summary_counter"]
            self._next_snapshot_step = runner_state["next_snapshot_step"]
            self._eval_count = runner_state.get("eval_count", 0)
            self.logger.info(
                f"Restored runner state: best_eval={self.best_eval}, "
                f"episode_summary_counter={self._episode_summary_counter}, "
                f"next_snapshot_step={self._next_snapshot_step}, "
                f"eval_count={self._eval_count}"
            )
        else:
            # Checkpoint predates runner_state support — compute safe defaults
            explore_steps = self.step_counter.exploration
            self._next_snapshot_step = (
                (explore_steps // self.policy_snapshot_every_steps) + 1
            ) * self.policy_snapshot_every_steps
            self.logger.warning(
                f"No runner_state in checkpoint. Setting _next_snapshot_step={self._next_snapshot_step}"
            )

        # Restore probe states from saved .npz if available
        if probe_states_dir is not None:
            probe_path = os.path.join(probe_states_dir, "probes", "probe_states.npz")
            if os.path.isfile(probe_path):
                try:
                    data = np.load(probe_path)
                    probe_states = data["probe_states"]
                    if hasattr(self.agent, 'set_probe_states'):
                        self.agent.set_probe_states(probe_states.tolist())
                        self._probe_states_set = True
                        self.logger.info(f"Restored {len(probe_states)} probe states from {probe_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to restore probe states: {e}")

    def load_replay_buffer(self):
        """Auto-load replay buffer from checkpoint folder if file exists."""
        replay_path = os.path.join(self.checkpoint_folder, "replay_buffer.npz")
        if os.path.isfile(replay_path):
            try:
                n = self.agent.replay_buffer.load_buffer_from_file(replay_path)
                self.logger.info(f"Loaded replay buffer ({n} episodes) from {replay_path}")
                return n
            except Exception as e:
                self.logger.warning(f"Failed to load replay buffer: {e}")
        return 0

    def heatup(self, steps: int):
        episodes = self.agent.heatup(
            steps=steps,
            custom_action_low=self.heatup_action_low,
            custom_action_high=self.heatup_action_high,
        )
        return episodes

    def explore(self, n_episodes: int):
        self.agent.explore(episodes=n_episodes)

    def update(self, n_steps: int):
        self.agent.update(n_steps)

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[List[int]] = None
    ):
        explore_steps = self.step_counter.exploration
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{explore_steps}.everl"
        )
        result_episodes = self.agent.evaluate(episodes=episodes, seeds=seeds)
        qualities, rewards = [], []
        results_for_info = {
            info_result_name: [] for info_result_name in self.info_results
        }
        eval_results = {"episodes": []}
        for episode in result_episodes:
            reward = episode.episode_reward
            quality = (
                episode.infos[-1][self.quality_info]
                if self.quality_info is not None
                else reward
            )
            qualities.append(quality)
            rewards.append(reward)
            for info_result_name in self.info_results:
                info = episode.infos[-1][info_result_name]
                results_for_info[info_result_name].append(info)
            eval_results["episodes"].append(
                {
                    "seed": episode.seed,
                    "options": episode.options,
                    "quality": episode.infos[-1][self.quality_info],
                }
            )

        reward = sum(rewards) / len(rewards)
        quality = sum(qualities) / len(qualities)
        for info_result_name, results in results_for_info.items():
            result = sum(results) / len(results)
            self._results[info_result_name] = round(result, 3)
        save_best = False
        if quality > self.best_eval["quality"]:
            save_best = True
            self.best_eval["quality"] = quality
            self.best_eval["steps"] = explore_steps

        self._results["episodes explore"] = self.episode_counter.exploration
        self._results["steps explore"] = explore_steps
        self._results["reward"] = round(reward, 3)
        self._results["quality"] = round(quality, 3)
        self._results["best quality"] = self.best_eval["quality"]
        self._results["best explore steps"] = self.best_eval["steps"]

        eval_results.update(self._results)
        eval_results.pop("best quality")
        eval_results.pop("best explore steps")

        self._eval_count += 1

        # Include runner state for resume support
        eval_results["runner_state"] = {
            "best_eval": self.best_eval.copy(),
            "episode_summary_counter": self._episode_summary_counter,
            "next_snapshot_step": self._next_snapshot_step,
            "eval_count": self._eval_count,
        }

        self.agent.save_checkpoint(checkpoint_file, eval_results)
        if save_best:
            checkpoint_file = os.path.join(
                self.checkpoint_folder, "best_checkpoint.everl"
            )
            self.agent.save_checkpoint(checkpoint_file, eval_results)

        log_info = (
            f"Quality: {quality}, Reward: {reward}, Exploration steps: {explore_steps}"
        )
        self.logger.info(log_info)
        with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(self._results.values())

        # Periodic replay buffer save
        if self._eval_count % self._replay_save_interval == 0:
            replay_path = os.path.join(self.checkpoint_folder, "replay_buffer.npz")
            try:
                n = self.agent.replay_buffer.save_buffer_to_file(replay_path)
                self.logger.info(f"Saved replay buffer ({n} episodes) at eval #{self._eval_count}")
            except Exception as e:
                self.logger.warning(f"Failed to save replay buffer: {e}")

        return quality, reward

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: int = None,
        explore_steps_limit: int = None,
    ):
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        while self.step_counter.exploration < explore_steps_limit:
            # Total experience = heatup + exploration (both contribute to replay buffer)
            total_experience_steps = (
                self.step_counter.heatup + self.step_counter.exploration
            )
            update_steps = (
                total_experience_steps * update_steps_per_explore_step
                - self.step_counter.update
            )
            result = self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )
            
            # Log episode summaries if we got results
            if result is not None:
                episodes, _ = result if isinstance(result, tuple) else (result, None)
                if episodes:
                    self._log_episode_summaries(episodes)
            
            # Check for policy snapshots after each exploration cycle
            # This ensures snapshots are saved at the actual milestone with correct weights
            self._maybe_save_policy_snapshot()

    def training_run(
        self,
        heatup_steps: int,
        training_steps: int,
        explore_steps_between_eval: int,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
        heatup_cache_save_path: Optional[str] = None,
    ):
        # TODO: Log Training Run Infos
        heatup_episodes = self.heatup(heatup_steps)

        # Save heatup cache if requested (before training starts)
        if heatup_cache_save_path and heatup_episodes:
            from ..util.experience_cache import save_episodes_npz
            tuples = [
                (np.array(e.flat_obs), np.array(e.actions),
                 np.array(e.rewards), np.array(e.terminals))
                for e in heatup_episodes
            ]
            save_episodes_npz(heatup_cache_save_path, tuples)
            self.logger.info(f"Saved heatup cache ({len(tuples)} episodes) to {heatup_cache_save_path}")

        # Capture and set probe states after heatup
        self._capture_and_set_probe_states(heatup_episodes)
        
        next_eval_step_limt = (
            self.agent.step_counter.exploration + explore_steps_between_eval
        )
        while self.agent.step_counter.exploration < training_steps:
            self.explore_and_update(
                explore_episodes_between_updates,
                update_steps_per_explore_step,
                explore_steps_limit=next_eval_step_limt,
            )
            
            # Note: policy snapshots are now saved inside explore_and_update()
            # at the actual milestone steps with correct weights
            
            quality, reward = self.eval(episodes=eval_episodes, seeds=eval_seeds)
            next_eval_step_limt += explore_steps_between_eval

        return quality, reward

    def _capture_and_set_probe_states(self, episodes: List):
        """Capture probe states from heatup episodes and set them in trainer."""
        if not episodes or self._probe_states_set:
            return
            
        try:
            probe_states = []
            n_episodes = min(len(episodes), self.n_probe_episodes)
            
            for ep_idx in range(n_episodes):
                episode = episodes[ep_idx]
                if hasattr(episode, 'flat_obs') and len(episode.flat_obs) > 0:
                    # Collect start state
                    probe_states.append(episode.flat_obs[0])
                    
                    # Collect near-start states
                    n_states = min(self.n_probe_near_start_steps, len(episode.flat_obs) - 1)
                    for step_idx in range(1, n_states + 1):
                        probe_states.append(episode.flat_obs[step_idx])
                        
            if probe_states:
                probe_states_array = np.array(probe_states)
                
                # Set probe states in agent (if it supports it)
                if hasattr(self.agent, 'set_probe_states'):
                    self.agent.set_probe_states(probe_states_array.tolist())
                    self.logger.info(f"Set {len(probe_states)} probe states for diagnostics")
                    
                # Save probe states to disk
                if self.diagnostics_folder is not None:
                    probes_dir = os.path.join(self.diagnostics_folder, "probes")
                    os.makedirs(probes_dir, exist_ok=True)
                    probe_path = os.path.join(probes_dir, "probe_states.npz")
                    np.savez(probe_path, probe_states=probe_states_array)
                    self.logger.info(f"Saved probe states to {probe_path}")
                    
                self._probe_states_set = True
                
        except Exception as e:
            self.logger.warning(f"Failed to capture probe states: {e}")

    def _maybe_save_policy_snapshot(self):
        """
        Save policy snapshot if we've reached the next snapshot step.
        
        FIXED: Uses a while loop to catch up on all missed snapshots when the
        eval cycle spans multiple snapshot intervals.
        """
        if self.diagnostics_folder is None:
            return
            
        current_step = self.step_counter.exploration
        
        # Save all missed snapshots (in case eval cycle spans multiple snapshot intervals)
        while current_step >= self._next_snapshot_step:
            try:
                snapshots_dir = os.path.join(self.diagnostics_folder, "policy_snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)
                
                # Use the scheduled snapshot step, not current step, for consistent naming
                snapshot_step = self._next_snapshot_step
                snapshot_path = os.path.join(snapshots_dir, f"policy_{snapshot_step}.pt")
                
                if hasattr(self.agent, 'save_policy_snapshot'):
                    self.agent.save_policy_snapshot(snapshot_path)
                    self.logger.info(f"Saved policy snapshot at scheduled step {snapshot_step} (current={current_step})")
                    
            except Exception as e:
                self.logger.warning(f"Failed to save policy snapshot at step {self._next_snapshot_step}: {e}")
                
            self._next_snapshot_step += self.policy_snapshot_every_steps
        
        # Flush diagnostics logs after all snapshots are saved
        if hasattr(self.agent, 'flush_diagnostics'):
            try:
                self.agent.flush_diagnostics()
            except Exception as e:
                self.logger.warning(f"Failed to flush diagnostics: {e}")

    def _log_episode_summaries(self, episodes: List):
        """
        Log episode summaries to episode_summary.jsonl with explore_step and update_step.
        
        This directly solves the timestamp correlation problem by providing exact
        step counters at episode completion time.
        """
        if self._episode_summary_file is None:
            return
            
        try:
            # Fix #4: Use per-episode step counters stamped by the worker at
            # completion time. Fall back to current shared counter for backward
            # compatibility with Episode objects that don't carry step snapshots.
            fallback_explore = self.step_counter.exploration
            fallback_update = self.step_counter.update

            with open(self._episode_summary_file, "a", encoding="utf-8") as f:
                for i, episode in enumerate(episodes):
                    try:
                        wall_time = time.time()  # Per-episode timestamp
                        # Use per-episode step snapshot if available, else fallback
                        explore_step = getattr(episode, 'explore_step_at_completion', None) or fallback_explore
                        update_step = getattr(episode, 'update_step_at_completion', None) or fallback_update
                        # Extract episode data
                        total_reward = getattr(episode, 'episode_reward', 0.0)
                        steps = len(getattr(episode, 'rewards', []))
                        terminated = getattr(episode, 'terminals', [False])[-1] if getattr(episode, 'terminals', []) else False
                        truncated = getattr(episode, 'truncations', [False])[-1] if getattr(episode, 'truncations', []) else False

                        # Try to get max_insertion from episode info if available
                        max_insertion = None
                        infos = getattr(episode, 'infos', [])
                        if infos:
                            last_info = infos[-1] if infos else {}
                            # Look for insertion depth in common info keys
                            for key in ['max_insertion', 'inserted_length', 'insertion_depth']:
                                if key in last_info:
                                    val = last_info[key]
                                    if hasattr(val, '__iter__') and not isinstance(val, str):
                                        max_insertion = float(max(val))
                                    else:
                                        max_insertion = float(val)
                                    break

                        # Fix #5: Monotonically increasing counter for episode_id.
                        # Replaces retroactive calculation that assumed sequential,
                        # non-interleaved episodes.
                        episode_id = self._episode_summary_counter
                        self._episode_summary_counter += 1

                        summary = {
                            "wall_time": wall_time,
                            "explore_step": explore_step,
                            "update_step": update_step,
                            "episode_id": episode_id,
                            "total_reward": float(total_reward),
                            "steps": steps,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                        if max_insertion is not None:
                            summary["max_insertion"] = max_insertion
                        
                        f.write(json.dumps(summary) + "\n")
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to log episode {i} summary: {e}")
                        
                f.flush()
                
        except Exception as e:
            self.logger.warning(f"Failed to log episode summaries: {e}")

```

**7. Update DualDeviceNav_train_vs.py with cache CLI args:**

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
    parser.add_argument(
        "--heuristic_cache_file",
        type=str,
        default=None,
        help="Path to load pre-generated heuristic episodes from (skips generation)",
    )
    parser.add_argument(
        "--save_heuristic_cache",
        type=str,
        default=None,
        help="Path to save generated heuristic episodes to (for reuse)",
    )
    parser.add_argument(
        "--heatup_cache_file",
        type=str,
        default=None,
        help="Path to load pre-generated heatup episodes from (skips heatup)",
    )
    parser.add_argument(
        "--save_heatup_cache",
        type=str,
        default=None,
        help="Path to save heatup episodes to (for reuse)",
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
    parser.add_argument(
        "--heuristic_cache_file",
        type=str,
        default=None,
        help="Path to load pre-generated heuristic episodes from (skips generation)",
    )
    parser.add_argument(
        "--save_heuristic_cache",
        type=str,
        default=None,
        help="Path to save generated heuristic episodes to (for reuse)",
    )
    parser.add_argument(
        "--heatup_cache_file",
        type=str,
        default=None,
        help="Path to load pre-generated heatup episodes from (skips heatup)",
    )
    parser.add_argument(
        "--save_heatup_cache",
        type=str,
        default=None,
        help="Path to save heatup episodes to (for reuse)",
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
        from eve_rl.replaybuffer import Episode, EpisodeReplay
        from eve_rl.util import flatten_obs
        from eve_rl.util.experience_cache import save_episodes_npz, load_episodes_npz

        # Try to load from cache first
        if args.heuristic_cache_file and os.path.isfile(args.heuristic_cache_file):
            print(f"Loading heuristic cache from {args.heuristic_cache_file}...")
            episodes_tuples, _ = load_episodes_npz(args.heuristic_cache_file)
            n_pushed = 0
            for ep_tuple in episodes_tuples:
                flat_obs, actions, rewards, terminals = ep_tuple
                replay_ep = EpisodeReplay(
                    flat_obs=list(flat_obs),
                    actions=list(actions),
                    rewards=list(rewards),
                    terminals=list(terminals),
                )
                agent.replay_buffer.push(replay_ep)
                n_pushed += 1
            print(f"Heuristic cache loaded: {n_pushed} episodes pushed to replay buffer.")
        else:
            # Generate heuristic episodes
            print(f"Seeding replay buffer with {args.heuristic_seeding} heuristic episodes...")
            seed_intervention = DualDeviceNav()
            seed_env = EnvClass(intervention=seed_intervention, mode="train", visualisation=False)
            heuristic = CenterlineFollowerHeuristic(
                pathfinder=seed_env.pathfinder,
                intervention=seed_intervention,
            )
            rng = np.random.default_rng(42)
            n_pushed = 0
            episodes_to_save = []

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

                # Collect for potential cache saving
                if args.save_heuristic_cache:
                    episodes_to_save.append((
                        np.array(episode.flat_obs),
                        np.array(episode.actions),
                        np.array(episode.rewards),
                        np.array(episode.terminals),
                    ))

                if (ep + 1) % 100 == 0:
                    print(f"  Heuristic seeding: {ep + 1}/{args.heuristic_seeding} episodes (reward={episode.episode_reward:.2f})")
            seed_env.close()
            print(f"Heuristic seeding complete: {n_pushed} episodes pushed to replay buffer.")

            # Save cache if requested
            if args.save_heuristic_cache and episodes_to_save:
                os.makedirs(os.path.dirname(args.save_heuristic_cache) or ".", exist_ok=True)
                save_episodes_npz(args.save_heuristic_cache, episodes_to_save)
                print(f"Saved heuristic cache to {args.save_heuristic_cache}")

    # Load heatup cache if provided (skips heatup phase)
    heatup_steps_effective = HEATUP_STEPS
    if args.heatup_cache_file and os.path.isfile(args.heatup_cache_file):
        from eve_rl.replaybuffer import EpisodeReplay
        from eve_rl.util.experience_cache import load_episodes_npz

        print(f"Loading heatup cache from {args.heatup_cache_file}...")
        episodes_tuples, _ = load_episodes_npz(args.heatup_cache_file)
        n_pushed = 0
        for ep_tuple in episodes_tuples:
            flat_obs, actions, rewards, terminals = ep_tuple
            replay_ep = EpisodeReplay(
                flat_obs=list(flat_obs),
                actions=list(actions),
                rewards=list(rewards),
                terminals=list(terminals),
            )
            agent.replay_buffer.push(replay_ep)
            n_pushed += 1
        print(f"Heatup cache loaded: {n_pushed} episodes. Skipping heatup phase.")
        heatup_steps_effective = 0

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
    parser.add_argument(
        "--heuristic_cache_file",
        type=str,
        default=None,
        help="Path to load pre-generated heuristic episodes from (skips generation)",
    )
    parser.add_argument(
        "--save_heuristic_cache",
        type=str,
        default=None,
        help="Path to save generated heuristic episodes to (for reuse)",
    )
    parser.add_argument(
        "--heatup_cache_file",
        type=str,
        default=None,
        help="Path to load pre-generated heatup episodes from (skips heatup)",
    )
    parser.add_argument(
        "--save_heatup_cache",
        type=str,
        default=None,
        help="Path to save heatup episodes to (for reuse)",
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
        from eve_rl.replaybuffer import Episode, EpisodeReplay
        from eve_rl.util import flatten_obs
        from eve_rl.util.experience_cache import save_episodes_npz, load_episodes_npz

        # Try to load from cache first
        if args.heuristic_cache_file and os.path.isfile(args.heuristic_cache_file):
            print(f"Loading heuristic cache from {args.heuristic_cache_file}...")
            episodes_tuples, _ = load_episodes_npz(args.heuristic_cache_file)
            n_pushed = 0
            for ep_tuple in episodes_tuples:
                flat_obs, actions, rewards, terminals = ep_tuple
                replay_ep = EpisodeReplay(
                    flat_obs=list(flat_obs),
                    actions=list(actions),
                    rewards=list(rewards),
                    terminals=list(terminals),
                )
                agent.replay_buffer.push(replay_ep)
                n_pushed += 1
            print(f"Heuristic cache loaded: {n_pushed} episodes pushed to replay buffer.")
        else:
            # Generate heuristic episodes
            print(f"Seeding replay buffer with {args.heuristic_seeding} heuristic episodes...")
            seed_intervention = DualDeviceNav()
            seed_env = EnvClass(intervention=seed_intervention, mode="train", visualisation=False)
            heuristic = CenterlineFollowerHeuristic(
                pathfinder=seed_env.pathfinder,
                intervention=seed_intervention,
            )
            rng = np.random.default_rng(42)
            n_pushed = 0
            episodes_to_save = []

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

                # Collect for potential cache saving
                if args.save_heuristic_cache:
                    episodes_to_save.append((
                        np.array(episode.flat_obs),
                        np.array(episode.actions),
                        np.array(episode.rewards),
                        np.array(episode.terminals),
                    ))

                if (ep + 1) % 100 == 0:
                    print(f"  Heuristic seeding: {ep + 1}/{args.heuristic_seeding} episodes (reward={episode.episode_reward:.2f})")
            seed_env.close()
            print(f"Heuristic seeding complete: {n_pushed} episodes pushed to replay buffer.")

            # Save cache if requested
            if args.save_heuristic_cache and episodes_to_save:
                os.makedirs(os.path.dirname(args.save_heuristic_cache) or ".", exist_ok=True)
                save_episodes_npz(args.save_heuristic_cache, episodes_to_save)
                print(f"Saved heuristic cache to {args.save_heuristic_cache}")

    # Load heatup cache if provided (skips heatup phase)
    heatup_steps_effective = HEATUP_STEPS
    if args.heatup_cache_file and os.path.isfile(args.heatup_cache_file):
        from eve_rl.replaybuffer import EpisodeReplay
        from eve_rl.util.experience_cache import load_episodes_npz

        print(f"Loading heatup cache from {args.heatup_cache_file}...")
        episodes_tuples, _ = load_episodes_npz(args.heatup_cache_file)
        n_pushed = 0
        for ep_tuple in episodes_tuples:
            flat_obs, actions, rewards, terminals = ep_tuple
            replay_ep = EpisodeReplay(
                flat_obs=list(flat_obs),
                actions=list(actions),
                rewards=list(rewards),
                terminals=list(terminals),
            )
            agent.replay_buffer.push(replay_ep)
            n_pushed += 1
        print(f"Heatup cache loaded: {n_pushed} episodes. Skipping heatup phase.")
        heatup_steps_effective = 0

    # Determine heatup cache save path (only if not loading from cache)
    heatup_cache_save = args.save_heatup_cache if not args.heatup_cache_file else None

    reward, success = runner.training_run(
        heatup_steps_effective,
        TRAINING_STEPS,
        EXPLORE_STEPS_BTW_EVAL,
        CONSECUTIVE_EXPLORE_EPISODES,
        UPDATE_PER_EXPLORE_STEP,
        eval_seeds=EVAL_SEEDS,
        heatup_cache_save_path=heatup_cache_save,
    )
    agent.close()

```

**8. Update DualDeviceNav_resume.py with auto-load replay buffer and cache args:**

```python
"""
DualDeviceNav training script with resume capability.

Usage:
  # Fresh training:
  python DualDeviceNav_resume.py -n my_experiment -d cuda:0 -nw 16

  # Resume from checkpoint (continues in the SAME experiment folder):
  python DualDeviceNav_resume.py -d cuda:0 -nw 16 \
    --resume /path/to/experiment_folder/checkpoints/checkpoint250000.everl
"""
import os
import logging
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch
from util.util import get_result_checkpoint_config_and_log_path
from util.env import BenchEnv
from util.env2 import BenchEnv2
from util.env3 import BenchEnv3
from util.env4 import BenchEnv4
from util.env5 import BenchEnv5
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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="DualDeviceNav training with resume capability")
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=4, help="Number of workers"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device of trainer, where the NN update is performed.",
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
        help="Environment version: 1=original (PathDelta), 2=waypoint+centerlines, 3=tuned waypoint, 4=arclength progress+guidance, 5=optimized env4",
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
        help="Number of heuristic episodes to seed replay buffer (0=disabled, recommended: 500). Only for env4/5.",
    )
    parser.add_argument(
        "--heuristic_cache_file",
        type=str,
        default=None,
        help="Path to load pre-generated heuristic episodes from (skips generation)",
    )
    # Resume from checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file (.everl) to resume training from. "
             "Continues in the same experiment folder. Skips heatup when provided."
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
    resume_checkpoint = args.resume
    worker_device = torch.device("cpu")
    
    # Select environment class based on version
    if env_version == 5:
        EnvClass = BenchEnv5
        print("Using BenchEnv5 (optimized env4 with shared projection cache)")
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
        "resumed_from": resume_checkpoint if resume_checkpoint else "fresh",
    }

    if resume_checkpoint:
        # Derive all paths from the checkpoint location to resume in the SAME folder.
        # Expected layout: .../experiment_folder/checkpoints/checkpoint*.everl
        checkpoint_folder = os.path.dirname(os.path.abspath(resume_checkpoint))
        results_folder = os.path.dirname(checkpoint_folder)
        results_file = results_folder + ".csv"
        log_file = os.path.join(results_folder, "main.log")
        config_folder = os.path.join(results_folder, "config")
        diagnostics_folder = os.path.join(results_folder, "diagnostics")
        os.makedirs(config_folder, exist_ok=True)
        os.makedirs(diagnostics_folder, exist_ok=True)
        print(f"Resuming in existing experiment folder: {results_folder}")
    else:
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

    # Load checkpoint if resuming
    if resume_checkpoint:
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT: {resume_checkpoint}")
        print(f"{'='*60}")
        agent.load_checkpoint(resume_checkpoint)
        print(f"  Loaded network weights, optimizer states, and schedulers")
        print(f"  Heatup steps:      {agent.step_counter.heatup:,}")
        print(f"  Exploration steps: {agent.step_counter.exploration:,}")
        print(f"  Update steps:      {agent.step_counter.update:,}")
        print(f"  Evaluation steps:  {agent.step_counter.evaluation:,}")
        print(f"  Heatup episodes:   {agent.episode_counter.heatup:,}")
        print(f"  Explore episodes:  {agent.episode_counter.exploration:,}")
        print(f"{'='*60}")

        # CRITICAL: Restore curriculum step count to avoid Stage regression
        if args.curriculum:
            restored_steps = agent.step_counter.exploration
            env_train._total_steps = restored_steps
            print(f"  Curriculum restored: _total_steps={restored_steps}, Stage={env_train.current_stage}")

        print()

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

    # When resuming, restore Runner-level state (best_eval, snapshot step, episode counter, probes)
    if resume_checkpoint:
        runner.restore_runner_state(resume_checkpoint, probe_states_dir=diagnostics_folder)

        # NOTE: Replay buffer is NOT saved in checkpoints (too large).
        # The agent resumes with an empty buffer and refills via exploration.
        # Gradient updates will be paused until enough new experience accumulates
        # to satisfy: (heatup + explore) * update_ratio > update_steps.
        # This means a "dead zone" of ~(prior_total_steps * update_ratio) updates
        # where exploration runs but no learning happens.

    # Optional: seed replay buffer with heuristic episodes (fresh training or resume)
    if args.heuristic_seeding > 0 and env_version in (4, 5):
        from util.heuristic_controller import CenterlineFollowerHeuristic
        from eve_rl.replaybuffer import Episode
        from eve_rl.util import flatten_obs

        print(f"Seeding replay buffer with {args.heuristic_seeding} heuristic episodes...")
        seed_intervention = DualDeviceNav()
        # Use the appropriate env class for seeding
        SeedEnvClass = BenchEnv5 if env_version == 5 else BenchEnv4
        seed_env = SeedEnvClass(intervention=seed_intervention, mode="train", visualisation=False)
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

    # When resuming, use heatup_steps=0 to skip heatup phase
    # The checkpoint already contains learned weights and the step counters
    # will pick up from where training left off
    effective_heatup = 0 if resume_checkpoint else HEATUP_STEPS

    if resume_checkpoint:
        print(f"Skipping heatup (checkpoint already has {agent.step_counter.heatup:,} heatup steps)")
        print(f"Continuing training from exploration step {agent.step_counter.exploration:,}")
    
    reward, success = runner.training_run(
        effective_heatup,
        TRAINING_STEPS,
        EXPLORE_STEPS_BTW_EVAL,
        CONSECUTIVE_EXPLORE_EPISODES,
        UPDATE_PER_EXPLORE_STEP,
        eval_seeds=EVAL_SEEDS,
    )
    agent.close()

```

```python
"""
DualDeviceNav training script with resume capability.

Usage:
  # Fresh training:
  python DualDeviceNav_resume.py -n my_experiment -d cuda:0 -nw 16

  # Resume from checkpoint (continues in the SAME experiment folder):
  python DualDeviceNav_resume.py -d cuda:0 -nw 16 \
    --resume /path/to/experiment_folder/checkpoints/checkpoint250000.everl
"""
import os
import logging
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch
from util.util import get_result_checkpoint_config_and_log_path
from util.env import BenchEnv
from util.env2 import BenchEnv2
from util.env3 import BenchEnv3
from util.env4 import BenchEnv4
from util.env5 import BenchEnv5
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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="DualDeviceNav training with resume capability")
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=4, help="Number of workers"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device of trainer, where the NN update is performed.",
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
        help="Environment version: 1=original (PathDelta), 2=waypoint+centerlines, 3=tuned waypoint, 4=arclength progress+guidance, 5=optimized env4",
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
        help="Number of heuristic episodes to seed replay buffer (0=disabled, recommended: 500). Only for env4/5.",
    )
    parser.add_argument(
        "--heuristic_cache_file",
        type=str,
        default=None,
        help="Path to load pre-generated heuristic episodes from (skips generation)",
    )
    # Resume from checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file (.everl) to resume training from. "
             "Continues in the same experiment folder. Skips heatup when provided."
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
    resume_checkpoint = args.resume
    worker_device = torch.device("cpu")
    
    # Select environment class based on version
    if env_version == 5:
        EnvClass = BenchEnv5
        print("Using BenchEnv5 (optimized env4 with shared projection cache)")
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
        "resumed_from": resume_checkpoint if resume_checkpoint else "fresh",
    }

    if resume_checkpoint:
        # Derive all paths from the checkpoint location to resume in the SAME folder.
        # Expected layout: .../experiment_folder/checkpoints/checkpoint*.everl
        checkpoint_folder = os.path.dirname(os.path.abspath(resume_checkpoint))
        results_folder = os.path.dirname(checkpoint_folder)
        results_file = results_folder + ".csv"
        log_file = os.path.join(results_folder, "main.log")
        config_folder = os.path.join(results_folder, "config")
        diagnostics_folder = os.path.join(results_folder, "diagnostics")
        os.makedirs(config_folder, exist_ok=True)
        os.makedirs(diagnostics_folder, exist_ok=True)
        print(f"Resuming in existing experiment folder: {results_folder}")
    else:
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

    # Load checkpoint if resuming
    if resume_checkpoint:
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT: {resume_checkpoint}")
        print(f"{'='*60}")
        agent.load_checkpoint(resume_checkpoint)
        print(f"  Loaded network weights, optimizer states, and schedulers")
        print(f"  Heatup steps:      {agent.step_counter.heatup:,}")
        print(f"  Exploration steps: {agent.step_counter.exploration:,}")
        print(f"  Update steps:      {agent.step_counter.update:,}")
        print(f"  Evaluation steps:  {agent.step_counter.evaluation:,}")
        print(f"  Heatup episodes:   {agent.episode_counter.heatup:,}")
        print(f"  Explore episodes:  {agent.episode_counter.exploration:,}")
        print(f"{'='*60}")

        # CRITICAL: Restore curriculum step count to avoid Stage regression
        if args.curriculum:
            restored_steps = agent.step_counter.exploration
            env_train._total_steps = restored_steps
            print(f"  Curriculum restored: _total_steps={restored_steps}, Stage={env_train.current_stage}")

        print()

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

    # When resuming, restore Runner-level state (best_eval, snapshot step, episode counter, probes)
    if resume_checkpoint:
        runner.restore_runner_state(resume_checkpoint, probe_states_dir=diagnostics_folder)

        # Auto-load replay buffer if saved (avoids "dead zone" with empty buffer)
        n_loaded = runner.load_replay_buffer()
        if n_loaded > 0:
            print(f"Loaded {n_loaded} episodes from replay buffer checkpoint")
        else:
            print("No replay buffer checkpoint found - buffer starts empty")

    # Optional: seed replay buffer with heuristic episodes (fresh training or resume)
    if args.heuristic_seeding > 0 and env_version in (4, 5):
        from util.heuristic_controller import CenterlineFollowerHeuristic
        from eve_rl.replaybuffer import Episode
        from eve_rl.util import flatten_obs

        print(f"Seeding replay buffer with {args.heuristic_seeding} heuristic episodes...")
        seed_intervention = DualDeviceNav()
        # Use the appropriate env class for seeding
        SeedEnvClass = BenchEnv5 if env_version == 5 else BenchEnv4
        seed_env = SeedEnvClass(intervention=seed_intervention, mode="train", visualisation=False)
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

    # When resuming, use heatup_steps=0 to skip heatup phase
    # The checkpoint already contains learned weights and the step counters
    # will pick up from where training left off
    effective_heatup = 0 if resume_checkpoint else HEATUP_STEPS

    if resume_checkpoint:
        print(f"Skipping heatup (checkpoint already has {agent.step_counter.heatup:,} heatup steps)")
        print(f"Continuing training from exploration step {agent.step_counter.exploration:,}")
    
    reward, success = runner.training_run(
        effective_heatup,
        TRAINING_STEPS,
        EXPLORE_STEPS_BTW_EVAL,
        CONSECUTIVE_EXPLORE_EPISODES,
        UPDATE_PER_EXPLORE_STEP,
        eval_seeds=EVAL_SEEDS,
    )
    agent.close()

```

```python
"""
DualDeviceNav training script with resume capability.

Usage:
  # Fresh training:
  python DualDeviceNav_resume.py -n my_experiment -d cuda:0 -nw 16

  # Resume from checkpoint (continues in the SAME experiment folder):
  python DualDeviceNav_resume.py -d cuda:0 -nw 16 \
    --resume /path/to/experiment_folder/checkpoints/checkpoint250000.everl
"""
import os
import logging
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch
from util.util import get_result_checkpoint_config_and_log_path
from util.env import BenchEnv
from util.env2 import BenchEnv2
from util.env3 import BenchEnv3
from util.env4 import BenchEnv4
from util.env5 import BenchEnv5
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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="DualDeviceNav training with resume capability")
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=4, help="Number of workers"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device of trainer, where the NN update is performed.",
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
        help="Environment version: 1=original (PathDelta), 2=waypoint+centerlines, 3=tuned waypoint, 4=arclength progress+guidance, 5=optimized env4",
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
        help="Number of heuristic episodes to seed replay buffer (0=disabled, recommended: 500). Only for env4/5.",
    )
    parser.add_argument(
        "--heuristic_cache_file",
        type=str,
        default=None,
        help="Path to load pre-generated heuristic episodes from (skips generation)",
    )
    # Resume from checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file (.everl) to resume training from. "
             "Continues in the same experiment folder. Skips heatup when provided."
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
    resume_checkpoint = args.resume
    worker_device = torch.device("cpu")
    
    # Select environment class based on version
    if env_version == 5:
        EnvClass = BenchEnv5
        print("Using BenchEnv5 (optimized env4 with shared projection cache)")
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
        "resumed_from": resume_checkpoint if resume_checkpoint else "fresh",
    }

    if resume_checkpoint:
        # Derive all paths from the checkpoint location to resume in the SAME folder.
        # Expected layout: .../experiment_folder/checkpoints/checkpoint*.everl
        checkpoint_folder = os.path.dirname(os.path.abspath(resume_checkpoint))
        results_folder = os.path.dirname(checkpoint_folder)
        results_file = results_folder + ".csv"
        log_file = os.path.join(results_folder, "main.log")
        config_folder = os.path.join(results_folder, "config")
        diagnostics_folder = os.path.join(results_folder, "diagnostics")
        os.makedirs(config_folder, exist_ok=True)
        os.makedirs(diagnostics_folder, exist_ok=True)
        print(f"Resuming in existing experiment folder: {results_folder}")
    else:
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

    # Load checkpoint if resuming
    if resume_checkpoint:
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT: {resume_checkpoint}")
        print(f"{'='*60}")
        agent.load_checkpoint(resume_checkpoint)
        print(f"  Loaded network weights, optimizer states, and schedulers")
        print(f"  Heatup steps:      {agent.step_counter.heatup:,}")
        print(f"  Exploration steps: {agent.step_counter.exploration:,}")
        print(f"  Update steps:      {agent.step_counter.update:,}")
        print(f"  Evaluation steps:  {agent.step_counter.evaluation:,}")
        print(f"  Heatup episodes:   {agent.episode_counter.heatup:,}")
        print(f"  Explore episodes:  {agent.episode_counter.exploration:,}")
        print(f"{'='*60}")

        # CRITICAL: Restore curriculum step count to avoid Stage regression
        if args.curriculum:
            restored_steps = agent.step_counter.exploration
            env_train._total_steps = restored_steps
            print(f"  Curriculum restored: _total_steps={restored_steps}, Stage={env_train.current_stage}")

        print()

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

    # When resuming, restore Runner-level state (best_eval, snapshot step, episode counter, probes)
    if resume_checkpoint:
        runner.restore_runner_state(resume_checkpoint, probe_states_dir=diagnostics_folder)

        # Auto-load replay buffer if saved (avoids "dead zone" with empty buffer)
        n_loaded = runner.load_replay_buffer()
        if n_loaded > 0:
            print(f"Loaded {n_loaded} episodes from replay buffer checkpoint")
        else:
            print("No replay buffer checkpoint found - buffer starts empty")

    # Optional: seed replay buffer with heuristic episodes (fresh training or resume)
    if args.heuristic_seeding > 0 and env_version in (4, 5):
        from util.heuristic_controller import CenterlineFollowerHeuristic
        from eve_rl.replaybuffer import Episode, EpisodeReplay
        from eve_rl.util import flatten_obs

        # Try to load from cache first
        if args.heuristic_cache_file and os.path.isfile(args.heuristic_cache_file):
            from eve_rl.util.experience_cache import load_episodes_npz
            print(f"Loading heuristic cache from {args.heuristic_cache_file}...")
            episodes_tuples, _ = load_episodes_npz(args.heuristic_cache_file)
            n_pushed = 0
            for ep_tuple in episodes_tuples:
                flat_obs, actions, rewards, terminals = ep_tuple
                replay_ep = EpisodeReplay(
                    flat_obs=list(flat_obs),
                    actions=list(actions),
                    rewards=list(rewards),
                    terminals=list(terminals),
                )
                agent.replay_buffer.push(replay_ep)
                n_pushed += 1
            print(f"Heuristic cache loaded: {n_pushed} episodes pushed to replay buffer.")
        else:
            # Generate heuristic episodes
            print(f"Seeding replay buffer with {args.heuristic_seeding} heuristic episodes...")
            seed_intervention = DualDeviceNav()
            # Use the appropriate env class for seeding
            SeedEnvClass = BenchEnv5 if env_version == 5 else BenchEnv4
            seed_env = SeedEnvClass(intervention=seed_intervention, mode="train", visualisation=False)
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

    # When resuming, use heatup_steps=0 to skip heatup phase
    # The checkpoint already contains learned weights and the step counters
    # will pick up from where training left off
    effective_heatup = 0 if resume_checkpoint else HEATUP_STEPS

    if resume_checkpoint:
        print(f"Skipping heatup (checkpoint already has {agent.step_counter.heatup:,} heatup steps)")
        print(f"Continuing training from exploration step {agent.step_counter.exploration:,}")
    
    reward, success = runner.training_run(
        effective_heatup,
        TRAINING_STEPS,
        EXPLORE_STEPS_BTW_EVAL,
        CONSECUTIVE_EXPLORE_EPISODES,
        UPDATE_PER_EXPLORE_STEP,
        eval_seeds=EVAL_SEEDS,
    )
    agent.close()

```

Now let me check for lint errors:

All warnings are just unresolved import warnings (normal for this project's structure). No actual errors.

Let me verify the changes are correct by checking the key files:

Let me verify the backup files were created:

All changes are complete. Here's a summary:

## Merge Summary

### Backups Created
| File | Backup |
|------|--------|
| `replaybuffer.py` | `replaybuffer_v1.py` |
| `vanillaepisode.py` | `vanillaepisode_v1.py` |
| `vanillashared.py` | `vanillashared_v1.py` |
| `__init__.py` | `__init__v1.py` |
| `runner.py` | `runner_v1.py` |
| `DualDeviceNav_train_vs.py` | `DualDeviceNav_train_vs_v2.py` |
| `DualDeviceNav_resume.py` | `DualDeviceNav_resume_v1.py` |

### New File Created
- `eve_rl/eve_rl/util/experience_cache.py` — `save_episodes_npz()` / `load_episodes_npz()` functions

### Changes Merged

1. **`replaybuffer.py`** — Added `EpisodeReplay.to_replay()` method for push compatibility

2. **`vanillaepisode.py`** — Added `export_all()` / `import_all()` for buffer serialization

3. **`vanillashared.py`** — Added `save_buffer_to_file()` / `load_buffer_from_file()` methods + IPC task handlers in subprocess loop. *Note: Clean push() method preserved (no verbose logging)*

4. **`__init__.py`** — Exported `EpisodeReplay`

5. **`runner.py`** — Added:
   - `_eval_count` and `_replay_save_interval`
   - `load_replay_buffer()` method
   - Periodic replay buffer save every 4 evals
   - `heatup_cache_save_path` param in `training_run()`
   - Fixed log message: `episode_counter` → `episode_summary_counter`

6. **`DualDeviceNav_train_vs.py`** — Added CLI args:
   - `--heuristic_cache_file` / `--save_heuristic_cache`
   - `--heatup_cache_file` / `--save_heatup_cache`

7. **`DualDeviceNav_resume.py`** — Added:
   - `--heuristic_cache_file` arg
   - Auto-load replay buffer via `runner.load_replay_buffer()`
   - Cache loading support in heuristic seeding

### Bugs Fixed
- ✅ Log message typo (`episode_counter` → `episode_summary_counter`)
- ✅ `.flatten()` already present in both training scripts (no action needed)

---

**User**

How many of these are done 1. ActionCurriculumWrapper Reset (P0)	Restore _total_steps from checkpoint	✅ DONE - Lines 296-300
2. Environment Logging Counters (P2)	Pass step counters to env	❌ NOT DONE
3. Replay Buffer Empty (P1)	Warmup or checkpoint buffer	⚠️ PARTIAL - Heuristic seeding optional, no auto-warmup
4. Missing Features (P1)	Add env4/5, curriculum, heuristic	⚠️ PARTIAL - env5 missing
5. Runner best_eval Lost (P2)	Store in checkpoint	✅ DONE - Lines 214-219 (save), 112-114 (restore)
6. Probe States Not Captured (P3)	Restore from saved file	✅ DONE - Lines 131-143
7. Diagnostics Counters Reset (P3)	Include in runner_state	✅ DONE - Line 114
8. Snapshot Step Misalignment (P3)	Compute aligned next step	✅ DONE - Lines 115, 122-129


---

**Cursor**

Let me verify each item by checking the current code:

Here's the verified status:

| # | Item | Priority | Status | Details |
|---|------|----------|--------|---------|
| 1 | ActionCurriculumWrapper Reset | P0 | ✅ **DONE** | Lines 307-310 restore `_total_steps = agent.step_counter.exploration` |
| 2 | Environment Logging Counters | P2 | ❌ **NOT DONE** | Env internal counters (`_step_count`, `_episode_count`) are per-worker, NOT synced with agent counters |
| 3 | Replay Buffer Empty | P1 | ✅ **DONE** | `runner.load_replay_buffer()` auto-loads `replay_buffer.npz` (line 339) |
| 4 | Missing Features (env5, curriculum, heuristic) | P1 | ✅ **DONE** | env5 imported (line 23), selectable (lines 166-167), heuristic cache support added |
| 5 | Runner best_eval Lost | P2 | ✅ **DONE** | Saved in `runner_state` and restored (line 115) |
| 6 | Probe States Not Captured | P3 | ✅ **DONE** | Restored from `probe_states.npz` (lines 136-147) |
| 7 | Diagnostics Counters Reset | P3 | ✅ **DONE** | `_episode_summary_counter` in `runner_state` (lines 116, 121) |
| 8 | Snapshot Step Misalignment | P3 | ✅ **DONE** | `_next_snapshot_step` restored or computed (lines 117, 127-132) |

---

**Summary: 7/8 DONE**

The only remaining item is **#2 Environment Logging Counters (P2)** — this is architectural: environments run in separate worker processes and maintain their own internal counters for subprocess-level diagnostic logging. These counters (`_step_count`, `_episode_count`) are not the same as the agent's shared counters and don't need to be synced for training correctness — they're only for per-worker log correlation.

If you want this fixed, it would require either:
- Passing shared memory counters to workers at spawn time
- Injecting counters into the environment after unpickling (via `__setstate__`)

But this is purely cosmetic for logging and doesn't affect training resumption.

---

**User**

Does it effect diagnose_collapse4/5/6 analysis

---

**Cursor**

Let me check what data the diagnose scripts rely on:

**No, it does NOT affect diagnose_collapse4/5/6 analysis.**

Here's why:

## What diagnose_collapse scripts use:

| Data Source | Fields Used | Restored on Resume? |
|-------------|-------------|---------------------|
| `episode_summary.jsonl` | `explore_step`, `update_step`, `episode_id`, `wall_time` | ✅ Yes - from Runner's `_episode_summary_counter` |
| `losses.jsonl` / `losses.csv` | `update_step`, `explore_step`, `timestamp` | ✅ Yes - from agent's shared counters |
| `probe_values.csv` | `explore_step`, `update_step` | ✅ Yes |
| `batch_samples.jsonl` | `explore_step`, `update_step` | ✅ Yes |

## What the unsynced env counters are for:

The environment's internal counters (`_step_count`, `_episode_count`) are **only** used for per-worker subprocess logs:

```
STEP | ep=5 | ep_step=32 | global=1532 | wall_time=... | pid=12345
```

These are written to `logs_subprocesses/worker_*.log` for **debugging individual worker processes** — they are **not parsed** by diagnose_collapse scripts.

## Summary

The diagnose_collapse scripts rely on:
- **`_episode_summary_counter`** → ✅ Restored in `runner_state`
- **`step_counter.exploration/update`** → ✅ Restored from checkpoint

The unsynced env counters only affect subprocess debug logs, which are not used for collapse forensics. The analysis will work correctly after resume.

```plaintext
STEP | ep=5 | ep_step=32 | global=1532 | wall_time=... | pid=12345
```

