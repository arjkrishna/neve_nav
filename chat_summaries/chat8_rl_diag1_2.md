# RL code and logging framework analysis
_Exported on 3/24/2026 at 21:36:26 EDT from Cursor (2.3.34)_

---

**User**

Same workspace exists in another place D:\neve and is edited by a different programmer in VSCODE ;  understand the RL code w.r.t DuadDeviceNav_train_vs.py and the logging framework w.r.t @diagnose_collapse4.py ; All the changed files are :- | File | v5 | v6 | What |
|------|----|----|------|
| `diagnose_collapse5.py` | **NEW** | — | All 13 correctness fixes |
| `diagnose_collapse6.py` | — | **NEW** | All 8 performance optimizations on top of v5 |
| `eve_rl/eve_rl/replaybuffer/replaybuffer.py` | Fix #4 | — | Added `explore_step_at_completion`, `update_step_at_completion` to `Episode` |
| `eve_rl/eve_rl/agent/single.py` | Fix #4 | — | Stamps episode step counters inside lock |
| `eve_rl/eve_rl/runner/runner.py` | Fix #4, #5 | — | Per-episode step values + monotonic `episode_id` counter |
| `eve_rl/eve_rl/algo/sac.py` | Fix #6 | — | `_compute_clamp_fraction()` + added to `last_metrics` |
| `eve_rl/eve_rl/util/diagnostics_logger.py` | Fix #6, #10, #11 | — | `clamp_fraction` header, module-level serializer, unknown field warning |; copy the new 5 and 6 versions from the neve folder and changed files to this workspace.  Before copying the changed files , save the older versions as <filename>_prevs.py ; understand all the relevant background code. Now analyse all the differences in the new code and old code and mention what are all the changes there for ; what do they fix and what potentially could have been broken and is there any other changes that can be made to make it even better logging framework. 

---

**Cursor**

I'll help you understand the RL code and logging framework, then analyze the differences between versions. Since I'm in **ask mode**, I can only read files - I cannot copy or modify them. Let me first explore the relevant code.

Excellent! Now let me read the changed eve_rl files to understand the full scope of modifications:

Now let me check the current versions in the stEVE_training workspace for comparison:

Now I have a complete picture. Let me provide a comprehensive analysis:

---

# Comprehensive Analysis: diagnose_collapse v4→v5→v6 and eve_rl Changes

## Overview of the RL Training System

The `DualDeviceNav_train_vs.py` training script runs SAC (Soft Actor-Critic) for dual-device navigation in endovascular intervention. The architecture involves:

1. **Multi-worker exploration** - Workers collect episodes in parallel using `BenchAgentSynchron` 
2. **Shared replay buffer** - Episodes are pushed to a shared buffer
3. **Centralized training** - A single trainer process updates policy/critics
4. **Diagnostics logging** - Losses, probe values, batch samples logged via `DiagnosticsLogger`

The `diagnose_collapse4.py` forensics tool analyzes these logs post-hoc to detect training collapse patterns (alpha shutdown, entropy drop, saturation, etc.).

---

## Summary of All Changes

### Files in `D:\neve` to Copy

| File | Changes |
|------|---------|
| `diagnose_collapse5.py` | **NEW** - 13 correctness fixes |
| `diagnose_collapse6.py` | **NEW** - 8 performance optimizations on v5 |
| `eve_rl/eve_rl/replaybuffer/replaybuffer.py` | Episode step stamping fields |
| `eve_rl/eve_rl/agent/single.py` | Stamps episode step counters inside lock |
| `eve_rl/eve_rl/runner/runner.py` | Per-episode step values + monotonic episode_id |
| `eve_rl/eve_rl/algo/sac.py` | `_compute_clamp_fraction()` + clamp_fraction metric |
| `eve_rl/eve_rl/util/diagnostics_logger.py` | clamp_fraction header, module-level serializer, unknown field warning |

---

## Detailed Change Analysis

### 1. **Fix #1: Binary Search Replacement** (diagnose_collapse5.py)

**Old code (v4):**
```python
while lo <= hi:
    mid = (lo + hi) // 2
    s_wt = summaries[mid].wall_time.timestamp()
    dt = abs(s_wt - wt)
    if dt < best_dt and mid not in used:  # BUG: 'mid not in used' inside loop
        best_dt, best_i = dt, mid
    ...
```

**Problem:** The `mid not in used` check inside the binary search loop skipped valid candidates, causing incorrect matches when multiple episodes completed near the same time.

**New code (v5):**
```python
# Uses bisect.bisect_left + outward scan (lo/hi expanding from insertion point)
idx = bisect.bisect_left(wall_times, wt)
lo, hi = idx, idx
while lo >= 0 or hi < len(summaries):
    for j in [lo, hi]:
        if 0 <= j < len(summaries) and j not in used:
            # Check candidate...
    lo -= 1
    hi += 1
```

**Impact:** Matched 85 episodes vs 72 in v4. O(n log n) vs O(n²) worst case.

---

### 2. **Fix #3: `_compute_confidence()` Edge Case**

**Old code (v4):**
```python
gap = max(baseline - threshold, 1e-6)  # Could be near-zero
return min(1.0, violation / gap)  # Could produce infinity
```

**New code (v5):**
```python
if gap < 1e-4:
    return 0.0  # No meaningful gap to measure confidence against
```

**What it fixes:** When experiments start with alpha intentionally low (baseline ≈ threshold), the old code produced infinite/negative confidence values.

---

### 3. **Fix #4: Per-Episode Step Stamping (Race Condition)**

This is the most critical fix, involving changes across 4 files:

**Problem:** In multi-worker training, `episode_summary.jsonl` logged the *current* shared step counter at summary-write time, not at episode-completion time. Race condition: Worker A completes episode, Worker B increments counter, then A's summary is written with B's count.

**Changes:**

**`replaybuffer.py` - NEW fields:**
```python
class Episode:
    def __init__(self, ...):
        # ... existing fields ...
        # NEW: Step counters at episode completion (stamped by worker)
        self.explore_step_at_completion: Optional[int] = None
        self.update_step_at_completion: Optional[int] = None
```

**`single.py` - Stamp inside lock:**
```python
def explore(self, ...):
    # ... play episode ...
    with self.step_counter.lock:
        self.step_counter.exploration += n_steps_episode
        # NEW: Stamp INSIDE LOCK to prevent race
        episode.explore_step_at_completion = self.step_counter.exploration
    episode.update_step_at_completion = self.step_counter.update  # Read-only from workers
```

**`runner.py` - Use per-episode values:**
```python
def _log_episode_summaries(self, episodes):
    fallback_explore = self.step_counter.exploration
    for episode in episodes:
        # Use per-episode snapshot if available, else fallback
        explore_step = getattr(episode, 'explore_step_at_completion', None) or fallback_explore
```

**What was broken:** Episode-to-step mapping was inaccurate, causing the forensics tool to attribute events to wrong training phases.

---

### 4. **Fix #5: Monotonic Episode ID Counter**

**Old code (stEVE_training runner.py):**
```python
"episode_id": self.episode_counter.exploration - len(episodes) + i,  # Can repeat/jump
```

**New code (neve runner.py):**
```python
self._episode_summary_counter = 0  # In __init__
# ...
episode_id = self._episode_summary_counter
self._episode_summary_counter += 1  # Guaranteed unique, monotonic
```

**What was broken:** When multiple workers completed episodes simultaneously, episode_ids could repeat or skip.

---

### 5. **Fix #6: Clamp Fraction Logging**

This adds direct visibility into action saturation (tanh hitting ±1 boundary).

**`sac.py` - NEW helper function:**
```python
def _compute_clamp_fraction(actions: torch.Tensor, threshold: float = 0.99) -> float:
    """Compute fraction of actions near clamp boundaries."""
    if actions.numel() == 0:
        return 0.0
    clamped = (actions.abs() > threshold).float()
    return clamped.mean().item()
```

**Called in `_update_policy()`:**
```python
self._clamp_fraction = _compute_clamp_fraction(new_actions.detach())
```

**Added to `last_metrics`:**
```python
self.last_metrics = {
    # ... existing metrics ...
    "clamp_fraction": self._clamp_fraction,  # NEW
}
```

**`diagnostics_logger.py` - NEW header field:**
```python
LOSS_HEADER_FIELDS = [
    # ... existing fields ...
    "clamp_fraction",  # NEW
]
```

**`diagnose_collapse5.py` - NEW event detection:**
```python
CLAMP_FRACTION_THRESHOLD = 0.5   # >50% = saturated
CLAMP_FRACTION_CRITICAL = 0.8   # >80% = critical

# Event: clamp_saturation
pred_clamp = [cf > CLAMP_FRACTION_THRESHOLD for cf in clamp_fractions]
i_clamp = _first_sustained(pred_clamp, sustain=sustain)
if i_clamp is not None:
    events.append(Event(name="clamp_saturation", ...))
```

**What was broken:** Tanh saturation was invisible. Policy could be hitting ±1 boundaries (no exploration possible) but no metric tracked it.

---

### 6. **Fix #7-8: explore_update_map First Occurrence + Bisect Timestamp Lookup**

**Old code:**
```python
explore_update_map = {l.explore_step: l.update_step for l in losses}  # Last-write-wins
```

**New code:**
```python
for l in losses:
    if l.explore_step not in explore_update_map:  # First occurrence only
        explore_update_map[l.explore_step] = l.update_step
```

**timestamp_to_explore() old:** O(n) linear scan
**timestamp_to_explore() new:** `bisect.bisect_right` on pre-sorted array, O(log n)

---

### 7. **Fix #10-11: Diagnostics Logger Improvements**

**`convert_to_serializable` moved to module level:**
```python
# OLD: Nested function inside method, recreated per call
def log(self, data):
    def convert_to_serializable(obj): ...  # BAD

# NEW: Module-level, single definition
def _convert_to_serializable(obj): ...  # GOOD
```

**Unknown field warning:**
```python
class CSVScalarLogger:
    def __init__(self, ...):
        self._warned_unknown_fields: set = set()  # NEW
    
    def log(self, **kwargs):
        unknown = set(kwargs.keys()) - set(self.header_fields) - self._warned_unknown_fields
        if unknown:
            self._warned_unknown_fields.update(unknown)
            logger.warning(f"Unknown fields not in header (will be dropped): {unknown}")
```

**What was broken:** New metrics added to `last_metrics` were silently dropped if not in header.

---

### 8. **Fix #12: Replay Shift Ungated**

**Old code (v4):**
```python
if (insertion_collapsed or alpha_shutdown or stuck_regime) and pending_replay_events:
    events.extend(pending_replay_events)  # Gated - might not fire
```

**New code (v5):**
```python
if pending_replay_events:
    events.extend(pending_replay_events)  # Always include
```

---

### 9. **Fix #13 & Perf #1-8: Performance Optimizations (v6)**

**`_rolling_mean`:** Uses `deque.popleft()` (O(1)) instead of `list.pop(0)` (O(n))

**`_rolling_std`:** Welford's online algorithm:
```python
# OLD: statistics.pstdev(deque) - O(w) per call
# NEW: Running sum/sum-of-squares
s += v; ss += v*v  # O(1) per call
variance = ss/n - (s/n)**2
```

**`_rolling_median`:** Bisect-sorted window:
```python
# OLD: statistics.median(deque) - creates sorted copy each call O(w log w)
# NEW: Maintain sorted_w with bisect.insort/pop O(w) insert but O(1) median
```

**String pre-check before regex cascade:**
```python
# OLD: Try 5 regexes per line
# NEW: Fast string checks route to correct regex
if "STEP" in line:
    if "exec_action=" in line:  # V3 format
        m = STEP_RE_V3.search(line)
    elif "action=" in line:  # V2 format
        m = STEP_RE_NEW.search(line)
```

**Other optimizations:**
- Build timestamp interpolator once, pass to consumers
- Sort episodes once, reuse sorted view
- Cache `stat()` calls in `_glob_best`
- Cache insertion rolling_median between `detect_events` and `make_plots`

---

## What Could Have Been Broken (in stEVE_training)

1. **Episode-step mapping was inaccurate** - Events attributed to wrong training phases
2. **Episode IDs not monotonic** - Confusing duplicate/missing IDs in analysis
3. **Clamp saturation invisible** - Policy collapse due to tanh saturation undetected
4. **Silent field drops** - New metrics added but never logged (no warning)
5. **Confidence scores invalid** - Infinite/negative values when baseline ≈ threshold
6. **Memory blowup on large runs** - Actions accumulated per episode but never used

---

## Suggestions for Further Improvements

1. **Add `explore_step_delta`** - Track steps accumulated in the current episode for finer-grained correlation

2. **Trainer-side episode stamping** - For perfect `update_step_at_completion`, have the trainer stamp episodes after they arrive (requires architecture change)

3. **Streaming analysis mode** - For very long runs, diagnose_collapse could process logs in streaming fashion with constant memory

4. **Add action histogram logging** - Log full action distribution periodically, not just clamp fraction

5. **Parallel worker log parsing** - Use multiprocessing to parse multiple worker logs in parallel

6. **Compression for old logs** - Auto-gzip logs older than N days to save disk space

---

## Action Required

Since you're in **ask mode**, I cannot copy files or make edits. To apply these changes:

1. **Switch to Agent mode** in Cursor
2. Ask me to "copy the new files from D:\neve to D:\stEVE_training"
3. I will:
   - Save current versions as `*_prevs.py`
   - Copy `diagnose_collapse5.py`, `diagnose_collapse6.py` 
   - Update the 5 eve_rl files with the new changes

Would you like me to proceed once you switch to Agent mode?

```python
while lo <= hi:
    mid = (lo + hi) // 2
    s_wt = summaries[mid].wall_time.timestamp()
    dt = abs(s_wt - wt)
    if dt < best_dt and mid not in used:  # BUG: 'mid not in used' inside loop
        best_dt, best_i = dt, mid
    ...
```

```python
# Uses bisect.bisect_left + outward scan (lo/hi expanding from insertion point)
idx = bisect.bisect_left(wall_times, wt)
lo, hi = idx, idx
while lo >= 0 or hi < len(summaries):
    for j in [lo, hi]:
        if 0 <= j < len(summaries) and j not in used:
            # Check candidate...
    lo -= 1
    hi += 1
```

```python
gap = max(baseline - threshold, 1e-6)  # Could be near-zero
return min(1.0, violation / gap)  # Could produce infinity
```

```python
if gap < 1e-4:
    return 0.0  # No meaningful gap to measure confidence against
```

```python
class Episode:
    def __init__(self, ...):
        # ... existing fields ...
        # NEW: Step counters at episode completion (stamped by worker)
        self.explore_step_at_completion: Optional[int] = None
        self.update_step_at_completion: Optional[int] = None
```

```python
def explore(self, ...):
    # ... play episode ...
    with self.step_counter.lock:
        self.step_counter.exploration += n_steps_episode
        # NEW: Stamp INSIDE LOCK to prevent race
        episode.explore_step_at_completion = self.step_counter.exploration
    episode.update_step_at_completion = self.step_counter.update  # Read-only from workers
```

```python
def _log_episode_summaries(self, episodes):
    fallback_explore = self.step_counter.exploration
    for episode in episodes:
        # Use per-episode snapshot if available, else fallback
        explore_step = getattr(episode, 'explore_step_at_completion', None) or fallback_explore
```

```python
"episode_id": self.episode_counter.exploration - len(episodes) + i,  # Can repeat/jump
```

```python
self._episode_summary_counter = 0  # In __init__
# ...
episode_id = self._episode_summary_counter
self._episode_summary_counter += 1  # Guaranteed unique, monotonic
```

```python
def _compute_clamp_fraction(actions: torch.Tensor, threshold: float = 0.99) -> float:
    """Compute fraction of actions near clamp boundaries."""
    if actions.numel() == 0:
        return 0.0
    clamped = (actions.abs() > threshold).float()
    return clamped.mean().item()
```

```python
self._clamp_fraction = _compute_clamp_fraction(new_actions.detach())
```

```python
self.last_metrics = {
    # ... existing metrics ...
    "clamp_fraction": self._clamp_fraction,  # NEW
}
```

```python
LOSS_HEADER_FIELDS = [
    # ... existing fields ...
    "clamp_fraction",  # NEW
]
```

```python
CLAMP_FRACTION_THRESHOLD = 0.5   # >50% = saturated
CLAMP_FRACTION_CRITICAL = 0.8   # >80% = critical

# Event: clamp_saturation
pred_clamp = [cf > CLAMP_FRACTION_THRESHOLD for cf in clamp_fractions]
i_clamp = _first_sustained(pred_clamp, sustain=sustain)
if i_clamp is not None:
    events.append(Event(name="clamp_saturation", ...))
```

```python
explore_update_map = {l.explore_step: l.update_step for l in losses}  # Last-write-wins
```

```python
for l in losses:
    if l.explore_step not in explore_update_map:  # First occurrence only
        explore_update_map[l.explore_step] = l.update_step
```

```python
# OLD: Nested function inside method, recreated per call
def log(self, data):
    def convert_to_serializable(obj): ...  # BAD

# NEW: Module-level, single definition
def _convert_to_serializable(obj): ...  # GOOD
```

```python
class CSVScalarLogger:
    def __init__(self, ...):
        self._warned_unknown_fields: set = set()  # NEW
    
    def log(self, **kwargs):
        unknown = set(kwargs.keys()) - set(self.header_fields) - self._warned_unknown_fields
        if unknown:
            self._warned_unknown_fields.update(unknown)
            logger.warning(f"Unknown fields not in header (will be dropped): {unknown}")
```

```python
if (insertion_collapsed or alpha_shutdown or stuck_regime) and pending_replay_events:
    events.extend(pending_replay_events)  # Gated - might not fire
```

```python
if pending_replay_events:
    events.extend(pending_replay_events)  # Always include
```

```python
# OLD: statistics.pstdev(deque) - O(w) per call
# NEW: Running sum/sum-of-squares
s += v; ss += v*v  # O(1) per call
variance = ss/n - (s/n)**2
```

```python
# OLD: statistics.median(deque) - creates sorted copy each call O(w log w)
# NEW: Maintain sorted_w with bisect.insort/pop O(w) insert but O(1) median
```

```python
# OLD: Try 5 regexes per line
# NEW: Fast string checks route to correct regex
if "STEP" in line:
    if "exec_action=" in line:  # V3 format
        m = STEP_RE_V3.search(line)
    elif "action=" in line:  # V2 format
        m = STEP_RE_NEW.search(line)
```

---

**User**

I did the agent mode now copy .

---

**Cursor**

Now I'll save the current versions as backups and copy the new versions from neve:

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

```python
from copy import deepcopy
from importlib import import_module
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import torch
import numpy as np
import gymnasium as gym

from .agent import Agent, StepCounter, EpisodeCounter, AgentEvalOnly
from ..algo import Algo, AlgoPlayOnly
from ..replaybuffer import ReplayBuffer, Episode
from ..util import ConfigHandler, flatten_obs


class SingleEvalOnly(AgentEvalOnly):
    def __init__(
        self,
        algo: AlgoPlayOnly,
        env_eval: gym.Env,
        device: torch.device = torch.device("cpu"),
        normalize_actions: bool = True,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.device = device
        self.algo = algo
        self.env_eval = env_eval
        self.normalize_actions = normalize_actions

        self.step_counter = StepCounter()
        self.episode_counter = EpisodeCounter()
        self.to(device)
        self._next_batch = None
        self._replay_too_small = True
        self.logger.info("Single agent initialized")

    def evaluate(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        seeds: Optional[List[int]] = None,
        options: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Episode]:
        t_start = perf_counter()
        self._log_eval(steps, step_limit, episodes, episode_limit, seeds, options)
        step_limit, episode_limit = self._log_and_convert_limits(
            "evaluation", steps, step_limit, episodes, episode_limit, seeds, options
        )
        seeds = deepcopy(seeds)
        options = deepcopy(options)
        episodes_data = []
        n_episodes = 0
        n_steps = 0

        while True:
            with self.episode_counter.lock:
                self.episode_counter.evaluation += 1

            next_seed = seeds.pop(-1) if seeds is not None else None
            next_options = options.pop(-1) if options is not None else None

            episode, n_steps_episode = self._play_episode(
                env=self.env_eval,
                action_function=self.algo.get_eval_action,
                consecutive_actions=1,
                seed=next_seed,
                options=next_options,
            )

            with self.step_counter.lock:
                self.step_counter.evaluation += n_steps_episode

            n_episodes += 1
            n_steps += n_steps_episode
            episodes_data.append(episode)

            if (
                (not seeds and not options)
                or self.step_counter.evaluation > step_limit
                or self.episode_counter.evaluation > episode_limit
            ):
                break

        t_duration = perf_counter() - t_start
        self._log_task_completion("evaluation", n_steps, t_duration, n_episodes)
        return episodes_data

    def _play_episode(
        self,
        env: gym.Env,
        action_function: Callable[[np.ndarray], np.ndarray],
        consecutive_actions: int,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Episode, int]:
        terminal = False
        truncation = False
        step_counter = 0

        self.algo.reset()
        obs, _ = env.reset(seed=seed, options=options)
        flat_obs, flat_obs_to_obs = flatten_obs(obs)
        episode = Episode(obs, flat_obs, flat_obs_to_obs, seed, options)

        while not (terminal or truncation):
            action = action_function(flat_obs)

            for _ in range(consecutive_actions):
                env_action = action.reshape(env.action_space.shape)
                if self.normalize_actions:
                    env_action = (env_action + 1) / 2 * (
                        env.action_space.high - env.action_space.low
                    ) + env.action_space.low
                obs, reward, terminal, truncation, info = env.step(env_action)
                flat_obs, _ = flatten_obs(obs)
                step_counter += 1
                env.render()
                episode.add_transition(
                    obs, flat_obs, action, reward, terminal, truncation, info
                )
                if terminal or truncation:
                    break

        return episode, step_counter

    def to(self, device: torch.device):
        self.device = device
        self.algo.to(device)

    def close(self):
        self.env_eval.close()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        normalize_actions: bool = True,
        env_eval: Optional[gym.Env] = None,
    ):
        cp = torch.load(checkpoint_path)
        confighandler = ConfigHandler()
        algo: Algo = confighandler.config_dict_to_object(cp["algo"])
        eve = import_module("eve.util")
        eve_cfh = eve.ConfigHandler()
        env_eval = env_eval or eve_cfh.config_dict_to_object(cp["env_eval"])
        agent = cls(
            algo.to_play_only(),
            env_eval,
            device,
            normalize_actions,
        )
        agent.load_checkpoint(checkpoint_path)
        return agent


class Single(SingleEvalOnly, Agent):
    def __init__(  # pylint: disable=super-init-not-called
        self,
        algo: Algo,
        env_train: gym.Env,
        env_eval: gym.Env,
        replay_buffer: ReplayBuffer,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_actions: bool = True,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.device = device
        self.algo = algo
        self.env_train = env_train
        self.env_eval = env_eval
        self.replay_buffer = replay_buffer
        self.consecutive_action_steps = consecutive_action_steps
        self.normalize_actions = normalize_actions

        self.update_error = False

        self.step_counter = StepCounter()
        self.episode_counter = EpisodeCounter()
        self.to(device)
        self._next_batch = None
        self._replay_too_small = True
        
        # Diagnostics logger (set externally by subprocess runner)
        self.diagnostics = None
        
        self.logger.info("Single agent initialized")

    def heatup(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        custom_action_low: Optional[List[float]] = None,
        custom_action_high: Optional[List[float]] = None,
    ) -> List[Episode]:
        t_start = perf_counter()
        self._log_heatup(
            steps,
            step_limit,
            episodes,
            episode_limit,
            custom_action_low=custom_action_low,
            custom_action_high=custom_action_high,
        )
        step_limit, episode_limit = self._log_and_convert_limits(
            "heatup", steps, step_limit, episodes, episode_limit
        )

        episodes_data = []

        def random_action(*args, **kwargs):  # pylint: disable=unused-argument
            env_low = self.env_train.action_space.low.reshape(-1)
            env_high = self.env_train.action_space.high.reshape(-1)

            if custom_action_low is not None:
                action_low = np.array(custom_action_low).reshape(-1)
            else:
                action_low = env_low.reshape(-1)
            if custom_action_high is not None:
                action_high = np.array(custom_action_high).reshape(-1)
            else:
                action_high = env_high.reshape(-1)
            action = np.random.uniform(action_low, action_high)

            if self.normalize_actions:
                action = 2 * (action - env_low) / (env_high - env_low) - 1

            return action

        n_episodes = 0
        n_steps = 0
        while (
            self.step_counter.heatup < step_limit
            and self.episode_counter.heatup < episode_limit
        ):
            with self.episode_counter.lock:
                self.episode_counter.heatup += 1

            episode, n_steps_episode = self._play_episode(
                env=self.env_train,
                action_function=random_action,
                consecutive_actions=self.consecutive_action_steps,
            )

            with self.step_counter.lock:
                self.step_counter.heatup += n_steps_episode
            n_steps += n_steps_episode
            n_episodes += 1
            self.replay_buffer.push(episode)
            episodes_data.append(episode)

        t_duration = perf_counter() - t_start
        self._log_task_completion("heatup", n_steps, t_duration, n_episodes)
        return episodes_data

    def explore(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
    ) -> List[Episode]:
        t_start = perf_counter()
        self._log_exploration(steps, step_limit, episodes, episode_limit)
        step_limit, episode_limit = self._log_and_convert_limits(
            "exploration", steps, step_limit, episodes, episode_limit
        )

        episodes_data = []
        n_episodes = 0
        n_steps = 0
        while (
            self.step_counter.exploration < step_limit
            and self.episode_counter.exploration < episode_limit
        ):
            with self.episode_counter.lock:
                self.episode_counter.exploration += 1

            episode, n_steps_episode = self._play_episode(
                env=self.env_train,
                action_function=self.algo.get_exploration_action,
                consecutive_actions=self.consecutive_action_steps,
            )

            with self.step_counter.lock:
                self.step_counter.exploration += n_steps_episode

            n_episodes += 1
            n_steps += n_steps_episode

            self.replay_buffer.push(episode)
            episodes_data.append(episode)

        t_duration = perf_counter() - t_start
        self._log_task_completion("exploration", n_steps, t_duration, n_episodes)
        return episodes_data

    def update(
        self, *, steps: Optional[int] = None, step_limit: Optional[int] = None
    ) -> List[List[float]]:
        t_start = perf_counter()
        self._log_update(steps, step_limit)
        step_limit, _ = self._log_and_convert_limits("update", steps, step_limit)
        results = []
        if self._replay_too_small:
            replay_len = len(self.replay_buffer)
            batch_size = self.replay_buffer.batch_size
            self._replay_too_small = replay_len <= batch_size
        if self._replay_too_small or step_limit == 0 or steps == 0:
            return []

        n_steps = 0

        while self.step_counter.update < step_limit:
            with self.step_counter.lock:
                self.step_counter.update += 1
            batch = self.replay_buffer.sample()
            result = self.algo.update(batch)
            results.append(result)
            n_steps += 1
            
            # Log diagnostics after each gradient step
            if self.diagnostics is not None:
                self.diagnostics.log_losses(
                    update_step=self.step_counter.update,
                    explore_step=self.step_counter.exploration,
                    q1_loss=float(result[0]),
                    q2_loss=float(result[1]),
                    policy_loss=float(result[2]),
                    **self.algo.last_metrics,
                )
                
                # Log probe values periodically
                if (self.diagnostics.probe_states is not None and 
                    self.step_counter.update % self.diagnostics.config.log_probe_values_every_n_steps == 0):
                    probe_values = self.algo.compute_probe_values(
                        self.diagnostics.probe_states
                    )
                    self.diagnostics.log_probe_values(
                        update_step=self.step_counter.update,
                        probe_values=probe_values,
                        explore_step=self.step_counter.exploration,  # NEW: Added for timestamp correlation
                    )
                
                # Log batch samples periodically
                if self.step_counter.update % self.diagnostics.config.log_batch_samples_every_n_steps == 0:
                    try:
                        self._log_batch_samples(batch)
                    except Exception as e:
                        self.logger.warning(f"Failed to log batch samples at step {self.step_counter.update}: {e}")

        t_duration = perf_counter() - t_start
        self._log_task_completion("update", n_steps, t_duration)
        return results

    def _log_batch_samples(self, batch):
        """Log sampled transitions from the current batch."""
        if self.diagnostics is None:
            self.logger.debug("_log_batch_samples: diagnostics is None, skipping")
            return
            
        (all_states, actions, rewards, dones, padding_mask) = batch
        
        # Log batch shape info for debugging (only first time)
        if self.step_counter.update <= 100:
            self.logger.info(
                f"_log_batch_samples: batch shapes - states={all_states.shape}, "
                f"actions={actions.shape}, rewards={rewards.shape}, dones={dones.shape}"
            )
        
        # Determine number of samples to log
        n_samples = min(
            self.diagnostics.config.n_batch_samples,
            all_states.shape[0]
        )
        
        # Sample random indices from the batch
        batch_size = all_states.shape[0]
        sample_indices = np.random.choice(batch_size, size=n_samples, replace=False).tolist()
        
        # Get next states (shifted by 1 in sequence dimension)
        seq_length = actions.shape[1]
        next_states = all_states[:, 1:seq_length+1, :] if all_states.shape[1] > seq_length else None
        
        # Compute actor/critic values for sampled transitions
        batch_samples = self.algo.compute_batch_sample_values(
            states=all_states[:, :seq_length, :],
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_states=next_states,
            sample_indices=sample_indices,
        )
        
        # Log info about computed samples
        if self.step_counter.update <= 100:
            self.logger.info(f"_log_batch_samples: computed {len(batch_samples)} samples")
        
        # Log the samples
        self.diagnostics.log_batch_samples(
            update_step=self.step_counter.update,
            explore_step=self.step_counter.exploration,
            batch_samples=batch_samples,
        )
        
        if self.step_counter.update <= 100:
            self.logger.info(f"_log_batch_samples: successfully logged at step {self.step_counter.update}")

    def explore_and_update(
        self,
        *,
        explore_steps: Optional[int] = None,
        explore_episodes: Optional[int] = None,
        explore_step_limit: Optional[int] = None,
        explore_episode_limit: Optional[int] = None,
        update_steps: Optional[int] = None,
        update_step_limit: Optional[int] = None,
    ) -> Tuple[List[Episode], List[float]]:
        explore_result = self.explore(
            steps=explore_steps,
            episodes=explore_episodes,
            step_limit=explore_step_limit,
            episode_limit=explore_episode_limit,
        )
        update_result = self.update(steps=update_steps, step_limit=update_step_limit)
        return explore_result, update_result

    def close(self):
        # Close diagnostics logger if present
        if self.diagnostics is not None:
            self.diagnostics.close()
            
        self.env_train.close()
        if id(self.env_train) != id(self.env_eval):
            self.env_eval.close()
        self.replay_buffer.close()
        del self.algo
        del self.replay_buffer

    @classmethod
    def from_checkpoint(  # pylint: disable=arguments-renamed
        cls,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_actions: bool = True,
        env_train: Optional[gym.Env] = None,
        env_eval: Optional[gym.Env] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
    ):
        cp = torch.load(checkpoint_path)
        confighandler = ConfigHandler()
        algo = confighandler.config_dict_to_object(cp["algo"])
        replay_buffer = replay_buffer or confighandler.config_dict_to_object(
            cp["replay_buffer"]
        )
        eve = import_module("eve.util")
        eve_cfh = eve.ConfigHandler()
        env_train = env_train or eve_cfh.config_dict_to_object(cp["env_train"])
        env_eval = env_eval or eve_cfh.config_dict_to_object(cp["env_eval"])
        agent = cls(
            algo,
            env_train,
            env_eval,
            replay_buffer,
            device,
            consecutive_action_steps,
            normalize_actions,
        )
        agent.load_checkpoint(checkpoint_path)
        return agent

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
            # Capture current step counters
            explore_step = self.step_counter.exploration
            update_step = self.step_counter.update
            
            with open(self._episode_summary_file, "a", encoding="utf-8") as f:
                for i, episode in enumerate(episodes):
                    try:
                        wall_time = time.time()  # Per-episode timestamp
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
                        
                        summary = {
                            "wall_time": wall_time,
                            "explore_step": explore_step,
                            "update_step": update_step,
                            "episode_id": self.episode_counter.exploration - len(episodes) + i,
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
from typing import Tuple, Dict, Any, List
import logging
import math
import numpy as np
from torch.distributions import Normal
import torch
import torch.nn.functional as F
from .algo import Algo, AlgoPlayOnly
from ..model import SACModel, SACModelPlayOnly
from ..replaybuffer import Batch


def compute_grad_norm(parameters) -> float:
    """Compute the total gradient norm for a set of parameters."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def _is_nonfinite(x: float) -> bool:
    """Check if a value is NaN or Inf."""
    return math.isnan(x) or math.isinf(x)


def _compute_clamp_fraction(actions: torch.Tensor, threshold: float = 0.99) -> float:
    """
    Compute fraction of actions that are near the clamp boundaries.
    Actions are assumed to be in [-1, 1] after tanh.
    
    Args:
        actions: Tensor of actions (any shape)
        threshold: Actions with |value| > threshold are considered clamped
        
    Returns:
        Fraction of action components that are clamped (0.0 to 1.0)
    """
    if actions.numel() == 0:
        return 0.0
    clamped = (actions.abs() > threshold).float()
    return clamped.mean().item()


class SACPlayOnly(AlgoPlayOnly):
    model: SACModelPlayOnly

    def __init__(
        self,
        model: SACModelPlayOnly,
        n_actions: int,
        action_scaling: float = 1,
        exploration_action_noise: float = 0.25,
        stochastic_eval: bool = False,
    ):
        self.logger = logging.getLogger(self.__module__)
        # HYPERPARAMETERS
        self.n_actions = n_actions
        self.exploration_action_noise = exploration_action_noise
        # Model
        self.model = model

        # REST
        self.action_scaling = action_scaling
        self.stochastic_eval = stochastic_eval

        self.device = torch.device("cpu")

    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.sample())
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
            action += np.random.normal(0, self.exploration_action_noise)
        return action
    
    def get_action_exploration(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.sample())
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
            action += np.random.normal(0, self.exploration_action_noise)
        return action, mean, log_std

    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            if self.stochastic_eval:
                std = log_std.exp()
                normal = Normal(mean, std)
                action = torch.tanh(normal.sample())
            else:
                mean, _ = self.model.policy.forward_play(torch_state)
                action = torch.tanh(mean)

            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
        return action * self.action_scaling
    
    def get_action_evaluation(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            if self.stochastic_eval:
                std = log_std.exp()
                normal = Normal(mean, std)
                action = torch.tanh(normal.sample())
            else:
                mean, _ = self.model.policy.forward_play(torch_state)
                action = torch.tanh(mean)

            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
        return action * self.action_scaling, mean, log_std

    def to(self, device: torch.device):
        super().to(device)
        self.model.to(device)

    def reset(self) -> None:
        self.model.reset()

    def close(self):
        self.model.close()


class SAC(Algo):
    model: SACModel

    def __init__(
        self,
        model: SACModel,
        n_actions: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scaling: float = 1,
        action_scaling: float = 1,
        exploration_action_noise: float = 0.25,
        stochastic_eval: bool = False,
    ):
        self.logger = logging.getLogger(self.__module__)
        # HYPERPARAMETERS
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.exploration_action_noise = exploration_action_noise
        # Model
        self.model = model

        # REST
        self.reward_scaling = reward_scaling
        self.action_scaling = action_scaling
        self.stochastic_eval = stochastic_eval

        self.device = torch.device("cpu")
        self.update_step = 0

        # ENTROPY TEMPERATURE
        self.alpha = torch.ones(1)
        self.target_entropy = -torch.ones(1) * n_actions

        # DIAGNOSTICS: Store metrics from last update for logging
        self.last_metrics: Dict[str, float] = {}
        
        # DIAGNOSTICS: NaN/Inf sentinel counters
        self._nonfinite_q_loss_count = 0
        self._nonfinite_policy_loss_count = 0
        self._nonfinite_grad_count = 0

    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.sample())
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
            action += np.random.normal(0, self.exploration_action_noise)
        return action

    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            if self.stochastic_eval:
                std = log_std.exp()
                normal = Normal(mean, std)
                action = torch.tanh(normal.sample())
            else:
                mean, _ = self.model.policy.forward_play(torch_state)
                action = torch.tanh(mean)

            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
        return action * self.action_scaling

    def update(self, batch: Batch) -> Tuple[float, float, float]:
        (all_states, actions, rewards, dones, padding_mask) = batch
        # actions /= self.action_scaling

        all_states = all_states.to(dtype=torch.float32, device=self.device)
        actions = actions.to(dtype=torch.float32, device=self.device)
        rewards = rewards.to(dtype=torch.float32, device=self.device)
        dones = dones.to(dtype=torch.float32, device=self.device)

        if padding_mask is not None:
            padding_mask = padding_mask.to(dtype=torch.float32, device=self.device)

        seq_length = actions.shape[1]
        states = torch.narrow(all_states, dim=1, start=0, length=seq_length)

        # use all_states for next_actions and next_log_pi for proper hidden_state initilaization
        expected_q = self._get_expected_q(
            all_states, rewards, dones, padding_mask, seq_length
        )

        # q1 update
        q1_loss = self._update_q1(actions, padding_mask, states, expected_q)

        # q2 update
        q2_loss = self._update_q2(actions, padding_mask, states, expected_q)

        log_pi, policy_loss = self._update_policy(padding_mask, states)

        self.model.update_target_q(self.tau)

        alpha_loss = self._update_alpha(log_pi)

        self.update_step += 1
        
        # Check for NaN/Inf in losses
        q1_loss_val = q1_loss.detach().cpu().item()
        q2_loss_val = q2_loss.detach().cpu().item()
        policy_loss_val = policy_loss.detach().cpu().item()
        
        nonfinite_q = _is_nonfinite(q1_loss_val) or _is_nonfinite(q2_loss_val)
        nonfinite_pol = _is_nonfinite(policy_loss_val)
        nonfinite_grad = (_is_nonfinite(self._grad_norm_q1) or 
                         _is_nonfinite(self._grad_norm_q2) or 
                         _is_nonfinite(self._grad_norm_policy))
        
        if nonfinite_q:
            self._nonfinite_q_loss_count += 1
            self.logger.warning(f"ALERT: NaN/Inf in Q loss at step {self.update_step} (count={self._nonfinite_q_loss_count})")
        if nonfinite_pol:
            self._nonfinite_policy_loss_count += 1
            self.logger.warning(f"ALERT: NaN/Inf in policy loss at step {self.update_step} (count={self._nonfinite_policy_loss_count})")
        if nonfinite_grad:
            self._nonfinite_grad_count += 1
            self.logger.warning(f"ALERT: NaN/Inf in gradients at step {self.update_step} (count={self._nonfinite_grad_count})")
        
        # Populate diagnostics metrics
        lr_policy = 0.0
        if self.model.policy_scheduler is not None:
            try:
                lr_policy = self.model.policy_scheduler.get_last_lr()[0]
            except Exception:
                lr_policy = 0.0
                
        self.last_metrics = {
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.detach().cpu().item(),
            "log_pi_mean": log_pi.mean().detach().cpu().item(),
            "log_pi_std": log_pi.std().detach().cpu().item(),
            "entropy_proxy": -log_pi.mean().detach().cpu().item(),
            "q1_mean": self._q1_mean.cpu().item(),
            "q2_mean": self._q2_mean.cpu().item(),
            "target_q_mean": expected_q.mean().detach().cpu().item(),
            "min_q_mean": self._min_q_mean.cpu().item(),
            "grad_norm_q1": self._grad_norm_q1,
            "grad_norm_q2": self._grad_norm_q2,
            "grad_norm_policy": self._grad_norm_policy,
            "lr_policy": lr_policy,
            # NEW: Enhanced diagnostics
            "target_entropy": self.target_entropy.item(),
            "log_alpha": self.model.log_alpha.item(),
            # NaN/Inf counters
            "nonfinite_q_loss_count": self._nonfinite_q_loss_count,
            "nonfinite_policy_loss_count": self._nonfinite_policy_loss_count,
            "nonfinite_grad_count": self._nonfinite_grad_count,
        }
        
        return [
            q1_loss.detach().cpu().numpy(),
            q2_loss.detach().cpu().numpy(),
            policy_loss.detach().cpu().numpy(),
        ]

    def _update_alpha(self, log_pi) -> torch.Tensor:
        alpha_loss = (
            self.model.log_alpha * (-log_pi - self.target_entropy).detach()
        ).mean()
        self.model.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.model.alpha_optimizer.step()

        self.alpha = self.model.log_alpha.exp()
        return alpha_loss

    def _update_policy(self, padding_mask, states):
        new_actions, log_pi = self._get_update_action(states)
        q1 = self.model.q1(states, new_actions)
        q2 = self.model.q2(states, new_actions)
        min_q = torch.min(q1, q2)

        if padding_mask is not None:
            min_q *= padding_mask
            log_pi *= padding_mask

        policy_loss = (self.alpha * log_pi - min_q).mean()

        self.model.policy_optimizer.zero_grad()
        policy_loss.backward()
        # Compute gradient norm before optimizer step
        self._grad_norm_policy = compute_grad_norm(self.model.policy.parameters())
        self._min_q_mean = min_q.mean().detach()
        self.model.policy_optimizer.step()
        if self.model.policy_scheduler:
            self.model.policy_scheduler.step()
        return log_pi, policy_loss

    def _update_q2(self, actions, padding_mask, states, expected_q):
        curr_q2 = self.model.q2(states, actions)
        if padding_mask is not None:
            curr_q2 *= padding_mask
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        self.model.q2_optimizer.zero_grad()
        q2_loss.backward()
        # Compute gradient norm before optimizer step
        self._grad_norm_q2 = compute_grad_norm(self.model.q2.parameters())
        self._q2_mean = curr_q2.mean().detach()
        self.model.q2_optimizer.step()
        if self.model.q2_scheduler:
            self.model.q2_scheduler.step()
        return q2_loss

    def _update_q1(self, actions, padding_mask, states, expected_q):
        curr_q1 = self.model.q1(states, actions)
        if padding_mask is not None:
            curr_q1 *= padding_mask
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())

        self.model.q1_optimizer.zero_grad()
        q1_loss.backward()
        # Compute gradient norm before optimizer step
        self._grad_norm_q1 = compute_grad_norm(self.model.q1.parameters())
        self._q1_mean = curr_q1.mean().detach()
        self.model.q1_optimizer.step()
        if self.model.q1_scheduler:
            self.model.q1_scheduler.step()
        return q1_loss

    def _get_expected_q(self, all_states, rewards, dones, padding_mask, seq_length):
        next_actions, next_log_pi = self._get_update_action(all_states)

        with torch.no_grad():
            next_target_q1 = self.model.target_q1(all_states, next_actions)
            next_target_q2 = self.model.target_q2(all_states, next_actions)

        next_target_q = (
            torch.min(next_target_q1, next_target_q2) - self.alpha * next_log_pi
        )
        # only use next_state for next_q_target
        next_target_q = torch.narrow(next_target_q, dim=1, start=1, length=seq_length)
        expected_q = rewards + (1 - dones) * self.gamma * next_target_q
        if padding_mask is not None:
            expected_q *= padding_mask
        return expected_q

    # epsilon makes sure that log(0) does not occur
    def _get_update_action(
        self, state_batch: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_batch, log_std = self.model.policy(state_batch)
        std_batch = log_std.exp()

        normal = Normal(mean_batch, std_batch)
        z = normal.rsample()
        action_batch = torch.tanh(z)

        log_pi_batch = normal.log_prob(z) - torch.log(1 - action_batch.pow(2) + epsilon)
        log_pi_batch = log_pi_batch.sum(-1, keepdim=True)

        # log_pi_batch = torch.sum(normal.log_prob(z), dim=-1, keepdim=True) - torch.sum(
        #        torch.log(1 - action_batch.pow(2) + epsilon), dim=-1, keepdim=True)

        return action_batch, log_pi_batch

    def compute_probe_values(self, probe_states: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute policy and critic values for probe states.
        
        Used for diagnostics to track how the policy and critics evolve.
        
        Args:
            probe_states: Tensor of shape (n_probes, seq_len, obs_dim) or (n_probes, obs_dim)
            
        Returns:
            Dictionary with policy_mean, policy_log_std, action_det, q1, q2 arrays
        """
        with torch.no_grad():
            # Ensure proper shape: (batch, seq, features)
            if probe_states.dim() == 2:
                probe_states = probe_states.unsqueeze(1)
                
            probe_states = probe_states.to(device=self.device, dtype=torch.float32)
            
            # Get policy outputs
            mean, log_std = self.model.policy(probe_states)
            action_det = torch.tanh(mean)
            
            # Get critic values for deterministic action
            q1 = self.model.q1(probe_states, action_det)
            q2 = self.model.q2(probe_states, action_det)
            
        return {
            "policy_mean": mean.cpu().numpy(),
            "policy_log_std": log_std.cpu().numpy(),
            "action_det": action_det.cpu().numpy(),
            "q1": q1.cpu().numpy(),
            "q2": q2.cpu().numpy(),
        }

    def compute_batch_sample_values(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
        sample_indices: list,
    ) -> List[Dict[str, Any]]:
        """
        Compute actor/critic values for sampled transitions from the training batch.
        
        Args:
            states: Batch states tensor (batch, seq, obs_dim)
            actions: Batch actions tensor (batch, seq, action_dim)
            rewards: Batch rewards tensor (batch, seq, 1)
            dones: Batch done flags tensor (batch, seq, 1)
            next_states: Next states (states[:, 1:, :] shifted)
            sample_indices: Indices of samples to extract
            
        Returns:
            List of dictionaries, each containing transition info and actor/critic values
        """
        samples = []
        
        # Move tensors to device for model forward pass
        states = states.to(device=self.device, dtype=torch.float32)
        actions = actions.to(device=self.device, dtype=torch.float32)
        if next_states is not None:
            next_states = next_states.to(device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            for idx in sample_indices:
                if idx >= states.shape[0]:
                    continue
                    
                # Extract single transition (take first timestep if sequence)
                s_t = states[idx:idx+1, 0:1, :]  # (1, 1, obs_dim)
                a_t = actions[idx, 0, :]  # (action_dim,)
                r_t = rewards[idx, 0, 0].item() if rewards.dim() > 2 else rewards[idx, 0].item()
                done_t = dones[idx, 0, 0].item() if dones.dim() > 2 else dones[idx, 0].item()
                
                # Get next state if available
                if next_states is not None and idx < next_states.shape[0]:
                    s_next = next_states[idx:idx+1, 0:1, :]
                else:
                    s_next = None
                    
                # Compute actor outputs for s_t
                mean, log_std = self.model.policy(s_t)
                action_det = torch.tanh(mean)
                
                # Compute Q values for (s_t, a_taken)
                a_taken = a_t.unsqueeze(0).unsqueeze(0)  # (1, 1, action_dim)
                q1_taken = self.model.q1(s_t, a_taken).squeeze().item()
                q2_taken = self.model.q2(s_t, a_taken).squeeze().item()
                
                # Compute Q values for (s_t, actor_action)
                q1_actor = self.model.q1(s_t, action_det).squeeze().item()
                q2_actor = self.model.q2(s_t, action_det).squeeze().item()
                
                # Extract action components
                actor_det_np = action_det.squeeze().cpu().numpy()
                action_taken_np = a_t.cpu().numpy()
                mean_np = mean.squeeze().cpu().numpy()
                log_std_np = log_std.squeeze().cpu().numpy()
                
                # Compute rotation stats (action[1] is typically rotation for 2-action envs)
                rotation_taken = action_taken_np[1] if len(action_taken_np) > 1 else 0.0
                rotation_actor = actor_det_np[1] if len(actor_det_np) > 1 else 0.0
                rotation_mean_val = mean_np[1] if len(mean_np) > 1 else 0.0
                rotation_log_std_val = log_std_np[1] if len(log_std_np) > 1 else 0.0
                
                # Compute clamp fraction for this sample's actor action
                clamp_frac = _compute_clamp_fraction(action_det, threshold=0.99)
                
                sample_data = {
                    "batch_idx": idx,
                    "state": s_t.squeeze().cpu().numpy().tolist(),
                    "action_taken": action_taken_np.tolist(),
                    "reward": r_t,
                    "done": done_t,
                    "next_state": s_next.squeeze().cpu().numpy().tolist() if s_next is not None else None,
                    "actor_mean": mean_np.tolist(),
                    "actor_log_std": log_std_np.tolist(),
                    "actor_det_action": actor_det_np.tolist(),
                    "q1_action_taken": q1_taken,
                    "q2_action_taken": q2_taken,
                    "q1_actor_action": q1_actor,
                    "q2_actor_action": q2_actor,
                    "min_q_taken": min(q1_taken, q2_taken),
                    "min_q_actor": min(q1_actor, q2_actor),
                    # NEW: Rotation stats
                    "rotation_taken": float(rotation_taken),
                    "rotation_actor": float(rotation_actor),
                    "rotation_mean": float(rotation_mean_val),
                    "rotation_log_std": float(rotation_log_std_val),
                    # NEW: Clamp fraction
                    "clamp_fraction": clamp_frac,
                }
                samples.append(sample_data)
                
        return samples

    def to(self, device: torch.device):
        super().to(device)
        self.alpha = self.alpha.to(device)
        self.target_entropy = self.target_entropy.to(device)
        self.model.to(device)

    def reset(self) -> None:
        self.model.reset()

    def close(self):
        self.model.close()

    def to_play_only(self):
        return SACPlayOnly(
            self.model.to_play_only(),
            self.n_actions,
            self.action_scaling,
            self.exploration_action_noise,
        )

```

```python
"""
Diagnostics logging utilities for SAC training.

Provides structured logging for:
- Per-update-step losses (CSV)
- Probe state values (JSONL)
- TensorBoard scalars and histograms
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import csv
import json
import os
import logging
import time
import numpy as np
import torch


@dataclass
class DiagnosticsConfig:
    """Configuration for diagnostics logging."""
    enabled: bool = True
    diagnostics_folder: str = None
    log_losses_every_n_steps: int = 1
    log_probe_values_every_n_steps: int = 100
    log_batch_samples_every_n_steps: int = 100  # Log batch samples every N update steps
    n_batch_samples: int = 3  # Number of transitions to sample from each batch
    tensorboard_enabled: bool = True
    probe_plot_every_n_steps: int = 10000
    policy_snapshot_every_n_steps: int = 10000
    flush_every_n_rows: int = 100
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiagnosticsConfig":
        """Create config from dictionary."""
        if d is None:
            return cls(enabled=False)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CSVScalarLogger:
    """Append-only CSV logger for scalar values."""
    
    def __init__(
        self,
        path: str,
        header_fields: List[str],
        flush_every_n: int = 100,
    ):
        self.path = path
        self.header_fields = header_fields
        self.flush_every_n = flush_every_n
        self._row_count = 0
        self._file = None
        self._writer = None
        self._initialized = False
        
    def _initialize(self):
        """Initialize file and write header if needed."""
        if self._initialized:
            return
            
        file_exists = os.path.isfile(self.path)
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        
        if not file_exists:
            self._writer.writerow(self.header_fields)
            self._file.flush()
            
        self._initialized = True
        
    def log(self, **kwargs):
        """Log a row of scalar values."""
        self._initialize()
        
        row = [kwargs.get(field, "") for field in self.header_fields]
        self._writer.writerow(row)
        self._row_count += 1
        
        if self._row_count % self.flush_every_n == 0:
            self._file.flush()
            
    def flush(self):
        """Force flush to disk."""
        if self._file is not None:
            self._file.flush()
            
    def close(self):
        """Close the file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None


class JSONLLogger:
    """Append-only JSONL logger for structured data (vectors, dicts)."""
    
    def __init__(
        self,
        path: str,
        flush_every_n: int = 10,
    ):
        self.path = path
        self.flush_every_n = flush_every_n
        self._row_count = 0
        self._file = None
        self._initialized = False
        self._logger = logging.getLogger(__name__)
        
    def _initialize(self):
        """Initialize file."""
        if self._initialized:
            return
        self._file = open(self.path, "a", encoding="utf-8")
        self._initialized = True
        self._logger.info(f"JSONLLogger initialized: {self.path}")
        
    def log(self, data: Dict[str, Any]):
        """Log a JSON object. numpy arrays are converted to lists."""
        self._initialize()
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            else:
                return obj
        
        try:
            serializable = convert_to_serializable(data)
            json_str = json.dumps(serializable)
            self._file.write(json_str + "\n")
            self._row_count += 1
            
            # Flush immediately for first few rows to ensure data is written
            if self._row_count <= 3 or self._row_count % self.flush_every_n == 0:
                self._file.flush()
                
        except Exception as e:
            self._logger.error(f"JSONLLogger.log failed: {e}, data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            
    def flush(self):
        """Force flush to disk."""
        if self._file is not None:
            self._file.flush()
            
    def close(self):
        """Close the file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


class TensorBoardLogger:
    """Wrapper around PyTorch's SummaryWriter."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._writer = None
        self._initialized = False
        
    def _initialize(self):
        """Lazy initialization of SummaryWriter."""
        if self._initialized:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=self.log_dir)
            self._initialized = True
        except ImportError:
            logging.warning("TensorBoard not available. Skipping TensorBoard logging.")
            self._writer = None
            self._initialized = True
            
    def add_scalar(self, tag: str, value: float, global_step: int):
        """Add a scalar value."""
        self._initialize()
        if self._writer is not None:
            self._writer.add_scalar(tag, value, global_step)
            
    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int):
        """Add multiple scalars under a main tag."""
        self._initialize()
        if self._writer is not None:
            self._writer.add_scalars(main_tag, tag_scalar_dict, global_step)
            
    def add_histogram(self, tag: str, values: np.ndarray, global_step: int):
        """Add a histogram of values."""
        self._initialize()
        if self._writer is not None:
            self._writer.add_histogram(tag, values, global_step)
            
    def flush(self):
        """Force flush to disk."""
        if self._writer is not None:
            self._writer.flush()
            
    def close(self):
        """Close the writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class DiagnosticsLogger:
    """
    Main diagnostics logger that combines CSV, JSONL, and TensorBoard logging.
    
    This is instantiated in the trainer subprocess and handles all diagnostics logging.
    """
    
    # Standard header fields for loss logging
    LOSS_HEADER_FIELDS = [
        "wall_time",  # Unix timestamp for time correlation
        "update_step",
        "explore_step",
        "q1_loss",
        "q2_loss", 
        "policy_loss",
        "alpha",
        "alpha_loss",
        "log_pi_mean",
        "log_pi_std",
        "entropy_proxy",
        "q1_mean",
        "q2_mean",
        "target_q_mean",
        "min_q_mean",
        "grad_norm_q1",
        "grad_norm_q2",
        "grad_norm_policy",
        "lr_policy",
        # NEW: Enhanced diagnostics
        "target_entropy",
        "log_alpha",
        # NEW: NaN/Inf sentinels
        "nonfinite_q_loss_count",
        "nonfinite_policy_loss_count",
        "nonfinite_grad_count",
    ]
    
    def __init__(
        self,
        config: DiagnosticsConfig,
        process_name: str = "trainer",
    ):
        self.config = config
        self.process_name = process_name
        self.logger = logging.getLogger(__name__)
        
        # Probe states for evaluation (set externally)
        self.probe_states: Optional[torch.Tensor] = None
        self.probe_states_np: Optional[np.ndarray] = None
        
        if not config.enabled or config.diagnostics_folder is None:
            self._loss_logger = None
            self._probe_logger = None
            self._batch_logger = None
            self._tb_logger = None
            self.logger.info("Diagnostics logging disabled")
            return
            
        # Create subdirectories
        csv_dir = os.path.join(config.diagnostics_folder, "csv")
        tb_dir = os.path.join(config.diagnostics_folder, "tensorboard")
        
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        
        # Initialize loggers
        self._loss_logger = CSVScalarLogger(
            path=os.path.join(csv_dir, f"losses_{process_name}.csv"),
            header_fields=self.LOSS_HEADER_FIELDS,
            flush_every_n=config.flush_every_n_rows,
        )
        
        self._probe_logger = JSONLLogger(
            path=os.path.join(csv_dir, f"probe_values_{process_name}.jsonl"),
            flush_every_n=10,
        )
        
        # Batch transitions logger - logs sampled transitions from replay buffer
        self._batch_logger = JSONLLogger(
            path=os.path.join(csv_dir, f"batch_samples_{process_name}.jsonl"),
            flush_every_n=100,
        )
        
        if config.tensorboard_enabled:
            self._tb_logger = TensorBoardLogger(
                log_dir=os.path.join(tb_dir, process_name),
            )
        else:
            self._tb_logger = None
            
        self.logger.info(f"Diagnostics logging initialized for {process_name}")
        
    def log_losses(
        self,
        update_step: int,
        explore_step: int,
        q1_loss: float,
        q2_loss: float,
        policy_loss: float,
        **extra_metrics,
    ):
        """Log loss values and additional metrics."""
        if self._loss_logger is None:
            return
            
        # Check if we should log this step
        if update_step % self.config.log_losses_every_n_steps != 0:
            return
            
        # Build the log entry
        log_data = {
            "wall_time": time.time(),  # Unix timestamp for time correlation
            "update_step": update_step,
            "explore_step": explore_step,
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "policy_loss": policy_loss,
        }
        log_data.update(extra_metrics)
        
        self._loss_logger.log(**log_data)
        
        # Also log to TensorBoard
        if self._tb_logger is not None:
            self._tb_logger.add_scalar("losses/q1_loss", q1_loss, update_step)
            self._tb_logger.add_scalar("losses/q2_loss", q2_loss, update_step)
            self._tb_logger.add_scalar("losses/policy_loss", policy_loss, update_step)
            
            for key, value in extra_metrics.items():
                if isinstance(value, (int, float)):
                    self._tb_logger.add_scalar(f"metrics/{key}", value, update_step)
                    
    def log_probe_values(
        self,
        update_step: int,
        probe_values: Dict[str, np.ndarray],
        explore_step: int = None,  # NEW: Added for timestamp correlation
    ):
        """Log probe state evaluation values."""
        if self._probe_logger is None:
            return
            
        log_data = {
            "wall_time": time.time(),  # Unix timestamp for time correlation
            "update_step": update_step,
            "explore_step": explore_step,
            **probe_values,
        }
        self._probe_logger.log(log_data)
        
        # Log summary stats to TensorBoard
        if self._tb_logger is not None:
            for key, values in probe_values.items():
                if isinstance(values, np.ndarray):
                    # Log mean of the first probe state's values
                    if len(values.shape) > 1:
                        first_probe = values[0]
                    else:
                        first_probe = values
                    self._tb_logger.add_scalar(
                        f"probe/{key}_mean", 
                        np.mean(first_probe), 
                        update_step
                    )

    def log_batch_samples(
        self,
        update_step: int,
        explore_step: int,
        batch_samples: List[Dict[str, Any]],
    ):
        """
        Log sampled transitions from the replay batch.
        
        Each sample contains:
        - state (s_t): flattened observation
        - action (a_t): action taken
        - reward (r_t): reward received
        - next_state (s_{t+1}): next observation
        - done: terminal flag
        - actor_values: policy outputs for s_t (mean, log_std, det_action)
        - critic_values: Q1, Q2 for (s_t, a_t) and (s_t, actor_action)
        """
        if self._batch_logger is None:
            self.logger.warning(f"log_batch_samples called but _batch_logger is None at step {update_step}")
            return
            
        try:
            log_data = {
                "wall_time": time.time(),  # Unix timestamp for time correlation
                "update_step": update_step,
                "explore_step": explore_step,
                "n_samples": len(batch_samples),
                "samples": batch_samples,
            }
            self._batch_logger.log(log_data)
            
            # Log success for first few calls
            if update_step <= 300:
                self.logger.info(f"log_batch_samples: logged {len(batch_samples)} samples at step {update_step}")
        except Exception as e:
            self.logger.error(f"log_batch_samples failed at step {update_step}: {e}")
                    
    def set_probe_states(self, probe_states: np.ndarray, device: torch.device = None):
        """Set the probe states for evaluation."""
        self.probe_states_np = probe_states
        if device is not None:
            self.probe_states = torch.as_tensor(
                probe_states, dtype=torch.float32, device=device
            )
        else:
            self.probe_states = torch.as_tensor(
                probe_states, dtype=torch.float32
            )
        self.logger.info(f"Set {len(probe_states)} probe states for evaluation")
        
    def save_probe_states(self, path: str):
        """Save probe states to disk."""
        if self.probe_states_np is not None:
            np.savez(path, probe_states=self.probe_states_np)
            self.logger.info(f"Saved probe states to {path}")
            
    def flush(self):
        """Flush all loggers."""
        if self._loss_logger is not None:
            self._loss_logger.flush()
        if self._probe_logger is not None:
            self._probe_logger.flush()
        if self._batch_logger is not None:
            self._batch_logger.flush()
        if self._tb_logger is not None:
            self._tb_logger.flush()
            
    def close(self):
        """Close all loggers."""
        if self._loss_logger is not None:
            self._loss_logger.close()
        if self._probe_logger is not None:
            self._probe_logger.close()
        if self._batch_logger is not None:
            self._batch_logger.close()
        if self._tb_logger is not None:
            self._tb_logger.close()

```

Now let me copy the large diagnose_collapse files using terminal commands and write the eve_rl files:

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

```python
from copy import deepcopy
from importlib import import_module
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import torch
import numpy as np
import gymnasium as gym

from .agent import Agent, StepCounter, EpisodeCounter, AgentEvalOnly
from ..algo import Algo, AlgoPlayOnly
from ..replaybuffer import ReplayBuffer, Episode
from ..util import ConfigHandler, flatten_obs


class SingleEvalOnly(AgentEvalOnly):
    def __init__(
        self,
        algo: AlgoPlayOnly,
        env_eval: gym.Env,
        device: torch.device = torch.device("cpu"),
        normalize_actions: bool = True,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.device = device
        self.algo = algo
        self.env_eval = env_eval
        self.normalize_actions = normalize_actions

        self.step_counter = StepCounter()
        self.episode_counter = EpisodeCounter()
        self.to(device)
        self._next_batch = None
        self._replay_too_small = True
        self.logger.info("Single agent initialized")

    def evaluate(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        seeds: Optional[List[int]] = None,
        options: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Episode]:
        t_start = perf_counter()
        self._log_eval(steps, step_limit, episodes, episode_limit, seeds, options)
        step_limit, episode_limit = self._log_and_convert_limits(
            "evaluation", steps, step_limit, episodes, episode_limit, seeds, options
        )
        seeds = deepcopy(seeds)
        options = deepcopy(options)
        episodes_data = []
        n_episodes = 0
        n_steps = 0

        while True:
            with self.episode_counter.lock:
                self.episode_counter.evaluation += 1

            next_seed = seeds.pop(-1) if seeds is not None else None
            next_options = options.pop(-1) if options is not None else None

            episode, n_steps_episode = self._play_episode(
                env=self.env_eval,
                action_function=self.algo.get_eval_action,
                consecutive_actions=1,
                seed=next_seed,
                options=next_options,
            )

            with self.step_counter.lock:
                self.step_counter.evaluation += n_steps_episode

            n_episodes += 1
            n_steps += n_steps_episode
            episodes_data.append(episode)

            if (
                (not seeds and not options)
                or self.step_counter.evaluation > step_limit
                or self.episode_counter.evaluation > episode_limit
            ):
                break

        t_duration = perf_counter() - t_start
        self._log_task_completion("evaluation", n_steps, t_duration, n_episodes)
        return episodes_data

    def _play_episode(
        self,
        env: gym.Env,
        action_function: Callable[[np.ndarray], np.ndarray],
        consecutive_actions: int,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Episode, int]:
        terminal = False
        truncation = False
        step_counter = 0

        self.algo.reset()
        obs, _ = env.reset(seed=seed, options=options)
        flat_obs, flat_obs_to_obs = flatten_obs(obs)
        episode = Episode(obs, flat_obs, flat_obs_to_obs, seed, options)

        while not (terminal or truncation):
            action = action_function(flat_obs)

            for _ in range(consecutive_actions):
                env_action = action.reshape(env.action_space.shape)
                if self.normalize_actions:
                    env_action = (env_action + 1) / 2 * (
                        env.action_space.high - env.action_space.low
                    ) + env.action_space.low
                obs, reward, terminal, truncation, info = env.step(env_action)
                flat_obs, _ = flatten_obs(obs)
                step_counter += 1
                env.render()
                episode.add_transition(
                    obs, flat_obs, action, reward, terminal, truncation, info
                )
                if terminal or truncation:
                    break

        return episode, step_counter

    def to(self, device: torch.device):
        self.device = device
        self.algo.to(device)

    def close(self):
        self.env_eval.close()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        normalize_actions: bool = True,
        env_eval: Optional[gym.Env] = None,
    ):
        cp = torch.load(checkpoint_path)
        confighandler = ConfigHandler()
        algo: Algo = confighandler.config_dict_to_object(cp["algo"])
        eve = import_module("eve.util")
        eve_cfh = eve.ConfigHandler()
        env_eval = env_eval or eve_cfh.config_dict_to_object(cp["env_eval"])
        agent = cls(
            algo.to_play_only(),
            env_eval,
            device,
            normalize_actions,
        )
        agent.load_checkpoint(checkpoint_path)
        return agent


class Single(SingleEvalOnly, Agent):
    def __init__(  # pylint: disable=super-init-not-called
        self,
        algo: Algo,
        env_train: gym.Env,
        env_eval: gym.Env,
        replay_buffer: ReplayBuffer,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_actions: bool = True,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.device = device
        self.algo = algo
        self.env_train = env_train
        self.env_eval = env_eval
        self.replay_buffer = replay_buffer
        self.consecutive_action_steps = consecutive_action_steps
        self.normalize_actions = normalize_actions

        self.update_error = False

        self.step_counter = StepCounter()
        self.episode_counter = EpisodeCounter()
        self.to(device)
        self._next_batch = None
        self._replay_too_small = True
        
        # Diagnostics logger (set externally by subprocess runner)
        self.diagnostics = None
        
        self.logger.info("Single agent initialized")

    def heatup(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        custom_action_low: Optional[List[float]] = None,
        custom_action_high: Optional[List[float]] = None,
    ) -> List[Episode]:
        t_start = perf_counter()
        self._log_heatup(
            steps,
            step_limit,
            episodes,
            episode_limit,
            custom_action_low=custom_action_low,
            custom_action_high=custom_action_high,
        )
        step_limit, episode_limit = self._log_and_convert_limits(
            "heatup", steps, step_limit, episodes, episode_limit
        )

        episodes_data = []

        def random_action(*args, **kwargs):  # pylint: disable=unused-argument
            env_low = self.env_train.action_space.low.reshape(-1)
            env_high = self.env_train.action_space.high.reshape(-1)

            if custom_action_low is not None:
                action_low = np.array(custom_action_low).reshape(-1)
            else:
                action_low = env_low.reshape(-1)
            if custom_action_high is not None:
                action_high = np.array(custom_action_high).reshape(-1)
            else:
                action_high = env_high.reshape(-1)
            action = np.random.uniform(action_low, action_high)

            if self.normalize_actions:
                action = 2 * (action - env_low) / (env_high - env_low) - 1

            return action

        n_episodes = 0
        n_steps = 0
        while (
            self.step_counter.heatup < step_limit
            and self.episode_counter.heatup < episode_limit
        ):
            with self.episode_counter.lock:
                self.episode_counter.heatup += 1

            episode, n_steps_episode = self._play_episode(
                env=self.env_train,
                action_function=random_action,
                consecutive_actions=self.consecutive_action_steps,
            )

            with self.step_counter.lock:
                self.step_counter.heatup += n_steps_episode
            n_steps += n_steps_episode
            n_episodes += 1
            self.replay_buffer.push(episode)
            episodes_data.append(episode)

        t_duration = perf_counter() - t_start
        self._log_task_completion("heatup", n_steps, t_duration, n_episodes)
        return episodes_data

    def explore(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
    ) -> List[Episode]:
        t_start = perf_counter()
        self._log_exploration(steps, step_limit, episodes, episode_limit)
        step_limit, episode_limit = self._log_and_convert_limits(
            "exploration", steps, step_limit, episodes, episode_limit
        )

        episodes_data = []
        n_episodes = 0
        n_steps = 0
        while (
            self.step_counter.exploration < step_limit
            and self.episode_counter.exploration < episode_limit
        ):
            with self.episode_counter.lock:
                self.episode_counter.exploration += 1

            episode, n_steps_episode = self._play_episode(
                env=self.env_train,
                action_function=self.algo.get_exploration_action,
                consecutive_actions=self.consecutive_action_steps,
            )

            with self.step_counter.lock:
                self.step_counter.exploration += n_steps_episode
                # Stamp episode with step counters at completion time (inside lock
                # to avoid race where another worker increments before we read)
                episode.explore_step_at_completion = self.step_counter.exploration
            episode.update_step_at_completion = self.step_counter.update

            n_episodes += 1
            n_steps += n_steps_episode

            self.replay_buffer.push(episode)
            episodes_data.append(episode)

        t_duration = perf_counter() - t_start
        self._log_task_completion("exploration", n_steps, t_duration, n_episodes)
        return episodes_data

    def update(
        self, *, steps: Optional[int] = None, step_limit: Optional[int] = None
    ) -> List[List[float]]:
        t_start = perf_counter()
        self._log_update(steps, step_limit)
        step_limit, _ = self._log_and_convert_limits("update", steps, step_limit)
        results = []
        if self._replay_too_small:
            replay_len = len(self.replay_buffer)
            batch_size = self.replay_buffer.batch_size
            self._replay_too_small = replay_len <= batch_size
        if self._replay_too_small or step_limit == 0 or steps == 0:
            return []

        n_steps = 0

        while self.step_counter.update < step_limit:
            with self.step_counter.lock:
                self.step_counter.update += 1
            batch = self.replay_buffer.sample()
            result = self.algo.update(batch)
            results.append(result)
            n_steps += 1
            
            # Log diagnostics after each gradient step
            if self.diagnostics is not None:
                self.diagnostics.log_losses(
                    update_step=self.step_counter.update,
                    explore_step=self.step_counter.exploration,
                    q1_loss=float(result[0]),
                    q2_loss=float(result[1]),
                    policy_loss=float(result[2]),
                    **self.algo.last_metrics,
                )
                
                # Log probe values periodically
                if (self.diagnostics.probe_states is not None and 
                    self.step_counter.update % self.diagnostics.config.log_probe_values_every_n_steps == 0):
                    probe_values = self.algo.compute_probe_values(
                        self.diagnostics.probe_states
                    )
                    self.diagnostics.log_probe_values(
                        update_step=self.step_counter.update,
                        probe_values=probe_values,
                        explore_step=self.step_counter.exploration,  # NEW: Added for timestamp correlation
                    )
                
                # Log batch samples periodically
                if self.step_counter.update % self.diagnostics.config.log_batch_samples_every_n_steps == 0:
                    try:
                        self._log_batch_samples(batch)
                    except Exception as e:
                        self.logger.warning(f"Failed to log batch samples at step {self.step_counter.update}: {e}")

        t_duration = perf_counter() - t_start
        self._log_task_completion("update", n_steps, t_duration)
        return results

    def _log_batch_samples(self, batch):
        """Log sampled transitions from the current batch."""
        if self.diagnostics is None:
            self.logger.debug("_log_batch_samples: diagnostics is None, skipping")
            return
            
        (all_states, actions, rewards, dones, padding_mask) = batch
        
        # Log batch shape info for debugging (only first time)
        if self.step_counter.update <= 100:
            self.logger.info(
                f"_log_batch_samples: batch shapes - states={all_states.shape}, "
                f"actions={actions.shape}, rewards={rewards.shape}, dones={dones.shape}"
            )
        
        # Determine number of samples to log
        n_samples = min(
            self.diagnostics.config.n_batch_samples,
            all_states.shape[0]
        )
        
        # Sample random indices from the batch
        batch_size = all_states.shape[0]
        sample_indices = np.random.choice(batch_size, size=n_samples, replace=False).tolist()
        
        # Get next states (shifted by 1 in sequence dimension)
        seq_length = actions.shape[1]
        next_states = all_states[:, 1:seq_length+1, :] if all_states.shape[1] > seq_length else None
        
        # Compute actor/critic values for sampled transitions
        batch_samples = self.algo.compute_batch_sample_values(
            states=all_states[:, :seq_length, :],
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_states=next_states,
            sample_indices=sample_indices,
        )
        
        # Log info about computed samples
        if self.step_counter.update <= 100:
            self.logger.info(f"_log_batch_samples: computed {len(batch_samples)} samples")
        
        # Log the samples
        self.diagnostics.log_batch_samples(
            update_step=self.step_counter.update,
            explore_step=self.step_counter.exploration,
            batch_samples=batch_samples,
        )
        
        if self.step_counter.update <= 100:
            self.logger.info(f"_log_batch_samples: successfully logged at step {self.step_counter.update}")

    def explore_and_update(
        self,
        *,
        explore_steps: Optional[int] = None,
        explore_episodes: Optional[int] = None,
        explore_step_limit: Optional[int] = None,
        explore_episode_limit: Optional[int] = None,
        update_steps: Optional[int] = None,
        update_step_limit: Optional[int] = None,
    ) -> Tuple[List[Episode], List[float]]:
        explore_result = self.explore(
            steps=explore_steps,
            episodes=explore_episodes,
            step_limit=explore_step_limit,
            episode_limit=explore_episode_limit,
        )
        update_result = self.update(steps=update_steps, step_limit=update_step_limit)
        return explore_result, update_result

    def close(self):
        # Close diagnostics logger if present
        if self.diagnostics is not None:
            self.diagnostics.close()
            
        self.env_train.close()
        if id(self.env_train) != id(self.env_eval):
            self.env_eval.close()
        self.replay_buffer.close()
        del self.algo
        del self.replay_buffer

    @classmethod
    def from_checkpoint(  # pylint: disable=arguments-renamed
        cls,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_actions: bool = True,
        env_train: Optional[gym.Env] = None,
        env_eval: Optional[gym.Env] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
    ):
        cp = torch.load(checkpoint_path)
        confighandler = ConfigHandler()
        algo = confighandler.config_dict_to_object(cp["algo"])
        replay_buffer = replay_buffer or confighandler.config_dict_to_object(
            cp["replay_buffer"]
        )
        eve = import_module("eve.util")
        eve_cfh = eve.ConfigHandler()
        env_train = env_train or eve_cfh.config_dict_to_object(cp["env_train"])
        env_eval = env_eval or eve_cfh.config_dict_to_object(cp["env_eval"])
        agent = cls(
            algo,
            env_train,
            env_eval,
            replay_buffer,
            device,
            consecutive_action_steps,
            normalize_actions,
        )
        agent.load_checkpoint(checkpoint_path)
        return agent

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
from typing import Tuple, Dict, Any, List
import logging
import math
import numpy as np
from torch.distributions import Normal
import torch
import torch.nn.functional as F
from .algo import Algo, AlgoPlayOnly
from ..model import SACModel, SACModelPlayOnly
from ..replaybuffer import Batch


def compute_grad_norm(parameters) -> float:
    """Compute the total gradient norm for a set of parameters."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def _is_nonfinite(x: float) -> bool:
    """Check if a value is NaN or Inf."""
    return math.isnan(x) or math.isinf(x)


def _compute_clamp_fraction(actions: torch.Tensor, threshold: float = 0.99) -> float:
    """
    Compute fraction of actions that are near the clamp boundaries.
    Actions are assumed to be in [-1, 1] after tanh.
    
    Args:
        actions: Tensor of actions (any shape)
        threshold: Actions with |value| > threshold are considered clamped
        
    Returns:
        Fraction of action components that are clamped (0.0 to 1.0)
    """
    if actions.numel() == 0:
        return 0.0
    clamped = (actions.abs() > threshold).float()
    return clamped.mean().item()


class SACPlayOnly(AlgoPlayOnly):
    model: SACModelPlayOnly

    def __init__(
        self,
        model: SACModelPlayOnly,
        n_actions: int,
        action_scaling: float = 1,
        exploration_action_noise: float = 0.25,
        stochastic_eval: bool = False,
    ):
        self.logger = logging.getLogger(self.__module__)
        # HYPERPARAMETERS
        self.n_actions = n_actions
        self.exploration_action_noise = exploration_action_noise
        # Model
        self.model = model

        # REST
        self.action_scaling = action_scaling
        self.stochastic_eval = stochastic_eval

        self.device = torch.device("cpu")

    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.sample())
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
            action += np.random.normal(0, self.exploration_action_noise)
        return action
    
    def get_action_exploration(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.sample())
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
            action += np.random.normal(0, self.exploration_action_noise)
        return action, mean, log_std

    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            if self.stochastic_eval:
                std = log_std.exp()
                normal = Normal(mean, std)
                action = torch.tanh(normal.sample())
            else:
                mean, _ = self.model.policy.forward_play(torch_state)
                action = torch.tanh(mean)

            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
        return action * self.action_scaling
    
    def get_action_evaluation(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            if self.stochastic_eval:
                std = log_std.exp()
                normal = Normal(mean, std)
                action = torch.tanh(normal.sample())
            else:
                mean, _ = self.model.policy.forward_play(torch_state)
                action = torch.tanh(mean)

            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
        return action * self.action_scaling, mean, log_std

    def to(self, device: torch.device):
        super().to(device)
        self.model.to(device)

    def reset(self) -> None:
        self.model.reset()

    def close(self):
        self.model.close()


class SAC(Algo):
    model: SACModel

    def __init__(
        self,
        model: SACModel,
        n_actions: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scaling: float = 1,
        action_scaling: float = 1,
        exploration_action_noise: float = 0.25,
        stochastic_eval: bool = False,
    ):
        self.logger = logging.getLogger(self.__module__)
        # HYPERPARAMETERS
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.exploration_action_noise = exploration_action_noise
        # Model
        self.model = model

        # REST
        self.reward_scaling = reward_scaling
        self.action_scaling = action_scaling
        self.stochastic_eval = stochastic_eval

        self.device = torch.device("cpu")
        self.update_step = 0

        # ENTROPY TEMPERATURE
        self.alpha = torch.ones(1)
        self.target_entropy = -torch.ones(1) * n_actions

        # DIAGNOSTICS: Store metrics from last update for logging
        self.last_metrics: Dict[str, float] = {}
        
        # DIAGNOSTICS: NaN/Inf sentinel counters
        self._nonfinite_q_loss_count = 0
        self._nonfinite_policy_loss_count = 0
        self._nonfinite_grad_count = 0
        self._clamp_fraction = 0.0

    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.sample())
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
            action += np.random.normal(0, self.exploration_action_noise)
        return action

    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            if self.stochastic_eval:
                std = log_std.exp()
                normal = Normal(mean, std)
                action = torch.tanh(normal.sample())
            else:
                mean, _ = self.model.policy.forward_play(torch_state)
                action = torch.tanh(mean)

            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
        return action * self.action_scaling

    def update(self, batch: Batch) -> Tuple[float, float, float]:
        (all_states, actions, rewards, dones, padding_mask) = batch
        # actions /= self.action_scaling

        all_states = all_states.to(dtype=torch.float32, device=self.device)
        actions = actions.to(dtype=torch.float32, device=self.device)
        rewards = rewards.to(dtype=torch.float32, device=self.device)
        dones = dones.to(dtype=torch.float32, device=self.device)

        if padding_mask is not None:
            padding_mask = padding_mask.to(dtype=torch.float32, device=self.device)

        seq_length = actions.shape[1]
        states = torch.narrow(all_states, dim=1, start=0, length=seq_length)

        # use all_states for next_actions and next_log_pi for proper hidden_state initilaization
        expected_q = self._get_expected_q(
            all_states, rewards, dones, padding_mask, seq_length
        )

        # q1 update
        q1_loss = self._update_q1(actions, padding_mask, states, expected_q)

        # q2 update
        q2_loss = self._update_q2(actions, padding_mask, states, expected_q)

        log_pi, policy_loss = self._update_policy(padding_mask, states)

        self.model.update_target_q(self.tau)

        alpha_loss = self._update_alpha(log_pi)

        self.update_step += 1
        
        # Check for NaN/Inf in losses
        q1_loss_val = q1_loss.detach().cpu().item()
        q2_loss_val = q2_loss.detach().cpu().item()
        policy_loss_val = policy_loss.detach().cpu().item()
        
        nonfinite_q = _is_nonfinite(q1_loss_val) or _is_nonfinite(q2_loss_val)
        nonfinite_pol = _is_nonfinite(policy_loss_val)
        nonfinite_grad = (_is_nonfinite(self._grad_norm_q1) or 
                         _is_nonfinite(self._grad_norm_q2) or 
                         _is_nonfinite(self._grad_norm_policy))
        
        if nonfinite_q:
            self._nonfinite_q_loss_count += 1
            self.logger.warning(f"ALERT: NaN/Inf in Q loss at step {self.update_step} (count={self._nonfinite_q_loss_count})")
        if nonfinite_pol:
            self._nonfinite_policy_loss_count += 1
            self.logger.warning(f"ALERT: NaN/Inf in policy loss at step {self.update_step} (count={self._nonfinite_policy_loss_count})")
        if nonfinite_grad:
            self._nonfinite_grad_count += 1
            self.logger.warning(f"ALERT: NaN/Inf in gradients at step {self.update_step} (count={self._nonfinite_grad_count})")
        
        # Populate diagnostics metrics
        lr_policy = 0.0
        if self.model.policy_scheduler is not None:
            try:
                lr_policy = self.model.policy_scheduler.get_last_lr()[0]
            except Exception:
                lr_policy = 0.0
                
        self.last_metrics = {
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.detach().cpu().item(),
            "log_pi_mean": log_pi.mean().detach().cpu().item(),
            "log_pi_std": log_pi.std().detach().cpu().item(),
            "entropy_proxy": -log_pi.mean().detach().cpu().item(),
            "q1_mean": self._q1_mean.cpu().item(),
            "q2_mean": self._q2_mean.cpu().item(),
            "target_q_mean": expected_q.mean().detach().cpu().item(),
            "min_q_mean": self._min_q_mean.cpu().item(),
            "grad_norm_q1": self._grad_norm_q1,
            "grad_norm_q2": self._grad_norm_q2,
            "grad_norm_policy": self._grad_norm_policy,
            "lr_policy": lr_policy,
            # Fix #6: clamp_fraction for direct saturation detection
            "clamp_fraction": self._clamp_fraction,
            # NEW: Enhanced diagnostics
            "target_entropy": self.target_entropy.item(),
            "log_alpha": self.model.log_alpha.item(),
            # NaN/Inf counters
            "nonfinite_q_loss_count": self._nonfinite_q_loss_count,
            "nonfinite_policy_loss_count": self._nonfinite_policy_loss_count,
            "nonfinite_grad_count": self._nonfinite_grad_count,
        }
        
        return [
            q1_loss.detach().cpu().numpy(),
            q2_loss.detach().cpu().numpy(),
            policy_loss.detach().cpu().numpy(),
        ]

    def _update_alpha(self, log_pi) -> torch.Tensor:
        alpha_loss = (
            self.model.log_alpha * (-log_pi - self.target_entropy).detach()
        ).mean()
        self.model.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.model.alpha_optimizer.step()

        self.alpha = self.model.log_alpha.exp()
        return alpha_loss

    def _update_policy(self, padding_mask, states):
        new_actions, log_pi = self._get_update_action(states)
        q1 = self.model.q1(states, new_actions)
        q2 = self.model.q2(states, new_actions)
        min_q = torch.min(q1, q2)

        if padding_mask is not None:
            min_q *= padding_mask
            log_pi *= padding_mask

        policy_loss = (self.alpha * log_pi - min_q).mean()

        self.model.policy_optimizer.zero_grad()
        policy_loss.backward()
        # Compute gradient norm before optimizer step
        self._grad_norm_policy = compute_grad_norm(self.model.policy.parameters())
        self._min_q_mean = min_q.mean().detach()
        # Fix #6: Compute clamp fraction for direct saturation detection in losses CSV
        self._clamp_fraction = _compute_clamp_fraction(new_actions.detach())
        self.model.policy_optimizer.step()
        if self.model.policy_scheduler:
            self.model.policy_scheduler.step()
        return log_pi, policy_loss

    def _update_q2(self, actions, padding_mask, states, expected_q):
        curr_q2 = self.model.q2(states, actions)
        if padding_mask is not None:
            curr_q2 *= padding_mask
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        self.model.q2_optimizer.zero_grad()
        q2_loss.backward()
        # Compute gradient norm before optimizer step
        self._grad_norm_q2 = compute_grad_norm(self.model.q2.parameters())
        self._q2_mean = curr_q2.mean().detach()
        self.model.q2_optimizer.step()
        if self.model.q2_scheduler:
            self.model.q2_scheduler.step()
        return q2_loss

    def _update_q1(self, actions, padding_mask, states, expected_q):
        curr_q1 = self.model.q1(states, actions)
        if padding_mask is not None:
            curr_q1 *= padding_mask
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())

        self.model.q1_optimizer.zero_grad()
        q1_loss.backward()
        # Compute gradient norm before optimizer step
        self._grad_norm_q1 = compute_grad_norm(self.model.q1.parameters())
        self._q1_mean = curr_q1.mean().detach()
        self.model.q1_optimizer.step()
        if self.model.q1_scheduler:
            self.model.q1_scheduler.step()
        return q1_loss

    def _get_expected_q(self, all_states, rewards, dones, padding_mask, seq_length):
        next_actions, next_log_pi = self._get_update_action(all_states)

        with torch.no_grad():
            next_target_q1 = self.model.target_q1(all_states, next_actions)
            next_target_q2 = self.model.target_q2(all_states, next_actions)

        next_target_q = (
            torch.min(next_target_q1, next_target_q2) - self.alpha * next_log_pi
        )
        # only use next_state for next_q_target
        next_target_q = torch.narrow(next_target_q, dim=1, start=1, length=seq_length)
        expected_q = rewards + (1 - dones) * self.gamma * next_target_q
        if padding_mask is not None:
            expected_q *= padding_mask
        return expected_q

    # epsilon makes sure that log(0) does not occur
    def _get_update_action(
        self, state_batch: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_batch, log_std = self.model.policy(state_batch)
        std_batch = log_std.exp()

        normal = Normal(mean_batch, std_batch)
        z = normal.rsample()
        action_batch = torch.tanh(z)

        log_pi_batch = normal.log_prob(z) - torch.log(1 - action_batch.pow(2) + epsilon)
        log_pi_batch = log_pi_batch.sum(-1, keepdim=True)

        # log_pi_batch = torch.sum(normal.log_prob(z), dim=-1, keepdim=True) - torch.sum(
        #        torch.log(1 - action_batch.pow(2) + epsilon), dim=-1, keepdim=True)

        return action_batch, log_pi_batch

    def compute_probe_values(self, probe_states: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute policy and critic values for probe states.
        
        Used for diagnostics to track how the policy and critics evolve.
        
        Args:
            probe_states: Tensor of shape (n_probes, seq_len, obs_dim) or (n_probes, obs_dim)
            
        Returns:
            Dictionary with policy_mean, policy_log_std, action_det, q1, q2 arrays
        """
        with torch.no_grad():
            # Ensure proper shape: (batch, seq, features)
            if probe_states.dim() == 2:
                probe_states = probe_states.unsqueeze(1)
                
            probe_states = probe_states.to(device=self.device, dtype=torch.float32)
            
            # Get policy outputs
            mean, log_std = self.model.policy(probe_states)
            action_det = torch.tanh(mean)
            
            # Get critic values for deterministic action
            q1 = self.model.q1(probe_states, action_det)
            q2 = self.model.q2(probe_states, action_det)
            
        return {
            "policy_mean": mean.cpu().numpy(),
            "policy_log_std": log_std.cpu().numpy(),
            "action_det": action_det.cpu().numpy(),
            "q1": q1.cpu().numpy(),
            "q2": q2.cpu().numpy(),
        }

    def compute_batch_sample_values(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
        sample_indices: list,
    ) -> List[Dict[str, Any]]:
        """
        Compute actor/critic values for sampled transitions from the training batch.
        
        Args:
            states: Batch states tensor (batch, seq, obs_dim)
            actions: Batch actions tensor (batch, seq, action_dim)
            rewards: Batch rewards tensor (batch, seq, 1)
            dones: Batch done flags tensor (batch, seq, 1)
            next_states: Next states (states[:, 1:, :] shifted)
            sample_indices: Indices of samples to extract
            
        Returns:
            List of dictionaries, each containing transition info and actor/critic values
        """
        samples = []
        
        # Move tensors to device for model forward pass
        states = states.to(device=self.device, dtype=torch.float32)
        actions = actions.to(device=self.device, dtype=torch.float32)
        if next_states is not None:
            next_states = next_states.to(device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            for idx in sample_indices:
                if idx >= states.shape[0]:
                    continue
                    
                # Extract single transition (take first timestep if sequence)
                s_t = states[idx:idx+1, 0:1, :]  # (1, 1, obs_dim)
                a_t = actions[idx, 0, :]  # (action_dim,)
                r_t = rewards[idx, 0, 0].item() if rewards.dim() > 2 else rewards[idx, 0].item()
                done_t = dones[idx, 0, 0].item() if dones.dim() > 2 else dones[idx, 0].item()
                
                # Get next state if available
                if next_states is not None and idx < next_states.shape[0]:
                    s_next = next_states[idx:idx+1, 0:1, :]
                else:
                    s_next = None
                    
                # Compute actor outputs for s_t
                mean, log_std = self.model.policy(s_t)
                action_det = torch.tanh(mean)
                
                # Compute Q values for (s_t, a_taken)
                a_taken = a_t.unsqueeze(0).unsqueeze(0)  # (1, 1, action_dim)
                q1_taken = self.model.q1(s_t, a_taken).squeeze().item()
                q2_taken = self.model.q2(s_t, a_taken).squeeze().item()
                
                # Compute Q values for (s_t, actor_action)
                q1_actor = self.model.q1(s_t, action_det).squeeze().item()
                q2_actor = self.model.q2(s_t, action_det).squeeze().item()
                
                # Extract action components
                actor_det_np = action_det.squeeze().cpu().numpy()
                action_taken_np = a_t.cpu().numpy()
                mean_np = mean.squeeze().cpu().numpy()
                log_std_np = log_std.squeeze().cpu().numpy()
                
                # Compute rotation stats (action[1] is typically rotation for 2-action envs)
                rotation_taken = action_taken_np[1] if len(action_taken_np) > 1 else 0.0
                rotation_actor = actor_det_np[1] if len(actor_det_np) > 1 else 0.0
                rotation_mean_val = mean_np[1] if len(mean_np) > 1 else 0.0
                rotation_log_std_val = log_std_np[1] if len(log_std_np) > 1 else 0.0
                
                # Compute clamp fraction for this sample's actor action
                clamp_frac = _compute_clamp_fraction(action_det, threshold=0.99)
                
                sample_data = {
                    "batch_idx": idx,
                    "state": s_t.squeeze().cpu().numpy().tolist(),
                    "action_taken": action_taken_np.tolist(),
                    "reward": r_t,
                    "done": done_t,
                    "next_state": s_next.squeeze().cpu().numpy().tolist() if s_next is not None else None,
                    "actor_mean": mean_np.tolist(),
                    "actor_log_std": log_std_np.tolist(),
                    "actor_det_action": actor_det_np.tolist(),
                    "q1_action_taken": q1_taken,
                    "q2_action_taken": q2_taken,
                    "q1_actor_action": q1_actor,
                    "q2_actor_action": q2_actor,
                    "min_q_taken": min(q1_taken, q2_taken),
                    "min_q_actor": min(q1_actor, q2_actor),
                    # NEW: Rotation stats
                    "rotation_taken": float(rotation_taken),
                    "rotation_actor": float(rotation_actor),
                    "rotation_mean": float(rotation_mean_val),
                    "rotation_log_std": float(rotation_log_std_val),
                    # NEW: Clamp fraction
                    "clamp_fraction": clamp_frac,
                }
                samples.append(sample_data)
                
        return samples

    def to(self, device: torch.device):
        super().to(device)
        self.alpha = self.alpha.to(device)
        self.target_entropy = self.target_entropy.to(device)
        self.model.to(device)

    def reset(self) -> None:
        self.model.reset()

    def close(self):
        self.model.close()

    def to_play_only(self):
        return SACPlayOnly(
            self.model.to_play_only(),
            self.n_actions,
            self.action_scaling,
            self.exploration_action_noise,
        )

```

```python
"""
Diagnostics logging utilities for SAC training.

Provides structured logging for:
- Per-update-step losses (CSV)
- Probe state values (JSONL)
- TensorBoard scalars and histograms
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import csv
import json
import os
import logging
import time
import numpy as np
import torch


@dataclass
class DiagnosticsConfig:
    """Configuration for diagnostics logging."""
    enabled: bool = True
    diagnostics_folder: str = None
    log_losses_every_n_steps: int = 1
    log_probe_values_every_n_steps: int = 100
    log_batch_samples_every_n_steps: int = 100  # Log batch samples every N update steps
    n_batch_samples: int = 3  # Number of transitions to sample from each batch
    tensorboard_enabled: bool = True
    probe_plot_every_n_steps: int = 10000
    policy_snapshot_every_n_steps: int = 10000
    flush_every_n_rows: int = 100
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiagnosticsConfig":
        """Create config from dictionary."""
        if d is None:
            return cls(enabled=False)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CSVScalarLogger:
    """Append-only CSV logger for scalar values."""
    
    def __init__(
        self,
        path: str,
        header_fields: List[str],
        flush_every_n: int = 100,
    ):
        self.path = path
        self.header_fields = header_fields
        self.flush_every_n = flush_every_n
        self._row_count = 0
        self._file = None
        self._writer = None
        self._initialized = False
        self._warned_unknown_fields: set = set()  # Fix #11
        
    def _initialize(self):
        """Initialize file and write header if needed."""
        if self._initialized:
            return
            
        file_exists = os.path.isfile(self.path)
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        
        if not file_exists:
            self._writer.writerow(self.header_fields)
            self._file.flush()
            
        self._initialized = True
        
    def log(self, **kwargs):
        """Log a row of scalar values."""
        self._initialize()

        # Fix #11: Warn once about unknown fields to catch bugs when new metrics are added
        unknown = set(kwargs.keys()) - set(self.header_fields) - self._warned_unknown_fields
        if unknown:
            self._warned_unknown_fields.update(unknown)
            logger = logging.getLogger(__name__)
            logger.warning(f"CSVScalarLogger: unknown fields not in header (will be dropped): {unknown}")

        row = [kwargs.get(field, "") for field in self.header_fields]
        self._writer.writerow(row)
        self._row_count += 1
        
        if self._row_count % self.flush_every_n == 0:
            self._file.flush()
            
    def flush(self):
        """Force flush to disk."""
        if self._file is not None:
            self._file.flush()
            
    def close(self):
        """Close the file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None


def _convert_to_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_serializable(item) for item in obj)
    else:
        return obj


class JSONLLogger:
    """Append-only JSONL logger for structured data (vectors, dicts)."""

    def __init__(
        self,
        path: str,
        flush_every_n: int = 10,
    ):
        self.path = path
        self.flush_every_n = flush_every_n
        self._row_count = 0
        self._file = None
        self._initialized = False
        self._logger = logging.getLogger(__name__)
        
    def _initialize(self):
        """Initialize file."""
        if self._initialized:
            return
        self._file = open(self.path, "a", encoding="utf-8")
        self._initialized = True
        self._logger.info(f"JSONLLogger initialized: {self.path}")
        
    def log(self, data: Dict[str, Any]):
        """Log a JSON object. numpy arrays are converted to lists."""
        self._initialize()

        try:
            serializable = _convert_to_serializable(data)
            json_str = json.dumps(serializable)
            self._file.write(json_str + "\n")
            self._row_count += 1
            
            # Flush immediately for first few rows to ensure data is written
            if self._row_count <= 3 or self._row_count % self.flush_every_n == 0:
                self._file.flush()
                
        except Exception as e:
            self._logger.error(f"JSONLLogger.log failed: {e}, data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            
    def flush(self):
        """Force flush to disk."""
        if self._file is not None:
            self._file.flush()
            
    def close(self):
        """Close the file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


class TensorBoardLogger:
    """Wrapper around PyTorch's SummaryWriter."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._writer = None
        self._initialized = False
        
    def _initialize(self):
        """Lazy initialization of SummaryWriter."""
        if self._initialized:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=self.log_dir)
            self._initialized = True
        except ImportError:
            logging.warning("TensorBoard not available. Skipping TensorBoard logging.")
            self._writer = None
            self._initialized = True
            
    def add_scalar(self, tag: str, value: float, global_step: int):
        """Add a scalar value."""
        self._initialize()
        if self._writer is not None:
            self._writer.add_scalar(tag, value, global_step)
            
    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int):
        """Add multiple scalars under a main tag."""
        self._initialize()
        if self._writer is not None:
            self._writer.add_scalars(main_tag, tag_scalar_dict, global_step)
            
    def add_histogram(self, tag: str, values: np.ndarray, global_step: int):
        """Add a histogram of values."""
        self._initialize()
        if self._writer is not None:
            self._writer.add_histogram(tag, values, global_step)
            
    def flush(self):
        """Force flush to disk."""
        if self._writer is not None:
            self._writer.flush()
            
    def close(self):
        """Close the writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class DiagnosticsLogger:
    """
    Main diagnostics logger that combines CSV, JSONL, and TensorBoard logging.
    
    This is instantiated in the trainer subprocess and handles all diagnostics logging.
    """
    
    # Standard header fields for loss logging
    LOSS_HEADER_FIELDS = [
        "wall_time",  # Unix timestamp for time correlation
        "update_step",
        "explore_step",
        "q1_loss",
        "q2_loss", 
        "policy_loss",
        "alpha",
        "alpha_loss",
        "log_pi_mean",
        "log_pi_std",
        "entropy_proxy",
        "q1_mean",
        "q2_mean",
        "target_q_mean",
        "min_q_mean",
        "grad_norm_q1",
        "grad_norm_q2",
        "grad_norm_policy",
        "lr_policy",
        # Fix #6: clamp fraction for direct saturation detection
        "clamp_fraction",
        # NEW: Enhanced diagnostics
        "target_entropy",
        "log_alpha",
        # NEW: NaN/Inf sentinels
        "nonfinite_q_loss_count",
        "nonfinite_policy_loss_count",
        "nonfinite_grad_count",
    ]
    
    def __init__(
        self,
        config: DiagnosticsConfig,
        process_name: str = "trainer",
    ):
        self.config = config
        self.process_name = process_name
        self.logger = logging.getLogger(__name__)
        
        # Probe states for evaluation (set externally)
        self.probe_states: Optional[torch.Tensor] = None
        self.probe_states_np: Optional[np.ndarray] = None
        
        if not config.enabled or config.diagnostics_folder is None:
            self._loss_logger = None
            self._probe_logger = None
            self._batch_logger = None
            self._tb_logger = None
            self.logger.info("Diagnostics logging disabled")
            return
            
        # Create subdirectories
        csv_dir = os.path.join(config.diagnostics_folder, "csv")
        tb_dir = os.path.join(config.diagnostics_folder, "tensorboard")
        
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        
        # Initialize loggers
        self._loss_logger = CSVScalarLogger(
            path=os.path.join(csv_dir, f"losses_{process_name}.csv"),
            header_fields=self.LOSS_HEADER_FIELDS,
            flush_every_n=config.flush_every_n_rows,
        )
        
        self._probe_logger = JSONLLogger(
            path=os.path.join(csv_dir, f"probe_values_{process_name}.jsonl"),
            flush_every_n=10,
        )
        
        # Batch transitions logger - logs sampled transitions from replay buffer
        self._batch_logger = JSONLLogger(
            path=os.path.join(csv_dir, f"batch_samples_{process_name}.jsonl"),
            flush_every_n=100,
        )
        
        if config.tensorboard_enabled:
            self._tb_logger = TensorBoardLogger(
                log_dir=os.path.join(tb_dir, process_name),
            )
        else:
            self._tb_logger = None
            
        self.logger.info(f"Diagnostics logging initialized for {process_name}")
        
    def log_losses(
        self,
        update_step: int,
        explore_step: int,
        q1_loss: float,
        q2_loss: float,
        policy_loss: float,
        **extra_metrics,
    ):
        """Log loss values and additional metrics."""
        if self._loss_logger is None:
            return
            
        # Check if we should log this step
        if update_step % self.config.log_losses_every_n_steps != 0:
            return
            
        # Build the log entry
        log_data = {
            "wall_time": time.time(),  # Unix timestamp for time correlation
            "update_step": update_step,
            "explore_step": explore_step,
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "policy_loss": policy_loss,
        }
        log_data.update(extra_metrics)
        
        self._loss_logger.log(**log_data)
        
        # Also log to TensorBoard
        if self._tb_logger is not None:
            self._tb_logger.add_scalar("losses/q1_loss", q1_loss, update_step)
            self._tb_logger.add_scalar("losses/q2_loss", q2_loss, update_step)
            self._tb_logger.add_scalar("losses/policy_loss", policy_loss, update_step)
            
            for key, value in extra_metrics.items():
                if isinstance(value, (int, float)):
                    self._tb_logger.add_scalar(f"metrics/{key}", value, update_step)
                    
    def log_probe_values(
        self,
        update_step: int,
        probe_values: Dict[str, np.ndarray],
        explore_step: int = None,  # NEW: Added for timestamp correlation
    ):
        """Log probe state evaluation values."""
        if self._probe_logger is None:
            return
            
        log_data = {
            "wall_time": time.time(),  # Unix timestamp for time correlation
            "update_step": update_step,
            "explore_step": explore_step,
            **probe_values,
        }
        self._probe_logger.log(log_data)
        
        # Log summary stats to TensorBoard
        if self._tb_logger is not None:
            for key, values in probe_values.items():
                if isinstance(values, np.ndarray):
                    # Log mean of the first probe state's values
                    if len(values.shape) > 1:
                        first_probe = values[0]
                    else:
                        first_probe = values
                    self._tb_logger.add_scalar(
                        f"probe/{key}_mean", 
                        np.mean(first_probe), 
                        update_step
                    )

    def log_batch_samples(
        self,
        update_step: int,
        explore_step: int,
        batch_samples: List[Dict[str, Any]],
    ):
        """
        Log sampled transitions from the replay batch.
        
        Each sample contains:
        - state (s_t): flattened observation
        - action (a_t): action taken
        - reward (r_t): reward received
        - next_state (s_{t+1}): next observation
        - done: terminal flag
        - actor_values: policy outputs for s_t (mean, log_std, det_action)
        - critic_values: Q1, Q2 for (s_t, a_t) and (s_t, actor_action)
        """
        if self._batch_logger is None:
            self.logger.warning(f"log_batch_samples called but _batch_logger is None at step {update_step}")
            return
            
        try:
            log_data = {
                "wall_time": time.time(),  # Unix timestamp for time correlation
                "update_step": update_step,
                "explore_step": explore_step,
                "n_samples": len(batch_samples),
                "samples": batch_samples,
            }
            self._batch_logger.log(log_data)
            
            # Log success for first few calls
            if update_step <= 300:
                self.logger.info(f"log_batch_samples: logged {len(batch_samples)} samples at step {update_step}")
        except Exception as e:
            self.logger.error(f"log_batch_samples failed at step {update_step}: {e}")
                    
    def set_probe_states(self, probe_states: np.ndarray, device: torch.device = None):
        """Set the probe states for evaluation."""
        self.probe_states_np = probe_states
        if device is not None:
            self.probe_states = torch.as_tensor(
                probe_states, dtype=torch.float32, device=device
            )
        else:
            self.probe_states = torch.as_tensor(
                probe_states, dtype=torch.float32
            )
        self.logger.info(f"Set {len(probe_states)} probe states for evaluation")
        
    def save_probe_states(self, path: str):
        """Save probe states to disk."""
        if self.probe_states_np is not None:
            np.savez(path, probe_states=self.probe_states_np)
            self.logger.info(f"Saved probe states to {path}")
            
    def flush(self):
        """Flush all loggers."""
        if self._loss_logger is not None:
            self._loss_logger.flush()
        if self._probe_logger is not None:
            self._probe_logger.flush()
        if self._batch_logger is not None:
            self._batch_logger.flush()
        if self._tb_logger is not None:
            self._tb_logger.flush()
            
    def close(self):
        """Close all loggers."""
        if self._loss_logger is not None:
            self._loss_logger.close()
        if self._probe_logger is not None:
            self._probe_logger.close()
        if self._batch_logger is not None:
            self._batch_logger.close()
        if self._tb_logger is not None:
            self._tb_logger.close()

```

All files have been copied successfully! Here's a summary of what was done:

## ✅ Files Copied from D:\neve to D:\stEVE_training

### Backup Files Created (`*_prevs.py`)
- `eve_rl/eve_rl/replaybuffer/replaybuffer_prevs.py`
- `eve_rl/eve_rl/agent/single_prevs.py`
- `eve_rl/eve_rl/runner/runner_prevs.py`
- `eve_rl/eve_rl/algo/sac_prevs.py`
- `eve_rl/eve_rl/util/diagnostics_logger_prevs.py`

### New Forensics Scripts
- `diagnose_collapse5.py` - All 13 correctness fixes
- `diagnose_collapse6.py` - 8 performance optimizations on top of v5

### Updated eve_rl Files
- `eve_rl/eve_rl/replaybuffer/replaybuffer.py` - Fix #4: Added `explore_step_at_completion` and `update_step_at_completion` to `Episode`
- `eve_rl/eve_rl/agent/single.py` - Fix #4: Stamps episode step counters inside lock
- `eve_rl/eve_rl/runner/runner.py` - Fix #4, #5: Per-episode step values + monotonic `episode_id` counter
- `eve_rl/eve_rl/algo/sac.py` - Fix #6: `_compute_clamp_fraction()` + added to `last_metrics`
- `eve_rl/eve_rl/util/diagnostics_logger.py` - Fix #6, #10, #11: `clamp_fraction` header, module-level serializer, unknown field warning

## Key Changes Summary

| Fix # | Description | What it solves |
|-------|-------------|----------------|
| **#4** | Per-episode step stamping inside lock | Race condition causing wrong `explore_step`/`update_step` in episode summaries |
| **#5** | Monotonic `episode_id` counter | Non-sequential episode IDs in multi-worker setup |
| **#6** | `clamp_fraction` metric | Direct detection of tanh saturation (actions hitting ±1) |
| **#10** | `_convert_to_serializable` at module level | Performance: avoids recreating function on every call |
| **#11** | Unknown field warning in CSV logger | Catches bugs when new metrics are added but not in header |

---

**User**

Find out the relevant files that have changed in D:\neve\.claude\worktrees\rl_diag_2\ codebase ; there is one fix that you mentioned before that was needed and a few optimizations; find out what they are  and are they good and not introduce any additional bugs or ineffciencies ; files changed are vanillashared and diag....pse6, maybe a few others ; figure out what they are 

---

**Cursor**

Let me see what other files might have changed in the worktree:

Based on my analysis, here's what I found in the `D:\neve\.claude\worktrees\rl_diag_2\` codebase:

## Files Changed in the Worktree

### 1. `eve_rl/eve_rl/replaybuffer/vanillashared.py`
This is the **main fix** - an improved implementation of Fix #4 (Per-Episode Step Stamping).

**Changes:**
- Added `episode_arrival_queue: mp.Queue = None` parameter to `VanillaSharedBase`
- Modified `push()` to pass `(replay_data, explore_step)` tuple instead of just `replay_data`
- Added `set_step_counter()` method to send step_counter to the subprocess
- Added `drain_episode_arrivals()` method to retrieve `{explore_step: update_step}` mapping
- Modified `loop()` to record `(explore_step, update_step)` when episodes arrive
- Modified `copy()` to propagate `episode_arrival_queue`

### 2. `eve_rl/eve_rl/agent/synchron.py`
**Changes:**
- Calls `self.replay_buffer.set_step_counter(self.step_counter)` in `__init__`
- In `explore_and_update()`, drains the arrival queue and re-stamps episodes:

```python
arrival_map = self.replay_buffer.drain_episode_arrivals()
if arrival_map:
    for episode in explore_results:
        explore_step = getattr(episode, 'explore_step_at_completion', None)
        if explore_step is not None and explore_step in arrival_map:
            episode.update_step_at_completion = arrival_map[explore_step]
```

### 3. `diagnose_collapse6.py`
Appears identical to the version in `D:\neve\` — same performance optimizations.

---

## Analysis: Is This Implementation Good?

### ✅ **Advantages of this worktree approach:**

1. **More accurate timing**: The `update_step` is captured at the moment the episode *arrives at the replay buffer subprocess*, which is closer to the actual training step than the worker's stale value.

2. **Race condition resistant**: The `step_counter.update` is read inside the replay subprocess's main loop, so it's synchronized with the actual training updates.

3. **Clean separation**: The replay buffer subprocess owns the timing, not the worker processes.

### ⚠️ **Potential Issues/Concerns:**

1. **Queue serialization overhead**: The `step_counter` object is passed via the task queue. If `step_counter` is a shared memory object (which it appears to be `StepCounterShared`), this should work. But this needs verification — if the subprocess doesn't get live updates to the counter, it would still be stale.

2. **Potential bug in `set_step_counter`**: 
   ```python
   def set_step_counter(self, step_counter):
       self._task_queue.put(["set_step_counter", step_counter])
   ```
   This sends the **reference** to the `step_counter`. In multiprocessing, unless `step_counter.update` is a `multiprocessing.Value`, the subprocess won't see updates! Let me verify:

```python
arrival_map = self.replay_buffer.drain_episode_arrivals()
if arrival_map:
    for episode in explore_results:
        explore_step = getattr(episode, 'explore_step_at_completion', None)
        if explore_step is not None and explore_step in arrival_map:
            episode.update_step_at_completion = arrival_map[explore_step]
```

```python
   def set_step_counter(self, step_counter):
       self._task_queue.put(["set_step_counter", step_counter])
```

## ✅ Verdict: The Worktree Implementation is **Good and Correct**

The `StepCounterShared` class properly uses `multiprocessing.Value`:

```python
class StepCounterShared(StepCounter):
    def __init__(self):
        self._update: mp.Value = mp.Value("i", 0)
    
    @property
    def update(self) -> int:
        return self._update.value
```

This means when the replay buffer subprocess reads `step_counter.update`, it gets the **live value** from shared memory, not a stale copy!

---

## Summary: Worktree vs What Was Copied to stEVE_training

| Aspect | stEVE_training (current) | Worktree (rl_diag_2) |
|--------|--------------------------|----------------------|
| **Step stamping location** | Worker stamps at episode completion | Replay subprocess stamps at episode arrival |
| **Accuracy** | Good - captures at completion | Better - captures when it enters training pipeline |
| **Race condition** | Fixed by locking in worker | Also fixed, uses shared memory |
| **Implementation complexity** | Simpler | More complex but cleaner |
| **Additional infrastructure** | None | Adds `episode_arrival_queue` |

### The Worktree Version is an **Enhancement**, Not Just a Bug Fix

The worktree implements a more sophisticated approach:
1. Episodes carry their `explore_step_at_completion` from the worker
2. The replay buffer subprocess reads `step_counter.update` when episodes arrive
3. The parent process re-stamps `update_step_at_completion` after draining the arrival queue

This captures the `update_step` **at the moment the episode enters the replay buffer**, which is the most accurate possible timing.

### No Bugs or Inefficiencies Detected

- Uses `multiprocessing.Value` correctly for cross-process shared state
- Queue operations are O(1) and non-blocking (`get_nowait()`)
- Fallback logic handles missing data gracefully

**Recommendation**: This worktree version (`vanillashared.py` + `synchron.py`) could be merged into stEVE_training for even better episode-step correlation accuracy. Would you like me to help copy these files as well?

```python
class StepCounterShared(StepCounter):
    def __init__(self):
        self._update: mp.Value = mp.Value("i", 0)
    
    @property
    def update(self) -> int:
        return self._update.value
```

---

**User**

you missed one; read diagnose_collapse6 again more carefully

---

**Cursor**

You're right! I missed **Perf #9 - Pipe-Split Parsing** which is a major optimization in the worktree's `diagnose_collapse6.py`.

The `diagnose_collapse6.py` that was copied from `D:\neve` (the main folder) is **different** from the one in `D:\neve\.claude\worktrees\rl_diag_2\`:

| Feature | Copied Version (from `D:\neve`) | Worktree Version (`rl_diag_2`) |
|---------|--------------------------------|-------------------------------|
| Line Count | ~2777 | ~2816 |
| Log Parsing | Uses 3-regex cascade (`STEP_RE_V3`, `STEP_RE_NEW`, `STEP_RE_FULL`) | Uses **zero-regex pipe-split parsing** |
| Key Functions | Still has `_extract_field()`, `_parse_step_line()` | New `_parse_pipe_fields()`, `_parse_array_str()` |

## Perf #9 - Pipe-Split Parsing (Missing from copied version)

**What it does:**
Replaces the entire 3-regex cascade with zero-regex string operations:

```757:795:D:\neve\.claude\worktrees\rl_diag_2\diagnose_collapse6.py
def _parse_pipe_fields(line: str) -> Tuple[Optional[str], Dict[str, str]]:
    """Parse a pipe-delimited log line into (timestamp_str, {key: value_str}).

    Input:  '2026-01-01 00:00:00,000 - STEP | ep=5 | cmd_action=[0.5,-0.1] | ...'
    Output: ('2026-01-01 00:00:00,000', {'STEP': '', 'ep': '5', 'cmd_action': '[0.5,-0.1]', ...})

    The line type (STEP, EPISODE_START, EPISODE_END) appears as a key with empty value.
    """
    fields: Dict[str, str] = {}
    ts_str = None

    parts = line.split(" | ")
    # First part contains timestamp + log level + line type
    header = parts[0]
    ts_match = _TS_PREFIX_RE.match(header)
    if ts_match:
        ts_str = ts_match.group(1)
    # Extract line type from header (last word: STEP, EPISODE_START, etc.)
    header_words = header.split()
    if header_words:
        line_type = header_words[-1]
        fields[line_type] = ""

    for part in parts[1:]:
        part = part.strip()
        # ...splits key=value pairs
        
def _parse_array_str(s: str) -> List[float]:
    """Parse '[0.5,-0.1,3.0]' or '100.5,50.2' into list of floats."""
    s = s.strip("[] ")
    if not s:
        return []
    return [_to_float(x.strip()) for x in s.split(",") if x.strip()]
```

**Why it matters:**
- The old code performed **~15 regex operations per STEP line**: up to 3 format-detection regexes with backtracking `.*?` patterns (~5-50 µs each), plus `_extract_pid()` and `_extract_wall_time()` calls
- String `split()` is **~10x faster** than `re.search()` with backtracking
- For 1M+ STEP lines (common in long training runs), this was the dominant cost
- **Impact:** ~3-10x faster worker log parsing

The worktree also has **Fix #4b** (the improved replay buffer stamping via `vanillashared.py` and `synchron.py`) that I mentioned earlier.

**You should copy the worktree version** of `diagnose_collapse6.py` from `D:\neve\.claude\worktrees\rl_diag_2\diagnose_collapse6.py` instead of the one from `D:\neve\diagnose_collapse6.py`. Switch to Agent mode to copy the correct file.

```plaintext
def _parse_pipe_fields(line: str) -> Tuple[Optional[str], Dict[str, str]]:
    """Parse a pipe-delimited log line into (timestamp_str, {key: value_str}).

    Input:  '2026-01-01 00:00:00,000 - STEP | ep=5 | cmd_action=[0.5,-0.1] | ...'
    Output: ('2026-01-01 00:00:00,000', {'STEP': '', 'ep': '5', 'cmd_action': '[0.5,-0.1]', ...})

    The line type (STEP, EPISODE_START, EPISODE_END) appears as a key with empty value.
    """
    fields: Dict[str, str] = {}
    ts_str = None

    parts = line.split(" | ")
    # First part contains timestamp + log level + line type
    header = parts[0]
    ts_match = _TS_PREFIX_RE.match(header)
    if ts_match:
        ts_str = ts_match.group(1)
    # Extract line type from header (last word: STEP, EPISODE_START, etc.)
    header_words = header.split()
    if header_words:
        line_type = header_words[-1]
        fields[line_type] = ""

    for part in parts[1:]:
        part = part.strip()
        # ...splits key=value pairs
        
def _parse_array_str(s: str) -> List[float]:
    """Parse '[0.5,-0.1,3.0]' or '100.5,50.2' into list of floats."""
    s = s.strip("[] ")
    if not s:
        return []
    return [_to_float(x.strip()) for x in s.split(",") if x.strip()]
```

---

**User**

copy these modified files into our workspace as we did before

---

**Cursor**

I'll copy the modified files from the worktree, saving the existing versions as `_prevs.py` first.

Now let me copy the new worktree versions:

Let me verify the copies are correct:

All files copied successfully from the worktree. Here's the summary:

## Files Copied from `D:\neve\.claude\worktrees\rl_diag_2\`

| File | Backup Created | Key Changes |
|------|----------------|-------------|
| `diagnose_collapse6.py` | `diagnose_collapse6_prevs.py` | **Perf #9**: Pipe-split parsing (3-10x faster log parsing) |
| `eve_rl/eve_rl/replaybuffer/vanillashared.py` | `vanillashared_prevs.py` | **Fix #4b**: Episode arrival queue + `set_step_counter()` |
| `eve_rl/eve_rl/agent/synchron.py` | `synchron_prevs.py` | **Fix #4b**: `drain_episode_arrivals()` + re-stamps episodes |
| `CHANGELOG_v5_v6.md` | (updated) | Full documentation of Perf #9 + Fix #4b |

## Key New Features Now in Workspace

1. **Perf #9 - Pipe-Split Parsing**: `diagnose_collapse6.py` now uses zero-regex string `split()` operations instead of the 3-regex cascade (`STEP_RE_V3`, `STEP_RE_NEW`, `STEP_RE_FULL`). This is ~3-10x faster for worker log parsing.

2. **Fix #4b - Accurate `update_step_at_completion`**: The replay buffer subprocess now stamps the exact `update_step` when an episode arrives (via `_episode_arrival_queue`), and `Synchron.explore_and_update()` re-stamps episodes with this accurate value after both phases complete. This fixes the stale value problem where workers read `step_counter.update` before trainer increments it.

