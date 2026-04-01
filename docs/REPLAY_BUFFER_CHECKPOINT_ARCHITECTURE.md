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
