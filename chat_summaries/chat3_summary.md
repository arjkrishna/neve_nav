# Chat 3 Summary

_Exported: 3/15/2026 from Cursor (2.3.34)_
_Title: "explore rewards"_
_~4,892 lines_

---

## High-Level Summary

This chat covers the **initial GPU training attempts, SOFA timeout debugging, and first discovery of policy collapse** in the DualDeviceNav RL training pipeline. It represents the early phase of experimentation before the extensive diagnostics tooling built in Chats 1-2.

### Key Activities (Chronological)

1. **Initial GPU Training Setup:** Started `first_run_gpu` experiment with `--gpus all` flag. GPU detected: NVIDIA RTX 4500 Ada Generation. Container stopped during heatup phase.

2. **Dual CPU+GPU Experiments:** Ran `second_run` (CPU) and `second_run_gpu` (GPU) simultaneously. Both ran for ~9 hours with **zero steps completed** due to SOFA timeouts.

3. **SOFA Timeout Investigation:**
   - **Root Cause 1:** `step_timeout=2` seconds too short for complex DualDeviceNav physics
   - **Attempted fix:** `intervention.make_mp(step_timeout=10.0)` — increased timeout to 10s
   - **Root Cause 2 (deeper):** Nested multiprocessing — 9 processes (Main + 4 Workers + 4 SOFA subprocesses) fighting for `/dev/shm`
   - **Final fix:** Changed `make_mp()` to `make_non_mp()` in `training_scripts/util/env.py` — reduced to 5 processes

4. **"Case 1" Error Investigation:** Recurring `InterventionalRadiologyController` "Case 1 should never happen" error (`xCurvAbs > prev_maxCurvAbs + threshold(0.01mm)`). Minimal test script proved these are just warnings, not crashes. Heatup action ranges temporarily reduced then reverted.

5. **Mount Fix:** Discovered mounted code was not being used (imports reference image's copy). Fixed by overlaying `env.py` directly over the image's installed copy.

6. **Systematic Training (`non_mp_first_run_gpu`):** With `make_non_mp()` fix:
   - ~700-1000 steps/minute with 4 workers, 0.4-0.75s per step
   - Zero SOFA timeouts
   - Heatup completed in ~21 hours (500,469 steps, 870 episodes)

7. **Policy Collapse Discovery:** After heatup, policy immediately collapsed:
   - Heatup (random actions): devices reached 900mm
   - Explore (policy + noise): only ~30-60mm
   - Evaluate (pure policy): only ~7mm (near-zero actions)
   - **Root cause:** Reward signal too weak — step penalty (-0.005/step) dominates PathDelta reward (0.001 * distance_change). Forward movement: -0.004/step vs stay still: -0.005/step — only 0.001 difference.

---

## Experiment Timeline

| Experiment | Device | Workers | Outcome |
|---|---|---|---|
| `first_run_gpu` | cuda:0 | 4 | Stopped during heatup |
| `second_run` | cpu | 4 | 9 hours, 0 steps — SOFA timeouts (make_mp issue) |
| `second_run_gpu` | cuda:0 | 4 | 9 hours, 0 steps — SOFA timeouts (make_mp issue) |
| `non_mp_first_run_gpu` | cuda:0 | 4 | Heatup completed ~21h (500K steps). Explore phase: policy collapsed immediately. |

---

## Docker Commands

### 1. GPU Training (Basic)

```bash
docker run --rm --gpus all --shm-size=8g \
  -v "D:\stEVE_training\training_scripts:/opt/eve_training/training_scripts_host" \
  -v "D:\stEVE_training\training_scripts/results:/opt/eve_training/results" \
  eve-training-fixed \
  python3 /opt/eve_training/training_scripts_host/DualDeviceNav_train.py \
  -n first_run_gpu -d cuda:0
```

**Purpose:** First GPU training attempt. Note the host mount at `/opt/eve_training/training_scripts_host` (different from later chats that overlay individual files).

### 2. CPU Training

```bash
docker run --rm --shm-size=8g \
  -v "D:\stEVE_training\training_scripts:/opt/eve_training/training_scripts_host" \
  -v "D:\stEVE_training\training_scripts/results:/opt/eve_training/results" \
  eve-training-fixed \
  python3 /opt/eve_training/training_scripts_host/DualDeviceNav_train.py \
  -n cpu_experiment -d cpu
```

**Purpose:** CPU-only experiment (no `--gpus all`).

### 3. GPU Training (Alternative Name)

```bash
docker run --rm --gpus all --shm-size=8g \
  -v "D:\stEVE_training\training_scripts:/opt/eve_training/training_scripts_host" \
  -v "D:\stEVE_training\training_scripts/results:/opt/eve_training/results" \
  eve-training-fixed \
  python3 /opt/eve_training/training_scripts_host/DualDeviceNav_train.py \
  -n gpu_experiment -d cuda:0
```

### 4. Systematic Run with env.py Overlay and Named Container

```bash
docker run --gpus all --shm-size=8g \
  --name non_mp_first_run_gpu \
  -v "D:\stEVE_training\training_scripts:/opt/eve_training/training_scripts_host" \
  -v "D:\stEVE_training\training_scripts/results:/opt/eve_training/results" \
  -v "D:\stEVE_training\training_scripts/util/env.py:/opt/eve_training/training_scripts/util/env.py" \
  -e STEP_LOG_DIR=/opt/eve_training/results \
  eve-training-fixed \
  python3 /opt/eve_training/training_scripts_host/DualDeviceNav_train.py \
  -n non_mp_first_run_gpu -d cuda:0
```

**Key changes from basic command:**
- `--rm` removed (container kept for log retrieval)
- `--name non_mp_first_run_gpu` — named container
- Additional `-v` mount for `env.py` directly overlaying the image's copy (make_non_mp fix)
- `-e STEP_LOG_DIR` — enables worker step logging

### 5. Docker Build

```bash
docker build -t eve-training-fixed .
```

**Purpose:** Rebuild the Docker image (suggested but typically avoided by using volume mounts instead).

### 6. Docker Exec (Diagnostic)

```bash
docker exec <container> df -h /dev/shm
```

**Purpose:** Check shared memory usage inside a running container (to diagnose /dev/shm exhaustion with nested multiprocessing).

### 7. Shared Memory Variant

```bash
docker run --shm-size=16g ...  # Even more shared memory
```

**Purpose:** Suggested alternative to `make_non_mp()` — increase shared memory to 16g instead of fixing the nested multiprocessing. Not recommended.

---

## Python Script Commands

### DualDeviceNav_train.py (Inside Docker)

```bash
# GPU training
python3 /opt/eve_training/training_scripts_host/DualDeviceNav_train.py \
  -n first_run_gpu -d cuda:0

# CPU training
python3 /opt/eve_training/training_scripts_host/DualDeviceNav_train.py \
  -n cpu_experiment -d cpu

# Named experiment variants
python3 /opt/eve_training/training_scripts_host/DualDeviceNav_train.py \
  -n second_run -d cpu

python3 /opt/eve_training/training_scripts_host/DualDeviceNav_train.py \
  -n second_run_gpu -d cuda:0

python3 /opt/eve_training/training_scripts_host/DualDeviceNav_train.py \
  -n non_mp_first_run_gpu -d cuda:0
```

**All flags:**
| Flag | Type | Default | Description |
|---|---|---|---|
| `-nw` / `--n_worker` | int | 4 | Number of parallel workers |
| `-d` / `--device` | str | cpu | Trainer device (`cpu`, `cuda:0`, `cuda:1`, `cuda`) |
| `-se` / `--stochastic_eval` | flag | false | Stochastic eval function |
| `-n` / `--name` | str | "test" | Training run name |
| `-lr` / `--learning_rate` | float | 0.00021989... | Optimizer learning rate |
| `--hidden` | int[] | [900,900,900,900] | Hidden layer sizes |
| `-en` / `--embedder_nodes` | int | 500 | Nodes per embedder layer |
| `-el` / `--embedder_layers` | int | 1 | Embedder layers |

---

## PowerShell Monitoring Commands

```powershell
# Check last 10 lines of training CSV
Get-Content "training_scripts\results\...\second_run\*.csv" | Select-Object -Last 10

# Check last 20 lines of main log
Get-Content "training_scripts\results\...\second_run\main.log" | Select-Object -Last 20

# List checkpoint files
Get-ChildItem "training_scripts\results\...\second_run\checkpoints\*.everl"
```

---

## Alternative Approaches Tried

| Approach | Description | Result |
|---|---|---|
| `make_mp(step_timeout=10.0)` | Increase SOFA timeout from 2s to 10s | Did not fix — nested multiprocessing was the real issue |
| `make_mp(step_timeout=30.0)` | Even longer timeout | Suggested but not tested |
| `make_non_mp()` | Remove SOFA multiprocessing entirely | **Working fix** — reduced 9 processes to 5 |
| `--shm-size=16g` | More shared memory | Suggested alternative to make_non_mp, not recommended |
| LCP solver tuning (`tolerance=1e-3, maxIt=5000`) | Adjust SOFA solver parameters | Suggested but not applied |
| Heatup range reduction (`[[15.0, 1.0], [12.0, 1.0]]`) | Reduce action ranges to avoid Case 1 errors | Tested, then reverted — not the root cause |
| Running with 1 worker (`-nw 1`) | Isolate multi-worker issues | Case 1 error still occurred |
| PYTHONPATH adjustment | Redirect imports to mounted code | Broke SOFA path setup |
| Direct env.py overlay mount | Mount file over installed package copy | **Working fix** |

---

## Key Technical Details

- **GPU:** NVIDIA RTX 4500 Ada Generation (24GB VRAM)
- **RAM:** 64GB
- **SOFA timeout root cause:** `make_mp()` spawns a subprocess per worker for SOFA simulation; with 4 workers = 9 total processes fighting for `/dev/shm`
- **make_non_mp() fix:** Runs SOFA simulation in the same process as the worker — eliminates subprocess overhead and `/dev/shm` contention
- **Training throughput (non_mp, 4 workers):** ~700-1000 steps/minute, ~480K steps in 20 hours
- **Policy collapse diagnosis:** Reward signal too weak for Q-function to distinguish forward vs backward movement. Step penalty dominates. `get_exploration_action` does NOT multiply by `action_scaling` but `get_eval_action` DOES — potential bug noted.
