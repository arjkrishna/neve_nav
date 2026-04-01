# Chat 1 Summary

_Exported: 3/10/2026 from Cursor (2.3.34)_
_Title: "Heatup and Logs"_
_~30,000 lines_

---

## High-Level Summary

This chat covers the **debugging, diagnostics instrumentation, and iterative improvement of RL training experiments** (diag_debug_test3 through diag_debug_test10) for the **DualDeviceNav** neurovascular catheter navigation task using the **eve_rl / SAC** reinforcement learning framework running in Docker on a GPU workstation.

### Key Activities (Chronological)

1. **Replay Buffer Debugging (test3-test4):** Discovered that completed episodes weren't reaching the replay buffer subprocess. Added logging to `vanillashared.py`'s `push()` method to trace episode flow through the multiprocessing queue.

2. **Diagnostics Infrastructure (test4-test5):** Built comprehensive diagnostics logging:
   - `diagnostics_logger.py` — logs losses, probe values, batch samples to CSV/JSONL
   - `probe_evaluator.py` — evaluates fixed "probe states" through the policy during training
   - Worker subprocess step-level logging (`STEP_LOG_DIR` environment variable)
   - Episode summary JSONL files from the Runner

3. **Experiment Validation Script:** Created `validate_experiment.py` to check experiment health:
   - Validates all log files, diagnostics, and probes
   - Detects training phase (heatup, training, evaluation)
   - Checks Docker container logs and file timestamps
   - Reports missing files with phase-appropriate context

4. **Volume Mount Pattern:** Established the pattern of mounting local edited Python files into the Docker container to override installed packages (eve_rl, eve_bench) without rebuilding the image.

5. **Multiple Environment Versions:** Worked with BenchEnv (v1 original PathDelta), BenchEnv2 (waypoint + centerlines), BenchEnv3 (tuned waypoint rewards).

6. **Policy Snapshot Saving:** Debugged and fixed policy snapshot saving logic — snapshots were not being saved due to the condition not being met. Fixed to save at every N exploration steps (catch-up logic for missed intervals).

7. **Training Analysis (test5-test9):** Ran experiments, monitored training progress, diagnosed slow episodes, checked for policy collapse, analyzed insertion depths and success rates.

8. **Checkpoint Resume (test10):** Created `DualDeviceNav_resume.py` — a copy of the training script with `--resume` flag to continue training from a saved checkpoint. Addressed:
   - Replay buffer starts fresh (not saved in checkpoints, too large)
   - Workers use the loaded trained policy (not random) to refill the buffer
   - Probe states need to be copied from the original experiment
   - Heatup is skipped when resuming

---

## Experiment Timeline

| Experiment | What Was Done | Outcome |
|---|---|---|
| **diag_debug_test3** | First test with CUDA device fix for `compute_batch_sample_values()` | Crashed after 725 updates (CUDA error on optimizer state dict transfer) |
| **diag_debug_test4** | Added all 4 CUDA fixes (batch_sample_values + 3 state_dict CPU transfers) | Ran successfully, 1975 updates, 0.56 sec/update. batch_samples=0 (logging bug) |
| **diag_debug_test5** | Fixed batch_samples logging (JSON serialization, exception handling, flush) | Ran 28+ hours, 59,588 updates, 1.19M exploration steps. All diagnostics working. |
| **diag_debug_test6** | Policy snapshot fix (catch-up logic for missed snapshot intervals) | Ran, validated with script |
| **diag_debug_test7** | Added monoplane volume mount (user noted it was changed) | Short run, status confirmed |
| **diag_debug_test8** | Quick test | Stopped by user ("stop this experiment and start the next one") |
| **diag_debug_test9** | Long-running experiment with all fixes | Ran to ~1.78M exploration steps; 0.26 success rate; episode summary JSONL missing (bug investigated); validated multiple times |
| **diag_debug_test10** | Resume from test9's `checkpoint1781693.everl` using new `DualDeviceNav_resume.py` | Container started, checkpoint loaded, replay buffer refilling from trained policy. 198 policy snapshots saved (3.1 GB). |

---

## Docker Commands

### 1. Training Experiment — Full Command with All Volume Mounts

This is the **canonical docker run command** used for training experiments (user-provided at line 6363). Container names changed per experiment (test3 through test10).

```powershell
docker run --name diag_debug_test4 --gpus all --shm-size=24g -d `
  -v "D:\stEVE_training\training_scripts\DualDeviceNav_train.py:/opt/eve_training/training_scripts/DualDeviceNav_train.py" `
  -v "D:\stEVE_training\training_scripts\util\env.py:/opt/eve_training/training_scripts/util/env.py" `
  -v "D:\stEVE_training\training_scripts\util\env2.py:/opt/eve_training/training_scripts/util/env2.py" `
  -v "D:\stEVE_training\training_scripts\util\env3.py:/opt/eve_training/training_scripts/util/env3.py" `
  -v "D:\stEVE_training\training_scripts\util\util.py:/opt/eve_training/training_scripts/util/util.py" `
  -v "D:\stEVE_training\training_scripts\util\agent.py:/opt/eve_training/training_scripts/util/agent.py" `
  -v "D:\stEVE_training\eve_rl\eve_rl\util\diagnostics_logger.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/diagnostics_logger.py" `
  -v "D:\stEVE_training\eve_rl\eve_rl\util\probe_evaluator.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/probe_evaluator.py" `
  -v "D:\stEVE_training\eve_rl\eve_rl\util\__init__.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/__init__.py" `
  -v "D:\stEVE_training\eve_rl\eve_rl\algo\sac.py:/usr/local/lib/python3.8/dist-packages/eve_rl/algo/sac.py" `
  -v "D:\stEVE_training\eve_rl\eve_rl\agent\single.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/single.py" `
  -v "D:\stEVE_training\eve_rl\eve_rl\agent\singelagentprocess.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/singelagentprocess.py" `
  -v "D:\stEVE_training\eve_rl\eve_rl\agent\synchron.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/synchron.py" `
  -v "D:\stEVE_training\eve_rl\eve_rl\runner\runner.py:/usr/local/lib/python3.8/dist-packages/eve_rl/runner/runner.py" `
  -v "D:\stEVE_training\eve_rl\eve_rl\replaybuffer\vanillashared.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillashared.py" `
  -v "D:\stEVE_training\training_scripts\results:/opt/eve_training/results" `
  -e STEP_LOG_DIR="/opt/eve_training/results" `
  eve-training-fixed `
  python3 /opt/eve_training/training_scripts/DualDeviceNav_train.py `
    --env_version 1 `
    -n diag_debug_test4 `
    -d cuda:0 `
    -nw 16
```

**Key flags:**
- `--gpus all` — enables GPU passthrough
- `--shm-size=24g` — large shared memory for SOFA physics + multiprocessing
- `-d` — detached (background) mode
- 16x `-v` mounts — override installed Python packages with local edits
- `-e STEP_LOG_DIR` — tells worker subprocesses where to write step logs
- Image: `eve-training-fixed`

**Python script flags:**
- `--env_version 1` — use BenchEnv (original PathDelta reward). Options: 1=original, 2=waypoint+centerlines, 3=tuned waypoint
- `-n <name>` — experiment name (becomes folder name in results)
- `-d cuda:0` — trainer device for NN updates
- `-nw 16` — number of parallel worker environments

### 2. Resume Training from Checkpoint (test10)

```powershell
docker run --name diag_debug_test10 --gpus all --shm-size=24g -d `
  # ... all the same volume mounts as above ... `
  eve-training-fixed `
  python3 /opt/eve_training/training_scripts/DualDeviceNav_resume.py `
  --env_version 1 `
  -n diag_debug_test10 `
  -d cuda:0 `
  -nw 16 `
  --resume "/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben/2026-02-21_185101_diag_debug_test9/checkpoints/checkpoint1781693.everl"
```

**Difference from standard:** Uses `DualDeviceNav_resume.py` with `--resume` pointing to a `.everl` checkpoint file. Skips heatup, loads trained weights into all workers.

### 3. Monitoring / Log Commands

```powershell
# Check container logs
docker logs diag_debug_test3

# Check for CUDA or exception errors (PowerShell)
docker logs diag_debug_test4 2>&1 | Select-String -Pattern "CUDA|Exception"
```

### 4. Docker Commands Inside validate_experiment.py (Programmatic)

These run inside Python via `subprocess.run()` and appear ~13 times across script iterations:

```python
# Check if container exists and its status
subprocess.run(["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Status}}"], ...)

# Get last 20 lines of container logs
subprocess.run(["docker", "logs", "--tail", "20", container_name], ...)
```

---

## Python Script Commands

### DualDeviceNav_train.py (Training Entrypoint)

```bash
# Inside Docker container:
python3 /opt/eve_training/training_scripts/DualDeviceNav_train.py \
  --env_version 1 \
  -n diag_debug_test5 \
  -d cuda:0 \
  -nw 16
```

**All flags:**
| Flag | Type | Default | Description |
|---|---|---|---|
| `-nw` / `--n_worker` | int | 4 | Number of parallel workers |
| `-d` / `--device` | str | cpu | Trainer device (`cpu`, `cuda:0`, `cuda:1`, `cuda`) |
| `-se` / `--stochastic_eval` | flag | false | Use stochastic eval function of SAC |
| `-n` / `--name` | str | "test" | Name of the training run |
| `-lr` / `--learning_rate` | float | 0.00021989... | Learning rate of optimizers |
| `--hidden` | int[] | [900,900,900,900] | Hidden layer sizes |
| `-en` / `--embedder_nodes` | int | 500 | Nodes per embedder layer |
| `-el` / `--embedder_layers` | int | 1 | Number of embedder layers |
| `--env_version` | int | 1 | Environment: 1=PathDelta, 2=waypoint+centerlines, 3=tuned waypoint |

### DualDeviceNav_resume.py (Resume Training)

```bash
# Fresh training:
python DualDeviceNav_resume.py -n my_experiment -d cuda:0 -nw 16

# Resume from checkpoint:
python DualDeviceNav_resume.py -n my_experiment_resumed -d cuda:0 -nw 16 \
  --resume /path/to/checkpoint.everl
```

**Additional flag (vs _train.py):**
| Flag | Type | Default | Description |
|---|---|---|---|
| `--resume` | str | None | Path to `.everl` checkpoint file to resume from. Skips heatup when provided. |

### validate_experiment.py (Experiment Validation)

```bash
# Basic validation
python validate_experiment.py diag_debug_test5

# Verbose output
python validate_experiment.py diag_debug_test5 --verbose

# With full path instead of name
python validate_experiment.py --full-path /path/to/experiment

# With Docker container check
python validate_experiment.py diag_debug_test5 --docker-container my_container

# Shorthand flags
python validate_experiment.py diag_debug_test6 -d diag_debug_test6 -v

# No color output
python validate_experiment.py diag_debug_test5 --no-color
```

**All flags:**
| Flag | Type | Description |
|---|---|---|
| positional `name` | str | Experiment name to search for |
| `--full-path` | str | Full path to experiment directory (instead of name search) |
| `-v` / `--verbose` | flag | Verbose output |
| `-d` / `--docker-container` | str | Docker container name to check logs/status |
| `--no-color` | flag | Disable colored output |

---

## Key Technical Details

- **Docker image:** `eve-training-fixed` (pre-built with eve, eve_rl, eve_bench, SOFA, PyTorch, CUDA)
- **Training algorithm:** SAC (Soft Actor-Critic) with dual Q-networks
- **Environment:** DualDeviceNav (catheter + guidewire in neurovascular mesh)
- **Results path inside container:** `/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben/`
- **Results path on host:** `D:\stEVE_training\training_scripts\results\`
- **Python version in container:** 3.8
- **Typical heatup:** 10,000 steps (random actions)
- **Typical training:** 20,000,000 steps
- **Workers:** 16 parallel environment instances
- **Checkpoint format:** `.everl` files (contain network weights, optimizer states, scheduler states, step/episode counters)
- **Key hyperparameters:** gamma=0.99, batch_size=32, replay_buffer_size=10,000, update_per_explore_step=1/20, lr_end_factor=0.15, lr_linear_end_steps=6M
