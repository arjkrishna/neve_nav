# Chat 2 Summary

_Exported: 3/10/2026 from Cursor (2.3.34)_
_Title: "Heatup and Logs"_
_~166,500 lines (very long session)_

---

## High-Level Summary

This chat is a **continuation and massive expansion** of the work in Chat 1. It covers the same DualDeviceNav RL training project but goes much deeper into **post-hoc training analysis, collapse forensics tooling, policy visualization, and diagnostic plot generation**. The bulk of the chat (~110,000 lines) is iterative development of sophisticated analysis scripts, with each revision shown in full.

### Project Context

- **Framework:** `eve_rl` (custom RL library built on top of EVE/SOFA simulation)
- **Task:** `DualDeviceNav` — dual-device catheter + guidewire navigation in patient-specific neurovascular meshes
- **Algorithm:** SAC (Soft Actor-Critic) with dual Q-networks
- **Training infrastructure:** Docker containers on GPU workstation, X11 forwarding for visualization to `10.113.123.221:0`

### Key Activities (Chronological)

1. **Experiment Validation Script Updates:** Extended `validate_experiment.py` with new file checks, phase detection, Docker log parsing, and support for in-progress experiments. Discovered STEP_LOG_DIR timing bug (env var must be set BEFORE agent creation, since workers are spawned in `__init__`).

2. **Collapse Forensics Tool (diagnose_collapse v1-v6):** The largest effort in this chat. Built a multi-version forensics tool to automatically detect and diagnose **policy collapse** in SAC training:
   - **v1-v2:** Basic loss parsing, alpha shutdown detection, entropy drop detection
   - **v3:** Added episode-level analysis from worker logs (insertion depth, stuck detection, reward tracking), batch sample analysis, policy snapshot correlation, event timeline with confidence scores. Worker log deduplication fix (3569 episodes after dedup for test9).
   - **v4:** Improved visualization — separate plots for grad norms (policy/q1), Q-losses (q1/q2), Q-means (q1/q2), critic preference (zoomed), translation speed, stuck fraction. Removed event lines from most plots (kept only for alpha). 22 plot files generated.
   - **v5-v6:** Refined thresholds, improved stuck detection, data-driven saturation thresholds (using early-training percentiles instead of fixed bounds), replay distribution shift detection (gated by insertion_collapse), episode summary correlation with explore/update steps via wall-time matching

   **Events detected by the tool:**
   - `alpha_shutdown` — entropy coefficient drops to near-zero
   - `entropy_drop` — policy entropy collapses
   - `policy_grad_collapse` — policy gradient norm drops dramatically
   - `probe_translation_saturated` — probe states show saturated actions
   - `probe_std_collapse` — probe action variance collapses
   - `probe_freeze` — probe actions freeze near zero
   - `actor_saturated` — batch actor mean saturates
   - `insertion_collapse` — rolling median insertion depth drops below threshold (200mm)
   - `stuck_regime` — policy repeatedly commands insertion but catheter doesn't move
   - `critic_loss_spike` — Q-function loss spikes
   - `replay_distribution_shift_positive` / `replay_distribution_shift_negative` — replay buffer distribution shifts (gated)

3. **Policy Visualization:** Created `visualize_policy.py` to render trained policies on the neurovascular mesh with SOFA/Pygame visualization via X11 forwarding. Iteratively fixed:
   - Container path resolution for experiment directories
   - Checkpoint loading with correct environment config (`get_env_from_checkpoint()`)
   - Policy snapshot loading (smaller `.pt` files vs full `.everl` checkpoints)
   - SOFA timeout issues
   - Observation structure mismatch between saved model and environment

4. **Training Collage Generator:** Created `create_training_collage.py` to arrange diagnostic plots into a 3x2 grid:
   - Left column: Critic preference (zoomed), Batch reward mean, Batch translation mean
   - Right column: Probe translation mean, Probe translation std, (empty)

5. **Episode Collage Generator:** Created `create_episode_collage.py` for episode-level metric plots.

6. **Worker Step Log Parsing:** Extensive work on parsing STEP-level logs from worker subprocesses:
   - Field-based extraction (order-independent, robust to format changes)
   - V2 format (action) vs V3 format (cmd_action/exec_action) support
   - Stuck detection: policy commands positive translation but delta insertion near-zero
   - Episode summary generation with max insertion, stuck fraction, jam detection, depth binning
   - Wall-time correlation to map episodes to explore/update step numbers
   - Estimate source priority system (summary_exact > wall_time_match > ts_interp > linear_interp)

7. **Cross-Experiment Comparison:** Read checkpoint success values across experiments 2-8 to compare performance. Found longer runs (2200+ episodes) achieved higher success rates.

---

## Experiment Timeline

| Experiment | What Was Done | Key Results |
|---|---|---|
| **diag_debug_test5** | Baseline experiment used as default example in all script docstrings | Used for collapse forensics development |
| **diag_debug_test6** | Testing three detection fixes (replay shift gating, confidence scoring, saturation thresholds) | Visualization: FAIL, reward -4.8627, 1000 steps max reached |
| **diag_debug_test7** | Validated with `validate_experiment.py -d diag_debug_test7 -v` | 19 passed, 1 failed (worker logs missing due to STEP_LOG_DIR timing bug). 82 policy snapshots, 294 episodes. |
| **diag_debug_test9** | Primary experiment for analysis and visualization | 3569 episodes (after dedup). 7 events detected, earliest: alpha_shutdown at update step 20814. Classification: policy_collapse_or_suboptimal_attractor. 0.26 success rate. Visualized at step 1,780,000. |
| **Experiments 2-8** | Cross-experiment checkpoint comparison | Longer runs achieved higher success rates |

---

## Docker Commands

### 1. Visualization — `eve-training-fixed` Image, 3 Volume Mounts, 5 Episodes (Cursor adaptation of user's pattern)

```bash
docker run --rm \
  -v "D:\stEVE_training\eve_bench:/opt/eve_training/eve_bench" \
  -v "D:\stEVE_training:/opt/eve_training_test" \
  -v "D:\vmr:/vmr_host" \
  -e DISPLAY=10.113.123.221:0 \
  eve-training-fixed \
  python3 /opt/eve_training_test/training_scripts/visualize_policy.py \
    --name diag_debug_test9 --snapshot-step 1780000 --episodes 5
```

**Purpose:** Visualize the trained policy navigating the neurovascular mesh. Adapted from user's existing `dual_human_play.py` command.

### 2. Visualization — 1 Episode, 1000 Steps

```bash
docker run --rm \
  -v "D:\stEVE_training\eve_bench:/opt/eve_training/eve_bench" \
  -v "D:\stEVE_training:/opt/eve_training_test" \
  -v "D:\vmr:/vmr_host" \
  -e DISPLAY=10.113.123.221:0 \
  eve-training-fixed \
  python3 /opt/eve_training_test/training_scripts/visualize_policy.py \
    --name diag_debug_test9 --snapshot-step 1780000 --episodes 1 --max-steps 1000
```

### 3. Visualization — `steve_training` Image, Single Workspace Mount

```bash
docker run -it --rm \
  -e DISPLAY=10.113.123.221:0 \
  -v D:\stEVE_training:/workspace \
  --shm-size=8g \
  steve_training \
  python training_scripts/visualize_policy.py \
    --name diag_debug_test9 --snapshot-step 1780000 --episodes 5
```

**Note:** Uses `steve_training` image (older), single workspace mount at `/workspace`, adds `-it` for interactive terminal and `--shm-size=8g`.

### 4. Visualization — Generic Template with Placeholder Paths

```bash
docker run -it --rm \
  -e DISPLAY=10.113.123.221:0 \
  -v /path/to/stEVE_training:/workspace \
  --shm-size=8g \
  your_image_name \
  python training_scripts/visualize_policy.py --name diag_debug_test9 --episodes 5
```

### 5. Visualization — Generic Template with Explicit Checkpoint

```bash
docker run -it --rm \
  -e DISPLAY=10.113.123.221:0 \
  -v /path/to/stEVE_training:/workspace \
  --shm-size=8g \
  your_image_name \
  python training_scripts/visualize_policy.py \
    --checkpoint /workspace/training_scripts/results/eve_paper/neurovascular/full/mesh_ben/2026-02-21_185101_diag_debug_test9/checkpoints/best_checkpoint.everl \
    --episodes 5
```

### 6. Human Play / Manual Control (User's Original Reference Command)

```bash
docker run --rm \
  -v "D:\stEVE_training\eve_bench:/opt/eve_training/eve_bench" \
  -v "D:\stEVE_training:/opt/eve_training_test" \
  -v "D:\vmr:/vmr_host" \
  -e DISPLAY=10.113.123.221:0 \
  eve-training-fixed \
  python3 /opt/eve_training_test/eve_bench/example/dual_human_play.py
```

**Purpose:** Manual control of the catheter/guidewire for testing environment setup. This is the user's pre-existing command that was adapted for visualization.

### 7. Programmatic Docker Commands (inside validate_experiment.py)

```bash
# Check container status
docker ps -a --filter "name=<container_name>" --format "{{.Status}}"

# Get last 20 lines of container logs
docker logs --tail 20 <container_name>
```

### 8. X11 Prerequisite

```bash
xhost +
```

**Purpose:** Allow X11 connections on the display host (10.113.123.221) before running GUI visualization.

---

## Python Script Commands

### visualize_policy.py (Policy Visualization)

```bash
# Auto-find best checkpoint by experiment name
python visualize_policy.py --name diag_debug_test9

# Use specific policy snapshot step (faster loading, smaller files)
python visualize_policy.py --name diag_debug_test9 --snapshot-step 500000

# Use latest snapshot
python visualize_policy.py --name diag_debug_test9 --snapshot-step 1780000

# Explicit paths for checkpoint and snapshot
python visualize_policy.py \
  --checkpoint /path/to/best_checkpoint.everl \
  --snapshot /path/to/policy_500000.pt

# With episode/step limits
python visualize_policy.py --name diag_debug_test9 --snapshot-step 1780000 \
  --episodes 1 --max-steps 1000

# Stochastic evaluation
python visualize_policy.py --name diag_debug_test9 --stochastic
```

**All flags:**
| Flag | Type | Default | Description |
|---|---|---|---|
| `--name` | str | — | Experiment name (auto-finds checkpoint in results dir) |
| `--checkpoint` | str | — | Explicit path to `.everl` checkpoint file |
| `--snapshot` | str | — | Explicit path to `.pt` policy snapshot file |
| `--snapshot-step` | int | — | Use policy snapshot from specific explore step |
| `--episodes` | int | 5 | Number of visualization episodes to run |
| `--max-steps` | int | 500 | Maximum steps per episode |
| `--seed` | int | 42 | Random seed |
| `--stochastic` | flag | false | Use stochastic (not deterministic) policy |

### diagnose_collapse3.py (Collapse Forensics v3)

```bash
# By run directory
python diagnose_collapse3.py --run-dir /path/to/run_dir

# By experiment name (auto-finds in results)
python diagnose_collapse3.py --name diag_debug_test5

# Positional argument (shorthand)
python diagnose_collapse3.py diag_debug_test6
```

**All flags:**
| Flag | Type | Default | Description |
|---|---|---|---|
| `--run-dir` | str | — | Training run directory |
| `--name` | str | — | Experiment name to search for |
| `--results-base` | str | `training_scripts/results/eve_paper/neurovascular/full/mesh_ben` | Base results directory |
| `--alpha-threshold` | float | 0.01 | Alpha shutdown threshold |
| `--sustain` | int | 50 | Sustained steps for event detection |
| `--plot` | flag | false | Generate diagnostic plots |

### diagnose_collapse4.py (Collapse Forensics v4)

```bash
# By run directory
python diagnose_collapse4.py --run-dir /path/to/run_dir

# By experiment name
python diagnose_collapse4.py --name diag_debug_test5

# With plots enabled
python diagnose_collapse4.py --name diag_debug_test9 --plot

# Custom thresholds
python diagnose_collapse4.py --name diag_debug_test9 --alpha-threshold 0.01 --sustain 50 --plot
```

**All flags:** Same as v3. Generates 22 plot files including: alpha, entropy, Q-means, Q-losses, grad norms (policy/q1), probe translation mean/std, batch translation, batch reward, critic preference (normal + zoomed), insertion depth, episode rewards, stuck fraction, max stuck run, batch done rate.

### validate_experiment.py (Experiment Validation)

```bash
# Basic validation
python validate_experiment.py diag_debug_test5

# Verbose output
python validate_experiment.py diag_debug_test5 --verbose

# With Docker container check (shorthand)
python validate_experiment.py diag_debug_test7 -d diag_debug_test7 -v

# With full path instead of name
python validate_experiment.py --full-path /path/to/experiment

# No color output
python validate_experiment.py diag_debug_test5 --no-color
```

### create_training_collage.py (Training Plot Collage)

```bash
# Create collage for an experiment
python create_training_collage.py --name diag_debug_test9

# Custom output path
python create_training_collage.py --name diag_debug_test9 --output training_collage.png

# Custom results base
python create_training_collage.py --name diag_debug_test9 \
  --results-base training_scripts/results/eve_paper/neurovascular/full/mesh_ben
```

**All flags:**
| Flag | Type | Default | Description |
|---|---|---|---|
| `--name` | str | required | Experiment name |
| `--results-base` | str | `training_scripts/results/eve_paper/neurovascular/full/mesh_ben` | Base results directory |
| `--output` | str | `diagnostics/analysis/training_collage.png` | Output path for collage |

### PowerShell Utility Commands

```powershell
# Count worker log files
Get-ChildItem "path\to\diagnostics\logs_subprocesses\" | Measure-Object
```

---

## Docker Images Used

| Image Name | Purpose |
|---|---|
| `eve-training-fixed` | Primary training/visualization image with EVE environment (user's actual image) |
| `steve_training` | Alternative/older image name (used in some Cursor suggestions) |
| `your_image_name` | Placeholder in generic templates |

## Volume Mounts Reference

| Host Path | Container Path | Purpose |
|---|---|---|
| `D:\stEVE_training` | `/workspace` or `/opt/eve_training_test` | Main training code and results |
| `D:\stEVE_training\eve_bench` | `/opt/eve_training/eve_bench` | EVE benchmark environment |
| `D:\vmr` | `/vmr_host` | Vascular mesh/model data |

## Key Docker Flags Reference

| Flag | Value | Purpose |
|---|---|---|
| `-e DISPLAY` | `10.113.123.221:0` | X11 forwarding for SOFA/Pygame GUI |
| `--shm-size` | `8g` (viz) / `24g` (training) | Shared memory for SOFA + multiprocessing |
| `--rm` | — | Auto-remove container after exit |
| `-it` | — | Interactive terminal (visualization) |
| `-d` | — | Detached mode (training) |
| `--gpus all` | — | GPU passthrough (training only) |

---

## Key Technical Details

- **Collapse forensics scripts:** `diagnose_collapse.py` through `diagnose_collapse6.py` (iterative versions, each shown in full multiple times)
- **Analysis output:** `diagnostics/analysis/` folder within each experiment directory
- **Report format:** Markdown report (`collapse_report_v3.md`) + PNG plots
- **Classification output:** `policy_collapse_or_suboptimal_attractor`, `critic_instability_precedes_policy`, or `no_clear_event_detected`
- **Key experiments analyzed:** `diag_debug_test5` (baseline), `diag_debug_test9` (primary focus)
- **Policy snapshots:** `.pt` files saved every 10,000 exploration steps (lighter than full `.everl` checkpoints)
- **Experiment 9 results:** ~0.26 success rate, ~1.78M exploration steps, 7 collapse events detected (earliest: alpha_shutdown at update step 20,814), classification: policy_collapse_or_suboptimal_attractor
- **Detection thresholds (v4+):** saturation_scale_factor=0.7, saturation_min_floor=0.15, std_collapse_fraction=0.5, freeze_mean_eps=0.05, insertion_collapse=200mm, stuck_fraction=30%, jam_episode=30%
- **Chat structure note:** ~110,000 lines of the chat are repeated iterations of Python source code (validate_experiment.py, diagnose_collapse3.py, diagnose_collapse4.py) being revised. The same docstring usage examples repeat dozens of times.
