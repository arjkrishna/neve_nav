# Chat 4 Summary

_Exported: 3/15/2026 from Cursor (2.3.34)_
_Title: (DualDeviceNav training continuation)_
_~9,306 lines_

---

## High-Level Summary

This chat covers the **creation and testing of alternative reward structures (env2/env3)** and a **systematic comparison of three environment versions**, all of which resulted in policy collapse. It also includes worker scaling analysis, mount path debugging, and the beginning of understanding why all reward structures fail.

### Key Activities (Chronological)

1. **File Inventory and Code Review:** Reviewed all new/modified files for state/reward modifications including `fixedpath.py`, `waypointprogress.py`, `centerlines2d.py`, `normalizecenterlines2depisode.py`, `env2.py`, and debug logging wrapper.

2. **Modified DualDeviceNav_train.py:** Added `--env_version` flag (choices: 1, 2, 3) to switch between env.py (v1 PathDelta), env2.py (v2 waypoint+centerlines), and env3.py (v3 tuned waypoint).

3. **Worker Scaling Analysis:** Analyzed GPU (RTX 4500 Ada 24GB) and RAM (64GB) to determine optimal worker count. Recommended 16 workers (up from 4).

4. **ENV2 Training (`non_mp_second_run_gpu`):** Launched with 16 workers. Encountered and fixed multiple bugs:
   - Mount path: eve installed at `/usr/local/lib/python3.8/dist-packages/eve/` not `/opt/eve_training/eve/`
   - Import error: `dijkstra2.py` needed despite not being directly used (imported by `__init__.py`)
   - `centerlineprogress` import removed from reward `__init__.py`
   - Bug: `vessel_tree.insertion.coordinates` changed to `vessel_tree.insertion.position`
   - Added `np.array()` conversion for insertion position
   - **Result:** Heatup OK (500K steps, ~8h). Policy collapsed after heatup — 88 target reaches during heatup, 0 after. Guidewire moved but policy was overly conservative.

5. **Reward Tuning Discussion:** User identified waypoint spacing too wide, rewards too small numerically relative to step penalty.

6. **Created env3.py:** New tuned parameters — `waypoint_spacing=5mm` (was 10), `branch_base_increment=1.0` (was 0.1), `step_penalty=-0.001` (was -0.005).

7. **ENV3 Training (`non_mp_third_run_gpu`):** Ran for 3 days. Found shallow local optimum at 7-30mm. Better reward metrics (6.9% positive) but worse actual insertion depth.

8. **ENV2 vs ENV3 Comparison:** At ~550K steps — ENV2 had better insertion (47mm avg) but ENV3 had better reward metrics.

9. **Visualization Run:** Human play with X11 forwarding. Ran 230 steps, confirmed vessel branch structure. Targets at 400-500mm depth.

10. **ENV Rerun (`non_mp_env_rerun_gpu`):** Original env.py with 16 workers for fair comparison. Heatup completed (506K steps, 7.47 hours). At 273K training steps: quality=0.0, reward=-4.996, deterministic eval shows 5.4mm (frozen policy). Same collapse pattern.

11. **Replay Buffer Caching Idea:** Discussed saving heatup replay buffer via `pickle.dump`/`pickle.load` to skip 15-hour heatup for subsequent runs. Not implemented.

---

## Experiment Timeline

| Experiment | Container | Env | Workers | Outcome |
|---|---|---|---|---|
| `non_mp_first_run_gpu` | (from chat3) | env.py (PathDelta) | 4 | Referenced — policy collapsed |
| `non_mp_second_run_gpu` | non_mp_second_run_gpu | env2.py (Waypoint 10mm, 0.1 incr, -0.005 penalty) | 16 | **FAILED:** 88 target reaches during heatup, 0 after. Avg insertion: 47mm (eval: 0.4mm) |
| `non_mp_third_run_gpu` | non_mp_third_run_gpu | env3.py (Waypoint 5mm, 1.0 incr, -0.001 penalty) | 16 | **FAILED:** Local optimum at 7-30mm. Better reward metrics (6.9% positive) but shallow insertion. Ran 3 days. |
| `non_mp_env_rerun_gpu` | non_mp_env_rerun_gpu | env.py (original PathDelta) | 16 | **FAILED:** Same collapse. Heatup OK (506K steps). Eval at 273K training steps: 5.4mm (frozen policy). |

**Conclusion:** All three reward structures produce identical policy collapse. Random heatup achieves ~1300mm avg, then training collapses 95-98%.

---

## Docker Commands

### 1. Reference Command — Original 4-Worker Run (User, Line 4676)

```bash
docker run --name non_mp_first_run_gpu --gpus all --shm-size=8g \
  -v "D:\stEVE_training\training_scripts\util\env.py:/opt/eve_training/training_scripts/util/env.py" \
  -v "D:\stEVE_training\training_scripts/results:/opt/eve_training/results" \
  -e STEP_LOG_DIR="/opt/eve_training/results" \
  eve-training-fixed \
  python3 /opt/eve_training/training_scripts/DualDeviceNav_train.py \
  -n non_mp_first_run_gpu -d cuda:0 -nw 4
```

### 2. ENV2 Training — 16 Workers (Cursor, ~Line 4760)

```bash
docker run --name non_mp_second_run_gpu --gpus all --shm-size=24g \
  -v "D:\stEVE_training\training_scripts\util\env2.py:/opt/eve_training/training_scripts/util/env2.py" \
  -v "D:\stEVE_training\training_scripts\util\env.py:/opt/eve_training/training_scripts/util/env.py" \
  -v "D:\stEVE_training\training_scripts\DualDeviceNav_train.py:/opt/eve_training/training_scripts/DualDeviceNav_train.py" \
  -v "D:\stEVE_training\training_scripts/results:/opt/eve_training/results" \
  -v "D:\stEVE_training\eve\eve\reward\fixedpath.py:/usr/local/lib/python3.8/dist-packages/eve/reward/fixedpath.py" \
  -v "D:\stEVE_training\eve\eve\reward\waypointprogress.py:/usr/local/lib/python3.8/dist-packages/eve/reward/waypointprogress.py" \
  -v "D:\stEVE_training\eve\eve\reward\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/reward/__init__.py" \
  -v "D:\stEVE_training\eve\eve\observation\centerlines2d.py:/usr/local/lib/python3.8/dist-packages/eve/observation/centerlines2d.py" \
  -v "D:\stEVE_training\eve\eve\observation\normalizecenterlines2depisode.py:/usr/local/lib/python3.8/dist-packages/eve/observation/normalizecenterlines2depisode.py" \
  -v "D:\stEVE_training\eve\eve\observation\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/observation/__init__.py" \
  -v "D:\stEVE_training\eve\eve\pathfinder\dijkstra2.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/dijkstra2.py" \
  -e STEP_LOG_DIR="/opt/eve_training/results" \
  eve-training-fixed \
  python3 /opt/eve_training/training_scripts/DualDeviceNav_train.py \
  --env_version 2 -n non_mp_second_run_gpu -d cuda:0 -nw 16
```

**Key differences from original:** `--shm-size=24g` (scaled for 16 workers), additional `-v` mounts for new eve reward/observation/pathfinder files at `/usr/local/lib/python3.8/dist-packages/eve/`, `--env_version 2`.

**Mount path lesson:** eve is installed at `/usr/local/lib/python3.8/dist-packages/eve/...` inside the container, NOT at `/opt/eve_training/eve/...`.

### 3. ENV3 Training (Cursor, ~Line 7054)

Same as ENV2 command but with:
- `--name non_mp_third_run_gpu`
- Additional mount: `env3.py`
- `--env_version 3`

### 4. ENV Rerun — Original with 16 Workers (Cursor, ~Line 8448)

Same volume mounts as original 4-worker run but with:
- `--name non_mp_env_rerun_gpu`
- `--shm-size=24g`
- `-nw 16`
- `--env_version 1`

### 5. Human Play / Visualization (User, Line 8057)

```bash
docker run --rm \
  -v "D:\stEVE_training\eve_bench:/opt/eve_training/eve_bench" \
  -v "D:\stEVE_training:/opt/eve_training_test" \
  -v "D:\vmr:/vmr_host" \
  -e DISPLAY=10.113.123.221:0 \
  eve-training-fixed \
  python3 /opt/eve_training_test/eve_bench/example/dual_human_play_general.py
```

**Note:** Uses `dual_human_play_general.py` (not `dual_human_play.py` from Chat 2).

### 6. Docker Logs (Monitoring)

```powershell
# PowerShell — last 20 lines of container logs
docker logs non_mp_second_run_gpu 2>&1 | Select-Object -Last 20

# Follow live logs
docker logs -f non_mp_env_rerun_gpu
```

### 7. Docker Stop/Remove

```bash
# Stop and remove before starting next experiment
docker stop non_mp_second_run_gpu
docker rm non_mp_second_run_gpu

docker stop non_mp_third_run_gpu
docker rm non_mp_third_run_gpu
```

---

## Python Script Commands

### DualDeviceNav_train.py with --env_version Flag

```bash
# ENV2 (waypoint + centerlines)
python training_scripts/DualDeviceNav_train.py --env_version 2 -n my_waypoint_run -d cuda -nw 4

# ENV1 (original PathDelta)
python training_scripts/DualDeviceNav_train.py --env_version 1 -n my_original_run -d cuda -nw 4

# Default (version 1)
python training_scripts/DualDeviceNav_train.py -n my_original_run -d cuda -nw 4
```

**New flag (added in this chat):**
| Flag | Type | Default | Choices | Description |
|---|---|---|---|---|
| `--env_version` | int | 1 | 1, 2, 3 | Environment: 1=PathDelta, 2=waypoint+centerlines, 3=tuned waypoint |

### dual_human_play_general.py (Visualization)

```bash
python3 /opt/eve_training_test/eve_bench/example/dual_human_play_general.py
```

**Purpose:** Interactive human control of catheter/guidewire in neurovascular mesh. No command-line flags.

---

## PowerShell Monitoring Commands

```powershell
# Check last 20 lines of main training log
Get-Content "D:\stEVE_training\training_scripts\results\...\non_mp_second_run_gpu\main.log" | Select-Object -Last 20

# Check last 20 lines of container logs
docker logs non_mp_second_run_gpu 2>&1 | Select-Object -Last 20

# Check if results directory exists
ls D:\stEVE_training\training_scripts\results\eve_paper\neurovascular\full\mesh_ben\*non_mp_env_rerun_gpu*
```

---

## Key Technical Details

- **Hardware:** NVIDIA RTX 4500 Ada (24GB VRAM), 64GB RAM
- **Worker scaling:** 4 workers -> 16 workers (with `--shm-size` 8g -> 24g)
- **Mount path correction:** eve packages installed at `/usr/local/lib/python3.8/dist-packages/eve/` inside Docker image (not `/opt/eve_training/eve/`)
- **dijkstra2.py inclusion:** Not directly used but imported by `__init__.py` — must be mounted
- **Result folder paths:**
  - `D:\stEVE_training\training_scripts\results\eve_paper\neurovascular\full\mesh_ben\2026-01-22_023221_non_mp_second_run_gpu`
  - `D:\stEVE_training\training_scripts\results\eve_paper\neurovascular\full\mesh_ben\2026-01-27_023329_non_mp_env_rerun_gpu`
- **Target depth:** 400-500mm; best achieved was ~60mm (exploration), 5.4mm (deterministic eval)
- **All three envs collapsed identically:** Random heatup achieves ~1300mm avg, training collapses 95-98%
- **Replay buffer caching:** Proposed (pickle.dump/load) but not implemented — would save ~15h heatup time
