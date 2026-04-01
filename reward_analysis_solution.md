

The “proper timestamp correlation” part is made much stronger in our since we have log wall_time (timestamp) into losses.csv / batch_samples.jsonl / probes.jsonl. This patch does not require any training-code changes 


This patch assumes we can use global= from worker step logs as the primary timeline anchor but since we have already logged wall_time (timestamp) and did Episode ↔ training timeline correlation; avoid implementing Episode ↔ training timeline correlation logic below but since "stuck" detection logic is linked to Episode ↔ training timeline correlation use it to understand "Stuck" Detection logic and implement the new stuck logic with our already implemented Episode ↔ training timeline correlation logic  



### Solution (Coding Edits) in diagnose_collapse3.py (only)


Below is a **function-by-function plan with exact edits** to implement (1) **proper timeline correlation** (so episode-derived events get valid `update_step` / `explore_step`) and (2) **“stuck-run” detection** using the **worker STEP logs**.




# A) What’s wrong in the current `diagnose_collapse3.py` (root issues)

## 1) Episode ↔ training timeline correlation is wrong

The current function `correlate_episodes_to_explore_step()` is **index-based**, not “true timestamp/global-step based”. That causes:

* `update_step=None` for episode-derived events (insertion collapse etc.)
* Events sort incorrectly (episodes appear “late” in the ordering)
* You get mismatched narratives (“insertion collapses before alpha shutdown” even if it didn’t)

✅ Fix: Use **`global=`** from worker logs whenever available (most accurate), else interpolate by real timestamps if available, else fallback.

---

## 2) No “stuck” detection exists despite having logs

You log step-level fields:

* `action=[...]`
* `inserted=[...]`
* `delta_ins=[...]`

…but the analysis script ignores `global`, ignores step-level rewards, and does not detect:

* long sequences of “commanded insert” + “no motion”
* stuck depth region (where it happens)
* whether it persists / becomes dominant

✅ Fix: Parse `global` and step reward, compute:

* `stuck_fraction`, `max_stuck_run`, depth-bin mode
* add an event `stuck_regime`

---

# B) Exact edits — function-by-function

## 1) **Constants (top-level)** — add stuck parameters

**Where:** right below `INSERTION_COLLAPSE_THRESHOLD_MM`

**Add:**

```python
# Stuck detection (worker STEP logs)
# Interprets "stuck" as: commanded positive translation but near-zero delta insertion.
STUCK_TRANSLATION_INDEX = 0          # action dim for translation/advance
STUCK_PUSH_THRESHOLD = 0.05          # cmd translation > this means "trying to insert"
STUCK_DELTA_INS_EPS_MM = 0.2         # |delta_ins| < this means "no motion"
STUCK_MIN_CONSEC_STEPS = 25          # min consecutive stuck steps to count as a run
STUCK_DEPTH_BIN_MM = 25.0            # bin size for reporting stuck depth mode
```

---

## 2) `EpisodeRowV3` dataclass — add fields for correlation + stuck stats

**Where:** inside `class EpisodeRowV3`

**Replace the old single field:**

```python
# Correlated explore_step (estimated via timestamp)
explore_step_estimate: Optional[int] = None
```

**With this expanded block (with correct indentation):**

```python
# Correlated explore/update steps
# explore_step_estimate prefers worker `global=` counter when available; falls back to timestamp/index.
explore_step_estimate: Optional[int] = None
update_step_estimate: Optional[int] = None

# From worker STEP logs (used for stuck diagnostics + better correlation)
start_global: Optional[int] = None
end_global: Optional[int] = None
stuck_steps: int = 0
stuck_fraction: float = 0.0
max_stuck_run: int = 0
stuck_depth_mode_mm: Optional[float] = None
mean_cmd_translation: Optional[float] = None
std_cmd_translation: Optional[float] = None
mean_step_reward: Optional[float] = None
min_step_reward: Optional[float] = None
```

---

## 3) `load_worker_logs_v3()` — parse `global`, step reward, compute stuck stats

### 3a) Parse `global_step` and `step_reward`

**In the STEP_RE_NEW block**, right after:

```python
ep_step = int(m_new.group(3))
```

**Insert:**

```python
global_step = _to_int(m_new.group(4), default=-1)
```

**Right after action parsing**, before cum_reward:

```python
step_reward = _to_float(m_new.group(6))
```

### 3b) Initialize episode dict with counters

When `key not in data`, after `actions=[action],` add:

```python
start_global=global_step if global_step >= 0 else None,
end_global=global_step if global_step >= 0 else None,
_cmd_trans_sum=(action[STUCK_TRANSLATION_INDEX] if len(action) > STUCK_TRANSLATION_INDEX else 0.0),
_cmd_trans_sumsq=(action[STUCK_TRANSLATION_INDEX] if len(action) > STUCK_TRANSLATION_INDEX else 0.0) ** 2,
_cmd_trans_n=1,
_step_reward_sum=(step_reward if not math.isnan(step_reward) else 0.0),
_step_reward_min=(step_reward if not math.isnan(step_reward) else float('inf')),
_step_reward_n=(0 if math.isnan(step_reward) else 1),
_stuck_steps=0,
_cur_stuck_run=0,
_max_stuck_run=0,
_stuck_depth_bins={},
```

### 3c) Update stuck counters per step

After the existing delta sums:

```python
d["total_delta_ins_2"] = d.get("total_delta_ins_2", 0.0) + delta2
```

Insert the stuck detection block:

```python
# Global step tracking (preferred correlation)
if global_step >= 0:
    if d.get("start_global") is None:
        d["start_global"] = global_step
    d["end_global"] = max(d.get("end_global") or global_step, global_step)

# Action translation stats
cmd_trans = action[STUCK_TRANSLATION_INDEX] if len(action) > STUCK_TRANSLATION_INDEX else 0.0
d["_cmd_trans_sum"] = d.get("_cmd_trans_sum", 0.0) + cmd_trans
d["_cmd_trans_sumsq"] = d.get("_cmd_trans_sumsq", 0.0) + cmd_trans * cmd_trans
d["_cmd_trans_n"] = d.get("_cmd_trans_n", 0) + 1

# Step reward stats
if not math.isnan(step_reward):
    d["_step_reward_sum"] = d.get("_step_reward_sum", 0.0) + step_reward
    d["_step_reward_min"] = min(d.get("_step_reward_min", float("inf")), step_reward)
    d["_step_reward_n"] = d.get("_step_reward_n", 0) + 1

# Stuck-run detection
# "stuck" = pushing forward but delta insertion is ~0
is_push = cmd_trans > STUCK_PUSH_THRESHOLD
is_no_motion = abs(delta1) < STUCK_DELTA_INS_EPS_MM
if is_push and is_no_motion:
    d["_stuck_steps"] = d.get("_stuck_steps", 0) + 1
    d["_cur_stuck_run"] = d.get("_cur_stuck_run", 0) + 1
    d["_max_stuck_run"] = max(d.get("_max_stuck_run", 0), d["_cur_stuck_run"])
    bin_id = int(ins1 // STUCK_DEPTH_BIN_MM) if not math.isnan(ins1) else -1
    if bin_id >= 0:
        bins = d.get("_stuck_depth_bins") or {}
        bins[bin_id] = bins.get(bin_id, 0) + 1
        d["_stuck_depth_bins"] = bins
else:
    d["_cur_stuck_run"] = 0
```

✅ This detects “stuck” even when the policy is trying to insert.

---

## 4) Still in `load_worker_logs_v3()` — propagate into `EpisodeRowV3`

Before `episodes.append(EpisodeRowV3(...))`, compute:

* `stuck_fraction`
* `max_stuck_run`
* `stuck_depth_mode_mm`
* mean/std translation
* reward stats

Then pass them as new args into `EpisodeRowV3(...)`.



## 5) `correlate_episodes_to_explore_step()` — fix correlation (critical)

**Replace the entire function** with:

### Preferred: use `end_global`

If `end_global` exists for episodes, set:

* `explore_step_estimate = end_global`
* `update_step_estimate = explore_to_update(explore_step_estimate)` from losses mapping

### Fallback: timestamps if both have timestamps

### Last-resort: index interpolation



## 6) `detect_events()` — fix episode ordering, insertion collapse threshold, add stuck event

### 6a) sort episodes by explore_step first

Replace:

```python
eps_sorted = sorted(episodes, key=lambda e: (e.end_timestamp or datetime.min, e.episode))
```

With:

```python
eps_sorted = sorted(episodes, key=lambda e: (
    e.explore_step_estimate if e.explore_step_estimate is not None else 10**18,
    e.end_timestamp or datetime.min,
    e.episode
))
```

### 6b) fix insertion collapse threshold (avoid false early triggers)

Currently it uses a fixed threshold `200mm` which falsely triggers if baseline is already shallow.

Add dynamic threshold:

```python
collapse_thresh = min(INSERTION_COLLAPSE_THRESHOLD_MM, 0.4 * base_insertion) if base_insertion > 0 else INSERTION_COLLAPSE_THRESHOLD_MM
pred_ins_collapse = [ins < collapse_thresh for ins in ins_roll]
```

### 6c) insertion event must use update_step_estimate

Set:

```python
update_step=ep.update_step_estimate
```

### 6d) Add stuck-regime event

Insert a new event block after insertion collapse:

* rolling mean of `stuck_fraction`
* baseline from early episodes
* sustained trigger



## 7) Event ordering key — fix sorting with `update_step=None`

Replace `_event_key` with:

```python
def _event_key(ev: Event):
    if ev.update_step is not None:
        return (0, ev.update_step, ev.name)
    if ev.explore_step is not None:
        return (1, ev.explore_step, ev.name)
    if ev.timestamp is not None:
        return (2, ev.timestamp.timestamp(), ev.name)
    return (3, 10**18, ev.name)
```


# C) “What happens when stuck occurs?” (re: your question)

From what the logging shows, **the environment does not automatically terminate** when stuck happens. Instead:

* The policy still outputs actions (not masked in logs)
* The sim reports near-zero `delta_ins`
* The episode typically ends via normal term/trunc condition (max steps, etc.)

⚠️ Important: With current logs, we cannot know if:

* action was internally clamped/masked
* reward was modified based on collision/constraint

✅ If you want to detect *true masking* vs *physics no-motion*, you should also log:

* executed action after clipping
* collision/contact flags
* “constraint active” boolean
* whether translation got set to 0 internally

---


