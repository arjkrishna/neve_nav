

## 1) Episode ↔ training timeline correlation

### What you *did* that’s correct

* **`runner.py` now writes `episode_summary.jsonl`** containing:

  * `wall_time`, `explore_step`, `update_step`, `episode_id`, `total_reward`, `steps`, terminal flags, and optional `max_insertion` 

* It’s called right after each `explore_and_update(...)` result, so the file is continuously populated .

* One version of **`diagnose_collapse3.py`** (see note below) **loads those summaries** as the preferred source (no interpolation)  and can **merge** them into worker episodes .

### What’s still missing / why it’s not “complete”

**A) ID mismatch risk (multi-worker / multi-env):**

* The merge logic assumes `worker_episode.ep == summary.episode_id` .
* But the env-side logs use `self._episode_count` (a per-env counter) for `ep=` , which will **not** equal the runner’s global `episode_id` in typical multi-worker setups.

➡️ Result: you may get

* accurate `(episode → explore_step/update_step)` from summaries, **and**
* detailed “stuck stats” from worker logs,
  but they **won’t attach to the same episode row** reliably unless you unify IDs (global episode id in worker logs, or worker/local episode fields in the summary, or time-based matching once timestamps are correct).

**B) Granularity (batch logging):**

* In `runner.py`, `explore_step`/`update_step` is captured **once per returned batch** , so all episodes in that returned list share the same step counters. That’s usually fine for “correlate to the right training phase/update,” but it’s not a perfect per-episode step position.

✅ Conclusion for (1): **Solved for “episode → training steps” at the summary level**, but **not complete** for stitching episode-level *worker-step diagnostics* to those exact steps in multi-worker runs.

---

## 2) “Stuck” detection and reporting

### What you *did* that’s correct (signals exist)

* `monoplanestatic.py` caches the *commanded* action, *executed* action, and mask reasons (`last_cmd_action`, `last_exec_action`, `last_mask_reasons`) .
* `env.py` logs per-step:

  * `cmd_action`, `exec_action`, `mask_reason`, `inserted=[..]`, and `delta_ins=[..]` .

These are exactly the ingredients needed to detect:

* “pushing but no insertion progress” (true stuck)
* “masked out repeatedly” (safety/constraint-induced stagnation)

### Diagnose side: the logic exists… but only if it can actually read your logs

One diagnose variant defines stuck thresholds  and marks stuck steps when `cmd_action[0]` indicates push but `delta_ins` is tiny .

### The two blockers that prevent this from working end-to-end

**A) Log filename / discovery mismatch**

* `env.py` writes to **`step_by_step.log`** 
* `diagnose_collapse3.py` (v3 variant) searches for **`logs_subprocesses/worker_*.log`** 

➡️ If nothing else copies/renames those logs, diagnose will simply **never see the step logs**, so the stuck detection won’t fire.

**B) Timestamp format mismatch**

* `env.py` formatter uses `datefmt='%H:%M:%S'`, producing timestamps like `12:34:56.789 - ...` 
* diagnose expects `YYYY-MM-DD HH:MM:SS,mmm` style timestamps in its regex  and parses with a year-month-day format 

➡️ Even if you point diagnose at the file, it may fail to parse timestamps, which then breaks ordering and (if used) timestamp→step correlation.

✅ Conclusion for (2): **The detection logic + logging signals are there, but the pipeline is not “complete”** until:

* diagnose can actually find the logs, and
* timestamps match what diagnose parses.

---



## Minimal fixes to make both problems truly “complete”

If you want this to fully close both issues with minimal disruption:

1. **Make diagnose read the same step logs that env writes**

* Either: change env filename to `worker_{id}.log`
* Or: change diagnose glob to include `**/step_by_step.log` and infer worker id from directory name.

2. **Make timestamps compatible**

* Either: change env formatter to include date (so it matches diagnose)
* Or: change diagnose regex/parser to accept time-only logs.

3. **Unify episode identity across runner summaries and worker logs**

* Add `(worker_id, local_ep)` into `episode_summary.jsonl`, or
* Log a global `episode_id` in env step logs (best), or
* Do time-based matching between summary `wall_time` and worker `EPISODE_END` timestamp (only works once timestamps are correct).

---

### Bottom line

* **Problem (1):** You’ve implemented the *right* mechanism (episode summaries with exact steps), but it’s **not fully complete** for multi-worker stitching unless episode IDs unify (or you time-match).
* **Problem (2):** The “stuck” signal + detection logic is there, but it’s **not complete end-to-end** until you fix **log discovery (worker_*.log vs step_by_step.log)** and **timestamp format**.




SOLUTION 

Yep — here’s the **smallest diff** (few-line edits) that makes the **logging format**, **worker-log file discovery**, and **episode↔training ID stitching** consistent across your four files.

What this patch does:

* **env.py**

  * Writes worker step logs as **`worker_<pid>.log`** (no collisions) into `STEP_LOG_DIR`
  * Uses **full datetime** format (`YYYY-mm-dd HH:MM:SS,mmm`) so your regex parses reliably
  * Adds **`wall_time=<epoch>`** and **`pid=<pid>`** to `EPISODE_START/END` and `STEP` lines → enables robust correlation

* **runner.py**

  * Ensures `diagnostics_folder/logs_subprocesses` exists and sets **`STEP_LOG_DIR`** early (so subprocesses inherit it, assuming they spawn after Runner init)
  * Writes **per-episode wall_time** (moves `time.time()` into the loop) to avoid duplicate timestamps

* **diagnose_collapse3.py**

  * Parses **`cmd_action=[...]`** as well as `action=[...]`
  * Loads `episode_summary.jsonl` and uses **`end_wall_time`** from worker logs to assign **`explore_step` and `update_step`** (proper timeline correlation)
  * Adds minimal fields to `EpisodeRowV3` (`update_step_estimate`, `end_wall_time`)

* **monoplanestatic.py**

  * **No changes required** for these two problems.

---

## Patch (unified diffs)

### 1) `runner.py`

```diff
--- a/runner.py
+++ b/runner.py
@@
         # Diagnostics settings
         self.diagnostics_folder = diagnostics_folder
+        # Ensure worker step logs land under diagnostics so diagnose_collapse3.py can find them
+        if self.diagnostics_folder is not None:
+            logs_subprocesses = os.path.join(self.diagnostics_folder, "logs_subprocesses")
+            os.makedirs(logs_subprocesses, exist_ok=True)
+            # Inherit into subprocesses if they are spawned after Runner init
+            os.environ.setdefault("STEP_LOG_DIR", logs_subprocesses)
         self.policy_snapshot_every_steps = policy_snapshot_every_steps
         self.n_probe_episodes = n_probe_episodes
         self.n_probe_near_start_steps = n_probe_near_start_steps
@@
-            wall_time = time.time()
-            
             with open(self._episode_summary_file, "a", encoding="utf-8") as f:
                 for i, episode in enumerate(episodes):
                     try:
+                        wall_time = time.time()
                         # Extract episode data
                         total_reward = getattr(episode, 'episode_reward', 0.0)
```

### 2) `env.py`

```diff
--- a/env.py
+++ b/env.py
@@
     # Create a handler that writes to a dedicated file
     # Use the current working directory's results folder if available
     log_dir = os.environ.get("STEP_LOG_DIR", "/tmp")
-    log_file = os.path.join(log_dir, "step_by_step.log")
+    log_file = os.path.join(log_dir, f"worker_{os.getpid()}.log")
@@
-        formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(message)s', datefmt='%H:%M:%S')
+        formatter = logging.Formatter('%(asctime)s,%(msecs)03d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
         handler.setFormatter(formatter)
         logger.addHandler(handler)
@@
-        formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(message)s', datefmt='%H:%M:%S')
+        formatter = logging.Formatter('%(asctime)s,%(msecs)03d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
         handler.setFormatter(formatter)
         logger.addHandler(handler)
@@
-        self._step_logger.info(f"EPISODE_START | ep={self._episode_count} | global_steps={self._step_count}")
+        self._step_logger.info(
+            f"EPISODE_START | ep={self._episode_count} | global_steps={self._step_count} | "
+            f"wall_time={time.time():.6f} | pid={os.getpid()}"
+        )
@@
             self._step_logger.info(
                 f"EPISODE_END | ep={self._episode_count} | steps={self._episode_step_count} | "
                 f"total_reward={self._episode_total_reward:.4f} | duration={episode_duration:.2f}s | "
-                f"avg_step_time={episode_duration/max(1,self._episode_step_count):.3f}s"
+                f"avg_step_time={episode_duration/max(1,self._episode_step_count):.3f}s | "
+                f"wall_time={time.time():.6f} | pid={os.getpid()}"
             )
@@
-        log_msg = (
-            f"STEP | ep={self._episode_count} | ep_step={self._episode_step_count} | "
-            f"global={self._step_count} | cmd_action={action_str} | "
-        )
+        log_msg = (
+            f"STEP | ep={self._episode_count} | ep_step={self._episode_step_count} | "
+            f"global={self._step_count} | wall_time={time.time():.6f} | pid={os.getpid()} | "
+            f"cmd_action={action_str} | "
+        )
```

### 3) `diagnose_collapse3.py`

```diff
--- a/diagnose_collapse3.py
+++ b/diagnose_collapse3.py
@@
 @dataclass
 class EpisodeRowV3:
@@
-    # Correlated explore_step (estimated via timestamp)
+    # Correlated explore_step (estimated via timestamp or episode_summary.jsonl)
     explore_step_estimate: Optional[int] = None
+    # Correlated update_step (estimated via episode_summary.jsonl when available)
+    update_step_estimate: Optional[int] = None
+    # Epoch wall_time from EPISODE_END line (used for robust ID stitching)
+    end_wall_time: Optional[float] = None
@@
 STEP_RE_NEW = re.compile(
@@
-    r"\s*action=\[([-\d.,]+)\]\s*\|.*?"
+    r"\s*(?:action|cmd_action)=\[([-\d.,]+)\]\s*\|.*?"
@@
 )
 
+WALL_TIME_RE = re.compile(r"wall_time=([0-9.]+)")
+
@@
 def _to_float(x: Any, default: float = float("nan")) -> float:
@@
     except Exception:
         return default
 
+def _extract_wall_time(line: str) -> Optional[float]:
+    """Extract epoch wall_time=... from a log line (if present)."""
+    m = WALL_TIME_RE.search(line) if 'WALL_TIME_RE' in globals() else None
+    if not m:
+        return None
+    wt = _to_float(m.group(1))
+    return None if math.isnan(wt) else wt
+
@@
                 m_end = EP_END_RE_FULL.search(line)
                 if m_end:
                     ts = _parse_timestamp(m_end.group(1))
                     ep = int(m_end.group(2))
                     steps = int(m_end.group(3))
                     total_reward = _to_float(m_end.group(4))
+                    end_wall_time = _extract_wall_time(line)
@@
                     if key not in data:
                         data[key] = dict(
                             worker=worker, episode=ep,
                             start_timestamp=None, end_timestamp=ts,
                             total_reward=total_reward,
+                            end_wall_time=end_wall_time,
                             max_insertion_1=0.0, max_insertion_2=0.0,
                             steps=steps, terminated=False, truncated=False
                         )
                     else:
                         d = data[key]
                         d["end_timestamp"] = ts
                         d["total_reward"] = total_reward
+                        if end_wall_time is not None:
+                            d["end_wall_time"] = end_wall_time
                         d["steps"] = max(d.get("steps", 0) or 0, steps)
@@
         episodes.append(EpisodeRowV3(
@@
             total_delta_ins_2=_to_float(d.get("total_delta_ins_2", 0.0)),
             actions=d.get("actions", []),
+            end_wall_time=d.get("end_wall_time"),
         ))
@@
+@dataclass
+class EpisodeSummaryRow:
+    wall_time: float
+    explore_step: int
+    update_step: int
+    episode_id: int
+
+def load_episode_summary_jsonl(path: Path) -> List[EpisodeSummaryRow]:
+    rows: List[EpisodeSummaryRow] = []
+    with path.open("r", encoding="utf-8") as f:
+        for line in f:
+            line = line.strip()
+            if not line:
+                continue
+            try:
+                j = json.loads(line)
+                rows.append(EpisodeSummaryRow(
+                    wall_time=_to_float(j.get("wall_time")),
+                    explore_step=_to_int(j.get("explore_step")),
+                    update_step=_to_int(j.get("update_step")),
+                    episode_id=_to_int(j.get("episode_id")),
+                ))
+            except Exception:
+                continue
+    rows.sort(key=lambda r: r.wall_time)
+    return rows
+
@@
 def correlate_episodes_to_explore_step(
     episodes: List[EpisodeRowV3],
     losses: List[LossRow],
+    episode_summaries: Optional[List[EpisodeSummaryRow]] = None,
 ) -> List[EpisodeRowV3]:
@@
-    Strategy:
+    # Fast path: if episode_summary.jsonl exists and EPISODE_END includes wall_time, use it.
+    if episode_summaries:
+        summaries = [s for s in episode_summaries if not math.isnan(s.wall_time)]
+        summaries.sort(key=lambda s: s.wall_time)
+        used = set()
+        eps = sorted([e for e in episodes if e.end_wall_time is not None], key=lambda e: e.end_wall_time or 0.0)
+        for e in eps:
+            wt = e.end_wall_time
+            if wt is None:
+                continue
+            lo, hi = 0, len(summaries) - 1
+            best_i = None
+            best_dt = float("inf")
+            while lo <= hi:
+                mid = (lo + hi) // 2
+                dt = abs(summaries[mid].wall_time - wt)
+                if dt < best_dt and mid not in used:
+                    best_dt, best_i = dt, mid
+                if summaries[mid].wall_time < wt:
+                    lo = mid + 1
+                else:
+                    hi = mid - 1
+            for j in [best_i-1, best_i, best_i+1]:
+                if j is None or j < 0 or j >= len(summaries) or j in used:
+                    continue
+                dt = abs(summaries[j].wall_time - wt)
+                if dt < best_dt:
+                    best_dt, best_i = dt, j
+            if best_i is not None and best_dt <= 10.0:
+                s = summaries[best_i]
+                e.explore_step_estimate = s.explore_step
+                e.update_step_estimate = s.update_step
+                used.add(best_i)
+        return episodes
+
+    Strategy:
@@
     worker_logs = _glob_all(run_dir, ["logs_subprocesses/worker_*.log", "**/worker_*.log"])
+    episode_summary_path = _glob_best(run_dir, ["diagnostics/csv/episode_summary.jsonl", "**/episode_summary.jsonl"])
@@
     episodes = load_worker_logs_v3(worker_logs) if worker_logs else []
+    episode_summaries = load_episode_summary_jsonl(episode_summary_path) if episode_summary_path else None
@@
-    episodes = correlate_episodes_to_explore_step(episodes, losses)
+    episodes = correlate_episodes_to_explore_step(episodes, losses, episode_summaries)
```

### 4) `monoplanestatic.py`

No diff needed for the two issues you called out (timeline correlation + stuck detection plumbing).

---

## Ready-to-use patched files

If you want to just drop-in replace, here are the patched versions:

* [runner_patched.py](sandbox:/mnt/data/runner_patched.py)
* [env_patched.py](sandbox:/mnt/data/env_patched.py)
* [diagnose_collapse3_patched.py](sandbox:/mnt/data/diagnose_collapse3_patched.py)
* [monoplanestatic_patched.py](sandbox:/mnt/data/monoplanestatic_patched.py)
