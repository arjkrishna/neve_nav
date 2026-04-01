#!/usr/bin/env python3
"""
diagnose_collapse6.py

Enhanced "collapse forensics" tool for stEVE / eve_rl SAC diagnostics.

Version 6 performance optimizations:
- Perf #1: Removed unused actions accumulation (massive memory savings)
- Perf #2: _rolling_std uses Welford's online algorithm O(n) instead of O(n*w)
- Perf #3: _rolling_median uses bisect-sorted window O(n*w) instead of O(n*w*log w)
- Perf #4: String pre-check before regex cascade in log parsing (~2-3x faster)
- Perf #5: build_timestamp_interpolator called once, passed to consumers
- Perf #6: Episodes sorted once, pre-sorted view reused
- Perf #7: _glob_best caches stat() calls
- Perf #8: Deduplicated insertion rolling_median between detect_events and make_plots

Version 5 fixes (inherited):
- Fix #1: Binary search in wall_time matching replaced with bisect + outward scan
- Fix #3: _compute_confidence edge case when baseline ~ threshold
- Fix #7: explore_update_map preserves first occurrence (not last)
- Fix #8: timestamp_to_explore uses bisect instead of O(n) linear scan
- Fix #13: Rolling functions use collections.deque for O(1) popleft

Version 4 improvements (inherited):
- Improved visualization with rolling averages and reference lines
- Separate graphs for grad_policy/grad_q1, q1/q2 loss, q1/q2 mean
- New graphs: q_preference zoomed, translation speed, stuck instances
- Removed batch done rate graph
- Removed event lines from most plots (kept only for alpha)

Usage
-----
python diagnose_collapse6.py --run-dir /path/to/run_dir
python diagnose_collapse6.py --name diag_debug_test5

Optional:
  --alpha-threshold 0.01
  --sustain 50
  --plot   (enable plots if matplotlib is installed)
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import re
import statistics
from collections import deque
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------
# Constants for detection thresholds
# -----------------------------

# Saturation: Data-driven thresholds based on action scale
# Infer scale from early training percentiles, not fixed bounds
SATURATION_SCALE_FACTOR = 0.7  # Saturation = 70% of observed max range (p95)
SATURATION_MIN_FLOOR = 0.15    # Absolute minimum threshold regardless of scale
SATURATION_K_STD = 30          # Multiplier for baseline std in threshold calculation

# Std collapse: Require meaningful drop, not statistical noise
STD_COLLAPSE_FRACTION = 0.5  # Std must drop to 50% of baseline (i.e., 50% reduction)

# Freeze detection
FREEZE_MEAN_EPSILON = 0.05  # Mean must be within +/-0.05
FREEZE_STD_MAX = 0.1        # Std must be below 0.1

# Insertion collapse
INSERTION_COLLAPSE_THRESHOLD_MM = 200.0  # Rolling median below this = collapse
INSERTION_COLLAPSE_DROP_RATIO = 0.4      # Collapse if below 40% of baseline

# Clamp saturation (from losses CSV clamp_fraction field, added by Fix #6)
CLAMP_FRACTION_THRESHOLD = 0.5   # >50% of actions at clamp boundaries = saturated
CLAMP_FRACTION_CRITICAL = 0.8    # >80% = critical saturation

# Replay distribution shift (independent — no longer gated)
REPLAY_SHIFT_FLOOR = 0.3  # Magnitude floor for replay shift detection

# Stuck detection (from worker STEP logs)
# "Stuck" = policy commands positive translation but delta insertion is near-zero
STUCK_TRANSLATION_INDEX = 0              # Action dimension for translation/advance
STUCK_PUSH_THRESHOLD = 0.05              # cmd_translation > this means "trying to insert"
STUCK_DELTA_INS_EPS_MM = 0.2             # |delta_ins| < this means "no motion"
STUCK_MIN_CONSEC_STEPS = 25              # Min consecutive stuck steps to count as a run
STUCK_DEPTH_BIN_MM = 25.0                # Bin size for reporting stuck depth mode
STUCK_FRACTION_THRESHOLD = 0.3           # Fraction of episode stuck to trigger event
STUCK_SUSTAINED_EPISODES = 10            # Episodes with high stuck fraction to trigger
JAM_EPISODE_THRESHOLD = 0.3              # Fraction of episodes with max_stuck_run >= STUCK_MIN_CONSEC_STEPS

# Estimate source priority (higher = more trustworthy)
ESTIMATE_PRIORITY = {
    "summary_exact": 4,    # Direct from episode_summary.jsonl (ground truth)
    "wall_time_match": 3,  # Wall_time correlation (high confidence)
    "ts_interp": 2,        # Timestamp-based interpolation
    "linear_interp": 1,    # Linear interpolation (fallback)
    None: 0,
}

# -----------------------------
# Small utilities
# -----------------------------

def _to_int(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default

def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return float(x)
        return float(x)
    except Exception:
        return default

def _extract_wall_time(line: str) -> Optional[float]:
    """Extract epoch wall_time=... from a log line (if present)."""
    m = WALL_TIME_RE.search(line)
    if not m:
        return None
    wt = _to_float(m.group(1))
    return None if math.isnan(wt) else wt

def _extract_pid(line: str) -> Optional[int]:
    """Extract pid=... from a log line (if present)."""
    m = PID_RE.search(line)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (ValueError, TypeError):
        return None

def _safe_mean(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)

def _safe_median(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan")
    return statistics.median(xs)

def _safe_pstdev(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) < 2:
        return 0.0
    return statistics.pstdev(xs)

def _rolling_mean(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values[:]
    out = []
    acc = 0.0
    q: deque = deque()
    for v in values:
        q.append(v)
        acc += v
        if len(q) > window:
            acc -= q.popleft()
        out.append(acc / len(q))
    return out

def _rolling_median(values: List[float], window: int) -> List[float]:
    """Perf #3: bisect-sorted window — O(n*w) insert/remove vs O(n*w*log w) re-sort."""
    if window <= 1:
        return values[:]
    out: List[float] = []
    q: deque = deque()
    sorted_w: list = []
    for v in values:
        q.append(v)
        bisect.insort(sorted_w, v)
        if len(q) > window:
            old = q.popleft()
            idx = bisect.bisect_left(sorted_w, old)
            sorted_w.pop(idx)
        n = len(sorted_w)
        if n % 2 == 1:
            out.append(sorted_w[n // 2])
        else:
            out.append((sorted_w[n // 2 - 1] + sorted_w[n // 2]) / 2.0)
    return out

def _rolling_std(values: List[float], window: int) -> List[float]:
    """Perf #2: Welford's online algorithm — O(n) total vs O(n*w)."""
    if window <= 1:
        return [0.0 for _ in values]
    out: List[float] = []
    q: deque = deque()
    s = 0.0
    ss = 0.0
    for v in values:
        q.append(v)
        s += v
        ss += v * v
        if len(q) > window:
            old = q.popleft()
            s -= old
            ss -= old * old
        n = len(q)
        if n < 2:
            out.append(0.0)
        else:
            variance = max(0.0, ss / n - (s / n) ** 2)
            out.append(math.sqrt(variance))
    return out

def _set_estimate_if_better(ep, explore: int, update: int, source: str) -> bool:
    """Set estimate only if source has higher priority than current. Returns True if set."""
    current_priority = ESTIMATE_PRIORITY.get(ep.estimate_source, 0)
    new_priority = ESTIMATE_PRIORITY.get(source, 0)
    if new_priority > current_priority:
        ep.explore_step_estimate = explore
        ep.update_step_estimate = update
        ep.estimate_source = source
        return True
    return False

def _extract_field(line: str, field_name: str) -> Optional[str]:
    """Extract field value from pipe-delimited log line.
    
    Captures everything until next | delimiter, handling:
    - Spaces in values
    - Rich reason strings (hyphens, dots, etc.)
    - Any field ordering
    """
    pattern = rf"{field_name}=([^|]+)"
    m = re.search(pattern, line)
    return m.group(1).strip() if m else None

def _extract_array(line: str, field_name: str) -> Optional[List[float]]:
    """Extract array field like cmd_action=[0.1, -0.2] or inserted=[100.5,50.2].
    
    Handles:
    - Spaces in arrays
    - Scientific notation (1e-5)
    - Variable bracket styles
    """
    raw = _extract_field(line, field_name)
    if raw is None:
        return None
    # Remove brackets and whitespace
    raw = raw.strip("[]")
    # Split and parse, handling spaces
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None

def _parse_step_line(line: str) -> Optional[dict]:
    """Parse STEP line using field-based extraction (order-independent).
    
    This is more robust than positional regex:
    - Handles field ordering changes
    - Handles added fields gracefully
    - Handles spaces in arrays
    - Handles rich reason strings (hyphens, dots, etc.)
    """
    if "STEP" not in line:
        return None
    
    # Extract timestamp from start of line
    ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
    timestamp_str = ts_match.group(1) if ts_match else None
    
    # Parse timestamp
    timestamp = None
    if timestamp_str:
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
        except ValueError:
            pass
    
    # Extract all fields by name (order-independent)
    ep = _to_int(_extract_field(line, "ep"))
    ep_step = _to_int(_extract_field(line, "ep_step"))
    global_step = _to_int(_extract_field(line, "global"))
    
    # Action fields (V3 has cmd_action/exec_action, V2 has action)
    cmd_action = _extract_array(line, "cmd_action")
    exec_action = _extract_array(line, "exec_action")
    action = _extract_array(line, "action")  # Fallback for V2
    
    # Use cmd_action if available, else fall back to action
    effective_cmd_action = cmd_action if cmd_action is not None else action
    
    # Reason/status fields
    mask_reason = _extract_field(line, "mask_reason")
    term_reason = _extract_field(line, "term_reason")
    term = _extract_field(line, "term")
    trunc = _extract_field(line, "trunc")
    
    # Reward fields
    reward = _to_float(_extract_field(line, "reward"))
    cum_reward = _to_float(_extract_field(line, "cum_reward"))
    
    # Position/insertion fields
    inserted = _extract_array(line, "inserted")
    delta_ins = _extract_array(line, "delta_ins")
    
    # Wall time
    wall_time = _to_float(_extract_field(line, "wall_time"))
    
    # Return None if we couldn't parse basic required fields
    if ep < 0 or ep_step < 0:
        return None
    
    return {
        "timestamp": timestamp,
        "ep": ep,
        "ep_step": ep_step,
        "global": global_step,
        "cmd_action": effective_cmd_action,
        "exec_action": exec_action,
        "mask_reason": mask_reason,
        "reward": reward,
        "cum_reward": cum_reward,
        "term": term,
        "trunc": trunc,
        "term_reason": term_reason,
        "inserted": inserted,
        "delta_ins": delta_ins,
        "wall_time": wall_time,
    }

def _first_sustained(predicate: List[bool], sustain: int) -> Optional[int]:
    """Return first index i such that predicate[i:i+sustain] are all True."""
    if sustain <= 1:
        for i, ok in enumerate(predicate):
            if ok:
                return i
        return None
    run = 0
    for i, ok in enumerate(predicate):
        run = run + 1 if ok else 0
        if run >= sustain:
            return i - sustain + 1
    return None

def _percent_change(a: float, b: float) -> float:
    if a is None or (isinstance(a, float) and math.isnan(a)) or a == 0:
        return float("nan")
    return (b - a) / abs(a) * 100.0

def _glob_best(root: Path, patterns: List[str]) -> Optional[Path]:
    """Pick the best match (largest file) among patterns."""
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(root.glob(pat))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    # Perf #7: cache stat() — avoid two syscalls per candidate
    def _stat_key(p: Path):
        st = p.stat()
        return (st.st_size, st.st_mtime)
    candidates.sort(key=_stat_key, reverse=True)
    return candidates[0]

def _glob_all(root: Path, patterns: List[str]) -> List[Path]:
    """Glob all patterns and deduplicate by resolved path."""
    seen: set = set()
    out: List[Path] = []
    for pat in patterns:
        for p in root.glob(pat):
            # Resolve to absolute path for deduplication
            resolved = p.resolve()
            if resolved not in seen and p.is_file():
                seen.add(resolved)
                out.append(p)
    out.sort(key=lambda p: p.stat().st_mtime)
    return out

def _extract_scalar_nested(x: Any) -> float:
    """Recursively take the first element until it's a scalar; then float()."""
    while isinstance(x, list) and len(x) > 0:
        x = x[0]
    return _to_float(x)

def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse timestamp from log line: '2026-02-04 20:52:14,042'"""
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
    except Exception:
        try:
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

def _compute_confidence(
    value: float, 
    threshold: float, 
    baseline: float, 
    direction: str = "above"
) -> float:
    """Compute confidence score (0-1) based on how far past threshold.
    
    Args:
        value: The observed value
        threshold: The threshold that was crossed
        baseline: The baseline value for normalization
        direction: "above" if violation is value > threshold, 
                   "below" if violation is value < threshold
    """
    if math.isnan(value) or math.isnan(threshold) or math.isnan(baseline):
        return 0.0

    if direction == "below":
        # For drop-type events: violation = how far below threshold
        violation = max(0.0, threshold - value)
        gap = baseline - threshold
    else:  # "above"
        # For spike-type events: violation = how far above threshold
        violation = max(0.0, value - threshold)
        gap = threshold - baseline

    # Fix #3: Return 0.0 when gap is near-zero (baseline ≈ threshold) to avoid
    # inflating confidence for trivially small violations. Also return 0.0 when
    # gap is negative (baseline already past threshold — no meaningful violation).
    if gap < 1e-4:
        return 0.0

    return min(1.0, violation / gap)

# -----------------------------
# Data loaders
# -----------------------------

@dataclass
class LossRow:
    update_step: int
    explore_step: int
    timestamp: Optional[datetime]  # NEW: parsed from file mtime or embedded
    q1_loss: float
    q2_loss: float
    policy_loss: float
    alpha: float
    alpha_loss: float
    log_pi_mean: float
    log_pi_std: float
    entropy_proxy: float
    q1_mean: float
    q2_mean: float
    target_q_mean: float
    min_q_mean: float
    grad_norm_q1: float
    grad_norm_q2: float
    grad_norm_policy: float
    lr_policy: float
    clamp_fraction: float  # Fraction of actions at clamp boundaries (from Fix #6)

def _parse_wall_time(val: Any) -> Optional[datetime]:
    """Parse wall_time (unix timestamp) into datetime."""
    if val is None:
        return None
    try:
        ts = float(val)
        return datetime.fromtimestamp(ts)
    except (ValueError, TypeError, OSError):
        return None

def load_losses_csv(path: Path) -> List[LossRow]:
    rows: List[LossRow] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Parse wall_time if present (new format)
            timestamp = _parse_wall_time(r.get("wall_time"))
            rows.append(LossRow(
                update_step=_to_int(r.get("update_step")),
                explore_step=_to_int(r.get("explore_step")),
                timestamp=timestamp,
                q1_loss=_to_float(r.get("q1_loss")),
                q2_loss=_to_float(r.get("q2_loss")),
                policy_loss=_to_float(r.get("policy_loss")),
                alpha=_to_float(r.get("alpha")),
                alpha_loss=_to_float(r.get("alpha_loss")),
                log_pi_mean=_to_float(r.get("log_pi_mean")),
                log_pi_std=_to_float(r.get("log_pi_std")),
                entropy_proxy=_to_float(r.get("entropy_proxy")),
                q1_mean=_to_float(r.get("q1_mean")),
                q2_mean=_to_float(r.get("q2_mean")),
                target_q_mean=_to_float(r.get("target_q_mean")),
                min_q_mean=_to_float(r.get("min_q_mean")),
                grad_norm_q1=_to_float(r.get("grad_norm_q1")),
                grad_norm_q2=_to_float(r.get("grad_norm_q2")),
                grad_norm_policy=_to_float(r.get("grad_norm_policy")),
                lr_policy=_to_float(r.get("lr_policy")),
                clamp_fraction=_to_float(r.get("clamp_fraction", 0.0), default=0.0),
            ))
    rows.sort(key=lambda x: x.update_step)
    return rows

@dataclass
class ProbeRow:
    update_step: int
    explore_step: Optional[int]
    timestamp: Optional[datetime]  # NEW: wall_time for correlation
    q1_avg: float
    q2_avg: float
    trans_mean: float
    trans_log_std_mean: float
    trans_std_mean: float

def load_probe_jsonl(path: Path) -> List[ProbeRow]:
    out: List[ProbeRow] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            u = _to_int(entry.get("update_step"))
            e = entry.get("explore_step")
            explore_step = _to_int(e) if e is not None else None
            timestamp = _parse_wall_time(entry.get("wall_time"))

            q1_list = entry.get("q1", [])
            q2_list = entry.get("q2", [])
            pm_list = entry.get("policy_mean", [])
            pls_list = entry.get("policy_log_std", [])

            q1_vals = [_extract_scalar_nested(q) for q in q1_list]
            q2_vals = [_extract_scalar_nested(q) for q in q2_list]

            trans_means = []
            for a in pm_list:
                try:
                    trans_means.append(_to_float(a[0][0]))
                except Exception:
                    trans_means.append(_extract_scalar_nested(a))

            trans_log_stds = []
            for s in pls_list:
                try:
                    trans_log_stds.append(_to_float(s[0][0]))
                except Exception:
                    trans_log_stds.append(_extract_scalar_nested(s))

            trans_log_std_mean = _safe_mean(trans_log_stds)
            trans_std_mean = math.exp(trans_log_std_mean) if not math.isnan(trans_log_std_mean) else float("nan")

            out.append(ProbeRow(
                update_step=u,
                explore_step=explore_step,
                timestamp=timestamp,
                q1_avg=_safe_mean(q1_vals),
                q2_avg=_safe_mean(q2_vals),
                trans_mean=_safe_mean(trans_means),
                trans_log_std_mean=trans_log_std_mean,
                trans_std_mean=trans_std_mean
            ))
    out.sort(key=lambda x: x.update_step)
    return out

@dataclass
class BatchRow:
    update_step: int
    explore_step: int
    timestamp: Optional[datetime]  # NEW: wall_time for correlation
    n_samples: int
    taken_trans_mean: float
    actor_trans_mean: float
    reward_mean: float
    done_rate: float
    min_q_taken_mean: float
    min_q_actor_mean: float
    min_q_actor_minus_taken: float

def load_batch_jsonl(path: Path) -> List[BatchRow]:
    out: List[BatchRow] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            u = _to_int(entry.get("update_step"))
            e = _to_int(entry.get("explore_step"))
            timestamp = _parse_wall_time(entry.get("wall_time"))
            samples = entry.get("samples", [])
            if not isinstance(samples, list) or len(samples) == 0:
                out.append(BatchRow(
                    update_step=u, explore_step=e, timestamp=timestamp, n_samples=0,
                    taken_trans_mean=float("nan"), actor_trans_mean=float("nan"),
                    reward_mean=float("nan"), done_rate=float("nan"),
                    min_q_taken_mean=float("nan"), min_q_actor_mean=float("nan"),
                    min_q_actor_minus_taken=float("nan")
                ))
                continue

            taken_trans = []
            actor_trans = []
            rewards = []
            dones = []
            min_q_taken = []
            min_q_actor = []
            diffs = []

            for s in samples:
                at = s.get("action_taken")
                ad = s.get("actor_det_action")
                if isinstance(at, list) and len(at) > 0:
                    taken_trans.append(_to_float(at[0]))
                if isinstance(ad, list) and len(ad) > 0:
                    actor_trans.append(_to_float(ad[0]))
                rewards.append(_to_float(s.get("reward")))
                dones.append(_to_float(s.get("done")))
                mqt = _to_float(s.get("min_q_taken"))
                mqa = _to_float(s.get("min_q_actor"))
                min_q_taken.append(mqt)
                min_q_actor.append(mqa)
                if not math.isnan(mqt) and not math.isnan(mqa):
                    diffs.append(mqa - mqt)

            out.append(BatchRow(
                update_step=u,
                explore_step=e,
                timestamp=timestamp,
                n_samples=_to_int(entry.get("n_samples"), default=len(samples)),
                taken_trans_mean=_safe_mean(taken_trans),
                actor_trans_mean=_safe_mean(actor_trans),
                reward_mean=_safe_mean(rewards),
                done_rate=_safe_mean(dones),
                min_q_taken_mean=_safe_mean(min_q_taken),
                min_q_actor_mean=_safe_mean(min_q_actor),
                min_q_actor_minus_taken=_safe_mean(diffs),
            ))
    out.sort(key=lambda x: x.update_step)
    return out

# -----------------------------
# Enhanced Episode data with timestamps
# -----------------------------

@dataclass
class EpisodeRowV3:
    """Enhanced episode data with timestamps for correlation."""
    worker: int
    episode: int
    start_timestamp: Optional[datetime]
    end_timestamp: Optional[datetime]
    total_reward: float
    max_insertion_1: float
    max_insertion_2: float
    steps: int
    terminated: bool
    truncated: bool
    # Correlated steps (estimated via timestamp or episode_summary.jsonl)
    explore_step_estimate: Optional[int] = None
    update_step_estimate: Optional[int] = None  # for proper event ordering
    estimate_source: Optional[str] = None  # "summary_exact", "wall_time_match", "ts_interp", "linear_interp"
    # Epoch wall_time from EPISODE_END line (used for robust ID stitching)
    end_wall_time: Optional[float] = None
    # Enhanced fields from updated env.py (may be None for old logs)
    term_reason: Optional[str] = None  # target_reached, max_steps, truncated, none
    total_delta_ins_1: float = 0.0     # Sum of delta insertions for device 1
    total_delta_ins_2: float = 0.0     # Sum of delta insertions for device 2
    # Perf #1: Removed actions accumulation — was never read, caused massive memory bloat

    # Stuck detection stats (computed from worker step logs)
    # "Stuck" = policy commands insert but delta_insertion is near-zero
    stuck_steps: int = 0                           # Total steps where stuck (delta_ins proxy)
    stuck_fraction: float = 0.0                    # stuck_steps / steps
    max_stuck_run: int = 0                         # Longest consecutive stuck run
    stuck_depth_mode_mm: Optional[float] = None    # Most common insertion depth when stuck
    mean_cmd_translation: Optional[float] = None   # Mean commanded translation action
    std_cmd_translation: Optional[float] = None    # Std of commanded translation action
    mean_step_reward: Optional[float] = None       # Mean per-step reward
    min_step_reward: Optional[float] = None        # Minimum per-step reward
    
    # NEW: Masking stats (from V3 logs with exec_action and mask_reason)
    # "Masked" = policy commands action but execution differs (action was masked)
    masked_steps: int = 0                          # Total steps where action was masked
    masked_fraction: float = 0.0                   # masked_steps / steps
    dominant_mask_reason: Optional[str] = None     # Most common mask reason
    mean_exec_translation: Optional[float] = None  # Mean executed translation action

# Regex to parse STEP lines with full timestamp
# OLD format: STEP | ep=X | ep_step=Y | global=Z | reward=R | cum_reward=CR | ... | term=T | trunc=TR | inserted=[A,B]
# V2 format: STEP | ep=X | ep_step=Y | global=Z | action=[a1,a2] | reward=R | cum_reward=CR | ... | term=T | trunc=TR | term_reason=REASON | inserted=[A,B] | delta_ins=[dA,dB]
# V3 format (NEW): STEP | ep=X | ep_step=Y | global=Z | cmd_action=[a1,a2] | exec_action=[b1,b2] | mask_reason=REASON | reward=R | ...

# V3 format regex (with cmd_action, exec_action, mask_reason)
# Made more permissive: allows spaces in arrays, hyphens in reason strings
# Uses .*? to skip wall_time, pid, and other optional fields between global= and cmd_action=
STEP_RE_V3 = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"STEP\s*\|\s*ep=(\d+)\s*\|\s*ep_step=(\d+)\s*\|\s*global=(\d+)\s*\|.*?"  # .*? to skip optional fields
    r"cmd_action=\[([-\d.,\seE+]+)\]\s*\|"         # Allow spaces, scientific notation
    r"\s*exec_action=\[([-\d.,\seE+]+)\]\s*\|"        # Allow spaces, scientific notation
    r"\s*mask_reason=([\w-]+)\s*\|.*?"                # Allow hyphens in reason
    r"reward=([-\d.eE+]+).*?cum_reward=([-\d.eE+]+).*?"
    r"term=(\w+)\s*\|\s*trunc=(\w+)\s*\|\s*term_reason=([\w-]+).*?"
    r"inserted=\[([\d.,\s]+),([\d.,\s]+)\].*?"        # Allow spaces
    r"delta_ins=\[([-\d.,\seE+]+),([-\d.,\seE+]+)\]"  # Allow spaces, scientific notation
)

# V2 format regex (with action/cmd_action, term_reason, delta_ins, no exec_action)
# Made more permissive: allows spaces in arrays, hyphens in reason strings
STEP_RE_NEW = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"STEP\s*\|\s*ep=(\d+)\s*\|\s*ep_step=(\d+)\s*\|\s*global=(\d+)\s*\|"
    r"\s*(?:cmd_)?action=\[([-\d.,\seE+]+)\]\s*\|.*?"   # Allow spaces, scientific notation
    r"reward=([-\d.eE+]+).*?cum_reward=([-\d.eE+]+).*?"
    r"term=(\w+)\s*\|\s*trunc=(\w+)\s*\|\s*term_reason=([\w-]+).*?"
    r"inserted=\[([\d.,\s]+),([\d.,\s]+)\].*?"          # Allow spaces
    r"delta_ins=\[([-\d.,\seE+]+),([-\d.,\seE+]+)\]"    # Allow spaces, scientific notation
)

# Old format regex (fallback for existing logs)
STEP_RE_FULL = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"STEP\s*\|\s*ep=(\d+)\s*\|\s*ep_step=(\d+)\s*\|\s*global=(\d+)\s*\|.*?"
    r"reward=([-\d.eE+]+).*?cum_reward=([-\d.eE+]+).*?"
    r"term=(\w+)\s*\|\s*trunc=(\w+).*?inserted=\[([\d.,\s]+),([\d.,\s]+)\]"
)

# Note: For more robust field-based parsing (handles field reordering, added fields),
# use the _parse_step_line() helper function instead of these regex patterns.

EP_END_RE_FULL = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"EPISODE_END\s*\|\s*ep=(\d+)\s*\|\s*steps=(\d+)\s*\|\s*total_reward=([-\d.]+)"
)

EP_START_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"EPISODE_START\s*\|\s*ep=(\d+)"
)

# Regex only needed for the timestamp prefix (not pipe-delimited)
_TS_PREFIX_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")

# Keep for backward compat with code that calls _extract_wall_time/_extract_pid directly
WALL_TIME_RE = re.compile(r"wall_time=([0-9.]+)")
PID_RE = re.compile(r"pid=(\d+)")


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
    # e.g. '2026-01-01 00:00:00,000 - STEP' or '2026-01-01 00:00:00,000 - INFO - STEP'
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
        if "=" in part:
            key, val = part.split("=", 1)
            fields[key.strip()] = val.strip()

    return ts_str, fields


def _parse_array_str(s: str) -> List[float]:
    """Parse '[0.5,-0.1,3.0]' or '100.5,50.2' into list of floats."""
    s = s.strip("[] ")
    if not s:
        return []
    return [_to_float(x.strip()) for x in s.split(",") if x.strip()]


def load_worker_logs_v3(worker_log_paths: List[Path]) -> List[EpisodeRowV3]:
    """Load worker logs with timestamps for correlation.
    
    Deduplicates episodes using (pid, ep) when PID is available in log lines.
    This handles the case where the same episode data exists in multiple log files
    (e.g., logs_subprocesses/worker_N.log and diagnostics/logs_subprocesses/worker_PID.log).
    """
    # Aggregate per (pid, ep) - using PID from log content for deduplication
    # This ensures the same episode from different log files is not counted twice
    data: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for p in worker_log_paths:
        # Extract worker number from filename as fallback
        try:
            filename_worker = int(p.stem.split("_")[1])
        except Exception:
            filename_worker = -1

        with p.open("r", errors="replace") as f:
            for line in f:
                # Fast string pre-check: skip lines without any known marker
                if "STEP" not in line and "EPISODE" not in line:
                    continue

                ts_str, f_ = _parse_pipe_fields(line)
                ts = _parse_timestamp(ts_str) if ts_str else None

                # Extract PID for deduplication
                pid_str = f_.get("pid")
                line_pid = _to_int(pid_str) if pid_str is not None else None
                worker_id = line_pid if line_pid is not None and line_pid >= 0 else filename_worker

                # --- EPISODE_START ---
                if "EPISODE_START" in f_:
                    ep = _to_int(f_.get("ep"))
                    if ep < 0:
                        continue
                    key = (worker_id, ep)
                    if key not in data:
                        data[key] = dict(
                            worker=worker_id, episode=ep,
                            start_timestamp=ts, end_timestamp=None,
                            total_reward=0.0,
                            max_insertion_1=0.0, max_insertion_2=0.0,
                            steps=0, terminated=False, truncated=False
                        )
                    else:
                        data[key]["start_timestamp"] = ts
                    continue

                # --- STEP (all formats: V3, V2, OLD) ---
                if "STEP" in f_:
                    ep = _to_int(f_.get("ep"))
                    ep_step = _to_int(f_.get("ep_step"))
                    if ep < 0 or ep_step < 0:
                        continue

                    # Actions: V3 has cmd_action+exec_action, V2 has action, OLD has neither
                    cmd_action_raw = f_.get("cmd_action")
                    exec_action_raw = f_.get("exec_action")
                    action_raw = f_.get("action")  # V2 fallback

                    cmd_action = _parse_array_str(cmd_action_raw) if cmd_action_raw else None
                    exec_action = _parse_array_str(exec_action_raw) if exec_action_raw else None
                    if cmd_action is None and action_raw:
                        cmd_action = _parse_array_str(action_raw)

                    mask_reason = f_.get("mask_reason")
                    term_reason = f_.get("term_reason")
                    step_reward = _to_float(f_.get("reward"))
                    cum_reward = _to_float(f_.get("cum_reward"))
                    term = f_.get("term", "").lower() == "true"
                    trunc = f_.get("trunc", "").lower() == "true"

                    # Insertion fields
                    inserted_raw = f_.get("inserted")
                    delta_ins_raw = f_.get("delta_ins")
                    inserted = _parse_array_str(inserted_raw) if inserted_raw else []
                    delta_ins = _parse_array_str(delta_ins_raw) if delta_ins_raw else []
                    ins1 = inserted[0] if len(inserted) > 0 else 0.0
                    ins2 = inserted[1] if len(inserted) > 1 else 0.0
                    delta1 = delta_ins[0] if len(delta_ins) > 0 else 0.0
                    delta2 = delta_ins[1] if len(delta_ins) > 1 else 0.0

                    key = (worker_id, ep)

                    # Stuck detection (requires cmd_action)
                    cmd_trans = 0.0
                    exec_trans = 0.0
                    is_stuck_step = False
                    is_masked = False
                    if cmd_action and len(cmd_action) > STUCK_TRANSLATION_INDEX:
                        cmd_trans = cmd_action[STUCK_TRANSLATION_INDEX]
                        is_push = cmd_trans > STUCK_PUSH_THRESHOLD
                        is_no_motion = abs(delta1) < STUCK_DELTA_INS_EPS_MM if not math.isnan(delta1) else False
                        is_stuck_step = is_push and is_no_motion
                    if exec_action and len(exec_action) > STUCK_TRANSLATION_INDEX:
                        exec_trans = exec_action[STUCK_TRANSLATION_INDEX]
                    if mask_reason is not None:
                        is_masked = (mask_reason != "none") or (abs(cmd_trans - exec_trans) > 0.001)

                    if key not in data:
                        init = dict(
                            worker=worker_id, episode=ep,
                            start_timestamp=None, end_timestamp=ts,
                            total_reward=cum_reward,
                            max_insertion_1=ins1, max_insertion_2=ins2,
                            steps=ep_step, terminated=term, truncated=trunc,
                            term_reason=term_reason,
                            total_delta_ins_1=delta1, total_delta_ins_2=delta2,
                        )
                        if cmd_action is not None:
                            init.update(
                                _stuck_steps=1 if is_stuck_step else 0,
                                _cur_stuck_run=1 if is_stuck_step else 0,
                                _max_stuck_run=1 if is_stuck_step else 0,
                                _stuck_depth_bins={int(ins1 // STUCK_DEPTH_BIN_MM): 1} if is_stuck_step and not math.isnan(ins1) else {},
                                _cmd_trans_sum=cmd_trans,
                                _cmd_trans_sumsq=cmd_trans * cmd_trans,
                                _cmd_trans_n=1,
                                _step_reward_sum=step_reward if not math.isnan(step_reward) else 0.0,
                                _step_reward_min=step_reward if not math.isnan(step_reward) else float('inf'),
                                _step_reward_n=0 if math.isnan(step_reward) else 1,
                            )
                        if exec_action is not None:
                            init.update(
                                _masked_steps=1 if is_masked else 0,
                                _mask_reason_counts={mask_reason: 1} if mask_reason and mask_reason != "none" else {},
                                _exec_trans_sum=exec_trans,
                                _exec_trans_n=1,
                            )
                        data[key] = init
                    else:
                        d = data[key]
                        d["end_timestamp"] = ts
                        d["total_reward"] = cum_reward
                        d["max_insertion_1"] = max(d.get("max_insertion_1", 0.0) or 0.0, ins1)
                        d["max_insertion_2"] = max(d.get("max_insertion_2", 0.0) or 0.0, ins2)
                        d["steps"] = max(d.get("steps", 0) or 0, ep_step)
                        d["total_delta_ins_1"] = d.get("total_delta_ins_1", 0.0) + delta1
                        d["total_delta_ins_2"] = d.get("total_delta_ins_2", 0.0) + delta2
                        if term:
                            d["terminated"] = True
                            d["term_reason"] = term_reason
                        if trunc:
                            d["truncated"] = True
                            d["term_reason"] = term_reason
                        if is_stuck_step:
                            d["_stuck_steps"] = d.get("_stuck_steps", 0) + 1
                            d["_cur_stuck_run"] = d.get("_cur_stuck_run", 0) + 1
                            d["_max_stuck_run"] = max(d.get("_max_stuck_run", 0), d["_cur_stuck_run"])
                            if not math.isnan(ins1):
                                bin_id = int(ins1 // STUCK_DEPTH_BIN_MM)
                                bins = d.get("_stuck_depth_bins", {})
                                bins[bin_id] = bins.get(bin_id, 0) + 1
                                d["_stuck_depth_bins"] = bins
                        elif cmd_action is not None:
                            d["_cur_stuck_run"] = 0
                        if cmd_action is not None:
                            d["_cmd_trans_sum"] = d.get("_cmd_trans_sum", 0.0) + cmd_trans
                            d["_cmd_trans_sumsq"] = d.get("_cmd_trans_sumsq", 0.0) + cmd_trans * cmd_trans
                            d["_cmd_trans_n"] = d.get("_cmd_trans_n", 0) + 1
                            if not math.isnan(step_reward):
                                d["_step_reward_sum"] = d.get("_step_reward_sum", 0.0) + step_reward
                                d["_step_reward_min"] = min(d.get("_step_reward_min", float('inf')), step_reward)
                                d["_step_reward_n"] = d.get("_step_reward_n", 0) + 1
                        if is_masked:
                            d["_masked_steps"] = d.get("_masked_steps", 0) + 1
                            reason_counts = d.get("_mask_reason_counts", {})
                            if mask_reason and mask_reason != "none":
                                reason_counts[mask_reason] = reason_counts.get(mask_reason, 0) + 1
                            d["_mask_reason_counts"] = reason_counts
                        if exec_action is not None:
                            d["_exec_trans_sum"] = d.get("_exec_trans_sum", 0.0) + exec_trans
                            d["_exec_trans_n"] = d.get("_exec_trans_n", 0) + 1
                    continue

                # --- EPISODE_END ---
                if "EPISODE_END" in f_:
                    ep = _to_int(f_.get("ep"))
                    if ep < 0:
                        continue
                    steps = _to_int(f_.get("steps"))
                    total_reward = _to_float(f_.get("total_reward"))
                    end_wall_time = _to_float(f_.get("wall_time"))
                    if math.isnan(end_wall_time):
                        end_wall_time = None

                    key = (worker_id, ep)
                    if key not in data:
                        data[key] = dict(
                            worker=worker_id, episode=ep,
                            start_timestamp=None, end_timestamp=ts,
                            total_reward=total_reward,
                            end_wall_time=end_wall_time,
                            max_insertion_1=0.0, max_insertion_2=0.0,
                            steps=steps, terminated=False, truncated=False
                        )
                    else:
                        d = data[key]
                        d["end_timestamp"] = ts
                        d["total_reward"] = total_reward
                        if end_wall_time is not None:
                            d["end_wall_time"] = end_wall_time
                        d["steps"] = max(d.get("steps", 0) or 0, steps)

    episodes: List[EpisodeRowV3] = []
    for d in data.values():
        # Compute final stuck stats from accumulators
        steps = _to_int(d.get("steps"))
        stuck_steps = d.get("_stuck_steps", 0)
        stuck_fraction = stuck_steps / steps if steps > 0 else 0.0
        max_stuck_run = d.get("_max_stuck_run", 0)
        
        # Find mode of stuck depth bins
        stuck_depth_bins = d.get("_stuck_depth_bins", {})
        stuck_depth_mode_mm = None
        if stuck_depth_bins:
            mode_bin = max(stuck_depth_bins.keys(), key=lambda k: stuck_depth_bins[k])
            stuck_depth_mode_mm = (mode_bin + 0.5) * STUCK_DEPTH_BIN_MM  # Center of bin
        
        # Compute command translation stats
        cmd_n = d.get("_cmd_trans_n", 0)
        mean_cmd_trans = d.get("_cmd_trans_sum", 0.0) / cmd_n if cmd_n > 0 else None
        std_cmd_trans = None
        if cmd_n > 1:
            variance = (d.get("_cmd_trans_sumsq", 0.0) / cmd_n) - (mean_cmd_trans ** 2) if mean_cmd_trans is not None else 0
            std_cmd_trans = math.sqrt(max(0.0, variance))
        
        # Compute step reward stats
        rew_n = d.get("_step_reward_n", 0)
        mean_step_rew = d.get("_step_reward_sum", 0.0) / rew_n if rew_n > 0 else None
        min_step_rew = d.get("_step_reward_min") if d.get("_step_reward_min", float('inf')) != float('inf') else None
        
        # Compute masking stats (from V3 logs)
        masked_steps = d.get("_masked_steps", 0)
        masked_fraction = masked_steps / steps if steps > 0 else 0.0
        
        # Find dominant mask reason
        mask_reason_counts = d.get("_mask_reason_counts", {})
        dominant_mask_reason = None
        if mask_reason_counts:
            dominant_mask_reason = max(mask_reason_counts.keys(), key=lambda k: mask_reason_counts[k])
        
        # Compute executed translation mean
        exec_n = d.get("_exec_trans_n", 0)
        mean_exec_trans = d.get("_exec_trans_sum", 0.0) / exec_n if exec_n > 0 else None
        
        episodes.append(EpisodeRowV3(
            worker=d["worker"],
            episode=d["episode"],
            start_timestamp=d.get("start_timestamp"),
            end_timestamp=d.get("end_timestamp"),
            total_reward=_to_float(d.get("total_reward")),
            max_insertion_1=_to_float(d.get("max_insertion_1")),
            max_insertion_2=_to_float(d.get("max_insertion_2")),
            steps=steps,
            terminated=d.get("terminated", False),
            truncated=d.get("truncated", False),
            # Enhanced fields (may be None/empty for old logs)
            term_reason=d.get("term_reason"),
            total_delta_ins_1=_to_float(d.get("total_delta_ins_1", 0.0)),
            total_delta_ins_2=_to_float(d.get("total_delta_ins_2", 0.0)),
            # Stuck detection stats (delta_ins proxy)
            stuck_steps=stuck_steps,
            stuck_fraction=stuck_fraction,
            max_stuck_run=max_stuck_run,
            stuck_depth_mode_mm=stuck_depth_mode_mm,
            mean_cmd_translation=mean_cmd_trans,
            std_cmd_translation=std_cmd_trans,
            mean_step_reward=mean_step_rew,
            min_step_reward=min_step_rew,
            # Masking stats (from V3 logs)
            masked_steps=masked_steps,
            masked_fraction=masked_fraction,
            dominant_mask_reason=dominant_mask_reason,
            mean_exec_translation=mean_exec_trans,
            # Wall time from EPISODE_END (for robust correlation)
            end_wall_time=d.get("end_wall_time"),
        ))
    
    # Sort by end_timestamp if available, else by (episode, worker)
    episodes.sort(key=lambda e: (e.end_timestamp or datetime.min, e.episode, e.worker))
    return episodes

def build_timestamp_interpolator(losses: List[LossRow]):
    """
    Build interpolator functions: timestamp -> explore_step and explore_step -> update_step.
    
    Returns:
        (timestamp_to_explore, explore_to_update, has_timestamps)
        
    If timestamps are not available in losses, returns None functions and has_timestamps=False.
    """
    # Check if we have timestamps in losses
    losses_with_ts = [l for l in losses if l.timestamp is not None]
    
    if len(losses_with_ts) < 2:
        # Not enough timestamps, can't build interpolator
        return None, None, False
    
    # Build (timestamp, explore_step) pairs
    ts_explore_pairs = [(l.timestamp.timestamp(), l.explore_step) for l in losses_with_ts]
    ts_explore_pairs.sort(key=lambda x: x[0])
    
    # Fix #7: Preserve FIRST occurrence per explore_step (not last).
    # Many consecutive loss rows share the same explore_step (it doesn't advance
    # during update phase). Using the first occurrence avoids biasing episode
    # mapping to end-of-batch.
    explore_update_map: Dict[int, int] = {}
    for l in losses:
        if l.explore_step not in explore_update_map:
            explore_update_map[l.explore_step] = l.update_step

    # Fix #8: Pre-extract sorted timestamp array for bisect lookup
    _ts_arr = [p[0] for p in ts_explore_pairs]
    _es_arr = [p[1] for p in ts_explore_pairs]

    def timestamp_to_explore(ts: datetime) -> int:
        """Interpolate timestamp to explore_step using bisect (O(log n))."""
        unix_ts = ts.timestamp()

        # Clamp to range
        if unix_ts <= _ts_arr[0]:
            return _es_arr[0]
        if unix_ts >= _ts_arr[-1]:
            return _es_arr[-1]

        # bisect to find interpolation bracket
        idx = bisect.bisect_right(_ts_arr, unix_ts)
        # idx is the first element > unix_ts, so bracket is [idx-1, idx]
        i = max(0, idx - 1)
        t1, e1 = _ts_arr[i], _es_arr[i]
        t2, e2 = _ts_arr[min(i + 1, len(_ts_arr) - 1)], _es_arr[min(i + 1, len(_es_arr) - 1)]
        if t2 == t1:
            return e1
        frac = (unix_ts - t1) / (t2 - t1)
        return int(e1 + frac * (e2 - e1))

    # Pre-sort explore_steps for bisect lookup in explore_to_update
    _explore_keys = sorted(explore_update_map.keys())

    def explore_to_update(explore_step: int) -> Optional[int]:
        """Find nearest update_step for given explore_step."""
        if explore_step in explore_update_map:
            return explore_update_map[explore_step]

        if not _explore_keys:
            return None

        idx = bisect.bisect_left(_explore_keys, explore_step)
        if idx == 0:
            return explore_update_map[_explore_keys[0]]
        if idx >= len(_explore_keys):
            return explore_update_map[_explore_keys[-1]]

        # Choose closer one
        left = _explore_keys[idx - 1]
        right = _explore_keys[idx]
        if abs(explore_step - left) <= abs(explore_step - right):
            return explore_update_map[left]
        return explore_update_map[right]
    
    return timestamp_to_explore, explore_to_update, True


def correlate_episodes_to_explore_step(
    episodes: List[EpisodeRowV3],
    losses: List[LossRow],
    episode_summaries: Optional[List["EpisodeSummaryRow"]] = None,
    interpolator: Optional[Tuple] = None,  # Perf #5: pre-built (ts_to_explore, explore_to_update, has_ts)
) -> List[EpisodeRowV3]:
    """
    Estimate explore_step and update_step for each episode using timestamp correlation.

    Strategy:
    1. If episode_summary.jsonl exists and episodes have end_wall_time, use wall_time matching
    2. Otherwise, if losses have wall_time timestamps, build interpolator: timestamp -> explore_step
    3. Map each episode's end_timestamp to explore_step_estimate
    4. Map explore_step_estimate to update_step_estimate via losses
    5. Fall back to linear interpolation if timestamps unavailable
    """
    if not episodes or not losses:
        return episodes
    
    # Fast path: if episode_summary.jsonl exists and episodes have end_wall_time, use it
    if episode_summaries:
        summaries = [s for s in episode_summaries if s.wall_time is not None]
        summaries.sort(key=lambda s: s.wall_time.timestamp() if s.wall_time else 0)
        
        # Count episodes with end_wall_time
        eps_with_wt = [e for e in episodes if e.end_wall_time is not None]
        
        if summaries and eps_with_wt:
            used = set()
            # Sort episodes by end_wall_time for efficient matching
            eps_sorted_by_wt = sorted(eps_with_wt, key=lambda e: e.end_wall_time or 0)
            
            # Fix #1: Pre-extract wall_time array for bisect, then scan outward
            # for nearest unused entry. The old binary search was corrupted by
            # checking `mid not in used` inside the loop, which could skip the
            # true closest entry and converge to a wrong branch.
            summary_wts = [s.wall_time.timestamp() if s.wall_time else 0.0 for s in summaries]

            for e in eps_sorted_by_wt:
                wt = e.end_wall_time
                if wt is None:
                    continue

                # Find insertion point via bisect (O(log n), ignores 'used')
                idx = bisect.bisect_left(summary_wts, wt)

                # Scan outward from insertion point to find nearest UNUSED entry
                best_i = None
                best_dt = float("inf")
                lo = max(0, idx - 1)
                hi = min(idx, len(summaries) - 1)
                # Expand outward until we can't improve
                while lo >= 0 or hi < len(summaries):
                    improved = False
                    if lo >= 0:
                        dt = abs(summary_wts[lo] - wt)
                        if dt < best_dt:
                            if lo not in used:
                                best_dt, best_i = dt, lo
                            lo -= 1
                            improved = True
                        else:
                            lo = -1  # stop expanding left
                    if hi < len(summaries):
                        dt = abs(summary_wts[hi] - wt)
                        if dt < best_dt:
                            if hi not in used:
                                best_dt, best_i = dt, hi
                            hi += 1
                            improved = True
                        else:
                            hi = len(summaries)  # stop expanding right
                    if not improved:
                        break

                # Accept if within 10 second tolerance
                if best_i is not None and best_dt <= 10.0:
                    s = summaries[best_i]
                    # Use priority-based setter (wall_time_match has lower priority than summary_exact)
                    if _set_estimate_if_better(e, s.explore_step, s.update_step, "wall_time_match"):
                        used.add(best_i)
            
            print(f"  [INFO] Matched {len(used)}/{len(eps_with_wt)} episodes using wall_time correlation")
            return episodes
    
    # Perf #5: Use pre-built interpolator if available
    if interpolator is not None:
        ts_to_explore, explore_to_update, has_timestamps = interpolator
    else:
        ts_to_explore, explore_to_update, has_timestamps = build_timestamp_interpolator(losses)
    
    min_explore = losses[0].explore_step
    max_explore = losses[-1].explore_step
    total_eps = len(episodes)
    
    if has_timestamps:
        # Use timestamp-based correlation
        for ep in episodes:
            if ep.end_timestamp is not None:
                explore_est = ts_to_explore(ep.end_timestamp)
                update_est = explore_to_update(explore_est)
                # Use priority-based setter (ts_interp has lower priority than summary_exact/wall_time_match)
                _set_estimate_if_better(ep, explore_est, update_est, "ts_interp")
            # If no timestamp and no existing estimate, leave as None
    else:
        # Fix #9: Weight linear interpolation by cumulative episode step counts
        # instead of assuming uniform episode distribution. Episodes with more
        # steps consumed more explore_steps, so they should span a larger range.
        print("  [WARNING] Losses do not have wall_time timestamps. Using step-weighted interpolation for episodes.")
        ep_steps = [max(ep.steps, 1) for ep in episodes]
        total_steps = sum(ep_steps)
        if total_steps > 0:
            cum_steps = 0
            for ep, steps in zip(episodes, ep_steps):
                # Use midpoint of this episode's step range for more accurate placement
                cum_steps += steps
                frac = (cum_steps - steps / 2) / total_steps
                explore_est = int(min_explore + frac * (max_explore - min_explore))
                _set_estimate_if_better(ep, explore_est, None, "linear_interp")
        else:
            # Degenerate case: no step data, fall back to uniform
            for i, ep in enumerate(episodes):
                frac = i / max(1, total_eps - 1)
                explore_est = int(min_explore + frac * (max_explore - min_explore))
                _set_estimate_if_better(ep, explore_est, None, "linear_interp")
    
    return episodes


# -----------------------------
# Episode Summary loader (from runner's episode_summary.jsonl)
# -----------------------------

@dataclass
class EpisodeSummaryRow:
    """Episode summary data from Runner's episode_summary.jsonl."""
    wall_time: datetime
    explore_step: int
    update_step: int
    episode_id: int
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    max_insertion: Optional[float] = None


def load_episode_summaries(run_dir: Path) -> List[EpisodeSummaryRow]:
    """
    Load episode summaries from Runner's episode_summary.jsonl.
    
    This is the preferred source for episode data because it has exact
    explore_step and update_step values (no interpolation needed).
    """
    summary_paths = _glob_all(run_dir, [
        "**/csv/episode_summary.jsonl",
        "**/episode_summary.jsonl",
    ])
    
    if not summary_paths:
        return []
    
    out: List[EpisodeSummaryRow] = []
    
    for path in summary_paths:
        try:
            with path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        wall_time = _parse_wall_time(entry.get("wall_time"))
                        if wall_time is None:
                            continue
                        
                        out.append(EpisodeSummaryRow(
                            wall_time=wall_time,
                            explore_step=_to_int(entry.get("explore_step")),
                            update_step=_to_int(entry.get("update_step")),
                            episode_id=_to_int(entry.get("episode_id")),
                            total_reward=_to_float(entry.get("total_reward")),
                            steps=_to_int(entry.get("steps")),
                            terminated=bool(entry.get("terminated", False)),
                            truncated=bool(entry.get("truncated", False)),
                            max_insertion=_to_float(entry.get("max_insertion")) if "max_insertion" in entry else None,
                        ))
                    except (json.JSONDecodeError, Exception):
                        continue
        except Exception:
            continue
    
    # Sort by explore_step
    out.sort(key=lambda x: (x.explore_step, x.episode_id))
    return out


def merge_episode_summaries_with_worker_logs(
    summaries: List[EpisodeSummaryRow],
    worker_episodes: List[EpisodeRowV3],
) -> List[EpisodeRowV3]:
    """
    Merge episode summaries (with exact step counters) with worker log episodes
    (which have detailed stuck detection data).
    
    Uses wall_time-based matching (not ID-based) to avoid worker/global ID mismatch.
    Summaries provide exact explore_step/update_step via "summary_exact" source.
    Worker episodes provide detailed per-step analysis (stuck detection, etc.).
    
    Returns merged list of EpisodeRowV3 with accurate step counters.
    """
    if not summaries:
        return worker_episodes
    
    # Filter and sort summaries with valid wall_time for binary search
    summaries_with_wt = [s for s in summaries if s.wall_time is not None]
    summaries_with_wt.sort(key=lambda s: s.wall_time.timestamp() if s.wall_time else 0)
    
    if not summaries_with_wt:
        # No wall_time available, cannot do robust matching
        return worker_episodes
    
    used = set()
    matched_count = 0
    
    # Fix #1: Pre-extract wall_time array for bisect + outward scan
    merge_wts = [s.wall_time.timestamp() if s.wall_time else 0.0 for s in summaries_with_wt]

    # Match worker episodes to summaries by wall_time
    for ep in worker_episodes:
        if ep.end_wall_time is None:
            continue

        wt = ep.end_wall_time
        best_i, best_dt = None, float("inf")

        # Find insertion point via bisect
        idx = bisect.bisect_left(merge_wts, wt)

        # Scan outward from insertion point to find nearest UNUSED entry
        lo = max(0, idx - 1)
        hi = min(idx, len(summaries_with_wt) - 1)
        while lo >= 0 or hi < len(summaries_with_wt):
            improved = False
            if lo >= 0:
                dt = abs(merge_wts[lo] - wt)
                if dt < best_dt:
                    if lo not in used:
                        best_dt, best_i = dt, lo
                    lo -= 1
                    improved = True
                else:
                    lo = -1
            if hi < len(summaries_with_wt):
                dt = abs(merge_wts[hi] - wt)
                if dt < best_dt:
                    if hi not in used:
                        best_dt, best_i = dt, hi
                    hi += 1
                    improved = True
                else:
                    hi = len(summaries_with_wt)
            if not improved:
                break

        # Accept if within 10 second tolerance
        if best_i is not None and best_dt <= 10.0:
            s = summaries_with_wt[best_i]
            # Use priority-based setter (summary_exact has highest priority)
            _set_estimate_if_better(ep, s.explore_step, s.update_step, "summary_exact")
            # Also update reward/insertion if worker logs are incomplete
            if math.isnan(ep.total_reward) or ep.total_reward == 0.0:
                ep.total_reward = s.total_reward
            if s.max_insertion is not None:
                if math.isnan(ep.max_insertion_1) or ep.max_insertion_1 == 0.0:
                    ep.max_insertion_1 = s.max_insertion
            used.add(best_i)
            matched_count += 1
    
    print(f"  [INFO] Matched {matched_count}/{len(worker_episodes)} worker episodes to summaries via wall_time")
    
    # Add any summaries that weren't matched to create placeholder episodes
    for i, s in enumerate(summaries_with_wt):
        if i not in used:
            worker_episodes.append(EpisodeRowV3(
                worker=-1,  # Unknown worker
                episode=s.episode_id,
                start_timestamp=None,
                end_timestamp=s.wall_time,
                total_reward=s.total_reward,
                max_insertion_1=s.max_insertion or 0.0,
                max_insertion_2=0.0,
                steps=s.steps,
                terminated=s.terminated,
                truncated=s.truncated,
                explore_step_estimate=s.explore_step,
                update_step_estimate=s.update_step,
                estimate_source="summary_exact",
            ))
    
    # Re-sort by explore_step_estimate
    worker_episodes.sort(key=lambda e: (
        e.explore_step_estimate if e.explore_step_estimate is not None else 10**18,
        e.episode,
        e.worker,
    ))
    
    return worker_episodes


# -----------------------------
# Snapshot metadata
# -----------------------------

@dataclass
class SnapshotRow:
    path: Path
    update_step: Optional[int]
    explore_step: Optional[int]

def load_policy_snapshots(run_dir: Path) -> List[SnapshotRow]:
    try:
        import torch
        torch_ok = True
    except Exception:
        torch_ok = False

    snap_paths = _glob_all(run_dir, [
        "**/policy_snapshots/*.pt",
        "**/policy_snapshots/*.pth",
        "**/policy_*.pt",
        "**/policy_*.pth",
    ])
    out: List[SnapshotRow] = []
    if not torch_ok:
        return [SnapshotRow(path=p, update_step=None, explore_step=None) for p in snap_paths]

    import torch

    for p in snap_paths:
        try:
            obj = torch.load(p, map_location="cpu")
            out.append(SnapshotRow(
                path=p,
                update_step=_to_int(obj.get("update_step")) if isinstance(obj, dict) else None,
                explore_step=_to_int(obj.get("explore_step")) if isinstance(obj, dict) else None,
            ))
        except Exception:
            out.append(SnapshotRow(path=p, update_step=None, explore_step=None))
    
    out.sort(key=lambda s: (s.update_step if s.update_step is not None and s.update_step >= 0 else 10**18, s.path.stat().st_mtime))
    return out

# -----------------------------
# Enhanced Event with severity and confidence
# -----------------------------

@dataclass
class Event:
    name: str
    update_step: Optional[int]
    explore_step: Optional[int]
    details: str
    severity: str = "info"  # "critical", "warning", "info"
    confidence: float = 0.0  # 0.0-1.0 based on how far past threshold
    timestamp: Optional[datetime] = None
    # Timeline inference fields (for unified sorting)
    update_step_inferred: Optional[int] = None  # Inferred from timestamp or explore_step
    timeline_source: Optional[str] = None       # "exact", "ts_inferred", "explore_inferred"

# -----------------------------
# Enhanced collapse detection
# -----------------------------

def detect_events(
    losses: List[LossRow],
    probes: List[ProbeRow],
    batches: List[BatchRow],
    episodes: List[EpisodeRowV3],
    alpha_threshold: float,
    sustain: int,
    interpolator: Optional[Tuple] = None,  # Perf #5: pre-built interpolator
    eps_time_sorted: Optional[List[EpisodeRowV3]] = None,  # Perf #6: pre-sorted episodes
) -> Tuple[List[Event], Dict[str, Any]]:
    """
    Enhanced event detection with:
    - Fixed thresholds (absolute bounds)
    - Severity and confidence scores
    - insertion_collapse from episodes
    - freeze detection
    - reward_decline and done_rate events
    - Gating for replay events
    """
    events: List[Event] = []

    if not losses:
        return events, {"classification": "insufficient_data"}

    # Build arrays for rolling stats
    update_steps = [r.update_step for r in losses]
    explore_steps = [r.explore_step for r in losses]
    alpha = [r.alpha for r in losses]
    entropy = [r.entropy_proxy for r in losses]
    q1l = [r.q1_loss for r in losses]
    q2l = [r.q2_loss for r in losses]
    gpol = [r.grad_norm_policy for r in losses]

    n = len(losses)
    
    # Baseline from stable training region (skip startup)
    startup_skip = min(100, n // 10)
    base_start = startup_skip
    base_end = min(base_start + 1000, n)
    
    if base_end - base_start < 20:
        base_start = 0
        base_end = min(200, n)
    
    base_alpha = _safe_median(alpha[base_start:base_end])
    base_entropy = _safe_median(entropy[base_start:base_end])
    base_gpol = _safe_median(gpol[base_start:base_end])
    base_q1l = _safe_median(q1l[base_start:base_end])
    base_q2l = _safe_median(q2l[base_start:base_end])

    # =========================================
    # Event 1: alpha shutdown
    # =========================================
    alpha_pred = [(a < alpha_threshold) for a in alpha]
    i_alpha = _first_sustained(alpha_pred, sustain=sustain)
    if i_alpha is not None:
        conf = _compute_confidence(alpha[i_alpha], alpha_threshold, base_alpha, direction="below")
        events.append(Event(
            name="alpha_shutdown",
            update_step=update_steps[i_alpha],
            explore_step=explore_steps[i_alpha],
            details=f"alpha fell below {alpha_threshold:g} and stayed for >= {sustain} updates (baseline~{base_alpha:.4g}).",
            severity="critical" if conf > 0.5 else "warning",
            confidence=conf,
        ))

    # =========================================
    # Event 2: entropy proxy collapse
    # =========================================
    ent_roll = _rolling_mean(entropy, window=max(5, sustain//2))
    base_entropy_std = _safe_pstdev(entropy[base_start:base_end])
    # Use at least 2x std or 10% of baseline
    ent_drop_thresh = base_entropy - max(2.0 * base_entropy_std, 0.1 * abs(base_entropy))
    ent_pred = [e < ent_drop_thresh for e in ent_roll]
    i_ent = _first_sustained(ent_pred, sustain=sustain)
    if i_ent is not None:
        conf = _compute_confidence(ent_roll[i_ent], ent_drop_thresh, base_entropy, direction="below")
        events.append(Event(
            name="entropy_drop",
            update_step=update_steps[i_ent],
            explore_step=explore_steps[i_ent],
            details=f"entropy_proxy dropped below {ent_drop_thresh:.3g} (baseline~{base_entropy:.3g}).",
            severity="warning",
            confidence=conf,
        ))

    # =========================================
    # Event 3: gradient collapse (policy)
    # =========================================
    gpol_pred = [(g < 0.25 * base_gpol) for g in gpol]
    i_gpol = _first_sustained(gpol_pred, sustain=sustain)
    if i_gpol is not None:
        conf = _compute_confidence(gpol[i_gpol], 0.25 * base_gpol, base_gpol, direction="below")
        events.append(Event(
            name="policy_grad_collapse",
            update_step=update_steps[i_gpol],
            explore_step=explore_steps[i_gpol],
            details=f"grad_norm_policy fell below 25% of baseline (baseline~{base_gpol:.3g}).",
            severity="warning",
            confidence=conf,
        ))

    # =========================================
    # Event 4: critic instability (loss spikes)
    # =========================================
    spike_factor = 8.0
    q_spike_pred = []
    for i, (ql1, ql2) in enumerate(zip(q1l, q2l)):
        if i < startup_skip:
            q_spike_pred.append(False)
        else:
            q_spike_pred.append((ql1 > spike_factor * base_q1l) or (ql2 > spike_factor * base_q2l))
    
    i_qspike = _first_sustained(q_spike_pred, sustain=max(5, sustain//2))
    if i_qspike is not None:
        events.append(Event(
            name="critic_loss_spike",
            update_step=update_steps[i_qspike],
            explore_step=explore_steps[i_qspike],
            details=f"q_loss exceeded {spike_factor}× baseline after startup.",
            severity="critical",
            confidence=0.8,
        ))

    # =========================================
    # Event 5: Clamp saturation (direct from losses CSV, Fix #6)
    # =========================================
    clamp_vals = [r.clamp_fraction for r in losses]
    has_clamp_data = any(c > 0 for c in clamp_vals)
    if has_clamp_data:
        clamp_roll = _rolling_mean(clamp_vals, window=max(5, sustain // 2))
        pred_clamp = [c > CLAMP_FRACTION_THRESHOLD for c in clamp_roll]
        i_clamp = _first_sustained(pred_clamp, sustain=sustain)
        if i_clamp is not None:
            peak_clamp = max(clamp_roll[i_clamp:min(i_clamp + sustain * 2, len(clamp_roll))])
            is_critical = peak_clamp > CLAMP_FRACTION_CRITICAL
            conf = min(1.0, clamp_roll[i_clamp] / CLAMP_FRACTION_CRITICAL)
            events.append(Event(
                name="clamp_saturation",
                update_step=update_steps[i_clamp],
                explore_step=explore_steps[i_clamp],
                details=f"clamp_fraction exceeded {CLAMP_FRACTION_THRESHOLD:.0%}: rolling={clamp_roll[i_clamp]:.2%}, peak={peak_clamp:.2%}.",
                severity="critical" if is_critical else "warning",
                confidence=conf,
            ))

    # =========================================
    # Probe events with FIXED thresholds
    # =========================================
    if probes:
        probe_u = [p.update_step for p in probes]
        probe_e = [p.explore_step for p in probes]
        probe_trans = [p.trans_mean for p in probes]
        probe_std = [p.trans_std_mean for p in probes]
        
        probe_trans_roll = _rolling_mean(probe_trans, window=max(5, sustain//3))
        probe_std_roll = _rolling_mean(probe_std, window=max(5, sustain//3))
        
        # Baseline for probes
        pb_start = min(10, len(probes) // 10)
        pb_end = min(pb_start + 100, len(probes))
        base_probe_trans = _safe_mean(probe_trans[pb_start:pb_end])
        base_probe_std = _safe_median(probe_std[pb_start:pb_end])
        base_probe_trans_std = _safe_pstdev(probe_trans[pb_start:pb_end])
        
        # Event: Translation saturation (DATA-DRIVEN threshold)
        # Infer action scale from early training percentiles
        probe_baseline_vals = [abs(t) for t in probe_trans[pb_start:pb_end] if not math.isnan(t)]
        probe_p95 = float(np.percentile(probe_baseline_vals, 95)) if len(probe_baseline_vals) > 10 else 0.5
        # Threshold = max(70% of p95 scale, baseline + K*std, minimum floor)
        sat_thresh = max(
            SATURATION_SCALE_FACTOR * probe_p95,
            abs(base_probe_trans) + SATURATION_K_STD * base_probe_trans_std,
            SATURATION_MIN_FLOOR
        )
        
        pred_saturated = [abs(t) > sat_thresh for t in probe_trans_roll]
        i_sat = _first_sustained(pred_saturated, sustain=max(10, sustain//2))
        if i_sat is not None:
            direction = "positive (max insert)" if probe_trans_roll[i_sat] > 0 else "negative (retract)"
            conf = _compute_confidence(abs(probe_trans_roll[i_sat]), sat_thresh, abs(base_probe_trans))
            events.append(Event(
                name="probe_translation_saturated",
                update_step=probe_u[i_sat],
                explore_step=probe_e[i_sat],
                details=f"probe translation saturated {direction}: |mean|>{sat_thresh:.3f} (value={probe_trans_roll[i_sat]:.3f}, baseline={base_probe_trans:.3f}).",
                severity="critical" if conf > 0.5 else "warning",
                confidence=conf,
            ))
        
        # Event: Std collapse (FIXED: require 50% reduction from baseline)
        if not math.isnan(base_probe_std) and base_probe_std > 0.1:
            std_collapse_thresh = STD_COLLAPSE_FRACTION * base_probe_std
            pred_std_collapse = [s < std_collapse_thresh for s in probe_std_roll]
            i_std = _first_sustained(pred_std_collapse, sustain=max(10, sustain//2))
            if i_std is not None:
                conf = _compute_confidence(probe_std_roll[i_std], std_collapse_thresh, base_probe_std, direction="below")
                events.append(Event(
                    name="probe_std_collapse",
                    update_step=probe_u[i_std],
                    explore_step=probe_e[i_std],
                    details=f"probe std collapsed to {probe_std_roll[i_std]:.3f} (threshold={std_collapse_thresh:.3f}, baseline={base_probe_std:.3f}, {100*(1-probe_std_roll[i_std]/base_probe_std):.0f}% reduction).",
                    severity="warning" if conf > 0.3 else "info",
                    confidence=conf,
                ))
        
        # Event: Freeze detection (mean near 0 AND std very low)
        pred_freeze = [
            abs(probe_trans_roll[i]) < FREEZE_MEAN_EPSILON and 
            probe_std_roll[i] < FREEZE_STD_MAX
            for i in range(len(probe_trans_roll))
        ]
        i_freeze = _first_sustained(pred_freeze, sustain=max(10, sustain//2))
        if i_freeze is not None:
            events.append(Event(
                name="probe_freeze",
                update_step=probe_u[i_freeze],
                explore_step=probe_e[i_freeze],
                details=f"probe translation frozen: |mean|<{FREEZE_MEAN_EPSILON}, std<{FREEZE_STD_MAX}.",
                severity="critical",
                confidence=0.9,
            ))

    # =========================================
    # Batch events with FIXED thresholds
    # =========================================
    insertion_collapsed = False  # For gating replay shift events
    pending_replay_events = []   # Will be added to events only if insertion_collapsed
    
    if batches:
        b_u = [b.update_step for b in batches]
        b_e = [b.explore_step for b in batches]
        actor_trans = [b.actor_trans_mean for b in batches]
        taken_trans = [b.taken_trans_mean for b in batches]  # Replay buffer actions
        reward_vals = [b.reward_mean for b in batches]
        done_vals = [b.done_rate for b in batches]
        
        actor_roll = _rolling_mean(actor_trans, window=max(5, sustain//3))
        taken_roll = _rolling_mean(taken_trans, window=max(5, sustain//3))
        reward_roll = _rolling_mean(reward_vals, window=max(5, sustain//3))
        done_roll = _rolling_mean(done_vals, window=max(5, sustain//3))
        
        # Baseline for batches
        bb_start = min(10, len(batches) // 10)
        bb_end = min(bb_start + 100, len(batches))
        base_actor = _safe_mean(actor_trans[bb_start:bb_end])
        base_actor_std = _safe_pstdev(actor_trans[bb_start:bb_end])
        base_taken = _safe_mean(taken_trans[bb_start:bb_end])
        base_taken_std = _safe_pstdev(taken_trans[bb_start:bb_end])
        base_reward = _safe_mean(reward_vals[bb_start:bb_end])
        base_reward_std = _safe_pstdev(reward_vals[bb_start:bb_end])
        base_done = _safe_mean(done_vals[bb_start:bb_end])
        base_done_std = _safe_pstdev(done_vals[bb_start:bb_end])
        
        # Event: Actor saturation (DATA-DRIVEN threshold)
        # Infer action scale from early training percentiles
        actor_baseline_vals = [abs(a) for a in actor_trans[bb_start:bb_end] if not math.isnan(a)]
        actor_p95 = float(np.percentile(actor_baseline_vals, 95)) if len(actor_baseline_vals) > 10 else 0.5
        # Threshold = max(70% of p95 scale, baseline + K*std, minimum floor)
        sat_thresh = max(
            SATURATION_SCALE_FACTOR * actor_p95,
            abs(base_actor) + SATURATION_K_STD * base_actor_std,
            SATURATION_MIN_FLOOR
        )
        
        pred_actor_saturated = [abs(a) > sat_thresh for a in actor_roll]
        i_sat = _first_sustained(pred_actor_saturated, sustain=max(10, sustain//2))
        if i_sat is not None:
            direction = "positive (max insert)" if actor_roll[i_sat] > 0 else "negative (retract)"
            conf = _compute_confidence(abs(actor_roll[i_sat]), sat_thresh, abs(base_actor))
            events.append(Event(
                name="actor_saturated",
                update_step=b_u[i_sat],
                explore_step=b_e[i_sat],
                details=f"actor translation saturated {direction}: |mean|>{sat_thresh:.3f} (value={actor_roll[i_sat]:.3f}, baseline={base_actor:.3f}).",
                severity="critical" if conf > 0.5 else "warning",
                confidence=conf,
            ))
        
        # Event: Reward decline
        reward_thresh = base_reward - max(2 * base_reward_std, 0.001)
        pred_reward_decline = [r < reward_thresh for r in reward_roll]
        i_rew = _first_sustained(pred_reward_decline, sustain=max(10, sustain//2))
        if i_rew is not None:
            events.append(Event(
                name="reward_decline",
                update_step=b_u[i_rew],
                explore_step=b_e[i_rew],
                details=f"batch reward dropped below {reward_thresh:.4f} (baseline={base_reward:.4f}).",
                severity="info",
                confidence=0.5,
            ))
        
        # Event: Done rate spike (indicates frequent terminations)
        done_spike_thresh = base_done + max(3 * base_done_std, 0.05)
        pred_done_spike = [d > done_spike_thresh for d in done_roll]
        i_done = _first_sustained(pred_done_spike, sustain=max(5, sustain//3))
        if i_done is not None:
            events.append(Event(
                name="done_rate_spike",
                update_step=b_u[i_done],
                explore_step=b_e[i_done],
                details=f"done_rate spiked above {done_spike_thresh:.3f} (baseline={base_done:.3f}).",
                severity="warning",
                confidence=0.6,
            ))
        
        # Replay distribution shift detection (PENDING - gated by insertion_collapse)
        # Require magnitude floor + sustained duration
        taken_margin = max(3.0 * base_taken_std, 0.2)
        
        pred_taken_pos = [(t > base_taken + taken_margin) and (t > REPLAY_SHIFT_FLOOR) for t in taken_roll]
        pred_taken_neg = [(t < base_taken - taken_margin) and (t < -REPLAY_SHIFT_FLOOR) for t in taken_roll]
        
        i_taken_pos = _first_sustained(pred_taken_pos, sustain=sustain)
        if i_taken_pos is not None:
            val = taken_roll[i_taken_pos]
            pending_replay_events.append(Event(
                name="replay_distribution_shift_positive",
                update_step=b_u[i_taken_pos],
                explore_step=b_e[i_taken_pos],
                details=f"replay buffer shifted positive: rolling_mean={val:.4f} > floor={REPLAY_SHIFT_FLOOR} (baseline={base_taken:.4f}).",
                severity="warning",
                confidence=0.6,
            ))
        
        i_taken_neg = _first_sustained(pred_taken_neg, sustain=sustain)
        if i_taken_neg is not None:
            val = taken_roll[i_taken_neg]
            pending_replay_events.append(Event(
                name="replay_distribution_shift_negative",
                update_step=b_u[i_taken_neg],
                explore_step=b_e[i_taken_neg],
                details=f"replay buffer shifted negative (retract/hold): rolling_mean={val:.4f} < -{REPLAY_SHIFT_FLOOR} (baseline={base_taken:.4f}).",
                severity="warning",
                confidence=0.6,
            ))

    # =========================================
    # Episode events (insertion collapse)
    # =========================================
    # Perf #8: Pre-compute insertion rolling median for reuse in make_plots
    _cached_ins_roll = None
    _cached_eps_sorted = None

    if episodes:
        # Perf #6: Use pre-sorted episodes if available
        eps_sorted = eps_time_sorted if eps_time_sorted is not None else sorted(episodes, key=lambda e: (e.end_timestamp or datetime.min, e.episode))
        _cached_eps_sorted = eps_sorted
        insertions = [e.max_insertion_1 for e in eps_sorted]

        if len(insertions) >= 20:
            ins_roll = _rolling_median(insertions, window=20)
            _cached_ins_roll = ins_roll
            
            # Baseline from first 20% of episodes
            ins_base_end = min(len(insertions) // 5, 50)
            base_insertion = _safe_median(insertions[:max(10, ins_base_end)])
            
            # Event: Insertion collapse (dynamic threshold - drop-based)
            # Use the lower of: absolute threshold OR drop_ratio * baseline
            # This avoids false triggers if baseline is already shallow
            dynamic_collapse_thresh = min(
                INSERTION_COLLAPSE_THRESHOLD_MM,
                INSERTION_COLLAPSE_DROP_RATIO * base_insertion
            ) if base_insertion > 0 else INSERTION_COLLAPSE_THRESHOLD_MM
            
            # Require baseline to be "good" (above threshold) to detect collapse
            # Otherwise we're already in a bad state and shouldn't report collapse
            baseline_is_good = base_insertion > INSERTION_COLLAPSE_THRESHOLD_MM * 0.8
            
            pred_ins_collapse = [ins < dynamic_collapse_thresh for ins in ins_roll]
            i_ins = _first_sustained(pred_ins_collapse, sustain=20)
            if i_ins is not None and baseline_is_good:
                insertion_collapsed = True
                ep = eps_sorted[i_ins]
                # Use update_step_estimate if available (from timestamp correlation)
                events.append(Event(
                    name="insertion_collapse",
                    update_step=getattr(ep, 'update_step_estimate', None),
                    explore_step=ep.explore_step_estimate,
                    timestamp=ep.end_timestamp,
                    details=f"rolling median insertion fell below {dynamic_collapse_thresh:.1f}mm (value={ins_roll[i_ins]:.1f}mm, baseline={base_insertion:.1f}mm, drop_ratio={INSERTION_COLLAPSE_DROP_RATIO}).",
                    severity="critical",
                    confidence=0.9,
                ))
            
            # Event: Stuck regime (policy pushing forward but making no progress)
            # Use dual-signal confidence: delta_ins proxy AND masking detection
            stuck_fractions = [e.stuck_fraction for e in eps_sorted]
            masked_fractions = [e.masked_fraction for e in eps_sorted]
            
            has_stuck_data = any(sf > 0 for sf in stuck_fractions)
            has_masked_data = any(mf > 0 for mf in masked_fractions)
            
            if has_stuck_data or has_masked_data:
                stuck_roll = _rolling_mean(stuck_fractions, window=max(5, len(eps_sorted) // 20))
                masked_roll = _rolling_mean(masked_fractions, window=max(5, len(eps_sorted) // 20)) if has_masked_data else [0.0] * len(eps_sorted)
                
                # Baseline stuck fraction from early episodes
                stuck_base_end = min(len(stuck_fractions) // 5, 30)
                base_stuck = _safe_mean(stuck_fractions[:max(5, stuck_base_end)])
                
                # TWO-LEVEL TRIGGER:
                # Level 1: high stuck/masked fraction (sensitivity)
                # Level 2: episodes must have actual consecutive jams (specificity)
                
                # Level 2: Compute jam indicator per episode (max_stuck_run >= threshold)
                has_jam = [e.max_stuck_run >= STUCK_MIN_CONSEC_STEPS for e in eps_sorted]
                jam_roll = _rolling_mean([1.0 if j else 0.0 for j in has_jam], 
                                         window=max(5, len(eps_sorted) // 20))
                
                # Two-level trigger: BOTH conditions must be met
                # Level 1: stuck_fraction OR masked_fraction above threshold
                # Level 2: jam_roll above JAM_EPISODE_THRESHOLD
                pred_stuck_regime = [
                    (sf > STUCK_FRACTION_THRESHOLD or mf > STUCK_FRACTION_THRESHOLD) 
                    and jf > JAM_EPISODE_THRESHOLD
                    for sf, mf, jf in zip(stuck_roll, masked_roll, jam_roll)
                ]
                i_stuck = _first_sustained(pred_stuck_regime, sustain=STUCK_SUSTAINED_EPISODES)
                
                if i_stuck is not None:
                    ep_stuck = eps_sorted[i_stuck]
                    # Find dominant stuck depth mode across affected episodes
                    affected_eps = eps_sorted[i_stuck:min(i_stuck + STUCK_SUSTAINED_EPISODES * 2, len(eps_sorted))]
                    depth_modes = [e.stuck_depth_mode_mm for e in affected_eps if e.stuck_depth_mode_mm is not None]
                    dominant_depth = _safe_median(depth_modes) if depth_modes else None
                    
                    # Compute average max stuck run in affected episodes
                    max_runs = [e.max_stuck_run for e in affected_eps if e.max_stuck_run > 0]
                    avg_max_run = _safe_mean(max_runs) if max_runs else 0
                    
                    # Find dominant mask reason in affected episodes
                    mask_reasons = [e.dominant_mask_reason for e in affected_eps if e.dominant_mask_reason is not None]
                    dominant_reason = max(set(mask_reasons), key=mask_reasons.count) if mask_reasons else None
                    
                    depth_str = f", dominant depth ~{dominant_depth:.0f}mm" if dominant_depth else ""
                    reason_str = f", dominant reason={dominant_reason}" if dominant_reason else ""
                    
                    # Get jam fraction at trigger point for confidence scaling
                    jam_fraction = jam_roll[i_stuck]
                    
                    # Dual-signal confidence:
                    # - Both signals agree: 0.95
                    # - Only masking signal: 0.85 (direct observation)
                    # - Only delta_ins signal: 0.70 (proxy)
                    delta_triggered = stuck_roll[i_stuck] > STUCK_FRACTION_THRESHOLD
                    masked_triggered = masked_roll[i_stuck] > STUCK_FRACTION_THRESHOLD if has_masked_data else False
                    
                    if delta_triggered and masked_triggered:
                        base_conf = 0.95
                        signal_str = "dual-signal (delta_ins + masking)"
                    elif masked_triggered:
                        base_conf = 0.85
                        signal_str = "masking detection"
                    else:
                        base_conf = 0.70
                        signal_str = "delta_ins proxy"
                    
                    # Scale confidence by how far past threshold AND jam fraction
                    combined_fraction = max(stuck_roll[i_stuck], masked_roll[i_stuck] if has_masked_data else 0)
                    # Scale by jam_factor: high jam fraction = high confidence
                    jam_factor = 0.5 + 0.5 * jam_fraction
                    confidence = min(base_conf, 0.5 + combined_fraction) * jam_factor
                    
                    events.append(Event(
                        name="stuck_regime",
                        update_step=getattr(ep_stuck, 'update_step_estimate', None),
                        explore_step=ep_stuck.explore_step_estimate,
                        timestamp=ep_stuck.end_timestamp,
                        details=f"policy stuck ({signal_str}): rolling stuck_fraction={stuck_roll[i_stuck]:.2%}, masked_fraction={masked_roll[i_stuck]:.2%}, jam_fraction={jam_fraction:.2%} (thresholds: stuck>{STUCK_FRACTION_THRESHOLD:.0%}, jam>{JAM_EPISODE_THRESHOLD:.0%}), avg max_run={avg_max_run:.0f} steps{depth_str}{reason_str}.",
                        severity="critical",
                        confidence=confidence,
                    ))

    # =========================================
    # Fix #12: Always include replay shift events (no longer gated).
    # Previously required insertion_collapse or alpha_shutdown to fire first,
    # which could miss independent distribution shifts.
    # =========================================
    if pending_replay_events:
        events.extend(pending_replay_events)

    # =========================================
    # Populate timeline inference fields for unified sorting
    # =========================================
    
    # Perf #5: Use pre-built interpolator if available
    if interpolator is not None:
        ts_to_explore, explore_to_update, has_timestamps = interpolator
    else:
        ts_to_explore, explore_to_update, has_timestamps = build_timestamp_interpolator(losses)
    
    def _populate_timeline_fields(events: List[Event]):
        """Infer update_step for events that only have timestamp or explore_step."""
        for ev in events:
            if ev.update_step is not None:
                ev.update_step_inferred = ev.update_step
                ev.timeline_source = "exact"
            elif ev.timestamp is not None and ts_to_explore is not None and explore_to_update is not None:
                # Convert timestamp -> explore_step -> update_step
                explore_est = ts_to_explore(ev.timestamp)
                update_est = explore_to_update(explore_est)
                ev.update_step_inferred = update_est
                ev.timeline_source = "ts_inferred"
            elif ev.explore_step is not None and explore_to_update is not None:
                ev.update_step_inferred = explore_to_update(ev.explore_step)
                ev.timeline_source = "explore_inferred"
            # else: leave as None
    
    _populate_timeline_fields(events)
    
    # =========================================
    # Order events by unified timeline
    # =========================================
    def _event_key(ev: Event):
        """
        Sort by unified timeline that respects actual timing:
        1. effective_update_step (exact or inferred) 
        2. timestamp (for tie-breaking when both have timestamps)
        3. explore_step
        4. event name
        
        Key insight: don't hard-prioritize "has exact update_step" over "earlier actual time".
        Events with inferred update_step sort correctly alongside exact ones.
        """
        # Use inferred update_step (which includes exact values)
        effective_step = ev.update_step_inferred if ev.update_step_inferred is not None else 10**18
        
        # Use timestamp for secondary sorting (handles same effective_step)
        ts_value = ev.timestamp.timestamp() if ev.timestamp else 10**18
        
        explore = ev.explore_step if ev.explore_step is not None else 10**18
        
        return (effective_step, ts_value, explore, ev.name)
    
    events.sort(key=_event_key)

    # =========================================
    # Classification
    # =========================================
    policy_names = {
        "alpha_shutdown", "entropy_drop", "policy_grad_collapse",
        "probe_translation_saturated", "probe_std_collapse", "probe_freeze",
        "actor_saturated", "clamp_saturation", "insertion_collapse", "stuck_regime",
        "replay_distribution_shift_positive", "replay_distribution_shift_negative",
    }
    critic_names = {"critic_loss_spike"}

    earliest_policy = next((ev for ev in events if ev.name in policy_names), None)
    earliest_critic = next((ev for ev in events if ev.name in critic_names), None)

    if earliest_policy and (not earliest_critic or _event_key(earliest_policy) <= _event_key(earliest_critic)):
        classification = "policy_collapse_or_suboptimal_attractor"
        rationale = f"earliest event is policy-side ({earliest_policy.name})"
    elif earliest_critic:
        classification = "critic_instability_precedes_policy"
        rationale = f"earliest event is critic-side ({earliest_critic.name})"
    else:
        classification = "no_clear_event_detected"
        rationale = "no sustained thresholds crossed"

    summary = dict(
        baseline=dict(
            alpha=base_alpha, entropy=base_entropy, grad_policy=base_gpol,
            q1_loss=base_q1l, q2_loss=base_q2l,
            computed_from_updates=f"{base_start}-{base_end}",
            startup_skip=startup_skip
        ),
        thresholds=dict(
            saturation_scale_factor=SATURATION_SCALE_FACTOR,
            saturation_min_floor=SATURATION_MIN_FLOOR,
            saturation_k_std=SATURATION_K_STD,
            std_collapse_fraction=STD_COLLAPSE_FRACTION,
            freeze_mean_eps=FREEZE_MEAN_EPSILON,
            freeze_std_max=FREEZE_STD_MAX,
            insertion_collapse_mm=INSERTION_COLLAPSE_THRESHOLD_MM,
            insertion_collapse_drop_ratio=INSERTION_COLLAPSE_DROP_RATIO,
            stuck_push_threshold=STUCK_PUSH_THRESHOLD,
            stuck_delta_ins_eps_mm=STUCK_DELTA_INS_EPS_MM,
            stuck_fraction_threshold=STUCK_FRACTION_THRESHOLD,
            replay_shift_floor=REPLAY_SHIFT_FLOOR,
        ),
        classification=classification,
        rationale=rationale,
        insertion_collapsed=insertion_collapsed,
        # Perf #6/#8: cached sorted episodes and insertion rolling median for reuse
        _cached_eps_sorted=_cached_eps_sorted,
        _cached_ins_roll=_cached_ins_roll,
    )
    return events, summary

# -----------------------------
# Reporting
# -----------------------------

def _format_float(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.{nd}f}"

def write_report(
    out_dir: Path,
    run_dir: Path,
    losses_path: Optional[Path],
    probe_path: Optional[Path],
    batch_path: Optional[Path],
    n_worker_logs: int,
    losses: List[LossRow],
    probes: List[ProbeRow],
    batches: List[BatchRow],
    episodes: List[EpisodeRowV3],
    snapshots: List[SnapshotRow],
    events: List[Event],
    summary: Dict[str, Any],
):
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "collapse_report_v6.md"

    # Key aggregate stats
    if losses:
        first = losses[0]
        last = losses[-1]
        q_change = _percent_change(first.q1_mean, last.q1_mean)
        alpha_change = _percent_change(first.alpha, last.alpha)
        ent_change = _percent_change(first.entropy_proxy, last.entropy_proxy)
    else:
        q_change = alpha_change = ent_change = float("nan")

    # Insertion trajectory
    if episodes:
        # Perf #6: Use cached sorted episodes if available
        eps_sorted = summary.get("_cached_eps_sorted") or sorted(episodes, key=lambda e: (e.end_timestamp or datetime.min, e.episode))
        first_n = eps_sorted[:min(50, len(eps_sorted)//4+1)]
        last_n = eps_sorted[-min(50, len(eps_sorted)//4+1):]
        ins_first = _safe_mean([e.max_insertion_1 for e in first_n])
        ins_last = _safe_mean([e.max_insertion_1 for e in last_n])
        ins_change = _percent_change(ins_first, ins_last)
        rew_first = _safe_mean([e.total_reward for e in first_n])
        rew_last = _safe_mean([e.total_reward for e in last_n])
        rew_change = _percent_change(rew_first, rew_last)
    else:
        ins_first = ins_last = ins_change = float("nan")
        rew_first = rew_last = rew_change = float("nan")

    collapse_step = events[0].update_step if events else None
    collapse_explore = events[0].explore_step if events else None

    # Pick snapshot before/after
    before_snap = after_snap = None
    if collapse_step is not None and snapshots:
        snaps_with_u = [s for s in snapshots if s.update_step is not None and s.update_step >= 0]
        if snaps_with_u:
            before = [s for s in snaps_with_u if s.update_step <= collapse_step]
            after = [s for s in snaps_with_u if s.update_step >= collapse_step]
            before_snap = before[-1] if before else snaps_with_u[0]
            after_snap = after[0] if after else snaps_with_u[-1]

    def rel(p: Optional[Path]) -> str:
        if p is None:
            return "not_found"
        try:
            return str(p.relative_to(run_dir))
        except Exception:
            return str(p)

    lines: List[str] = []
    lines.append("# Collapse Forensics Report (v6)\n")
    lines.append(f"**Run directory:** `{run_dir}`\n")
    lines.append("## Inputs discovered\n")
    lines.append(f"- Losses CSV: `{rel(losses_path)}`")
    lines.append(f"- Probe JSONL: `{rel(probe_path)}`")
    lines.append(f"- Batch samples JSONL: `{rel(batch_path)}`")
    lines.append(f"- Worker logs: `{n_worker_logs}` files ({len(episodes)} episodes with timestamps)")
    lines.append(f"- Policy snapshots: `{len(snapshots)}` files\n")

    lines.append("## High-level summary\n")
    lines.append(f"- Classification: **{summary.get('classification','?')}**")
    lines.append(f"- Rationale: {summary.get('rationale','')}")
    lines.append(f"- Insertion collapsed: {summary.get('insertion_collapsed', False)}")
    if collapse_step is not None:
        lines.append(f"- Earliest detected event at **update_step={collapse_step}**, explore_step={collapse_explore}")
    else:
        lines.append("- Earliest detected event: none\n")

    lines.append("")
    lines.append("### Detection thresholds used\n")
    thresh = summary.get("thresholds", {})
    lines.append(f"- Saturation: |mean| > max(p95*{thresh.get('saturation_scale_factor', 0.7):.1f}, baseline+{thresh.get('saturation_k_std', 30):.0f}σ, floor={thresh.get('saturation_min_floor', 0.15):.2f})")
    lines.append(f"- Std collapse: std < {thresh.get('std_collapse_fraction', 0.5)*100:.0f}% of baseline")
    lines.append(f"- Freeze: |mean| < {thresh.get('freeze_mean_eps', 0.05):.2f} AND std < {thresh.get('freeze_std_max', 0.1):.2f}")
    lines.append(f"- Insertion collapse: rolling median < {thresh.get('insertion_collapse_mm', 200):.0f}mm\n")

    lines.append("### Key metric deltas\n")
    if losses:
        lines.append(f"- Alpha: {_format_float(losses[0].alpha,4)} → {_format_float(losses[-1].alpha,6)} ({_format_float(alpha_change,1)}%)")
        lines.append(f"- Entropy: {_format_float(losses[0].entropy_proxy,2)} → {_format_float(losses[-1].entropy_proxy,2)} ({_format_float(ent_change,1)}%)")
    lines.append(f"- Max insertion: {_format_float(ins_first,1)} → {_format_float(ins_last,1)} mm ({_format_float(ins_change,1)}%)")
    lines.append(f"- Episode reward: {_format_float(rew_first,3)} → {_format_float(rew_last,3)} ({_format_float(rew_change,1)}%)\n")

    lines.append("## Event timeline\n")
    if not events:
        lines.append("_No sustained events detected._\n")
    else:
        lines.append("| # | Event | Severity | Confidence | update_step | explore_step | Details |")
        lines.append("|---:|---|---|---:|---:|---:|---|")
        for i, ev in enumerate(events, 1):
            sev_icon = {"critical": "[!]", "warning": "[~]", "info": "[i]"}.get(ev.severity, "[-]")
            lines.append(f"| {i} | `{ev.name}` | {sev_icon} {ev.severity} | {ev.confidence:.2f} | {ev.update_step or ''} | {ev.explore_step or ''} | {ev.details} |")
        lines.append("")

    lines.append("## Snapshots to inspect\n")
    if before_snap or after_snap:
        if before_snap:
            lines.append(f"- **Before collapse:** `{rel(before_snap.path)}` (update={before_snap.update_step})")
        if after_snap:
            lines.append(f"- **After collapse:** `{rel(after_snap.path)}` (update={after_snap.update_step})")
    else:
        lines.append("_No snapshots found._")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path

# -----------------------------
# Plotting (v4 - enhanced visualization)
# -----------------------------

def make_plots(out_dir: Path, losses: List[LossRow], probes: List[ProbeRow],
               batches: List[BatchRow], episodes: List[EpisodeRowV3], events: List[Event],
               summary: Optional[Dict[str, Any]] = None) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    def vline_alpha_events(ax, events_list):
        """Only show alpha_shutdown event lines."""
        for ev in events_list:
            if ev.name != "alpha_shutdown" or ev.update_step is None:
                continue
            ax.axvline(ev.update_step, linestyle="--", linewidth=1, 
                      color="red", alpha=0.7)

    if losses:
        us = [r.update_step for r in losses]

        # Alpha (keep event lines - only alpha_shutdown)
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.alpha for r in losses])
        plt.xlabel("update_step")
        plt.ylabel("alpha")
        plt.title("Alpha over training")
        vline_alpha_events(plt.gca(), events)
        p = out_dir / "alpha_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Entropy with rolling(50) and rolling(100)
        entropy_vals = [r.entropy_proxy for r in losses]
        plt.figure(figsize=(10, 4))
        plt.plot(us, entropy_vals, alpha=0.3, linewidth=0.5, label="raw")
        if len(entropy_vals) >= 50:
            plt.plot(us, _rolling_mean(entropy_vals, 50), label="rolling(50)", linewidth=1.5)
        if len(entropy_vals) >= 100:
            plt.plot(us, _rolling_mean(entropy_vals, 100), label="rolling(100)", linewidth=2)
        plt.xlabel("update_step")
        plt.ylabel("entropy_proxy")
        plt.title("Entropy proxy over training")
        plt.legend()
        p = out_dir / "entropy_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Q1 mean (separate graph)
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.q1_mean for r in losses], linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("Q1 mean")
        plt.title("Critic Q1 mean over training")
        p = out_dir / "q1_mean_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Q2 mean (separate graph)
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.q2_mean for r in losses], linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("Q2 mean")
        plt.title("Critic Q2 mean over training")
        p = out_dir / "q2_mean_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Q1 loss (separate graph)
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.q1_loss for r in losses], linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("Q1 loss")
        plt.title("Critic Q1 loss over training")
        p = out_dir / "q1_loss_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Q1 loss (zoomed [0, 1] with rolling(50))
        q1_loss_vals = [r.q1_loss for r in losses]
        plt.figure(figsize=(10, 4))
        plt.plot(us, q1_loss_vals, alpha=0.3, linewidth=0.5, label="raw")
        if len(q1_loss_vals) >= 50:
            plt.plot(us, _rolling_mean(q1_loss_vals, 50), label="rolling(50)", linewidth=1.5)
        plt.ylim(0, 1)
        plt.xlabel("update_step")
        plt.ylabel("Q1 loss")
        plt.title("Critic Q1 loss over training (zoomed [0, 1])")
        plt.legend()
        p = out_dir / "q1_loss_zoomed_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Q2 loss (separate graph)
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.q2_loss for r in losses], linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("Q2 loss")
        plt.title("Critic Q2 loss over training")
        p = out_dir / "q2_loss_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Q2 loss (zoomed [0, 1] with rolling(50))
        q2_loss_vals = [r.q2_loss for r in losses]
        plt.figure(figsize=(10, 4))
        plt.plot(us, q2_loss_vals, alpha=0.3, linewidth=0.5, label="raw")
        if len(q2_loss_vals) >= 50:
            plt.plot(us, _rolling_mean(q2_loss_vals, 50), label="rolling(50)", linewidth=1.5)
        plt.ylim(0, 1)
        plt.xlabel("update_step")
        plt.ylabel("Q2 loss")
        plt.title("Critic Q2 loss over training (zoomed [0, 1])")
        plt.legend()
        p = out_dir / "q2_loss_zoomed_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Grad policy (separate graph with rolling(50))
        grad_policy_vals = [r.grad_norm_policy for r in losses]
        plt.figure(figsize=(10, 4))
        plt.plot(us, grad_policy_vals, alpha=0.3, linewidth=0.5, label="raw")
        if len(grad_policy_vals) >= 50:
            plt.plot(us, _rolling_mean(grad_policy_vals, 50), label="rolling(50)", linewidth=1.5)
        plt.xlabel("update_step")
        plt.ylabel("grad norm (policy)")
        plt.title("Policy gradient norm over training")
        plt.legend()
        p = out_dir / "grad_policy_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Grad Q1 (separate graph with rolling(50))
        grad_q1_vals = [r.grad_norm_q1 for r in losses]
        plt.figure(figsize=(10, 4))
        plt.plot(us, grad_q1_vals, alpha=0.3, linewidth=0.5, label="raw")
        if len(grad_q1_vals) >= 50:
            plt.plot(us, _rolling_mean(grad_q1_vals, 50), label="rolling(50)", linewidth=1.5)
        plt.xlabel("update_step")
        plt.ylabel("grad norm (Q1)")
        plt.title("Q1 gradient norm over training")
        plt.legend()
        p = out_dir / "grad_q1_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

    if probes:
        probe_us = [p.update_step for p in probes]
        
        # Probe trans mean with rolling(50)
        probe_trans_vals = [p.trans_mean for p in probes]
        plt.figure(figsize=(10, 4))
        plt.plot(probe_us, probe_trans_vals, alpha=0.3, linewidth=0.5, label="raw")
        if len(probe_trans_vals) >= 50:
            plt.plot(probe_us, _rolling_mean(probe_trans_vals, 50), label="rolling(50)", linewidth=2)
        plt.axhline(SATURATION_MIN_FLOOR, color='orange', linestyle='--', alpha=0.5)
        plt.axhline(-SATURATION_MIN_FLOOR, color='orange', linestyle='--', alpha=0.5)
        plt.xlabel("update_step")
        plt.ylabel("probe translation mean")
        plt.title("Probe translation mean over training")
        plt.legend()
        p = out_dir / "probe_trans_mean_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Probe trans std with rolling(50)
        probe_std_vals = [p.trans_std_mean for p in probes]
        plt.figure(figsize=(10, 4))
        plt.plot(probe_us, probe_std_vals, alpha=0.3, linewidth=0.5, label="raw")
        if len(probe_std_vals) >= 50:
            plt.plot(probe_us, _rolling_mean(probe_std_vals, 50), label="rolling(50)", linewidth=2)
        plt.xlabel("update_step")
        plt.ylabel("probe translation std")
        plt.title("Probe translation std over training")
        plt.legend()
        p = out_dir / "probe_trans_std_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

    if batches:
        batch_us = [b.update_step for b in batches]
        
        # Batch trans means with rolling(100)
        actor_trans = [b.actor_trans_mean for b in batches]
        taken_trans = [b.taken_trans_mean for b in batches]
        plt.figure(figsize=(10, 4))
        plt.plot(batch_us, actor_trans, alpha=0.3, linewidth=0.5, label="actor raw")
        plt.plot(batch_us, taken_trans, alpha=0.3, linewidth=0.5, label="taken raw")
        if len(actor_trans) >= 100:
            plt.plot(batch_us, _rolling_mean(actor_trans, 100), label="actor rolling(100)", linewidth=2)
            plt.plot(batch_us, _rolling_mean(taken_trans, 100), label="taken rolling(100)", linewidth=2)
        plt.axhline(SATURATION_MIN_FLOOR, color='orange', linestyle='--', alpha=0.5)
        plt.axhline(-SATURATION_MIN_FLOOR, color='orange', linestyle='--', alpha=0.5)
        plt.xlabel("update_step")
        plt.ylabel("translation mean")
        plt.title("Batch translation means")
        plt.legend()
        p = out_dir / "batch_trans_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Batch done rate - REMOVED per plan

        # Batch reward mean with rolling(100) and rolling(1000)
        reward_vals = [b.reward_mean for b in batches]
        plt.figure(figsize=(10, 4))
        plt.plot(batch_us, reward_vals, alpha=0.3, linewidth=0.5, label="raw")
        if len(reward_vals) >= 100:
            plt.plot(batch_us, _rolling_mean(reward_vals, 100), label="rolling(100)", linewidth=1.5)
        if len(reward_vals) >= 1000:
            plt.plot(batch_us, _rolling_mean(reward_vals, 1000), label="rolling(1000)", linewidth=2)
        plt.xlabel("update_step")
        plt.ylabel("reward mean")
        plt.title("Batch reward mean over training")
        plt.legend()
        p = out_dir / "batch_reward_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Critic preference (actor vs taken) - original
        q_pref = [b.min_q_actor_minus_taken for b in batches]
        plt.figure(figsize=(10, 4))
        plt.plot(batch_us, q_pref, linewidth=0.5)
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel("update_step")
        plt.ylabel("min_q_actor - min_q_taken")
        plt.title("Critic preference: actor vs taken actions")
        p = out_dir / "batch_q_preference_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # NEW: Critic preference (zoomed) with rolling(100) and rolling(1000)
        plt.figure(figsize=(10, 4))
        plt.plot(batch_us, q_pref, alpha=0.3, linewidth=0.5, label="raw")
        if len(q_pref) >= 100:
            plt.plot(batch_us, _rolling_mean(q_pref, 100), label="rolling(100)", linewidth=1.5)
        if len(q_pref) >= 1000:
            plt.plot(batch_us, _rolling_mean(q_pref, 1000), label="rolling(1000)", linewidth=2)
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.ylim(-0.05, 0.05)
        plt.xlabel("update_step")
        plt.ylabel("min_q_actor - min_q_taken")
        plt.title("Critic preference: actor vs taken (zoomed)")
        plt.legend()
        p = out_dir / "batch_q_preference_zoomed_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

    if episodes:
        # Perf #6/#8: Use cached sorted episodes and insertion rolling median
        eps_sorted = (summary.get("_cached_eps_sorted") if summary else None) or sorted(episodes, key=lambda e: (e.end_timestamp or datetime.min, e.episode))
        x = list(range(len(eps_sorted)))

        # Insertion depth plot with green dashed line at 900
        plt.figure(figsize=(10, 4))
        ins_vals = [e.max_insertion_1 for e in eps_sorted]
        plt.plot(x, ins_vals, alpha=0.5, label="raw")
        if len(eps_sorted) >= 20:
            ins_roll = (summary.get("_cached_ins_roll") if summary else None) or _rolling_median(ins_vals, window=20)
            plt.plot(x, ins_roll, label="rolling median (20)", linewidth=2)
        plt.axhline(INSERTION_COLLAPSE_THRESHOLD_MM, color='red', linestyle='--', 
                   label=f'collapse threshold ({INSERTION_COLLAPSE_THRESHOLD_MM}mm)')
        plt.axhline(900, color='green', linestyle='--', linewidth=2, label='target (900mm)')
        plt.xlabel("episode index (time-sorted)")
        plt.ylabel("max insertion (mm)")
        plt.title("Insertion depth over episodes")
        plt.legend(loc='center left', bbox_to_anchor=(0.02, 0.5))
        p = out_dir / "insertion_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Episode reward plot with rolling(20) and green dashed line at 0
        rewards = [e.total_reward for e in eps_sorted]
        plt.figure(figsize=(10, 4))
        plt.plot(x, rewards, alpha=0.3, label="raw")
        if len(eps_sorted) >= 20:
            plt.plot(x, _rolling_mean(rewards, 20), label="rolling(20)", linewidth=2)
        plt.axhline(0, color='green', linestyle='--', linewidth=2, label="y=0")
        plt.xlabel("episode index (time-sorted)")
        plt.ylabel("episode reward")
        plt.title("Episode rewards over training")
        plt.legend()
        p = out_dir / "episode_rewards_v4.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # NEW: Translation speed per episode
        trans_speeds = []
        for e in eps_sorted:
            speed = (e.total_delta_ins_1 + e.total_delta_ins_2) / max(e.steps, 1)
            trans_speeds.append(speed)
        
        if any(s != 0 for s in trans_speeds):
            plt.figure(figsize=(10, 4))
            plt.plot(x, trans_speeds, alpha=0.5, label="raw")
            if len(trans_speeds) >= 20:
                plt.plot(x, _rolling_mean(trans_speeds, 20), label="rolling(20)", linewidth=2)
            plt.xlabel("episode index (time-sorted)")
            plt.ylabel("translation speed (mm/step)")
            plt.title("Average translation speed per episode")
            plt.legend()
            p = out_dir / "translation_speed_v4.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            paths.append(p)

        # NEW: Stuck instances per episode
        stuck_counts = [e.stuck_steps for e in eps_sorted]
        if any(sc > 0 for sc in stuck_counts):
            plt.figure(figsize=(10, 4))
            plt.plot(x, stuck_counts, alpha=0.5, label="raw")
            if len(stuck_counts) >= 20:
                plt.plot(x, _rolling_mean(stuck_counts, 20), label="rolling(20)", linewidth=2)
            plt.xlabel("episode index (time-sorted)")
            plt.ylabel("stuck steps")
            plt.title("Stuck instances per episode")
            plt.legend()
            p = out_dir / "stuck_instances_v4.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            paths.append(p)

        # Stuck fraction plot (if stuck data available)
        stuck_fractions = [e.stuck_fraction for e in eps_sorted]
        if any(sf > 0 for sf in stuck_fractions):
            plt.figure(figsize=(10, 4))
            plt.plot(x, stuck_fractions, alpha=0.5, label="raw")
            if len(eps_sorted) >= 20:
                stuck_roll = _rolling_mean(stuck_fractions, window=20)
                plt.plot(x, stuck_roll, label="rolling mean (20)", linewidth=2)
            plt.axhline(STUCK_FRACTION_THRESHOLD, color='red', linestyle='--', 
                       label=f'threshold ({STUCK_FRACTION_THRESHOLD:.0%})')
            plt.xlabel("episode index (time-sorted)")
            plt.ylabel("stuck fraction")
            plt.title("Episode stuck fraction (push cmd + no motion)")
            plt.legend()
            p = out_dir / "stuck_fraction_v4.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            paths.append(p)

            # Max stuck run plot
            plt.figure(figsize=(10, 4))
            max_runs = [e.max_stuck_run for e in eps_sorted]
            plt.plot(x, max_runs, alpha=0.5)
            if len(eps_sorted) >= 20:
                runs_roll = _rolling_mean(max_runs, window=20)
                plt.plot(x, runs_roll, label="rolling mean (20)", linewidth=2)
            plt.axhline(STUCK_MIN_CONSEC_STEPS, color='orange', linestyle='--', 
                       label=f'min run threshold ({STUCK_MIN_CONSEC_STEPS})')
            plt.xlabel("episode index (time-sorted)")
            plt.ylabel("max consecutive stuck steps")
            plt.title("Longest stuck run per episode")
            plt.legend()
            p = out_dir / "stuck_max_run_v4.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            paths.append(p)

    return paths

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Enhanced collapse forensics tool (v6)")
    ap.add_argument("--run-dir", type=str, help="Training run directory.")
    ap.add_argument("--name", type=str, help="Experiment name to search for.")
    ap.add_argument("--results-base", type=str, 
                   default="training_scripts/results/eve_paper/neurovascular/full/mesh_ben")
    ap.add_argument("--alpha-threshold", type=float, default=1e-2)
    ap.add_argument("--sustain", type=int, default=50)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    # Resolve run directory
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    elif args.name:
        base_dir = Path(args.results_base).expanduser().resolve()
        if not base_dir.exists():
            raise SystemExit(f"Results base dir does not exist: {base_dir}")
        matching = [d for d in base_dir.iterdir() if d.is_dir() and args.name in d.name]
        if not matching:
            raise SystemExit(f"No folder found matching name '{args.name}'")
        run_dir = max(matching, key=lambda p: p.stat().st_mtime)
        print(f"Found: {run_dir}")
    else:
        raise SystemExit("Must specify --run-dir or --name")
    
    if not run_dir.exists():
        raise SystemExit(f"Run dir does not exist: {run_dir}")

    # Load data
    losses_path = _glob_best(run_dir, ["diagnostics/csv/losses_*.csv", "**/losses_*.csv"])
    probe_path = _glob_best(run_dir, ["diagnostics/csv/probe_values_*.jsonl", "**/probe_values_*.jsonl"])
    batch_path = _glob_best(run_dir, ["diagnostics/csv/batch_samples_*.jsonl", "**/batch_samples_*.jsonl"])
    worker_logs = _glob_all(run_dir, ["logs_subprocesses/worker_*.log", "**/worker_*.log"])

    losses = load_losses_csv(losses_path) if losses_path else []
    probes = load_probe_jsonl(probe_path) if probe_path else []
    batches = load_batch_jsonl(batch_path) if batch_path else []
    
    # Load worker episodes (with stuck detection data)
    worker_episodes = load_worker_logs_v3(worker_logs) if worker_logs else []
    
    # Load episode summaries (preferred source for exact explore_step/update_step)
    episode_summaries = load_episode_summaries(run_dir)
    if episode_summaries:
        print(f"  [INFO] Loaded {len(episode_summaries)} episode summaries from Runner")
    
    # Merge episode summaries with worker logs (summaries provide exact step counters)
    episodes = merge_episode_summaries_with_worker_logs(episode_summaries, worker_episodes)

    # Perf #5: Build interpolator once, pass to both correlation and detection
    interpolator = build_timestamp_interpolator(losses) if losses else (None, None, False)

    # Correlate episodes to explore_step using wall_time matching (if available)
    episodes = correlate_episodes_to_explore_step(episodes, losses, episode_summaries, interpolator=interpolator)

    # Perf #6: Sort episodes once by timestamp, reuse everywhere
    eps_time_sorted = sorted(episodes, key=lambda e: (e.end_timestamp or datetime.min, e.episode))

    snapshots = load_policy_snapshots(run_dir)

    # Detect events
    events, summary = detect_events(
        losses=losses,
        probes=probes,
        batches=batches,
        episodes=episodes,
        alpha_threshold=args.alpha_threshold,
        sustain=args.sustain,
        interpolator=interpolator,
        eps_time_sorted=eps_time_sorted,
    )

    # Write report
    analysis_dir = run_dir / "diagnostics" / "analysis"
    md_path = write_report(
        out_dir=analysis_dir,
        run_dir=run_dir,
        losses_path=losses_path,
        probe_path=probe_path,
        batch_path=batch_path,
        n_worker_logs=len(worker_logs),
        losses=losses,
        probes=probes,
        batches=batches,
        episodes=episodes,
        snapshots=snapshots,
        events=events,
        summary=summary,
    )

    # Generate plots
    plot_paths: List[Path] = []
    if args.plot:
        plot_paths = make_plots(analysis_dir, losses, probes, batches, episodes, events, summary=summary)

    # Console summary
    print("=" * 90)
    print("Collapse Forensics Summary (v6)")
    print("=" * 90)
    print(f"run_dir: {run_dir}")
    print(f"losses: {losses_path or 'NOT FOUND'} (n={len(losses)})")
    print(f"probes: {probe_path or 'NOT FOUND'} (n={len(probes)})")
    print(f"batch_samples: {batch_path or 'NOT FOUND'} (n={len(batches)})")
    print(f"episodes: {len(episodes)} (with timestamps)")
    print(f"snapshots: {len(snapshots)}")
    print("")
    print(f"classification: {summary.get('classification')} ({summary.get('rationale')})")
    print(f"insertion_collapsed: {summary.get('insertion_collapsed', False)}")
    if events:
        print(f"\nEvents detected ({len(events)}):")
        for i, ev in enumerate(events[:10], 1):
            sev = {"critical": "[CRIT]", "warning": "[WARN]", "info": "[INFO]"}.get(ev.severity, "[----]")
            print(f"  {i}. {sev} {ev.name} @ update={ev.update_step} (conf={ev.confidence:.2f})")
        if len(events) > 10:
            print(f"  ... and {len(events)-10} more")
    else:
        print("earliest_event: none")
    print("")
    print(f"report: {md_path}")
    if plot_paths:
        print(f"plots: {len(plot_paths)} files in {analysis_dir}")
    print("=" * 90)

if __name__ == "__main__":
    main()
