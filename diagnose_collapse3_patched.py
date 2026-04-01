#!/usr/bin/env python3
"""
diagnose_collapse3.py

Enhanced "collapse forensics" tool for stEVE / eve_rl SAC diagnostics.

Version 3 improvements:
- Timestamp-based correlation between worker logs and diagnostics
- Fixed overly-sensitive detection thresholds (absolute bounds)
- Event severity and confidence scores
- insertion_collapse event detection
- freeze detection (mean near 0, std low)
- reward_decline and done_rate events
- Gating for replay_distribution_shift events

Usage
-----
python diagnose_collapse3.py --run-dir /path/to/run_dir
python diagnose_collapse3.py --name diag_debug_test5

Optional:
  --alpha-threshold 0.01
  --sustain 50
  --plot   (enable plots if matplotlib is installed)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from dataclasses import dataclass, field, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------
# Constants for detection thresholds
# -----------------------------

# Saturation: Actor mean should be clearly biased, not just statistically different
# For actions normalized to [-1, 1], "saturation" means near the bounds
SATURATION_ABSOLUTE_THRESHOLD = 0.5  # Actor mean > 0.5 = clearly biased
SATURATION_MIN_THRESHOLD = 0.3       # At minimum, mean must exceed 0.3

# Std collapse: Require meaningful drop, not statistical noise
STD_COLLAPSE_FRACTION = 0.5  # Std must drop to 50% of baseline (i.e., 50% reduction)

# Freeze detection
FREEZE_MEAN_EPSILON = 0.05  # Mean must be within +/-0.05
FREEZE_STD_MAX = 0.1        # Std must be below 0.1

# Insertion collapse
INSERTION_COLLAPSE_THRESHOLD_MM = 200.0  # Rolling median below this = collapse

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
    m = WALL_TIME_RE.search(line) if 'WALL_TIME_RE' in globals() else None
    if not m:
        return None
    wt = _to_float(m.group(1))
    return None if math.isnan(wt) else wt

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
    q = []
    for v in values:
        q.append(v)
        acc += v
        if len(q) > window:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out

def _rolling_median(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values[:]
    out = []
    q = []
    for v in values:
        q.append(v)
        if len(q) > window:
            q.pop(0)
        out.append(statistics.median(q))
    return out

def _rolling_std(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return [0.0 for _ in values]
    out = []
    q = []
    for v in values:
        q.append(v)
        if len(q) > window:
            q.pop(0)
        if len(q) < 2:
            out.append(0.0)
        else:
            out.append(statistics.pstdev(q))
    return out

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
    candidates.sort(key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
    return candidates[0]

def _glob_all(root: Path, patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(root.glob(pat))
    out = [p for p in out if p.is_file()]
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

def _compute_confidence(value: float, threshold: float, baseline: float) -> float:
    """Compute confidence score (0-1) based on how far past threshold."""
    if math.isnan(value) or math.isnan(threshold) or math.isnan(baseline):
        return 0.0
    if baseline == threshold:
        return 1.0 if value >= threshold else 0.0
    # Linear scaling from threshold to 2x the deviation
    deviation = abs(threshold - baseline)
    excess = abs(value - threshold)
    conf = min(1.0, excess / max(deviation, 1e-6))
    return conf

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

def load_losses_csv(path: Path) -> List[LossRow]:
    rows: List[LossRow] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(LossRow(
                update_step=_to_int(r.get("update_step")),
                explore_step=_to_int(r.get("explore_step")),
                timestamp=None,  # Could be added if we embed timestamps in CSV
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
            ))
    rows.sort(key=lambda x: x.update_step)
    return rows

@dataclass
class ProbeRow:
    update_step: int
    explore_step: Optional[int]
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
            samples = entry.get("samples", [])
            if not isinstance(samples, list) or len(samples) == 0:
                out.append(BatchRow(
                    update_step=u, explore_step=e, n_samples=0,
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
    # Correlated explore_step (estimated via timestamp or episode_summary.jsonl)
    explore_step_estimate: Optional[int] = None
    # Correlated update_step (estimated via episode_summary.jsonl when available)
    update_step_estimate: Optional[int] = None
    # Epoch wall_time from EPISODE_END line (used for robust ID stitching)
    end_wall_time: Optional[float] = None
    # NEW: Enhanced fields from updated env.py (may be None for old logs)
    term_reason: Optional[str] = None  # target_reached, max_steps, truncated, none
    total_delta_ins_1: float = 0.0     # Sum of delta insertions for device 1
    total_delta_ins_2: float = 0.0     # Sum of delta insertions for device 2
    actions: List[List[float]] = field(default_factory=list)  # Actions taken

# Regex to parse STEP lines with full timestamp
# OLD format: STEP | ep=X | ep_step=Y | global=Z | reward=R | cum_reward=CR | ... | term=T | trunc=TR | inserted=[A,B]
# NEW format: STEP | ep=X | ep_step=Y | global=Z | action=[a1,a2] | reward=R | cum_reward=CR | ... | term=T | trunc=TR | term_reason=REASON | inserted=[A,B] | delta_ins=[dA,dB]

# New format regex (with action, term_reason, delta_ins)
STEP_RE_NEW = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"STEP\s*\|\s*ep=(\d+)\s*\|\s*ep_step=(\d+)\s*\|\s*global=(\d+)\s*\|"
    r"\s*(?:action|cmd_action)=\[([-\d.,]+)\]\s*\|.*?"
    r"reward=([-\d.]+).*?cum_reward=([-\d.]+).*?"
    r"term=(\w+)\s*\|\s*trunc=(\w+)\s*\|\s*term_reason=(\w+).*?"
    r"inserted=\[([\d.]+),([\d.]+)\].*?"
    r"delta_ins=\[([-\d.]+),([-\d.]+)\]"
)

# Old format regex (fallback for existing logs)
STEP_RE_FULL = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"STEP\s*\|\s*ep=(\d+)\s*\|\s*ep_step=(\d+)\s*\|\s*global=(\d+)\s*\|.*?"
    r"reward=([-\d.]+).*?cum_reward=([-\d.]+).*?"
    r"term=(\w+)\s*\|\s*trunc=(\w+).*?inserted=\[([\d.]+),([\d.]+)\]"
)

EP_END_RE_FULL = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"EPISODE_END\s*\|\s*ep=(\d+)\s*\|\s*steps=(\d+)\s*\|\s*total_reward=([-\d.]+)"
)

WALL_TIME_RE = re.compile(r"wall_time=([0-9.]+)")

EP_START_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"EPISODE_START\s*\|\s*ep=(\d+)"
)

def load_worker_logs_v3(worker_log_paths: List[Path]) -> List[EpisodeRowV3]:
    """Load worker logs with timestamps for correlation."""
    # Aggregate per (worker, ep)
    data: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for p in worker_log_paths:
        try:
            worker = int(p.stem.split("_")[1])
        except Exception:
            worker = -1

        with p.open("r", errors="replace") as f:
            for line in f:
                # Try EPISODE_START
                m_start = EP_START_RE.search(line)
                if m_start:
                    ts = _parse_timestamp(m_start.group(1))
                    ep = int(m_start.group(2))
                    key = (worker, ep)
                    if key not in data:
                        data[key] = dict(
                            worker=worker, episode=ep,
                            start_timestamp=ts, end_timestamp=None,
                            total_reward=0.0,
                            max_insertion_1=0.0, max_insertion_2=0.0,
                            steps=0, terminated=False, truncated=False
                        )
                    else:
                        data[key]["start_timestamp"] = ts
                    continue

                # Try STEP line - NEW format first (with action, term_reason, delta_ins)
                m_new = STEP_RE_NEW.search(line)
                if m_new:
                    ts = _parse_timestamp(m_new.group(1))
                    ep = int(m_new.group(2))
                    ep_step = int(m_new.group(3))
                    # Parse action array
                    action_str = m_new.group(5)
                    action = [_to_float(a.strip()) for a in action_str.split(",")]
                    cum_reward = _to_float(m_new.group(7))
                    term = m_new.group(8).lower() == "true"
                    trunc = m_new.group(9).lower() == "true"
                    term_reason = m_new.group(10)  # NEW: termination reason
                    ins1 = _to_float(m_new.group(11))
                    ins2 = _to_float(m_new.group(12))
                    delta1 = _to_float(m_new.group(13))  # NEW: delta insertion
                    delta2 = _to_float(m_new.group(14))
                    
                    key = (worker, ep)
                    if key not in data:
                        data[key] = dict(
                            worker=worker, episode=ep,
                            start_timestamp=None, end_timestamp=ts,
                            total_reward=cum_reward,
                            max_insertion_1=ins1, max_insertion_2=ins2,
                            steps=ep_step, terminated=term, truncated=trunc,
                            term_reason=term_reason,
                            total_delta_ins_1=delta1, total_delta_ins_2=delta2,
                            actions=[action],
                        )
                    else:
                        d = data[key]
                        d["end_timestamp"] = ts
                        d["total_reward"] = cum_reward
                        d["max_insertion_1"] = max(d.get("max_insertion_1", 0.0) or 0.0, ins1)
                        d["max_insertion_2"] = max(d.get("max_insertion_2", 0.0) or 0.0, ins2)
                        d["steps"] = max(d.get("steps", 0) or 0, ep_step)
                        d["total_delta_ins_1"] = d.get("total_delta_ins_1", 0.0) + delta1
                        d["total_delta_ins_2"] = d.get("total_delta_ins_2", 0.0) + delta2
                        if "actions" not in d:
                            d["actions"] = []
                        d["actions"].append(action)
                        if term:
                            d["terminated"] = True
                            d["term_reason"] = term_reason
                        if trunc:
                            d["truncated"] = True
                            d["term_reason"] = term_reason
                    continue
                
                # Try STEP line - OLD format (fallback for existing logs)
                m = STEP_RE_FULL.search(line)
                if m:
                    ts = _parse_timestamp(m.group(1))
                    ep = int(m.group(2))
                    ep_step = int(m.group(3))
                    cum_reward = _to_float(m.group(6))
                    term = m.group(7).lower() == "true"
                    trunc = m.group(8).lower() == "true"
                    ins1 = _to_float(m.group(9))
                    ins2 = _to_float(m.group(10))
                    
                    key = (worker, ep)
                    if key not in data:
                        data[key] = dict(
                            worker=worker, episode=ep,
                            start_timestamp=None, end_timestamp=ts,
                            total_reward=cum_reward,
                            max_insertion_1=ins1, max_insertion_2=ins2,
                            steps=ep_step, terminated=term, truncated=trunc,
                            term_reason=None, total_delta_ins_1=0.0, total_delta_ins_2=0.0,
                            actions=[],
                        )
                    else:
                        d = data[key]
                        d["end_timestamp"] = ts
                        d["total_reward"] = cum_reward
                        d["max_insertion_1"] = max(d.get("max_insertion_1", 0.0) or 0.0, ins1)
                        d["max_insertion_2"] = max(d.get("max_insertion_2", 0.0) or 0.0, ins2)
                        d["steps"] = max(d.get("steps", 0) or 0, ep_step)
                        if term:
                            d["terminated"] = True
                        if trunc:
                            d["truncated"] = True
                    continue

                # Try EPISODE_END
                m_end = EP_END_RE_FULL.search(line)
                if m_end:
                    ts = _parse_timestamp(m_end.group(1))
                    ep = int(m_end.group(2))
                    steps = int(m_end.group(3))
                    total_reward = _to_float(m_end.group(4))
                    end_wall_time = _extract_wall_time(line)
                    
                    key = (worker, ep)
                    if key not in data:
                        data[key] = dict(
                            worker=worker, episode=ep,
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
        episodes.append(EpisodeRowV3(
            worker=d["worker"],
            episode=d["episode"],
            start_timestamp=d.get("start_timestamp"),
            end_timestamp=d.get("end_timestamp"),
            total_reward=_to_float(d.get("total_reward")),
            max_insertion_1=_to_float(d.get("max_insertion_1")),
            max_insertion_2=_to_float(d.get("max_insertion_2")),
            steps=_to_int(d.get("steps")),
            terminated=d.get("terminated", False),
            truncated=d.get("truncated", False),
            # NEW: Enhanced fields (may be None/empty for old logs)
            term_reason=d.get("term_reason"),
            total_delta_ins_1=_to_float(d.get("total_delta_ins_1", 0.0)),
            total_delta_ins_2=_to_float(d.get("total_delta_ins_2", 0.0)),
            actions=d.get("actions", []),
            end_wall_time=d.get("end_wall_time"),
        ))
    
    # Sort by end_timestamp if available, else by (episode, worker)
    episodes.sort(key=lambda e: (e.end_timestamp or datetime.min, e.episode, e.worker))
    return episodes

def correlate_episodes_to_explore_step(
    episodes: List[EpisodeRowV3],
    losses: List[LossRow],
    episode_summaries: Optional[List[EpisodeSummaryRow]] = None,
) -> List[EpisodeRowV3]:
    """
    Estimate explore_step for each episode using timestamp correlation.
    
    
    # Fast path: if episode_summary.jsonl exists and EPISODE_END includes wall_time, use it.
    if episode_summaries:
        # Greedy time-based assignment: match each episode end_wall_time to nearest unused summary wall_time.
        summaries = [s for s in episode_summaries if not math.isnan(s.wall_time)]
        summaries.sort(key=lambda s: s.wall_time)
        used = set()
        # Sort episodes by end_wall_time when available
        eps = sorted([e for e in episodes if e.end_wall_time is not None], key=lambda e: e.end_wall_time or 0.0)
        for e in eps:
            wt = e.end_wall_time
            if wt is None:
                continue
            # binary search nearest
            lo, hi = 0, len(summaries) - 1
            best_i = None
            best_dt = float("inf")
            while lo <= hi:
                mid = (lo + hi) // 2
                dt = abs(summaries[mid].wall_time - wt)
                if dt < best_dt and mid not in used:
                    best_dt, best_i = dt, mid
                if summaries[mid].wall_time < wt:
                    lo = mid + 1
                else:
                    hi = mid - 1
            # also check neighbors for unused
            for j in [best_i-1, best_i, best_i+1]:
                if j is None or j < 0 or j >= len(summaries) or j in used:
                    continue
                dt = abs(summaries[j].wall_time - wt)
                if dt < best_dt:
                    best_dt, best_i = dt, j
            # accept if within tolerance
            if best_i is not None and best_dt <= 10.0:
                s = summaries[best_i]
                e.explore_step_estimate = s.explore_step
                e.update_step_estimate = s.update_step
                used.add(best_i)
        return episodes

Strategy:
    - We know heatup ends around explore_step ~95000 (configurable)
    - Worker logs cover heatup phase (timestamps before training start)
    - Use losses CSV first row timestamp as training start reference
    - Interpolate explore_step based on episode timestamps relative to training timeline
    """
    if not episodes or not losses:
        return episodes
    
    # Get explore_step range from losses
    min_explore = losses[0].explore_step
    max_explore = losses[-1].explore_step
    
    # For episodes without timestamps, we can't correlate
    # For now, estimate based on episode index relative to total
    total_eps = len(episodes)
    
    for i, ep in enumerate(episodes):
        # Simple linear interpolation based on episode position
        # This assumes episodes are roughly uniformly distributed over time
        if ep.end_timestamp is None:
            # Use episode index as proxy
            frac = i / max(1, total_eps - 1)
            ep.explore_step_estimate = int(min_explore + frac * (max_explore - min_explore))
        else:
            # Could use timestamp interpolation if we had loss timestamps
            # For now, use episode index
            frac = i / max(1, total_eps - 1)
            ep.explore_step_estimate = int(min_explore + frac * (max_explore - min_explore))
    
    return episodes

# -----------------------------
# Snapshot metadata
# -----------------------------

@dataclass
class SnapshotRow:
    path: Path
    update_step: Optional[int]
    explore_step: Optional[int]


@dataclass
class EpisodeSummaryRow:
    wall_time: float
    explore_step: int
    update_step: int
    episode_id: int

def load_episode_summary_jsonl(path: Path) -> List[EpisodeSummaryRow]:
    rows: List[EpisodeSummaryRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                rows.append(EpisodeSummaryRow(
                    wall_time=_to_float(j.get("wall_time")),
                    explore_step=_to_int(j.get("explore_step")),
                    update_step=_to_int(j.get("update_step")),
                    episode_id=_to_int(j.get("episode_id")),
                ))
            except Exception:
                continue
    # sort by wall_time
    rows.sort(key=lambda r: r.wall_time)
    return rows

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
        conf = _compute_confidence(alpha[i_alpha], alpha_threshold, base_alpha)
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
        conf = _compute_confidence(ent_roll[i_ent], ent_drop_thresh, base_entropy)
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
        conf = _compute_confidence(gpol[i_gpol], 0.25 * base_gpol, base_gpol)
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
        
        # Event: Translation saturation (FIXED: use absolute threshold 0.5)
        # Actor mean must exceed 0.3 minimum, ideally 0.5
        sat_thresh = max(SATURATION_MIN_THRESHOLD, min(SATURATION_ABSOLUTE_THRESHOLD, 
                        abs(base_probe_trans) + max(30 * base_probe_trans_std, 0.3)))
        
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
                conf = _compute_confidence(probe_std_roll[i_std], std_collapse_thresh, base_probe_std)
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
    insertion_collapsed = False  # For gating
    
    if batches:
        b_u = [b.update_step for b in batches]
        b_e = [b.explore_step for b in batches]
        actor_trans = [b.actor_trans_mean for b in batches]
        reward_vals = [b.reward_mean for b in batches]
        done_vals = [b.done_rate for b in batches]
        
        actor_roll = _rolling_mean(actor_trans, window=max(5, sustain//3))
        reward_roll = _rolling_mean(reward_vals, window=max(5, sustain//3))
        done_roll = _rolling_mean(done_vals, window=max(5, sustain//3))
        
        # Baseline for batches
        bb_start = min(10, len(batches) // 10)
        bb_end = min(bb_start + 100, len(batches))
        base_actor = _safe_mean(actor_trans[bb_start:bb_end])
        base_actor_std = _safe_pstdev(actor_trans[bb_start:bb_end])
        base_reward = _safe_mean(reward_vals[bb_start:bb_end])
        base_reward_std = _safe_pstdev(reward_vals[bb_start:bb_end])
        base_done = _safe_mean(done_vals[bb_start:bb_end])
        base_done_std = _safe_pstdev(done_vals[bb_start:bb_end])
        
        # Event: Actor saturation (FIXED threshold)
        sat_thresh = max(SATURATION_MIN_THRESHOLD, min(SATURATION_ABSOLUTE_THRESHOLD,
                        abs(base_actor) + max(30 * base_actor_std, 0.3)))
        
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

    # =========================================
    # Episode events (insertion collapse)
    # =========================================
    if episodes:
        # Sort by timestamp
        eps_sorted = sorted(episodes, key=lambda e: (e.end_timestamp or datetime.min, e.episode))
        insertions = [e.max_insertion_1 for e in eps_sorted]
        
        if len(insertions) >= 20:
            ins_roll = _rolling_median(insertions, window=20)
            
            # Baseline from first 20% of episodes
            ins_base_end = min(len(insertions) // 5, 50)
            base_insertion = _safe_median(insertions[:max(10, ins_base_end)])
            
            # Event: Insertion collapse
            pred_ins_collapse = [ins < INSERTION_COLLAPSE_THRESHOLD_MM for ins in ins_roll]
            i_ins = _first_sustained(pred_ins_collapse, sustain=20)
            if i_ins is not None:
                insertion_collapsed = True
                ep = eps_sorted[i_ins]
                events.append(Event(
                    name="insertion_collapse",
                    update_step=None,
                    explore_step=ep.explore_step_estimate,
                    timestamp=ep.end_timestamp,
                    details=f"rolling median insertion fell below {INSERTION_COLLAPSE_THRESHOLD_MM}mm (value={ins_roll[i_ins]:.1f}mm, baseline={base_insertion:.1f}mm).",
                    severity="critical",
                    confidence=0.9,
                ))

    # =========================================
    # Order events by update_step
    # =========================================
    def _event_key(ev: Event):
        return (ev.update_step if ev.update_step is not None else 10**18,
                ev.explore_step if ev.explore_step is not None else 10**18,
                ev.name)
    events.sort(key=_event_key)

    # =========================================
    # Classification
    # =========================================
    policy_names = {
        "alpha_shutdown", "entropy_drop", "policy_grad_collapse",
        "probe_translation_saturated", "probe_std_collapse", "probe_freeze",
        "actor_saturated", "insertion_collapse",
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
            saturation_min=SATURATION_MIN_THRESHOLD,
            saturation_abs=SATURATION_ABSOLUTE_THRESHOLD,
            std_collapse_fraction=STD_COLLAPSE_FRACTION,
            freeze_mean_eps=FREEZE_MEAN_EPSILON,
            freeze_std_max=FREEZE_STD_MAX,
            insertion_collapse_mm=INSERTION_COLLAPSE_THRESHOLD_MM,
        ),
        classification=classification,
        rationale=rationale,
        insertion_collapsed=insertion_collapsed,
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
    md_path = out_dir / "collapse_report_v3.md"

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
        eps_sorted = sorted(episodes, key=lambda e: (e.end_timestamp or datetime.min, e.episode))
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
    lines.append("# Collapse Forensics Report (v3)\n")
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
    lines.append(f"- Saturation: |mean| > {thresh.get('saturation_min', 0.3):.2f} (min) to {thresh.get('saturation_abs', 0.5):.2f} (absolute)")
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
# Plotting
# -----------------------------

def make_plots(out_dir: Path, losses: List[LossRow], probes: List[ProbeRow], 
               batches: List[BatchRow], episodes: List[EpisodeRowV3], events: List[Event]) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    def vline_events(ax, events_list):
        colors = {"critical": "red", "warning": "orange", "info": "blue"}
        for ev in events_list:
            if ev.update_step is None:
                continue
            ax.axvline(ev.update_step, linestyle="--", linewidth=1, 
                      color=colors.get(ev.severity, "gray"), alpha=0.7)

    if losses:
        us = [r.update_step for r in losses]

        # Alpha
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.alpha for r in losses])
        plt.xlabel("update_step")
        plt.ylabel("alpha")
        plt.title("Alpha over training")
        vline_events(plt.gca(), events)
        p = out_dir / "alpha_v3.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Entropy
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.entropy_proxy for r in losses])
        plt.xlabel("update_step")
        plt.ylabel("entropy_proxy")
        plt.title("Entropy proxy over training")
        vline_events(plt.gca(), events)
        p = out_dir / "entropy_v3.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Q means
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.q1_mean for r in losses], label="q1_mean", linewidth=0.5)
        plt.plot(us, [r.q2_mean for r in losses], label="q2_mean", linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("Q mean")
        plt.title("Critic Q means over training")
        plt.legend()
        vline_events(plt.gca(), events)
        p = out_dir / "q_means_v3.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Q losses
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.q1_loss for r in losses], label="q1_loss", linewidth=0.5)
        plt.plot(us, [r.q2_loss for r in losses], label="q2_loss", linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("Q loss")
        plt.title("Critic Q losses over training")
        plt.legend()
        vline_events(plt.gca(), events)
        p = out_dir / "q_losses_v3.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Grad norms
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.grad_norm_policy for r in losses], label="grad_policy", linewidth=0.5)
        plt.plot(us, [r.grad_norm_q1 for r in losses], label="grad_q1", linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("grad norm")
        plt.title("Gradient norms over training")
        plt.legend()
        vline_events(plt.gca(), events)
        p = out_dir / "grad_norms_v3.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

    if probes:
        plt.figure(figsize=(10, 4))
        plt.plot([p.update_step for p in probes], [p.trans_mean for p in probes], label="mean")
        plt.axhline(SATURATION_MIN_THRESHOLD, color='orange', linestyle='--', label=f'sat_min={SATURATION_MIN_THRESHOLD}')
        plt.axhline(-SATURATION_MIN_THRESHOLD, color='orange', linestyle='--')
        plt.xlabel("update_step")
        plt.ylabel("probe translation mean")
        plt.title("Probe translation mean (with saturation thresholds)")
        plt.legend()
        vline_events(plt.gca(), events)
        p = out_dir / "probe_trans_mean_v3.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        plt.figure(figsize=(10, 4))
        plt.plot([p.update_step for p in probes], [p.trans_std_mean for p in probes])
        plt.xlabel("update_step")
        plt.ylabel("probe translation std")
        plt.title("Probe translation std over training")
        vline_events(plt.gca(), events)
        p = out_dir / "probe_trans_std_v3.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

    if batches:
        plt.figure(figsize=(10, 4))
        plt.plot([b.update_step for b in batches], [b.actor_trans_mean for b in batches], label="actor_det")
        plt.plot([b.update_step for b in batches], [b.taken_trans_mean for b in batches], label="taken", alpha=0.7)
        plt.axhline(SATURATION_MIN_THRESHOLD, color='orange', linestyle='--', alpha=0.5)
        plt.axhline(-SATURATION_MIN_THRESHOLD, color='orange', linestyle='--', alpha=0.5)
        plt.xlabel("update_step")
        plt.ylabel("translation mean")
        plt.title("Batch translation means")
        plt.legend()
        vline_events(plt.gca(), events)
        p = out_dir / "batch_trans_v3.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        plt.figure(figsize=(10, 4))
        plt.plot([b.update_step for b in batches], [b.done_rate for b in batches])
        plt.xlabel("update_step")
        plt.ylabel("done_rate")
        plt.title("Batch done rate over training")
        vline_events(plt.gca(), events)
        p = out_dir / "batch_done_rate_v3.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Batch reward mean
        plt.figure(figsize=(10, 4))
        plt.plot([b.update_step for b in batches], [b.reward_mean for b in batches], linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("reward mean")
        plt.title("Batch reward mean over training")
        vline_events(plt.gca(), events)
        p = out_dir / "batch_reward_v3.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Critic preference (actor vs taken)
        plt.figure(figsize=(10, 4))
        plt.plot([b.update_step for b in batches], [b.min_q_actor_minus_taken for b in batches], linewidth=0.5)
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel("update_step")
        plt.ylabel("min_q_actor - min_q_taken")
        plt.title("Critic preference: actor vs taken actions")
        vline_events(plt.gca(), events)
        p = out_dir / "batch_q_preference_v3.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

    if episodes:
        eps_sorted = sorted(episodes, key=lambda e: (e.end_timestamp or datetime.min, e.episode))
        x = range(len(eps_sorted))
        
        # Insertion depth plot
        plt.figure(figsize=(10, 4))
        plt.plot(x, [e.max_insertion_1 for e in eps_sorted], alpha=0.5, label="raw")
        if len(eps_sorted) >= 20:
            ins_roll = _rolling_median([e.max_insertion_1 for e in eps_sorted], window=20)
            plt.plot(x, ins_roll, label="rolling median (20)", linewidth=2)
        plt.axhline(INSERTION_COLLAPSE_THRESHOLD_MM, color='red', linestyle='--', 
                   label=f'collapse threshold ({INSERTION_COLLAPSE_THRESHOLD_MM}mm)')
        plt.xlabel("episode index (time-sorted)")
        plt.ylabel("max insertion (mm)")
        plt.title("Insertion depth over episodes")
        plt.legend()
        p = out_dir / "insertion_v3.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # Episode reward plot
        plt.figure(figsize=(10, 4))
        rewards = [e.total_reward for e in eps_sorted]
        plt.plot(x, rewards, alpha=0.5, label="raw")
        if len(eps_sorted) >= 20:
            rew_roll = _rolling_median(rewards, window=20)
            plt.plot(x, rew_roll, label="rolling median (20)", linewidth=2)
        plt.xlabel("episode index (time-sorted)")
        plt.ylabel("episode reward")
        plt.title("Episode rewards over training")
        plt.legend()
        p = out_dir / "episode_rewards_v3.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

    return paths

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Enhanced collapse forensics tool (v3)")
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
    episode_summary_path = _glob_best(run_dir, ["diagnostics/csv/episode_summary.jsonl", "**/episode_summary.jsonl"])

    losses = load_losses_csv(losses_path) if losses_path else []
    probes = load_probe_jsonl(probe_path) if probe_path else []
    batches = load_batch_jsonl(batch_path) if batch_path else []
    episodes = load_worker_logs_v3(worker_logs) if worker_logs else []
    episode_summaries = load_episode_summary_jsonl(episode_summary_path) if episode_summary_path else None
    snapshots = load_policy_snapshots(run_dir)

    # Correlate episodes to explore_step
    episodes = correlate_episodes_to_explore_step(episodes, losses, episode_summaries)

    # Detect events
    events, summary = detect_events(
        losses=losses,
        probes=probes,
        batches=batches,
        episodes=episodes,
        alpha_threshold=args.alpha_threshold,
        sustain=args.sustain,
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
        plot_paths = make_plots(analysis_dir, losses, probes, batches, episodes, events)

    # Console summary
    print("=" * 90)
    print("Collapse Forensics Summary (v3)")
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
