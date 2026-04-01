#!/usr/bin/env python3
"""
diagnose_collapse2.py

Unified "collapse forensics" tool for stEVE / eve_rl SAC diagnostics.
(v2: fixed event detection with noise-adaptive thresholds, wider windows,
     proper baseline comparison, and saturation detection.)

What it does
------------
Given a training run directory (the folder that contains diagnostics/ and logs_subprocesses/),
this script:

1) Loads trainer losses CSV (alpha, entropy_proxy, Q losses/means, grad norms, etc.).
2) Loads probe JSONL (fixed probe-state actor/critic monitoring).
3) Loads batch-sample JSONL (sampled transitions from training batches).
4) Loads worker env logs (episode rewards and insertion depths).
5) Loads policy snapshots metadata (policy_*.pt) and picks "last good" and "first bad".

Then it:
- aligns signals by update_step / explore_step,
- runs robust change-point / event detection heuristics,
- classifies likely failure mechanism (policy collapse vs critic instability),
- writes a Markdown report + optional plots.

v2 fixes over v1
-----------------
- Noise-adaptive thresholds: no more hard-coded zero thresholds for batch/probe events.
- Wider rolling windows: scaled to actual data noise (n_batch_samples=3 is very noisy).
- Proper baseline comparison: events fire on deviation from baseline, not absolute position.
- Saturation detection: detects policy saturating to extreme actions (positive or negative).
- Full sustain requirement: batch/probe events use full sustain, not sustain//2.
- Startup skip for all baselines: first 100 loss updates (or 5% of probe/batch) skipped.

Usage
-----
python diagnose_collapse2.py --run-dir /path/to/run_dir
python diagnose_collapse2.py --name diag_debug_test5

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

# -----------------------------
# Data loaders
# -----------------------------

@dataclass
class LossRow:
    update_step: int
    explore_step: int
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

            # Average across probe states
            q1_vals = []
            for q in q1_list:
                q1_vals.append(_extract_scalar_nested(q))
            q2_vals = []
            for q in q2_list:
                q2_vals.append(_extract_scalar_nested(q))

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

@dataclass
class EpisodeRow:
    worker: int
    episode: int
    end_global_step: Optional[int]
    total_reward: float
    max_insertion_1: float
    max_insertion_2: float
    steps: int

STEP_RE = re.compile(
    r"STEP\s*\|\s*ep=(\d+)\s*\|\s*ep_step=(\d+)\s*\|\s*global=(\d+)\s*\|.*?cum_reward=([-\d.]+).*?inserted=\[([\d.]+),([\d.]+)\]"
)
EP_END_RE = re.compile(r"EPISODE_END\s*\|\s*ep=(\d+)\s*\|\s*steps=(\d+)\s*\|\s*total_reward=([-\d.]+)(?:.*?global=(\d+))?")

def load_worker_logs(worker_log_paths: List[Path]) -> List[EpisodeRow]:
    data: Dict[Tuple[int,int], Dict[str, Any]] = {}

    for p in worker_log_paths:
        try:
            worker = int(p.stem.split("_")[1])
        except Exception:
            worker = -1

        with p.open("r", errors="replace") as f:
            for line in f:
                m = STEP_RE.search(line)
                if m:
                    ep = int(m.group(1))
                    ep_step = int(m.group(2))
                    global_step = int(m.group(3))
                    cum_reward = _to_float(m.group(4))
                    ins1 = _to_float(m.group(5))
                    ins2 = _to_float(m.group(6))
                    key = (worker, ep)
                    if key not in data:
                        data[key] = dict(
                            worker=worker, episode=ep,
                            end_global_step=global_step,
                            total_reward=cum_reward,
                            max_insertion_1=ins1, max_insertion_2=ins2,
                            steps=ep_step
                        )
                    else:
                        d = data[key]
                        d["end_global_step"] = max(d.get("end_global_step", global_step) or 0, global_step)
                        d["total_reward"] = cum_reward
                        d["max_insertion_1"] = max(d.get("max_insertion_1", 0.0) or 0.0, ins1)
                        d["max_insertion_2"] = max(d.get("max_insertion_2", 0.0) or 0.0, ins2)
                        d["steps"] = max(d.get("steps", 0) or 0, ep_step)
                    continue

                m2 = EP_END_RE.search(line)
                if m2:
                    ep = int(m2.group(1))
                    steps = int(m2.group(2))
                    total_reward = _to_float(m2.group(3))
                    g = m2.group(4)
                    gstep = int(g) if g is not None else None
                    key = (worker, ep)
                    if key not in data:
                        data[key] = dict(
                            worker=worker, episode=ep,
                            end_global_step=gstep,
                            total_reward=total_reward,
                            max_insertion_1=0.0, max_insertion_2=0.0,
                            steps=steps
                        )
                    else:
                        d = data[key]
                        d["total_reward"] = total_reward
                        d["steps"] = max(d.get("steps", 0) or 0, steps)
                        if gstep is not None:
                            d["end_global_step"] = gstep
                    continue

    episodes: List[EpisodeRow] = []
    for d in data.values():
        episodes.append(EpisodeRow(
            worker=d["worker"],
            episode=d["episode"],
            end_global_step=d.get("end_global_step"),
            total_reward=_to_float(d.get("total_reward")),
            max_insertion_1=_to_float(d.get("max_insertion_1")),
            max_insertion_2=_to_float(d.get("max_insertion_2")),
            steps=_to_int(d.get("steps"))
        ))
    episodes.sort(key=lambda x: (x.episode, x.worker))
    return episodes

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
        import torch  # noqa: F401
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

    import torch  # type: ignore

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
# Baseline computation helper
# -----------------------------

def _compute_baseline_region(n: int, skip_frac: float = 0.05, base_frac: float = 0.20) -> Tuple[int, int]:
    """
    Compute a stable baseline region for a signal with n data points.
    Skips first skip_frac of data (startup noise), then uses the next base_frac.
    Returns (start_idx, end_idx).
    """
    skip = max(5, int(n * skip_frac))
    base_len = max(20, int(n * base_frac))
    start = min(skip, n - 1)
    end = min(start + base_len, n)
    if end - start < 10:
        # Fallback: use first half
        start = 0
        end = max(10, n // 2)
    return start, end

# -----------------------------
# Collapse detection (v2)
# -----------------------------

@dataclass
class Event:
    name: str
    update_step: Optional[int]
    explore_step: Optional[int]
    details: str

def detect_events(
    losses: List[LossRow],
    probes: List[ProbeRow],
    batches: List[BatchRow],
    alpha_threshold: float,
    sustain: int,
) -> Tuple[List[Event], Dict[str, Any]]:
    """
    Returns:
      events: ordered list of detected events (earliest-first)
      summary: metrics used for final classification
    """
    events: List[Event] = []

    if not losses:
        return events, {"classification": "insufficient_data"}

    # ---------------------------------------------------------------
    # Loss-based arrays
    # ---------------------------------------------------------------
    update_steps = [r.update_step for r in losses]
    explore_steps = [r.explore_step for r in losses]
    alpha = [r.alpha for r in losses]
    entropy = [r.entropy_proxy for r in losses]
    q1l = [r.q1_loss for r in losses]
    q2l = [r.q2_loss for r in losses]
    gq1 = [r.grad_norm_q1 for r in losses]
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
    base_gq1 = _safe_median(gq1[base_start:base_end])
    base_gpol = _safe_median(gpol[base_start:base_end])
    base_q1l = _safe_median(q1l[base_start:base_end])
    base_q2l = _safe_median(q2l[base_start:base_end])

    # ---------------------------------------------------------------
    # Loss-based events (these are dense: 1 row per update step)
    # ---------------------------------------------------------------

    # Event: alpha shutdown
    alpha_pred = [(a < alpha_threshold) for a in alpha]
    i_alpha = _first_sustained(alpha_pred, sustain=sustain)
    if i_alpha is not None:
        events.append(Event(
            name="alpha_shutdown",
            update_step=update_steps[i_alpha],
            explore_step=explore_steps[i_alpha],
            details=f"alpha fell below {alpha_threshold:g} for >= {sustain} updates (baseline~{base_alpha:.4g}).",
        ))

    # Event: entropy proxy collapse (relative to baseline)
    ent_roll = _rolling_mean(entropy, window=max(5, sustain // 2))
    base_entropy_std = _safe_pstdev(entropy[base_start:base_end])
    ent_drop_thresh = base_entropy - 2.0 * max(1e-6, base_entropy_std)
    ent_pred = [e < ent_drop_thresh for e in ent_roll]
    i_ent = _first_sustained(ent_pred, sustain=sustain)
    if i_ent is not None:
        events.append(Event(
            name="entropy_drop",
            update_step=update_steps[i_ent],
            explore_step=explore_steps[i_ent],
            details=f"entropy_proxy dropped below {ent_drop_thresh:.3g} (baseline~{base_entropy:.3g} +/- {base_entropy_std:.3g}).",
        ))

    # Event: gradient collapse (policy)
    gpol_pred = [(g < 0.25 * base_gpol) for g in gpol]
    i_gpol = _first_sustained(gpol_pred, sustain=sustain)
    if i_gpol is not None:
        events.append(Event(
            name="policy_grad_collapse",
            update_step=update_steps[i_gpol],
            explore_step=explore_steps[i_gpol],
            details=f"grad_norm_policy fell below 25% of baseline (baseline~{base_gpol:.3g}).",
        ))

    # Event: critic instability (loss spikes after startup)
    spike_factor = 8.0
    q_spike_pred = []
    for i, (ql1, ql2) in enumerate(zip(q1l, q2l)):
        if i < startup_skip:
            q_spike_pred.append(False)
        else:
            q_spike_pred.append((ql1 > spike_factor * base_q1l) or (ql2 > spike_factor * base_q2l))
    i_qspike = _first_sustained(q_spike_pred, sustain=max(5, sustain // 2))
    if i_qspike is not None:
        events.append(Event(
            name="critic_loss_spike",
            update_step=update_steps[i_qspike],
            explore_step=explore_steps[i_qspike],
            details=f"q_loss exceeded {spike_factor}x baseline (q1_base~{base_q1l:.3g}, q2_base~{base_q2l:.3g}) after startup.",
        ))

    # ---------------------------------------------------------------
    # Probe events (sparse: 1 row per 100 update steps typically)
    # Key fix: noise-adaptive thresholds, wider windows, full sustain
    # ---------------------------------------------------------------
    if probes:
        probe_u = [p.update_step for p in probes]
        probe_trans = [p.trans_mean for p in probes]
        probe_std = [p.trans_std_mean for p in probes]
        n_probes = len(probes)

        # Baseline region for probe data (skip startup, use next 20%)
        pb_start, pb_end = _compute_baseline_region(n_probes)

        base_probe_trans_mean = _safe_mean(probe_trans[pb_start:pb_end])
        base_probe_trans_std = _safe_pstdev(probe_trans[pb_start:pb_end])
        base_probe_std_mean = _safe_mean(probe_std[pb_start:pb_end])
        base_probe_std_std = _safe_pstdev(probe_std[pb_start:pb_end])

        # Wider rolling window: use sustain (50 points = 5000 update steps)
        # This properly smooths the per-entry noise
        probe_window = max(sustain, 30)
        probe_trans_roll = _rolling_mean(probe_trans, window=probe_window)
        probe_std_roll = _rolling_mean(probe_std, window=probe_window)

        # Saturation detection: |rolling_mean| departs from baseline by meaningful amount
        # Threshold = baseline_mean +/- max(3 * baseline_std, 0.5)
        # The 0.5 floor ensures we don't flag noise even if baseline_std is tiny
        sat_margin = max(3.0 * base_probe_trans_std, 0.5)
        sat_upper = base_probe_trans_mean + sat_margin
        sat_lower = base_probe_trans_mean - sat_margin

        pred_sat_pos = [t > sat_upper for t in probe_trans_roll]
        pred_sat_neg = [t < sat_lower for t in probe_trans_roll]

        i_sat_pos = _first_sustained(pred_sat_pos, sustain=sustain)
        if i_sat_pos is not None:
            val = probe_trans_roll[i_sat_pos]
            events.append(Event(
                name="probe_translation_saturated",
                update_step=probe_u[i_sat_pos],
                explore_step=probes[i_sat_pos].explore_step,
                details=(
                    f"probe translation saturated positive (max insert): "
                    f"rolling_mean={val:.3f} > threshold={sat_upper:.3f} "
                    f"(baseline={base_probe_trans_mean:.3f} +/- {base_probe_trans_std:.3f}, margin={sat_margin:.3f})."
                ),
            ))

        i_sat_neg = _first_sustained(pred_sat_neg, sustain=sustain)
        if i_sat_neg is not None:
            val = probe_trans_roll[i_sat_neg]
            events.append(Event(
                name="probe_translation_saturated",
                update_step=probe_u[i_sat_neg],
                explore_step=probes[i_sat_neg].explore_step,
                details=(
                    f"probe translation saturated negative (retract): "
                    f"rolling_mean={val:.3f} < threshold={sat_lower:.3f} "
                    f"(baseline={base_probe_trans_mean:.3f} +/- {base_probe_trans_std:.3f}, margin={sat_margin:.3f})."
                ),
            ))

        # Variance collapse: std drops below max(50% of baseline, or baseline - 3*std_of_std)
        if not math.isnan(base_probe_std_mean) and base_probe_std_mean > 0:
            std_collapse_thresh = max(
                0.5 * base_probe_std_mean,
                base_probe_std_mean - 3.0 * max(base_probe_std_std, 0.01)
            )
            pred_std_collapse = [s < std_collapse_thresh for s in probe_std_roll]
            i_std = _first_sustained(pred_std_collapse, sustain=sustain)
            if i_std is not None:
                events.append(Event(
                    name="probe_std_collapse",
                    update_step=probe_u[i_std],
                    explore_step=probes[i_std].explore_step,
                    details=(
                        f"probe translation std collapsed: "
                        f"rolling_std={probe_std_roll[i_std]:.4f} < threshold={std_collapse_thresh:.4f} "
                        f"(baseline std={base_probe_std_mean:.4f} +/- {base_probe_std_std:.4f})."
                    ),
                ))

    # ---------------------------------------------------------------
    # Batch events (sparse: 1 row per 100 update steps, n_samples=3)
    # Key fix: noise-adaptive thresholds, wider windows, full sustain
    # ---------------------------------------------------------------
    if batches:
        b_u = [b.update_step for b in batches]
        taken = [b.taken_trans_mean for b in batches]
        actor = [b.actor_trans_mean for b in batches]
        n_batches = len(batches)

        # Baseline region for batch data
        bb_start, bb_end = _compute_baseline_region(n_batches)

        base_actor_mean = _safe_mean(actor[bb_start:bb_end])
        base_actor_std = _safe_pstdev(actor[bb_start:bb_end])
        base_taken_mean = _safe_mean(taken[bb_start:bb_end])
        base_taken_std = _safe_pstdev(taken[bb_start:bb_end])

        # Wider rolling window (sustain = 50 points = 5000 update steps)
        batch_window = max(sustain, 30)
        taken_roll = _rolling_mean(taken, window=batch_window)
        actor_roll = _rolling_mean(actor, window=batch_window)

        # Actor saturation: actor output departs significantly from baseline
        # Use max(3 * baseline_std, 0.1) as minimum margin
        # 0.1 is the absolute floor because with n_samples=3, actor_std ~ 0.003
        # and we don't want 3*0.003 = 0.009 as a threshold
        actor_margin = max(3.0 * base_actor_std, 0.1)
        actor_sat_upper = base_actor_mean + actor_margin
        actor_sat_lower = base_actor_mean - actor_margin

        pred_actor_sat_pos = [a > actor_sat_upper for a in actor_roll]
        pred_actor_sat_neg = [a < actor_sat_lower for a in actor_roll]

        i_actor_pos = _first_sustained(pred_actor_sat_pos, sustain=sustain)
        if i_actor_pos is not None:
            val = actor_roll[i_actor_pos]
            events.append(Event(
                name="actor_saturated_positive",
                update_step=b_u[i_actor_pos],
                explore_step=batches[i_actor_pos].explore_step,
                details=(
                    f"actor translation on batch states saturated positive: "
                    f"rolling_mean={val:.4f} > threshold={actor_sat_upper:.4f} "
                    f"(baseline={base_actor_mean:.4f} +/- {base_actor_std:.4f})."
                ),
            ))

        i_actor_neg = _first_sustained(pred_actor_sat_neg, sustain=sustain)
        if i_actor_neg is not None:
            val = actor_roll[i_actor_neg]
            events.append(Event(
                name="actor_saturated_negative",
                update_step=b_u[i_actor_neg],
                explore_step=batches[i_actor_neg].explore_step,
                details=(
                    f"actor translation on batch states saturated negative: "
                    f"rolling_mean={val:.4f} < threshold={actor_sat_lower:.4f} "
                    f"(baseline={base_actor_mean:.4f} +/- {base_actor_std:.4f})."
                ),
            ))

        # Replay distribution shift: buffer actions drift meaningfully from baseline
        # Use larger margin because taken actions are much noisier (n_samples=3, raw values +/-0.8)
        taken_margin = max(3.0 * base_taken_std, 0.2)
        taken_shift_upper = base_taken_mean + taken_margin
        taken_shift_lower = base_taken_mean - taken_margin

        pred_taken_pos = [t > taken_shift_upper for t in taken_roll]
        pred_taken_neg = [t < taken_shift_lower for t in taken_roll]

        i_taken_pos = _first_sustained(pred_taken_pos, sustain=sustain)
        if i_taken_pos is not None:
            val = taken_roll[i_taken_pos]
            events.append(Event(
                name="replay_distribution_shift_positive",
                update_step=b_u[i_taken_pos],
                explore_step=batches[i_taken_pos].explore_step,
                details=(
                    f"replay buffer action distribution shifted positive: "
                    f"rolling_mean={val:.4f} > threshold={taken_shift_upper:.4f} "
                    f"(baseline={base_taken_mean:.4f} +/- {base_taken_std:.4f})."
                ),
            ))

        i_taken_neg = _first_sustained(pred_taken_neg, sustain=sustain)
        if i_taken_neg is not None:
            val = taken_roll[i_taken_neg]
            events.append(Event(
                name="replay_distribution_shift_negative",
                update_step=b_u[i_taken_neg],
                explore_step=batches[i_taken_neg].explore_step,
                details=(
                    f"replay buffer action distribution shifted negative: "
                    f"rolling_mean={val:.4f} < threshold={taken_shift_lower:.4f} "
                    f"(baseline={base_taken_mean:.4f} +/- {base_taken_std:.4f})."
                ),
            ))

    # ---------------------------------------------------------------
    # Order events and classify
    # ---------------------------------------------------------------
    def _event_key(ev: Event):
        return (ev.update_step if ev.update_step is not None else 10**18,
                ev.explore_step if ev.explore_step is not None else 10**18,
                ev.name)
    events.sort(key=_event_key)

    policy_names = {
        "alpha_shutdown", "entropy_drop", "policy_grad_collapse",
        "probe_translation_saturated", "probe_std_collapse",
        "actor_saturated_positive", "actor_saturated_negative",
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

    # Build baseline summary for report
    baseline_info = dict(
        alpha=base_alpha, entropy=base_entropy, grad_q1=base_gq1, grad_policy=base_gpol,
        q1_loss=base_q1l, q2_loss=base_q2l,
        loss_baseline_updates=f"{base_start}-{base_end}",
        startup_skip=startup_skip,
    )
    if probes:
        baseline_info["probe_trans_mean"] = base_probe_trans_mean
        baseline_info["probe_trans_std"] = base_probe_trans_std
        baseline_info["probe_std_mean"] = base_probe_std_mean
    if batches:
        baseline_info["actor_mean"] = base_actor_mean
        baseline_info["actor_std"] = base_actor_std
        baseline_info["taken_mean"] = base_taken_mean
        baseline_info["taken_std"] = base_taken_std

    summary = dict(
        baseline=baseline_info,
        classification=classification,
        rationale=rationale,
    )
    return events, summary

# -----------------------------
# Reporting + plots
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
    episodes: List[EpisodeRow],
    snapshots: List[SnapshotRow],
    events: List[Event],
    summary: Dict[str, Any],
):
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "collapse_report.md"

    if losses:
        first = losses[0]
        last = losses[-1]
        q_change = _percent_change(first.q1_mean, last.q1_mean)
        alpha_change = _percent_change(first.alpha, last.alpha)
        ent_change = _percent_change(first.entropy_proxy, last.entropy_proxy)
    else:
        q_change = alpha_change = ent_change = float("nan")

    if episodes:
        eps_sorted = sorted(episodes, key=lambda e: (e.episode, e.worker))
        first100 = eps_sorted[:100] if len(eps_sorted) >= 100 else eps_sorted[:max(1, len(eps_sorted)//2)]
        last100 = eps_sorted[-100:] if len(eps_sorted) >= 100 else eps_sorted[-max(1, len(eps_sorted)//2):]
        ins_first = _safe_mean([e.max_insertion_1 for e in first100])
        ins_last = _safe_mean([e.max_insertion_1 for e in last100])
        ins_change = _percent_change(ins_first, ins_last)
        rew_first = _safe_mean([e.total_reward for e in first100])
        rew_last = _safe_mean([e.total_reward for e in last100])
        rew_change = _percent_change(rew_first, rew_last)
    else:
        ins_first = ins_last = ins_change = float("nan")
        rew_first = rew_last = rew_change = float("nan")

    collapse_step = events[0].update_step if events else None
    collapse_explore = events[0].explore_step if events else None

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
    lines.append(f"# Collapse Forensics Report (v2)\n")
    lines.append(f"**Run directory:** `{run_dir}`\n")
    lines.append("## Inputs discovered\n")
    lines.append(f"- Losses CSV: `{rel(losses_path)}`")
    lines.append(f"- Probe JSONL: `{rel(probe_path)}`")
    lines.append(f"- Batch samples JSONL: `{rel(batch_path)}`")
    lines.append(f"- Worker logs: `{n_worker_logs}` files")
    lines.append(f"- Policy snapshots: `{len(snapshots)}` files\n")

    lines.append("## High-level summary\n")
    lines.append(f"- Classification: **{summary.get('classification','?')}**")
    lines.append(f"- Rationale: {summary.get('rationale','')}")
    if collapse_step is not None:
        lines.append(f"- Earliest detected event at **update_step={collapse_step}**, explore_step={collapse_explore}")
    else:
        lines.append(f"- Earliest detected event: none (no sustained thresholds crossed)")

    # Baseline info
    bl = summary.get("baseline", {})
    lines.append("")
    lines.append("### Baselines used for detection\n")
    lines.append(f"- Loss baseline computed from updates {bl.get('loss_baseline_updates', '?')} (startup_skip={bl.get('startup_skip', '?')})")
    if "probe_trans_mean" in bl:
        lines.append(f"- Probe translation baseline: mean={_format_float(bl['probe_trans_mean'],4)}, std={_format_float(bl['probe_trans_std'],4)}")
        lines.append(f"- Probe action-std baseline: mean={_format_float(bl['probe_std_mean'],4)}")
    if "actor_mean" in bl:
        lines.append(f"- Batch actor baseline: mean={_format_float(bl['actor_mean'],5)}, std={_format_float(bl['actor_std'],5)}")
        lines.append(f"- Batch taken baseline: mean={_format_float(bl['taken_mean'],4)}, std={_format_float(bl['taken_std'],4)}")

    lines.append("")
    lines.append("### Key metric deltas (losses CSV)\n")
    lines.append(f"- Q1 mean: { _format_float(losses[0].q1_mean,2) if losses else 'nan' } -> { _format_float(losses[-1].q1_mean,2) if losses else 'nan' } ({_format_float(q_change,1)}%)")
    lines.append(f"- Alpha: { _format_float(losses[0].alpha,4) if losses else 'nan' } -> { _format_float(losses[-1].alpha,6) if losses else 'nan' } ({_format_float(alpha_change,1)}%)")
    lines.append(f"- Entropy proxy: { _format_float(losses[0].entropy_proxy,2) if losses else 'nan' } -> { _format_float(losses[-1].entropy_proxy,2) if losses else 'nan' } ({_format_float(ent_change,1)}%)\n")

    lines.append("### Episode behavior deltas (worker logs)\n")
    lines.append(f"- Max insertion (device 1): { _format_float(ins_first,1) } -> { _format_float(ins_last,1) } mm ({_format_float(ins_change,1)}%)")
    lines.append(f"- Total reward: { _format_float(rew_first,3) } -> { _format_float(rew_last,3) } ({_format_float(rew_change,1)}%)\n")

    lines.append("## Event timeline\n")
    if not events:
        lines.append("_No sustained events detected._\n")
    else:
        lines.append("| # | Event | update_step | explore_step | Details |")
        lines.append("|---:|---|---:|---:|---|")
        for i, ev in enumerate(events, 1):
            lines.append(f"| {i} | `{ev.name}` | {ev.update_step if ev.update_step is not None else ''} | {ev.explore_step if ev.explore_step is not None else ''} | {ev.details} |")
        lines.append("")

    lines.append("## Snapshots to inspect\n")
    if before_snap or after_snap:
        if before_snap:
            lines.append(f"- **Last snapshot at/before collapse:** `{rel(before_snap.path)}` (update={before_snap.update_step}, explore={before_snap.explore_step})")
        if after_snap:
            lines.append(f"- **First snapshot at/after collapse:** `{rel(after_snap.path)}` (update={after_snap.update_step}, explore={after_snap.explore_step})")
        lines.append("")
        lines.append("Recommended workflow:")
        lines.append("1) Load the 'before' snapshot in your mesh playback to confirm it still inserts/navigates.")
        lines.append("2) Load the 'after' snapshot to confirm degenerate behavior.")
        lines.append("3) Compare probe translation + alpha around those steps to validate causal ordering.\n")
    else:
        lines.append("_No snapshots with readable metadata were found (or no collapse step detected)._")

    lines.append("## Notes on interpreting collapse\n")
    lines.append("- **Saturation** (positive or negative): policy output saturates toward one extreme. The device may be stuck at vessel end, unable to navigate.")
    lines.append("- **Variance collapse**: policy becomes deterministic (low std) and can no longer explore alternative trajectories.")
    lines.append("- **Combined saturation + variance collapse**: strongest indicator of policy collapse to a degenerate attractor.")
    lines.append("- **Replay distribution shift**: buffer fills with actions from the degenerate policy, creating a self-reinforcing cycle.")
    lines.append("- If **critic losses spike** *before* policy saturation, critic instability may be the root cause.\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path

def make_plots(out_dir: Path, losses: List[LossRow], probes: List[ProbeRow], batches: List[BatchRow], episodes: List[EpisodeRow], events: List[Event]) -> List[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    event_colors = {
        "alpha_shutdown": "red",
        "entropy_drop": "orange",
        "policy_grad_collapse": "purple",
        "critic_loss_spike": "black",
        "probe_translation_saturated": "blue",
        "probe_std_collapse": "cyan",
        "actor_saturated_positive": "green",
        "actor_saturated_negative": "green",
        "replay_distribution_shift_positive": "brown",
        "replay_distribution_shift_negative": "brown",
    }

    def vline_events(ax):
        for ev in events:
            if ev.update_step is None:
                continue
            color = event_colors.get(ev.name, "gray")
            ax.axvline(ev.update_step, linestyle="--", linewidth=0.8, color=color, alpha=0.7)

    if losses:
        us = [r.update_step for r in losses]

        # alpha
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.alpha for r in losses], linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("alpha")
        plt.title("Alpha over training")
        vline_events(plt.gca())
        p = out_dir / "alpha.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # entropy
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.entropy_proxy for r in losses], linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("entropy_proxy")
        plt.title("Entropy proxy over training")
        vline_events(plt.gca())
        p = out_dir / "entropy_proxy.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # q mean
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.q1_mean for r in losses], label="q1_mean", linewidth=0.5)
        plt.plot(us, [r.q2_mean for r in losses], label="q2_mean", linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("Q mean")
        plt.title("Critic Q means over training")
        plt.legend()
        vline_events(plt.gca())
        p = out_dir / "q_means.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # q losses
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.q1_loss for r in losses], label="q1_loss", linewidth=0.5)
        plt.plot(us, [r.q2_loss for r in losses], label="q2_loss", linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("Q loss")
        plt.title("Critic Q losses over training")
        plt.legend()
        vline_events(plt.gca())
        p = out_dir / "q_losses.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # grad norms
        plt.figure(figsize=(10, 4))
        plt.plot(us, [r.grad_norm_policy for r in losses], label="grad_policy", linewidth=0.5)
        plt.plot(us, [r.grad_norm_q1 for r in losses], label="grad_q1", linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("grad norm")
        plt.title("Gradient norms over training")
        plt.legend()
        vline_events(plt.gca())
        p = out_dir / "grad_norms.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

    # Probe plots
    if probes:
        pu = [p.update_step for p in probes]

        # Translation mean with rolling mean overlay
        plt.figure(figsize=(10, 4))
        raw = [p.trans_mean for p in probes]
        roll = _rolling_mean(raw, window=50)
        plt.plot(pu, raw, alpha=0.3, linewidth=0.5, label="raw")
        plt.plot(pu, roll, linewidth=1.5, label="rolling(50)")
        plt.xlabel("update_step")
        plt.ylabel("probe translation mean")
        plt.title("Probe translation mean over training")
        plt.legend()
        vline_events(plt.gca())
        pth = out_dir / "probe_translation_mean.png"
        plt.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(pth)

        # Translation std with rolling overlay
        plt.figure(figsize=(10, 4))
        raw_std = [p.trans_std_mean for p in probes]
        roll_std = _rolling_mean(raw_std, window=50)
        plt.plot(pu, raw_std, alpha=0.3, linewidth=0.5, label="raw")
        plt.plot(pu, roll_std, linewidth=1.5, label="rolling(50)")
        plt.xlabel("update_step")
        plt.ylabel("probe translation std (exp(log_std))")
        plt.title("Probe translation std over training")
        plt.legend()
        vline_events(plt.gca())
        pth = out_dir / "probe_translation_std.png"
        plt.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(pth)

    # Batch plots
    if batches:
        bu = [b.update_step for b in batches]

        # Translation means with rolling overlay
        plt.figure(figsize=(10, 4))
        taken_raw = [b.taken_trans_mean for b in batches]
        actor_raw = [b.actor_trans_mean for b in batches]
        taken_roll = _rolling_mean(taken_raw, window=50)
        actor_roll = _rolling_mean(actor_raw, window=50)
        plt.plot(bu, taken_raw, alpha=0.15, linewidth=0.3, color="C0")
        plt.plot(bu, actor_raw, alpha=0.15, linewidth=0.3, color="C1")
        plt.plot(bu, taken_roll, linewidth=1.5, label="taken rolling(50)", color="C0")
        plt.plot(bu, actor_roll, linewidth=1.5, label="actor_det rolling(50)", color="C1")
        plt.xlabel("update_step")
        plt.ylabel("translation mean")
        plt.title("Batch sample translation means (raw + rolling)")
        plt.legend()
        vline_events(plt.gca())
        pth = out_dir / "batch_translation_means.png"
        plt.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(pth)

        plt.figure(figsize=(10, 4))
        plt.plot(bu, [b.min_q_actor_minus_taken for b in batches], linewidth=0.5)
        plt.xlabel("update_step")
        plt.ylabel("min_q_actor - min_q_taken")
        plt.title("Critic preference: actor vs taken actions (batch samples)")
        vline_events(plt.gca())
        pth = out_dir / "batch_q_preference.png"
        plt.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(pth)

    # Episode insertion vs global step
    if episodes and any(e.end_global_step is not None for e in episodes):
        eps = [e for e in episodes if e.end_global_step is not None]
        eps.sort(key=lambda e: e.end_global_step or 0)
        plt.figure(figsize=(10, 4))
        plt.plot([e.end_global_step for e in eps], [e.max_insertion_1 for e in eps], linewidth=0.5)
        plt.xlabel("global/explore step (from worker logs)")
        plt.ylabel("max insertion device1 (mm)")
        plt.title("Insertion depth vs global step")
        pth = out_dir / "insertion_vs_global_step.png"
        plt.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(pth)

    return paths

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Collapse forensics tool v2 (noise-adaptive thresholds)")
    ap.add_argument("--run-dir", type=str, help="Training run directory (contains diagnostics/ and logs_subprocesses/).")
    ap.add_argument("--name", type=str, help="Experiment name (searches for latest folder with this name in results dir).")
    ap.add_argument("--results-base", type=str, default="training_scripts/results/eve_paper/neurovascular/full/mesh_ben", help="Base results directory for --name search.")
    ap.add_argument("--alpha-threshold", type=float, default=1e-2)
    ap.add_argument("--sustain", type=int, default=50, help="How many consecutive data points a condition must hold to count as an event.")
    ap.add_argument("--plot", action="store_true", help="Generate plots under diagnostics/analysis/")
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
            raise SystemExit(f"No folder found matching name '{args.name}' in {base_dir}")
        run_dir = max(matching, key=lambda p: p.stat().st_mtime)
        print(f"Found matching folder: {run_dir}")
    else:
        raise SystemExit("Must specify either --run-dir or --name")

    if not run_dir.exists():
        raise SystemExit(f"Run dir does not exist: {run_dir}")

    # Discover files
    losses_path = _glob_best(run_dir, [
        "diagnostics/csv/losses_*.csv",
        "**/diagnostics/csv/losses_*.csv",
        "**/losses_*.csv",
    ])
    probe_path = _glob_best(run_dir, [
        "diagnostics/csv/probe_values_*.jsonl",
        "**/diagnostics/csv/probe_values_*.jsonl",
        "**/probe_values_*.jsonl",
    ])
    batch_path = _glob_best(run_dir, [
        "diagnostics/csv/batch_samples_*.jsonl",
        "**/diagnostics/csv/batch_samples_*.jsonl",
        "**/batch_samples_*.jsonl",
    ])

    worker_logs = _glob_all(run_dir, [
        "logs_subprocesses/worker_*.log",
        "**/logs_subprocesses/worker_*.log",
        "**/worker_*.log",
    ])

    losses = load_losses_csv(losses_path) if losses_path else []
    probes = load_probe_jsonl(probe_path) if probe_path else []
    batches = load_batch_jsonl(batch_path) if batch_path else []
    episodes = load_worker_logs(worker_logs) if worker_logs else []
    snapshots = load_policy_snapshots(run_dir)

    events, summary = detect_events(
        losses=losses,
        probes=probes,
        batches=batches,
        alpha_threshold=args.alpha_threshold,
        sustain=args.sustain,
    )

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

    plot_paths: List[Path] = []
    if args.plot:
        plot_paths = make_plots(analysis_dir, losses, probes, batches, episodes, events)

    # Console summary
    print("=" * 90)
    print("Collapse Forensics Summary (v2)")
    print("=" * 90)
    print(f"run_dir: {run_dir}")
    print(f"losses: {losses_path if losses_path else 'NOT FOUND'} (n={len(losses)})")
    print(f"probes: {probe_path if probe_path else 'NOT FOUND'} (n={len(probes)})")
    print(f"batch_samples: {batch_path if batch_path else 'NOT FOUND'} (n={len(batches)})")
    print(f"worker_logs: {len(worker_logs)} files (episodes={len(episodes)})")
    print(f"snapshots: {len(snapshots)} files")
    print("")
    bl = summary.get("baseline", {})
    print("baselines:")
    print(f"  loss region: updates {bl.get('loss_baseline_updates', '?')}")
    if "probe_trans_mean" in bl:
        print(f"  probe_trans: mean={bl['probe_trans_mean']:.4f}, std={bl['probe_trans_std']:.4f}")
    if "actor_mean" in bl:
        print(f"  batch_actor: mean={bl['actor_mean']:.5f}, std={bl['actor_std']:.5f}")
        print(f"  batch_taken: mean={bl['taken_mean']:.4f}, std={bl['taken_std']:.4f}")
    print("")
    print(f"classification: {summary.get('classification')} ({summary.get('rationale')})")
    if events:
        print(f"\nevents ({len(events)}):")
        for i, ev in enumerate(events, 1):
            print(f"  {i}. {ev.name} at update={ev.update_step}, explore={ev.explore_step}")
            print(f"     {ev.details}")
    else:
        print("earliest_event: none")
    print("")
    print(f"report: {md_path}")
    if plot_paths:
        print("plots:")
        for p in plot_paths:
            print(f"  - {p}")
    print("=" * 90)

if __name__ == "__main__":
    main()
