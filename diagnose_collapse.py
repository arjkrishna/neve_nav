#!/usr/bin/env python3
"""
diagnose_collapse.py

Unified "collapse forensics" tool for stEVE / eve_rl SAC diagnostics.

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

Usage
-----
python diagnose_collapse.py --run-dir /path/to/run_dir

Optional:
  --alpha-threshold 0.01
  --probe-trans-threshold -0.5
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
    # pick largest; tie-break by latest mtime
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
                # a expected ~ [[action_dim]] or similar
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
                # action_taken: list(action_dim)
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
    # We aggregate per (worker, ep)
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
                        d["total_reward"] = cum_reward  # overwrite with latest seen
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
        # We can still list paths but without metadata
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
    # Sort by update_step if present else mtime
    out.sort(key=lambda s: (s.update_step if s.update_step is not None and s.update_step >= 0 else 10**18, s.path.stat().st_mtime))
    return out

# -----------------------------
# Collapse detection
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
    probe_trans_threshold: float,
    sustain: int,
) -> Tuple[List[Event], Dict[str, Any]]:
    """
    Returns:
      events: ordered list of detected events (earliest-first)
      summary: metrics used for final classification
    """
    events: List[Event] = []

    # If no losses, nothing to do
    if not losses:
        return events, {"classification": "insufficient_data"}

    # Build arrays for rolling stats
    update_steps = [r.update_step for r in losses]
    explore_steps = [r.explore_step for r in losses]
    alpha = [r.alpha for r in losses]
    entropy = [r.entropy_proxy for r in losses]
    q1m = [r.q1_mean for r in losses]
    q2m = [r.q2_mean for r in losses]
    q1l = [r.q1_loss for r in losses]
    q2l = [r.q2_loss for r in losses]
    gq1 = [r.grad_norm_q1 for r in losses]
    gpol = [r.grad_norm_policy for r in losses]

    # Baselines from stable training region (skip initial startup phase)
    # Neural networks start with high losses that decrease rapidly in first 100-200 updates.
    # We need baseline from STABLE training, not from startup.
    n = len(losses)
    
    # Skip first 100 updates (startup phase where losses decrease from ~6.0 to ~0.06)
    # Then take next 200-1000 updates as baseline
    startup_skip = min(100, n // 10)  # Skip first 100 or 10% of training
    base_start = startup_skip
    base_end = min(base_start + 1000, n)  # Use up to 1000 updates for baseline
    
    if base_end - base_start < 20:
        # Fallback: if too few samples, use everything we have
        base_start = 0
        base_end = min(200, n)
    
    base_alpha = _safe_median(alpha[base_start:base_end])
    base_entropy = _safe_median(entropy[base_start:base_end])
    base_gq1 = _safe_median(gq1[base_start:base_end])
    base_gpol = _safe_median(gpol[base_start:base_end])
    base_q1l = _safe_median(q1l[base_start:base_end])
    base_q2l = _safe_median(q2l[base_start:base_end])

    # Event 1: alpha shutdown
    alpha_pred = [(a < alpha_threshold) for a in alpha]
    i_alpha = _first_sustained(alpha_pred, sustain=sustain)
    if i_alpha is not None:
        events.append(Event(
            name="alpha_shutdown",
            update_step=update_steps[i_alpha],
            explore_step=explore_steps[i_alpha],
            details=f"alpha fell below {alpha_threshold:g} and stayed for >= {sustain} updates (baseline~{base_alpha:.4g}).",
        ))

    # Event 2: entropy proxy collapse (relative)
    # If entropy is already negative baseline, we detect further drop by std.
    ent_roll = _rolling_mean(entropy, window=max(5, sustain//2))
    base_entropy_std = statistics.pstdev(entropy[base_start:base_end]) if (base_end - base_start) >= 2 else 0.0
    ent_drop_thresh = base_entropy - 2.0 * max(1e-6, base_entropy_std)
    ent_pred = [e < ent_drop_thresh for e in ent_roll]
    i_ent = _first_sustained(ent_pred, sustain=sustain)
    if i_ent is not None:
        events.append(Event(
            name="entropy_drop",
            update_step=update_steps[i_ent],
            explore_step=explore_steps[i_ent],
            details=f"entropy_proxy dropped below dynamic threshold {ent_drop_thresh:.3g} (baseline~{base_entropy:.3g}).",
        ))

    # Event 3: gradient collapse (policy)
    gpol_pred = [(g < 0.25 * base_gpol) for g in gpol]  # <25% baseline
    i_gpol = _first_sustained(gpol_pred, sustain=sustain)
    if i_gpol is not None:
        events.append(Event(
            name="policy_grad_collapse",
            update_step=update_steps[i_gpol],
            explore_step=explore_steps[i_gpol],
            details=f"grad_norm_policy fell below 25% of baseline (baseline~{base_gpol:.3g}).",
        ))

    # Event 4: critic instability (loss spikes)
    # Only check for spikes AFTER startup phase (first 100 updates are expected to be high)
    spike_factor = 8.0
    q_spike_pred = []
    for i, (ql1, ql2) in enumerate(zip(q1l, q2l)):
        # Skip startup phase - don't flag normal high initial losses
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
            details=f"q_loss exceeded {spike_factor}× baseline (q1_base~{base_q1l:.3g}, q2_base~{base_q2l:.3g}) AFTER startup phase (skipped first {startup_skip} updates).",
        ))

    # Probe events - detect SATURATION (extreme positive OR negative) + variance collapse
    probe_by_u = {p.update_step: p for p in probes}
    if probes:
        probe_u = [p.update_step for p in probes]
        probe_trans = [p.trans_mean for p in probes]
        probe_std = [p.trans_std_mean for p in probes]
        
        probe_trans_roll = _rolling_mean(probe_trans, window=max(3, sustain//3))
        probe_std_roll = _rolling_mean(probe_std, window=max(3, sustain//3))
        
        # Calculate baseline std from stable early region
        base_std = _safe_median(probe_std[base_start:min(base_end, len(probe_std))])
        
        # Event: Translation saturation (|mean| > 0.7, which saturates tanh to ~0.6)
        saturation_threshold = 0.7
        pred_saturated = [abs(t) > saturation_threshold for t in probe_trans_roll]
        i_sat = _first_sustained(pred_saturated, sustain=max(5, sustain//2))
        if i_sat is not None:
            direction = "positive (max insert)" if probe_trans_roll[i_sat] > 0 else "negative (retract)"
            events.append(Event(
                name="probe_translation_saturated",
                update_step=probe_u[i_sat],
                explore_step=probes[i_sat].explore_step,
                details=f"probe translation mean saturated {direction}: |mean|>{saturation_threshold:g} (value={probe_trans_roll[i_sat]:.3f}).",
            ))
        
        # Event: Variance collapse (std drops below 50% of baseline)
        if not math.isnan(base_std) and base_std > 0:
            std_collapse_thresh = 0.5 * base_std
            pred_std_collapse = [s < std_collapse_thresh for s in probe_std_roll]
            i_std = _first_sustained(pred_std_collapse, sustain=max(5, sustain//2))
            if i_std is not None:
                events.append(Event(
                    name="probe_std_collapse",
                    update_step=probe_u[i_std],
                    explore_step=probes[i_std].explore_step,
                    details=f"probe translation std collapsed below {std_collapse_thresh:.3f} (baseline={base_std:.3f}).",
                ))
        
        # Legacy: also check for negative (for backwards compat with some envs)
        pred_neg = [t < probe_trans_threshold for t in probe_trans_roll]
        i_neg = _first_sustained(pred_neg, sustain=max(5, sustain//2))
        if i_neg is not None:
            events.append(Event(
                name="probe_translation_negative",
                update_step=probe_u[i_neg],
                explore_step=probes[i_neg].explore_step,
                details=f"probe translation mean < {probe_trans_threshold:g} (sustained).",
            ))

    # Batch events - detect both negative AND saturated positive actions
    if batches:
        b_u = [b.update_step for b in batches]
        taken = [b.taken_trans_mean for b in batches]
        actor = [b.actor_trans_mean for b in batches]
        
        taken_roll = _rolling_mean(taken, window=max(3, sustain//3))
        actor_roll = _rolling_mean(actor, window=max(3, sustain//3))
        
        # Event: Actor actions saturated (|mean| > 0.7)
        saturation_threshold = 0.7
        pred_actor_saturated = [abs(a) > saturation_threshold for a in actor_roll]
        i_sat = _first_sustained(pred_actor_saturated, sustain=max(5, sustain//2))
        if i_sat is not None:
            direction = "positive (max insert)" if actor_roll[i_sat] > 0 else "negative (retract)"
            events.append(Event(
                name="actor_actions_saturated",
                update_step=b_u[i_sat],
                explore_step=batches[i_sat].explore_step,
                details=f"actor translation saturated {direction} on batch states: |mean|>{saturation_threshold:g} (value={actor_roll[i_sat]:.3f}).",
            ))
        
        # Legacy: also check for negative actions (for some envs where retract is the failure mode)
        pred_taken_neg = [t < 0.0 for t in taken_roll]
        pred_actor_neg = [a < 0.0 for a in actor_roll]
        i_t = _first_sustained(pred_taken_neg, sustain=max(5, sustain//2))
        if i_t is not None:
            events.append(Event(
                name="replay_actions_negative",
                update_step=b_u[i_t],
                explore_step=batches[i_t].explore_step,
                details="mean translation(action_taken) in batch samples became negative (replay dominated by retract/hold).",
            ))
        i_a = _first_sustained(pred_actor_neg, sustain=max(5, sustain//2))
        if i_a is not None:
            events.append(Event(
                name="actor_actions_negative_on_batch_states",
                update_step=b_u[i_a],
                explore_step=batches[i_a].explore_step,
                details="mean translation(actor_det_action) on sampled batch states became negative.",
            ))

    # Order events by update_step (fallback by explore_step)
    def _event_key(ev: Event):
        return (ev.update_step if ev.update_step is not None else 10**18,
                ev.explore_step if ev.explore_step is not None else 10**18,
                ev.name)
    events.sort(key=_event_key)

    # Classification
    # Determine the earliest "policy signal" and earliest "critic signal"
    policy_names = {
        "alpha_shutdown", "entropy_drop", "policy_grad_collapse",
        # Saturation events (policy converged to extreme action)
        "probe_translation_saturated", "probe_std_collapse", "actor_actions_saturated",
        # Negative translation events (legacy, for some envs)
        "probe_translation_negative", "replay_actions_negative", "actor_actions_negative_on_batch_states",
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
            alpha=base_alpha, entropy=base_entropy, grad_q1=base_gq1, grad_policy=base_gpol,
            q1_loss=base_q1l, q2_loss=base_q2l,
            computed_from_updates=f"{base_start}-{base_end}",
            startup_skip=startup_skip
        ),
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

    # Key aggregate stats
    if losses:
        first = losses[0]
        last = losses[-1]
        q_change = _percent_change(first.q1_mean, last.q1_mean)
        alpha_change = _percent_change(first.alpha, last.alpha)
        ent_change = _percent_change(first.entropy_proxy, last.entropy_proxy)
    else:
        q_change = alpha_change = ent_change = float("nan")

    # Insertion trajectory (device 1)
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

    # Determine collapse onset step (earliest event)
    collapse_step = events[0].update_step if events else None
    collapse_explore = events[0].explore_step if events else None

    # Pick snapshot before/after
    before_snap = after_snap = None
    if collapse_step is not None and snapshots:
        # snapshots sorted by update_step
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
    lines.append(f"# Collapse Forensics Report\n")
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

    lines.append("")
    lines.append("### Key metric deltas (losses CSV)\n")
    lines.append(f"- Q1 mean: { _format_float(losses[0].q1_mean,2) if losses else 'nan' } → { _format_float(losses[-1].q1_mean,2) if losses else 'nan' } ({_format_float(q_change,1)}%)")
    lines.append(f"- Alpha: { _format_float(losses[0].alpha,4) if losses else 'nan' } → { _format_float(losses[-1].alpha,6) if losses else 'nan' } ({_format_float(alpha_change,1)}%)")
    lines.append(f"- Entropy proxy: { _format_float(losses[0].entropy_proxy,2) if losses else 'nan' } → { _format_float(losses[-1].entropy_proxy,2) if losses else 'nan' } ({_format_float(ent_change,1)}%)\n")

    lines.append("### Episode behavior deltas (worker logs)\n")
    lines.append(f"- Max insertion (device 1): { _format_float(ins_first,1) } → { _format_float(ins_last,1) } mm ({_format_float(ins_change,1)}%)")
    lines.append(f"- Total reward: { _format_float(rew_first,3) } → { _format_float(rew_last,3) } ({_format_float(rew_change,1)}%)\n")

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
        lines.append("2) Load the 'after' snapshot to confirm degenerate retract/freeze behavior.")
        lines.append("3) Compare probe translation + alpha around those steps to validate causal ordering.\n")
    else:
        lines.append("_No snapshots with readable metadata were found (or no collapse step detected)._")

    lines.append("## Notes on interpreting policy vs critic failure\n")
    lines.append("- If **probe translation** and **batch action translation** degrade *before* critic loss spikes, it indicates the **actor converged to a suboptimal attractor** (policy collapse).")
    lines.append("- If **critic losses spike/diverge** *before* actor/probe/batch degradation, it suggests **critic instability/overestimation/underestimation** drove policy failure.")
    lines.append("- If both happen together, reward shaping + entropy tuning may be interacting (e.g., alpha shutdown + truncation/penalty attractor).\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path

def make_plots(out_dir: Path, losses: List[LossRow], probes: List[ProbeRow], batches: List[BatchRow], episodes: List[EpisodeRow], events: List[Event]) -> List[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    def vline_events(ax):
        for ev in events:
            if ev.update_step is None:
                continue
            ax.axvline(ev.update_step, linestyle="--", linewidth=1)

    # Plot alpha, entropy, q1_mean
    if losses:
        us = [r.update_step for r in losses]

        # alpha
        plt.figure()
        plt.plot(us, [r.alpha for r in losses])
        plt.xlabel("update_step")
        plt.ylabel("alpha")
        plt.title("Alpha over training")
        vline_events(plt.gca())
        p = out_dir / "alpha.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # entropy
        plt.figure()
        plt.plot(us, [r.entropy_proxy for r in losses])
        plt.xlabel("update_step")
        plt.ylabel("entropy_proxy")
        plt.title("Entropy proxy over training")
        vline_events(plt.gca())
        p = out_dir / "entropy_proxy.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # q mean
        plt.figure()
        plt.plot(us, [r.q1_mean for r in losses], label="q1_mean")
        plt.plot(us, [r.q2_mean for r in losses], label="q2_mean")
        plt.xlabel("update_step")
        plt.ylabel("Q mean")
        plt.title("Critic Q means over training")
        plt.legend()
        vline_events(plt.gca())
        p = out_dir / "q_means.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

        # grad norms
        plt.figure()
        plt.plot(us, [r.grad_norm_policy for r in losses], label="grad_policy")
        plt.plot(us, [r.grad_norm_q1 for r in losses], label="grad_q1")
        plt.xlabel("update_step")
        plt.ylabel("grad norm")
        plt.title("Gradient norms over training")
        plt.legend()
        vline_events(plt.gca())
        p = out_dir / "grad_norms.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

    # Probe translation mean
    if probes:
        plt.figure()
        plt.plot([p.update_step for p in probes], [p.trans_mean for p in probes])
        plt.xlabel("update_step")
        plt.ylabel("probe translation mean")
        plt.title("Probe translation mean over training")
        vline_events(plt.gca())
        pth = out_dir / "probe_translation_mean.png"
        plt.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(pth)

        plt.figure()
        plt.plot([p.update_step for p in probes], [p.trans_std_mean for p in probes])
        plt.xlabel("update_step")
        plt.ylabel("probe translation std (exp(log_std))")
        plt.title("Probe translation std over training")
        vline_events(plt.gca())
        pth = out_dir / "probe_translation_std.png"
        plt.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(pth)

    # Batch translation means
    if batches:
        plt.figure()
        plt.plot([b.update_step for b in batches], [b.taken_trans_mean for b in batches], label="taken")
        plt.plot([b.update_step for b in batches], [b.actor_trans_mean for b in batches], label="actor_det")
        plt.xlabel("update_step")
        plt.ylabel("translation mean")
        plt.title("Batch sample translation means")
        plt.legend()
        vline_events(plt.gca())
        pth = out_dir / "batch_translation_means.png"
        plt.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(pth)

        plt.figure()
        plt.plot([b.update_step for b in batches], [b.min_q_actor_minus_taken for b in batches])
        plt.xlabel("update_step")
        plt.ylabel("min_q_actor - min_q_taken")
        plt.title("Critic preference: actor vs taken actions (batch samples)")
        vline_events(plt.gca())
        pth = out_dir / "batch_q_preference.png"
        plt.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(pth)

    # Episode insertion vs global step (if available)
    if episodes and any(e.end_global_step is not None for e in episodes):
        eps = [e for e in episodes if e.end_global_step is not None]
        eps.sort(key=lambda e: e.end_global_step or 0)
        plt.figure()
        plt.plot([e.end_global_step for e in eps], [e.max_insertion_1 for e in eps])
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, help="Training run directory (contains diagnostics/ and logs_subprocesses/).")
    ap.add_argument("--name", type=str, help="Experiment name (searches for latest folder with this name in results dir).")
    ap.add_argument("--results-base", type=str, default="training_scripts/results/eve_paper/neurovascular/full/mesh_ben", help="Base results directory for --name search.")
    ap.add_argument("--alpha-threshold", type=float, default=1e-2)
    ap.add_argument("--probe-trans-threshold", type=float, default=-0.5)
    ap.add_argument("--sustain", type=int, default=50, help="How many consecutive updates a condition must hold to count as an event.")
    ap.add_argument("--plot", action="store_true", help="Generate plots under diagnostics/analysis/")
    args = ap.parse_args()

    # Resolve run directory
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    elif args.name:
        # Search for folder by name
        base_dir = Path(args.results_base).expanduser().resolve()
        if not base_dir.exists():
            raise SystemExit(f"Results base dir does not exist: {base_dir}")
        
        # Find all folders matching the name
        matching = [d for d in base_dir.iterdir() if d.is_dir() and args.name in d.name]
        if not matching:
            raise SystemExit(f"No folder found matching name '{args.name}' in {base_dir}")
        
        # Pick the most recent one
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
        probe_trans_threshold=args.probe_trans_threshold,
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
    print("Collapse Forensics Summary")
    print("=" * 90)
    print(f"run_dir: {run_dir}")
    print(f"losses: {losses_path if losses_path else 'NOT FOUND'} (n={len(losses)})")
    print(f"probes: {probe_path if probe_path else 'NOT FOUND'} (n={len(probes)})")
    print(f"batch_samples: {batch_path if batch_path else 'NOT FOUND'} (n={len(batches)})")
    print(f"worker_logs: {len(worker_logs)} files (episodes={len(episodes)})")
    print(f"snapshots: {len(snapshots)} files")
    print("")
    print(f"classification: {summary.get('classification')} ({summary.get('rationale')})")
    if events:
        ev0 = events[0]
        print(f"earliest_event: {ev0.name} at update={ev0.update_step}, explore={ev0.explore_step}")
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
