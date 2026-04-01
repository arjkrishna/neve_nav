

## What your new diagnostics enables and what is still missing

Your logging implementation (as described in your file list) structurally matches the correct architecture for high-fidelity, multi-process SAC diagnostics:

- **Losses are logged in the trainer subprocess**, where `SAC.update(batch)` actually runs. This is the only location that can capture per-update-step `q1_loss`, `q2_loss`, `policy_loss`, `alpha`, `alpha_loss`, and gradient norms without losing data due to IPC granularity.
- **Probe evaluation** (fixed probe states) is computed in the trainer, enabling consistent time-series evaluation of the actor distribution (mean/log_std) and critic values on the exact same states as training progresses.
- **Batch sample tracing** adds a “what data am I training on?” lens: action distribution drift, terminal/done frequency drift, and “actor vs replay action” critic preference can all be tracked.
- **Policy snapshots every 10k steps** enable two crucial workflows:
  - mesh playback for qualitative diagnosis,
  - and objective selection of “last good” vs “first bad” models around the detected collapse point.

What your current separate scripts still lack (by design) is **cross-signal alignment** and **automatic collapse onset detection**:
- `analyze_losses_detail.py` reads only losses and summarizes by segments. citeturn0file2  
- `analyze_probes.py` reads only probe JSONL and prints selected points; it does not align probes against α/loss evolution or snapshots. citeturn0file3  
- `analyze_episode_rewards.py` and `analyze_insertions.py` parse worker logs but do not fuse reward/insertion trends with actor/critic diagnostics or identify the earliest cause signal. citeturn0file0turn0file1  

The most actionable improvement is to unify these scripts into a single program that produces:
- a **timeline of events** (alpha shutdown, probe action inversion, batch action inversion, critic instability spikes),
- a **collapse onset** (update step + explore step),
- a **mechanism classification** (policy collapse to suboptimal attractor vs critic instability vs mixed),
- and a **“what to inspect next” pack** (the two snapshots bracketing collapse + the log windows around the event).

## Single-script workflow to pinpoint failure timing and mechanism

### Unified tool: `diagnose_collapse.py`

I merged and extended the logic of your four scripts into one runnable program that:

- Auto-discovers your diagnostics artifacts in `<run_dir>/diagnostics/`:
  - losses: `diagnostics/csv/losses_*.csv`
  - probes: `diagnostics/csv/probe_values_*.jsonl`
  - batch samples: `diagnostics/csv/batch_samples_*.jsonl`
  - snapshots: `diagnostics/policy_snapshots/policy_*.pt`
  - worker env logs: `<run_dir>/logs_subprocesses/worker_*.log`
- Aligns everything on **update_step** and **explore_step** (via the losses CSV, which is the canonical join key because it includes both).
- Detects “collapse onset” and emits an ordered event list.
- Picks the **last snapshot before collapse** and the **first snapshot after collapse** (for playback).
- Writes:
  - a markdown report to `diagnostics/analysis/collapse_report.md`
  - a set of PNG plots to `diagnostics/analysis/` (if matplotlib is available)



### How to run it

```bash
python diagnose_collapse.py --run-dir training_scripts/results/<...>/<run_name>
```

Optional parameters:
- `--process-hint synchron` (if you have multiple trainer logs and want to select a specific one)
- `--alpha-threshold 0.01` (default is 1e-2; tune to your expectations)

This produces a concise console summary, plus a durable markdown + plot artifact you can share or archive per run.

## Event detection methodology and cause classification rules

### Why event detection works well for your collapse type

Many SAC collapses are not a “single metric blow-up.” They are a *sequence*:
1. α trends downward (entropy term weakens),
2. policy distribution tightens (log_std decreases; entropy proxy drops),
3. probe actions become extreme (e.g., retract/freeze),
4. replay distribution shifts (batch sampled actions become retract-heavy),
5. critics converge stably on a bad local optimum (losses decrease, grads shrink).

Entropy regularization in SAC is explicitly intended to prevent premature convergence to bad local optima; therefore, tracking α/log_pi/log_std alongside probe actions is a principled collapse detector. citeturn0search1

### Signals used in the unified script

The script computes and detects four key event types:

- **Alpha shutdown**: rolling α falls below a threshold (default 1e-2) and stays there for a sustained number of updates.
- **Entropy proxy collapse**: rolling `entropy_proxy = -E[log π(a|s)]` falls materially below its initial baseline (threshold is baseline-driven; not a hardcoded constant).
- **Probe action degeneration**: probe translation mean falls below a negative threshold (default -0.5) for sustained updates. This is your “start state becomes retract” signature.
- **Replay distribution shift** (batch samples): sampled batch action translation mean becomes negative in a sustained way, indicating the buffer is filling with retract-heavy behavior.

Additionally, it detects **critic instability spikes** (not the same as critic collapse, but useful):
- sustained `q1_loss` or `q2_loss` above `baseline_median * factor` (default factor 8×).

### Mechanism classification heuristic

The tool emits a *mechanism label* based on ordering:

- **critic instability precedes policy signals** if sustained critic loss spikes appear before alpha/probe/replay degeneration signals.
- **policy collapse or suboptimal attractor** if alpha/probe/replay degeneration occurs first and critic losses remain bounded (common when SAC becomes nearly greedy and gets stuck).

This reflects what SAC is optimizing: if α collapses, the entropy term becomes negligible and the actor converges quickly; if in parallel rewards/truncation penalties create a “safe” attractor (freeze/retract), the learned policy “locks in.” citeturn0search1

## How to leverage probe states, batch samples, and policy snapshots together

This is where your new artifacts matter most—especially because you noted probe states, batch samples, and saved policy files aren’t being exploited yet as a unified triage system.

### Probes

Probe values answer:
- “What does the policy do on the same start(-like) states as training progresses?”
- “Do the critics value that action more or less over time?”

This is the most direct way to distinguish:
- critic pessimism on unchanged actions (critic drift), vs
- actor action drift into degeneracy (policy collapse).

The unified script reduces each JSONL probe entry to:
- mean Q1/Q2 on probe set (under actor deterministic action)
- translation mean and std (from policy_mean / policy_log_std)

It then detects the **first sustained inversion** of translation (or other configured components).

### Batch samples

Batch sample logs are your “training distribution microscope.” The script uses these for:
- **done rate drift** in sampled transitions (proxy for more terminal transitions in buffer),
- **action distribution drift** (buffer dominated by retract/hold actions),
- **critic preference of actor vs replay actions** (`min_q_actor_minus_taken`).

This is exactly the missing link between “policy degenerates” and “why training reinforces it.” When the buffer becomes dominated by the degenerate behavior, the critics converge with low loss and low gradients even though the policy is terrible—because the data distribution has collapsed.

### Policy snapshots (every 10k)

The unified script loads the metadata saved in your `policy_*.pt` snapshots (update_step/explore_step) and picks:
- **last snapshot before collapse**
- **first snapshot after collapse**

Those are the two models to load into your mesh playback and visually compare trajectories.

This directly supports your goal: “pinpoint timing and cause,” then immediately reproduce the behavior qualitatively.





## How the new system helps interpret policy collapse vs critic collapse in practice

A few “expected collapse signatures” your combined tool is designed to surface:

### Policy collapse (suboptimal attractor) signature

- α trends to ~0 (entropy term effectively disappears)
- probe translation mean becomes extreme (retract/freeze)
- probe std decreases
- batch sample action distribution shifts (buffer mirrors the degenerate policy)
- critic losses decrease and gradients shrink (critics converge on the degenerate data distribution)
- policy snapshots bracketing the collapse show the visible behavior change in playback

This is consistent with SAC’s known failure mode when entropy pressure is effectively removed and reward shaping makes “do less” locally appealing. citeturn0search1

### Critic instability / collapse signature

- Q losses spike or become highly volatile
- Q1/Q2 diverge (large Q-gap)
- gradients explode or become non-finite
- policy changes may follow, but it is “downstream damage”

Your unified script explicitly detects critic loss spikes and compares their timing against alpha/probe/replay degeneration signals to distinguish “critic breaks first” from “policy breaks first.”

