# A Diagnostic Logging and Checkpointing Plan to Debug Fast SAC Policy Collapse in stEVE

## Why policy collapse is hard to debug without losses and probes

In Soft Actor-Critic (SAC), the “policy collapsing” you describe usually shows up first as one (or more) of these measurable symptoms: critic targets and Q-values blow up or drift, actor entropy collapses (policy becomes nearly deterministic), temperature/entropy weight (α) drifts to an extreme, or the actor learns to exploit a spurious Q-peak and then becomes brittle. SAC’s core objects are explicitly defined by the critic MSBE losses and the actor objective that trades off expected Q value vs entropy (via α), so these are the first signals you want recorded at high frequency. citeturn5view2turn7view0turn4search0

Your repo already *computes* the three primary losses on every gradient update (q1 loss, q2 loss, policy loss), but the training orchestration currently doesn’t persist them in a structured way. Meanwhile, your evaluation/checkpoint cycle happens at a relatively large cadence (hundreds of thousands of exploration steps), which is too coarse to pinpoint “the exact moment” a collapse begins.

The plan below adds:

- A **comprehensive structured logging system** (CSV + optional TensorBoard).
- Two new per-update-step log streams:
  - **Losses** (actor + both critics + α/entropy diagnostics).
  - **Values and states** (probe states + sampled batch transitions with actor/critic outputs).
- A **probe-state evaluation pipeline** (start state + “near-start” states), logged continuously and plotted every 10k steps.
- **Frequent policy snapshots** every 10k steps for post-hoc mesh playback, without exploding disk usage.

TensorBoard is especially useful here because it supports scalars and histograms/distributions over time, which helps you see whether weights, Q-values, and action distributions are changing in plausible ways (and not saturating). citeturn8view0turn6view0turn5view1

---

## Where to hook logging in your current training architecture

Your training stack (as implemented in the ZIP) has a critical architectural detail that determines *where* logging must live:

- **Exploration / heatup rollouts** run in multiple worker processes.
- **Network updates** (SAC gradient steps) happen in a dedicated **trainer subprocess** inside the `Synchron` agent.
- The main process `Runner` orchestrates explore/update/eval loops, but it currently does not persist the per-gradient-step outputs.

This implies:

- **Loss logging must be written from the trainer subprocess** (or be transmitted back step-by-step). Logging from the main process is insufficient if you want one row per gradient update step without losing intermediate steps.
- **Actor/critic value traces on probe states** can be computed either:
  - in the trainer subprocess (best fidelity: every update step, no extra IPC), or
  - in the main process at the cadence you sync weights (lower fidelity, but simpler).

To diagnose fast collapse, you want maxima fidelity: **write a row per update step from the trainer process.**

---

## Logging outputs you should add

### Loss logging stream

At minimum, record these values **every gradient update**:

- `q1_loss`, `q2_loss`, `policy_loss` (the essentials)
- `alpha` and `alpha_loss` (SAC’s entropy/temperature behavior is often central to collapse) citeturn4search0turn7view0
- `log_pi` stats (mean/std) and implied entropy proxy `-log_pi_mean`
- basic Q stats: `q1_mean`, `q2_mean`, `min_q_mean`, `target_q_mean`

These correspond directly to the SAC equations and are the canonical “is SAC behaving?” signals. citeturn5view2turn7view0

Strongly recommended additional debug fields (very helpful when collapse is sudden):

- `grad_norm_q1`, `grad_norm_q2`, `grad_norm_policy` (to detect exploding gradients)
- `param_norm_q1`, `param_norm_q2`, `param_norm_policy` (to detect nil/decaying weights)
- learning rates from optimizers/schedulers
- fraction of policy log-std values clamped at min/max (a common early warning sign)

### Values + states logging stream

You asked for: “log state and state transitioned to and actor and critic values for those states, every step.”

If you try to log *all* states for *all* env steps, you’ll generate an impractical amount of data for a 20M-step run. The right way to meet your diagnostic intent without killing I/O is:

- Log **(A)** a small set of **fixed probe states** (start + “near-start”), *every update step*.
- Log **(B)** one (or a few) **sampled rollout transitions** per update step, pulled from the replay batch (so you see what the critic is training on).

This gives you both:
- **consistent time-series** (probe states), and
- **representative training distribution** (batch samples).

For each logged state `s` (and optionally `s_next`), record:

- actor:
  - `mu(s)` (mean)
  - `log_std(s)`
  - `a_det(s)=tanh(mu(s))` (deterministic action)
- critics:
  - `q1(s, a_det(s))`, `q2(s, a_det(s))`
  - and for sampled transitions: `q1(s, a_taken)`, `q2(s, a_taken)`

This directly answers: “is the actor evolving toward maximum insertion speed at start states?” and “are critics valuing that action higher over time?”

### Periodic visualization artifacts every 10k steps

Write derived plots every 10k exploration steps (or every N updates):

- actor deterministic action vs training time (for start probe state): translation and rotation components
- `q1/q2` values on start probe state under actor action
- α, entropy proxy, and losses over time

TensorBoard can do most of this interactively (scalars + histograms). PyTorch’s `SummaryWriter` is designed to be called inside the training loop and writes asynchronously, which is ideal for frequent logging. citeturn8view0turn5view1

---

## Detailed plan for code changes

### Add a diagnostics directory and filenames per run

In `training_scripts/util/util.py`:

- Extend `get_result_checkpoint_config_and_log_path(...)` to also create and return:
  - `diagnostics_folder = os.path.join(results_folder, "diagnostics")`
  - subfolders:
    - `diagnostics/csv/`
    - `diagnostics/tensorboard/`
    - `diagnostics/probes/`
    - `diagnostics/plots/`
    - `diagnostics/policy_snapshots/`

Update all training entrypoints (`DualDeviceNav_train.py`, `BasicWireNav_train.py`, etc.) accordingly.

Deliverables created on disk per run:
- `diagnostics/csv/losses_trainer.csv`
- `diagnostics/csv/values_trainer.jsonl` (or csv; JSONL is much nicer for vectors)
- `diagnostics/probes/probe_states.npz`
- `diagnostics/plots/*.png`
- `diagnostics/policy_snapshots/policy_<explore_steps>.everl`

### Introduce a lightweight structured logger utility

Add a new module inside `eve_rl/eve_rl/util/`, for example:

- `eve_rl/eve_rl/util/diagnostics_logger.py`

It should provide:

- `CSVScalarLogger(path, header_fields, flush_every_n=...)`
- `JSONLLogger(path, flush_every_n=...)`
- an optional TensorBoard writer (only if enabled):
  - `TensorBoardLogger(log_dir)` using `torch.utils.tensorboard.SummaryWriter` citeturn8view0turn5view1

Design goals:
- open files once per process, append-only
- flush periodically (not every line if performance becomes an issue)
- write a “run metadata” JSON with git hash, hyperparams, obs/action dims

### Log per-update-step SAC losses in the trainer subprocess

This is the core change.

#### Add a diagnostics hook to the subprocess runner

In `eve_rl/eve_rl/agent/singelagentprocess.py`:

- Extend `SingleAgentProcess.__init__` to accept:
  - `diagnostics_config: Optional[dict or dataclass] = None`
- Pass it into `run(...)`.
- In `run(...)`, after creating the `Single(...)` agent, attach:
  - `agent.diagnostics = DiagnosticsLogger(diagnostics_config, process_name=name)`
- Create the trainer’s files using `name` so you can distinguish processes cleanly.

#### Log losses inside `Single.update()`

In `eve_rl/eve_rl/agent/single.py`, in `Single.update(...)`:

- Right now it does:
  - sample batch
  - `result = self.algo.update(batch)`
  - append result
- Change it to additionally do (per update step):
  - call `self.diagnostics.log_losses(...)` with:
    - `update_step` = shared `step_counter.update`
    - `exploration_step` = shared `step_counter.exploration`
    - `q1_loss, q2_loss, policy_loss`
    - plus any other metrics available (see next subsection)

You will need the trainer subprocess to know:
- its update step index
- the “current” exploration step for alignment

Because the step counters are shared, the trainer can read them directly each update step.

### Add richer SAC metrics (α, entropy, Q stats) without breaking interfaces

Right now, `SAC.update(batch)` returns only `[q1_loss, q2_loss, policy_loss]`.

To avoid breaking any call sites, keep the return value stable, but add:

- `self.last_metrics: Dict[str, float]` on the SAC object (updated each call)

In `eve_rl/eve_rl/algo/sac.py`:

- Capture and store:
  - α and α-loss (modify `_update_alpha` to return its loss value)
  - `log_pi_mean`
  - summary statistics of:
    - `curr_q1`, `curr_q2`, `expected_q`
    - policy mean/log_std distributions
- Optionally compute gradient norms inside `_update_q1/_update_q2/_update_policy`
  - (before optimizer step)
- Store these scalar metrics in `self.last_metrics`.

Then in `Single.update`, after calling `algo.update(batch)`:
- read `algo.last_metrics`
- write them to the same `losses_trainer.csv` row

This gives you SAC-native collapse signals. SAC’s α/entropy behavior is central to its explore/exploit balance, and tracking it is standard practice. citeturn7view0turn4search0

### Add “values + states” logging from two sources

You want two perspectives:

#### Probe state values (fixed states, stable time series)

Implement in the trainer subprocess:

- Add a new SAC method:
  - `set_probe_states(np.ndarray probes)` storing a torch tensor on device
- Add a new SAC method:
  - `compute_probe_values()` returning values for each probe:
    - policy mean/log_std, deterministic action
    - q1/q2 on (s, actor_action)

Add a new task type in `singelagentprocess.run(...)`:

- `"set_probe_states"`: receives probe arrays once, stores them in algo
- (optional) `"clear_probe_states"` for resets

Where do probes come from?
- In the main process (Runner), on the first exploration batch after heatup:
  - take the start state (`flat_obs[0]`) and first N “near-start” states from a handful of episodes
  - save them to `diagnostics/probes/probe_states.npz`
  - send them to the trainer subprocess via the new `"set_probe_states"` task

Then in `Single.update`, each update step:
- compute `probe_values = algo.compute_probe_values()`
- append a JSONL entry per probe (or one entry containing arrays)

#### Sampled replay-batch transitions (moving window into training distribution)

Each update step, pick 1–K transitions from the batch, log:

- `s_t`, `a_t`, `r_t`, `done_t`, `s_{t+1}`
- actor outputs at `s_t`
- critic values at `(s_t, a_t)` and `(s_t, actor_action(s_t))`

Log these to `values_trainer.jsonl`. This gives you a “recording” of what the actor and critics think about the same kind of transitions used for training.

### Make Runner stop discarding update outputs, and add periodic actions every 10k steps

In `eve_rl/eve_rl/runner/runner.py`:

#### Capture update results
In `Runner.explore_and_update(...)`, currently you call:
- `self.agent.explore_and_update(...)` and ignore the return.

Change it to:
- `explore_episodes, update_losses = self.agent.explore_and_update(...)`

Even if your trainer subprocess is logging losses, capturing them here is still useful (sanity check / redundancy / quick summaries).

#### Add periodic “every 10k steps” hooks
Add new Runner constructor args:

- `checkpoint_every_explore_steps: int = 10_000`
- `probe_plot_every_explore_steps: int = 10_000`
- `policy_snapshot_every_explore_steps: int = 10_000`

Then inside `training_run(...)`, do not rely only on eval cadence. Add a “tick” after each explore/update call:

- if `step_counter.exploration >= next_policy_snapshot_step`:
  - save a **policy-only snapshot**
  - update `next_policy_snapshot_step += interval`
- if `step_counter.exploration >= next_probe_plot_step`:
  - write plots from probe logs (or write a “plot requested” marker file)
  - update `next_probe_plot_step += interval`

This is the mechanism that gives you the “after every 10k steps” artifacts you requested.

### Save policy snapshots every 10k steps without exploding storage

Saving a full `.everl` checkpoint every 10k steps can become enormous (you’d create ~2000 snapshots in a 20M-step run, each storing large parameters).

Instead implement *policy-only snapshots* that are still loadable for mesh playback:

- Save a torch checkpoint dict that includes:
  - `algo` config (so the network architecture can be reconstructed)
  - `env_eval` config
  - `network_state_dicts = {"policy": policy_state_dict}` only
  - step/episode counters
  - `additional_info` with the exact explore step and timestamp

This works because your evaluation-only loading path ultimately only needs the policy weights for running the actor, and SAC’s play-only model loads only the `policy` entry. (Your design already supports a “play-only” SAC model.) citeturn7view0turn4search0

Also add retention controls:

- keep every 10k snapshot for the first N steps (where collapse occurs)
- after that, keep every 50k or 100k
- always keep:
  - best checkpoint
  - the snapshot right before detected collapse (if you implement collapse detection)

---

## Probe-state visualization workflow and what “collapse” will look like in these logs

### What you will see when things are healthy

- `q1_loss` and `q2_loss` oscillate but remain bounded
- `policy_loss` trends smoothly, not exploding
- α remains in a reasonable range; entropy proxy doesn’t slam to ~0 immediately
- actor `a_det(start_state)` shifts gradually toward the expected “fast insert” behavior (translation dimension → +1), while rotation stays plausible

SAC’s entropy regularization is meant to prevent premature convergence to a brittle policy; when entropy collapses too early, you’ll see it in α and log-std. citeturn7view0turn4search0

### What collapse will look like

Common signatures:

- Q targets spike → `q1_loss/q2_loss` jump sharply
- Q values drift wildly → `q*_mean` and `target_q_mean` explode in magnitude
- actor saturates:
  - `log_std` clamps hard at minimum (many values at clamp)
  - or action mean saturates to ±1 too quickly
- α drifts toward near-zero (policy becomes effectively deterministic) or becomes too large (overly random)

TensorBoard histograms/distributions are especially good at showing saturation (e.g., weights exploding, log-std hitting clamp boundaries). citeturn8view0turn6view0

---

## A few high-impact “sanity fixes” to consider alongside logging

You asked primarily for logging, but since your goal is “pinpoint exactly when and why policy collapses,” it’s worth making sure your instrumentation isn’t hiding a mechanical issue:

- Verify your optimizers are attached to the intended parameter sets. In other parts of your repo (example SAC scripts), the optimizer is constructed on the full Q/policy network modules, which is the typical approach, and mismatches here can produce strange training behavior.
- Ensure checkpoints store the correct env configs (train vs eval) so playback is faithful.

Even with perfect logging, if something is structurally off (like an optimizer not updating important parameters), you’ll only learn “it collapses” without “why.” Logging + a quick audit of optimizer parameter coverage is the fastest route to a conclusive diagnosis.

---

If you want, I can turn this plan into an implementation-ready checklist with exact function signatures and log schemas (headers/JSONL keys), including a recommended minimal set of probe states for DualDeviceNav (start state + first N steps from fixed-seed episodes) and a plotting script that generates the “every 10k steps” visuals automatically from the logs.