❌ True “stuck” detection (needs delta-insertion + action coupling).

## Freeze / reduced variance / “stuck for several actions”: v3 detects only a narrow proxy

v3’s “freeze detection” is literally:
- mean near 0 (`±0.05`)
- std below `0.1` :contentReference[oaicite:14]{index=14}

That will **not** detect many real “stuck” situations in guidewire environments, where:
- actions can still vary, but insertion delta stays ~0 due to collision/friction, or
- the policy saturates to “push” but environment clips/blocks motion.

So: **v3 does not reliably detect “stuck in anatomy”** unless your action distribution itself collapses near 0.


## Reward analysis: v3 finally starts it, but it’s too shallow to localize *where* it gets stuck

v3 adds:
- `reward_decline` on rolling mean reward crossing a baseline-derived lower threshold :contentReference[oaicite:15]{index=15}
- `done_rate_spike` based on done fraction :contentReference[oaicite:16]{index=16}

But it still **doesn’t answer your question**: “where is the guidewire getting stuck?”

To locate “where,” you need step-level coupling like:
- insertion delta ~ 0 for long spans
- action translation is positive (trying to push)
- reward becomes constant penalty / progress reward stops changing
- termination reason distribution shifts (timeout vs collision vs invalid)

The script parses per-step lines (there’s timestamp parsing and worker log regex machinery), but the detection logic doesn’t actually compute “stuck segments by insertion depth bin.”

### No “stuck” detection exists despite having logs

You log step-level fields:

action=[...]

inserted=[...]

delta_ins=[...]

…but the analysis script ignores global, ignores step-level rewards, and does not detect:

long sequences of “commanded insert” + “no motion”

stuck depth region (where it happens)

whether it persists / becomes dominant

✅ Fix: Parse global and step reward, compute:

stuck_fraction, max_stuck_run, depth-bin mode

add an event stuck_regime


### Possible Solutions
3) **Detect “stuck” using worker step logs**
Add a new event like `stuck_segment`:
- Find contiguous segments where:
  - `abs(delta_insertion) < eps_mm` for K steps
  - `action_translation > +push_threshold` (policy is trying)
  - reward <= some penalty band or flatlines
- Report the **insertion depth range** (from inserted values) where this happens most.


### Possible Solutions on the training/logging side 
### eve_rl/eve_rl/runner/runner.py
Crucial: If we want episode-level logs aligned to explore/update steps, Runner is the best place to emit them (not the env).

### eve/eve/intervention/monoplanestatic.py (why “stuck” can happen)

This matters for  “detect stuck” request.

The intervention explicitly masks translation in some conditions:

	prevents retracting below 0 insertion
	prevents inserting beyond max length
	enforces constraints between devices (e.g., longer device cannot move further if constrained)

So there are cases where:

	the agent commands positive translation, but the environment sets translation to 0 internally.

That can look like “stuck” unless you log:

	commanded action vs executed (masked) action
	actual delta insertion / delta tip motion

Right now our forensics mostly infers stuck from action stats and insertion outcomes, but you can make it much more reliable.


### Log “executed action” and “action clipping/masking”

Right now we can detect saturation, but we can’t separate:

	“policy wants max insert” vs “env masks it to zero because max-length / constraint”

Add per-step or per-episode aggregates:

	cmd_action_mean, exec_action_mean
	fraction_translation_masked
	fraction_rotation_clipped
	mean_abs_delta_insertion_per_cmd_translation

This is exactly what we need to diagnose “stuck”.

### Log truncation reasons in batch samples and episode summaries

Our env produces both (terminated, truncated) plus info.
Our batch logs currently only carry “done” from the replay batch.

Add fields to batch samples:

	terminated, truncated, term_reason (if available in info)
	OR at least timeout_truncated from MaxSteps

Then in forensics we can answer:

	“did policy collapse correlate with timeouts increasing?”
	“do we see more vessel-end truncations?”
	“are we hitting sim errors?”


### Log reward decomposition (the best “stuck locator”)

	A scalar reward is too ambiguous. To locate stuck, you want components, e.g.:

	progress_reward
	step_penalty
	collision penalty
	retract_penalty

If our reward module can expose a dict of components each step, store it in:

	worker step logs (cheap)
	and optionally batch sample logs (small sample)

Then stuck becomes obvious:

	collision penalty spikes
	progress term flatlines
	step penalty accumulates


### Make insertion collapse detection “drop-based”, not “absolute threshold”

Replace:

“rolling median < 200mm”

With something like:

find best baseline window after heatup: baseline = median(top_k rolling medians)

require “good phase exists”: rolling median > good_threshold for ≥W episodes

collapse when rolling median < baseline * drop_ratio for ≥W episodes

This prevents false early triggers.


### Detect “stuck” from worker step logs (and make it actionable)

You already parse per-step actions + delta insertion in v3’s regex, but you don’t compute stuck runs.

Add per-episode stuck features computed while parsing:

stuck_step = (cmd_translation > t) AND (abs(delta_ins) < eps)

max_consecutive_stuck_steps

fraction_stuck_steps

mean_reward_during_stuck (use per-step reward from cum_reward differences)

optionally dominant_action_during_stuck (insert vs retract vs rotate)

Then define stuck events:

stuck_fraction_jump (e.g., >30% stuck steps sustained)

long_stuck_run (e.g., max run > 50 steps)

What happens in env when stuck?

episodes generally do not auto-terminate due to stuck

termination is typically only on TargetReached

truncation is usually MaxSteps or VesselEnd or SimError

action translation can be masked to 0 by the intervention (bounds/constraints), but reward is still computed normally

So if the agent is stuck, you’ll typically see:

delta insertion ~0 for many steps

step counter increases → truncation at MaxSteps

rewards drift based on penalties (or stagnate)

### Use batch_samples reward values to locate stuck even without env logs

Right now we’re not doing much with batch reward.

We can add analyses like:

reward conditional on |translation| bins (do big inserts produce worse reward later?)

reward conditional on “action effective” proxy (if we log delta insertion)

### What to add next (most useful, in order)

Add an episode_summary.jsonl emitted by Runner with explore/update step at episode end

In worker step logs, add executed action + per-step reward (not only cumulative)

Add “stuck features” computed online (fraction stuck, longest run) to episode summary

Then update diagnose_collapsev3.py to:

	detect collapse as a drop from a good baseline

	add stuck events + include “dominant termination reasons” around collapse time








 