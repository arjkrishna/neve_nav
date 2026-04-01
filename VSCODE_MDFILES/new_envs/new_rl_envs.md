


## 1) Main wiring (DualDeviceNav_train.py → env → agent → runner)

### Training script (`DualDeviceNav_train.py`)

* Builds `intervention = DualDeviceNav()` (two J-shaped devices, TrackingOnly fluoroscopy, random target on one of 4 supra-aortic branches).
* Picks env class by `--env_version`:

  * **v1:** `BenchEnv` (PathLengthDelta + step penalty + target reward)
  * **v2/v3:** `BenchEnv2/3` (adds centerlines obs + waypoint reward)
* Creates `env_train`, `env_eval`, then `BenchAgentSynchron(...)`, then `Runner(...).training_run(...)`.

### Env v1 (`util/env.py`)

* **Pathfinder:** `BruteForceBFS` (recomputes shortest path *from current tip → target* every step)
* **Reward:** `TargetReached(+1.0)` + `PathLengthDelta(factor=0.001)` + `Step(-0.005)`

  * This is *dense and smooth* because reward is proportional to **decrease in current shortest-path length**.

### Env v2/v3 (`util/env2.py`, `util/env3.py`)

* **Pathfinder:** `FixedPathfinder` (computes insertion→target path at reset, then fixed)
* **Observation adds:** `Centerlines2D` (all branches downsampled + path mask/order)
* **Reward replaces PathLengthDelta with:** `CenterlineWaypointProgress`

  * v2: waypoint spacing 10mm, branch increment 0.1, step penalty -0.005
  * v3: spacing 5mm, increment 1.0, step penalty -0.001

### Agent / algo (`util/agent.py` + `eve_rl`)

* `BenchAgentSynchron` spawns workers, uses SAC.
* Replay is **episode-based** (`VanillaEpisodeShared`): batches are padded sequences.
* SAC consumes `(batch, seq, obs_dim)` and learns with an LSTM head by default.

So the only *behavior-changing* differences between “works sometimes” (env.py) and “stuck at start” (env2/env3) are:

1. **reward definition** (PathLengthDelta vs waypoint-score delta)
2. **observation size/structure** (centerlines added)

---

## 2) Why env2/env3 can “backfire” into “wire doesn’t move”

Even with v3’s “balanced reward magnitudes”, there’s a structural failure mode in `CenterlineWaypointProgress` that can easily create a *do-nothing attractor*:

### A) The waypoint reward uses **global nearest waypoint across ALL branches**

`CenterlineWaypointProgress` finds the nearest waypoint among:

* positive-scored waypoints on the correct path, AND
* **negative-scored waypoints on wrong branches**

In an aortic-arch tree, multiple branches run close in Euclidean distance near the root / bifurcations. So “nearest waypoint” can **flip branches** purely due to geometry, even if the device is behaving correctly.

That creates:

* noisy score jumps (sometimes negative) when you start moving,
* inconsistent gradient signal early in training.

### B) Once SAC discovers “negative translation is safe”, you get “stuck at start”

In `MonoPlaneStatic.step()`, retract below zero is masked (`below_zero`), so **negative translation commands become executed translation = 0** at the start.

If the shaped reward makes “moving” risky/noisy and the step penalty exists, SAC can converge to:

* command retract / tiny motions → executed insertion stays ~0
* wire appears “stuck at start” even though the policy is “doing something” (it’s just masked)

This effect is *much* easier to fall into when the reward is discontinuous/noisy (nearest-waypoint flips), compared to env.py’s smooth `PathLengthDelta`.

### C) Observation explosion can worsen it

`Centerlines2D` adds a big, structured input. Since your networks treat everything as one flat vector, it’s easy for learning to become slower/more brittle unless you compress this information into low-dim “guidance features”.

---

## 3) If you already know the path: how to make training **more reliable + easier + faster**

Here are the highest-impact moves, ordered by “likely to help you immediately”:

---

### 3.1 Replace waypoint scoring with **continuous progress along the path**

You want the benefit of “known path” **without** nearest-waypoint discontinuities.

**Best option:** potential-style shaping using *projection onto the correct path polyline*.

Let `P` be the centerline polyline from insertion→target (you already have it via `FixedPathfinder.path_points_vessel_cs`).

At each step:

1. Project tip position onto the polyline → get arclength coordinate `s`
2. Remaining distance: `d_rem = L - s`
3. Reward: `r_prog = k * (d_rem_prev - d_rem_curr)` (positive if you make progress)
4. Add a small lateral penalty: `r_lat = -λ * dist_to_polyline` (keeps you near centerline)
5. Keep your terminal target reward.

This gives you:

* dense reward every step,
* smooth gradients,
* no branch-flip noise.

**Why this is better than env2/env3’s waypoint approach:** you’re measuring progress *along the path*, not “closest point in Euclidean space across the whole tree”.

---

### 3.2 Don’t feed the whole centerline — feed **local guidance features**

Instead of `Centerlines2D` (big flattened geometry), feed compact features like:

* `d_rem` (remaining arclength to goal)
* `dist_to_path` (cross-track error)
* `tangent_dir_2d` (path tangent at projection)
* `heading_error` (angle between device direction and tangent)
* `next_turn_angle` (curvature / upcoming bend)
* optionally: “distance to next bifurcation” + “which branch is correct” at that bifurcation

This reduces the function-approximation burden massively and makes training faster and more stable.

---

### 3.3 Make the action space easier (big win for DualDeviceNav)

DualDeviceNav is hard partly because you’re learning coordination of 2 devices from scratch.

Practical curricula:

1. **Stage 1:** control **only guidewire** (2D action). Make catheter follow with a rule:

   * catheter insertion = clamp(guidewire insertion − margin, 0, …)
   * catheter rotation = damped follow
2. **Stage 2:** enable catheter action but keep it “weak” (scale its action by 0.1 initially)
3. **Stage 3:** full control

This often boosts success rate dramatically because the learner doesn’t have to discover basic coordination.

---

### 3.4 Seed the replay buffer with a simple centerline follower (imitation-lite)

Since you know the path, you can generate “okay” trajectories with a heuristic controller:

* translate forward at a modest constant speed
* rotate to minimize cross-track error / align to tangent

Then:

* run ~200–1000 episodes of this to fill replay
* start SAC updates afterward

This usually avoids the early “collapse to masked no-op” failure.

---

### 3.5 Make training faster by changing replay + model choices

Right now you use **episode replay with padding** (slow) and an LSTM head.

If you don’t truly need sequence modeling (you already include some history in obs):

* switch to **step replay** (`VanillaStepShared`) and a pure **MLP head**
* increase batch size (e.g., 128–512)
* increase update ratio (e.g., from 1/20 → 1/5) once replay has enough data

This alone can make learning much faster per wall-clock.

---

## 4) Concrete “fix env2/env3 without throwing them away”

If you want to salvage the waypoint idea (instead of replacing it):

1. **Stop using global nearest waypoint across all branches.**

   * determine the **current tip branch** first
   * only consider waypoints on that branch (or on the current shortest-path branch)
2. Add **hysteresis** so branch assignment doesn’t flicker at bifurcations
3. Add explicit penalty for “masked translation at start”:

   * if commanded translation < 0 and executed translation == 0 → small penalty
   * this prevents the “retract-at-zero” attractor

But honestly, the continuous arclength-progress shaping is the cleaner solution.

---

## 5) What I’d do next (minimal changes, maximum payoff)

If your current env.py gives 10–20% after 80k–200k updates, I’d try this path-known upgrade first:

1. Keep env.py structure, but **replace** `CenterlineWaypointProgress` with:

   * projection-based `progress_delta` along the fixed path polyline
2. Add just 3–6 low-dim guidance features (don’t add full centerlines)
3. Use a 2-stage curriculum: guidewire-only → dual device

That combination is the most likely to turn “10–20% after 2k–5k episodes” into something substantially better with fewer updates.

If you want, I can sketch the exact code changes needed (which class/file to add the projection-based progress reward + what to wire into `BenchEnv2/3`) using your current `FixedPathfinder.path_points_vessel_cs` and tip position in vessel CS.
