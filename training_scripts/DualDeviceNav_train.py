import os
import logging
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch
from util.util import get_result_checkpoint_config_and_log_path
from util.env import BenchEnv
from util.env2 import BenchEnv2  # NEW: Centerline-aware environment with waypoint rewards
from util.env3 import BenchEnv3  # NEW: Tuned rewards for better balance
from util.env4 import BenchEnv4  # Arclength progress + local guidance
from util.env5 import BenchEnv5  # Optimized env4 with shared projection cache
from util.agent import BenchAgentSynchron
from eve_rl import Runner
from eve_bench import DualDeviceNav


RESULTS_FOLDER = os.getcwd() + "/results/eve_paper/neurovascular/full/mesh_ben"

# Target branches for DualDeviceNav (4 supra-aortic branches)
TARGET_BRANCHES = [
    "Centerline curve - LCCA.mrk",
    "Centerline curve - LVA.mrk",
    "Centerline curve - RCCA.mrk",
    "Centerline curve - RVA.mrk",
]


def build_episode_schedule(n_episodes, branches, base_seed=42, heuristic_mode=False):
    """Build a branch-balanced, reproducibly-shuffled episode schedule.

    Returns a list of (seed, options) tuples where branches are assigned
    round-robin so each branch gets exactly n_episodes // len(branches)
    episodes (plus remainder distributed across the first branches).

    Args:
        n_episodes: Total number of episodes to schedule.
        branches: List of target branch names.
        base_seed: Base seed for reproducible seed generation.
        heuristic_mode: If True, add heuristic_mode=True to options
            (enables env-side abort detectors during heuristic seeding).

    Returns:
        List of (seed, options) tuples, one per episode.
    """
    rng = np.random.default_rng(base_seed)
    schedule = []
    for i in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        branch = branches[i % len(branches)]
        options = {"target_branch": branch}
        if heuristic_mode:
            options["heuristic_mode"] = True
        schedule.append((ep_seed, options))
    # Shuffle to avoid workers getting only one branch each,
    # but deterministically so runs are reproducible.
    rng.shuffle(schedule)
    return schedule

EVAL_SEEDS = "1,2,3,5,6,7,8,9,10,12,13,14,16,17,18,21,22,23,27,31,34,35,37,39,42,43,44,47,48,50,52,55,56,58,61,62,63,68,69,70,71,73,79,80,81,84,89,91,92,93,95,97,102,103,108,109,110,115,116,117,118,120,122,123,124,126,127,128,129,130,131,132,134,136,138,139,140,141,142,143,144,147,148,149,150,151,152,154,155,156,158,159,161,162,167,168,171,175"
EVAL_SEEDS = EVAL_SEEDS.split(",")
EVAL_SEEDS = [int(seed) for seed in EVAL_SEEDS]
HEATUP_STEPS = 2e4  # 20k steps
TRAINING_STEPS = 2e7
CONSECUTIVE_EXPLORE_EPISODES = 100
EXPLORE_STEPS_BTW_EVAL = 2.5e5


GAMMA = 0.99
REWARD_SCALING = 1
REPLAY_BUFFER_SIZE = 1e4
CONSECUTIVE_ACTION_STEPS = 1
BATCH_SIZE = 32
UPDATE_PER_EXPLORE_STEP = 1 / 20


LR_END_FACTOR = 0.15
LR_LINEAR_END_STEPS = 6e6

DEBUG_LEVEL = logging.DEBUG

# HEATUP_STEPS = 5e3
# TRAINING_STEPS = 1e7
# CONSECUTIVE_EXPLORE_EPISODES = 10
# EXPLORE_STEPS_BTW_EVAL = 7.5e3
# EVAL_SEEDS = list(range(20))
# RESULTS_FOLDER = os.getcwd() + "/results/test"
# BATCH_SIZE = 8
# UPDATE_PER_EXPLORE_STEP = 1 / 200


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="perform IJCARS23 training")
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=4, help="Number of workers"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device of trainer, wehre the NN update is performed. ",
        choices=["cpu", "cuda:0", "cuda:1", "cuda"],
    )
    parser.add_argument(
        "-se",
        "--stochastic_eval",
        action="store_true",
        help="Runs optuna run with stochastic eval function of SAC.",
    )
    parser.add_argument(
        "-n", "--name", type=str, default="test", help="Name of the training run"
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.00021989352630306626,
        help="Learning Rate of Optimizers",
    )
    parser.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=[900, 900, 900, 900],
        help="Hidden Layers",
    )
    parser.add_argument(
        "-en",
        "--embedder_nodes",
        type=int,
        default=500,
        help="Number of nodes per layer in embedder",
    )
    parser.add_argument(
        "-el",
        "--embedder_layers",
        type=int,
        default=1,
        help="Number of layers in embedder",
    )
    parser.add_argument(
        "--env_version",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Environment version: 1=original (PathDelta), 2=waypoint+centerlines, 3=tuned waypoint, 4=arclength progress+guidance, 5=optimized env4 with shared projection cache",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable action-space curriculum (Stage 1: gw-only, Stage 2: scaled catheter, Stage 3: full)",
    )
    parser.add_argument(
        "--curriculum_stage1",
        type=int,
        default=200_000,
        help="Steps in curriculum Stage 1 (guidewire-only)",
    )
    parser.add_argument(
        "--curriculum_stage2",
        type=int,
        default=500_000,
        help="Cumulative steps at which curriculum Stage 2 ends",
    )
    parser.add_argument(
        "--heuristic_seeding",
        type=int,
        default=0,
        help="Number of heuristic episodes to seed replay buffer (0=disabled, recommended: 500)",
    )
    parser.add_argument(
        "--min_success_rate",
        type=float,
        default=0.3,
        help="Minimum fraction of successful episodes in heuristic seeding (default: 0.3)",
    )
    parser.add_argument(
        "--max_seeding_multiplier",
        type=int,
        default=5,
        help="Max total episodes = heuristic_seeding * this (safety cap, default: 5)",
    )
    parser.add_argument(
        "--heuristic_cache_file",
        type=str,
        default=None,
        help="Path to load pre-generated heuristic episodes from (skips generation)",
    )
    parser.add_argument(
        "--save_heuristic_cache",
        type=str,
        default=None,
        help="Path to save generated heuristic episodes to (for reuse)",
    )
    parser.add_argument(
        "--heatup_cache_file",
        type=str,
        default=None,
        help="Path to load pre-generated heatup episodes from (skips heatup)",
    )
    parser.add_argument(
        "--save_heatup_cache",
        type=str,
        default=None,
        help="Path to save heatup episodes to (for reuse)",
    )
    args = parser.parse_args()

    trainer_device = torch.device(args.device)
    n_worker = args.n_worker
    trial_name = args.name
    stochastic_eval = args.stochastic_eval
    lr = args.learning_rate
    hidden_layers = args.hidden
    embedder_nodes = args.embedder_nodes
    embedder_layers = args.embedder_layers
    env_version = args.env_version
    worker_device = torch.device("cpu")

    # Select environment class based on version
    if env_version == 5:
        EnvClass = BenchEnv5
        print("Using BenchEnv5 (optimized: shared projection cache + logging fixes)")
    elif env_version == 4:
        EnvClass = BenchEnv4
        print("Using BenchEnv4 (arclength progress reward + local guidance observation)")
    elif env_version == 3:
        EnvClass = BenchEnv3
        print("Using BenchEnv3 (TUNED waypoint rewards: 5mm spacing, 1.0 increment, -0.001 step penalty)")
    elif env_version == 2:
        EnvClass = BenchEnv2
        print("Using BenchEnv2 (waypoint rewards + centerline observations)")
    else:
        EnvClass = BenchEnv
        print("Using BenchEnv (original PathDelta reward)")

    custom_parameters = {
        "lr": lr,
        "hidden_layers": hidden_layers,
        "embedder_nodes": embedder_nodes,
        "embedder_layers": embedder_layers,
        "env_version": env_version,
        "HEATUP_STEPS": HEATUP_STEPS,
        "EXPLORE_STEPS_BTW_EVAL": EXPLORE_STEPS_BTW_EVAL,
        "CONSECUTIVE_EXPLORE_EPISODES": CONSECUTIVE_EXPLORE_EPISODES,
        "BATCH_SIZE": BATCH_SIZE,
        "UPDATE_PER_EXPLORE_STEP": UPDATE_PER_EXPLORE_STEP,
    }

    (
        results_file,
        checkpoint_folder,
        config_folder,
        log_file,
        _config_folder_dup,
        diagnostics_folder,
    ) = get_result_checkpoint_config_and_log_path(
        all_results_folder=RESULTS_FOLDER, name=trial_name, create_diagnostics=True
    )

    # CRITICAL: Set STEP_LOG_DIR BEFORE agent is created, so worker processes inherit it
    # Workers are spawned in BenchAgentSynchron.__init__, which happens before Runner.__init__
    if diagnostics_folder is not None:
        logs_subprocesses = os.path.join(diagnostics_folder, "logs_subprocesses")
        os.makedirs(logs_subprocesses, exist_ok=True)
        os.environ["STEP_LOG_DIR"] = logs_subprocesses
        print(f"Set STEP_LOG_DIR={logs_subprocesses}")

    logging.basicConfig(
        filename=log_file,
        level=DEBUG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Diagnostics configuration for SAC training monitoring
    diagnostics_config = {
        "enabled": True,
        "diagnostics_folder": diagnostics_folder,
        "log_losses_every_n_steps": 1,
        "log_probe_values_every_n_steps": 100,
        "log_batch_samples_every_n_steps": 100,  # Log batch samples every 100 gradient steps
        "n_batch_samples": 3,  # Number of transitions to sample from each batch
        "tensorboard_enabled": True,
        "probe_plot_every_n_steps": 10000,
        "policy_snapshot_every_n_steps": 10000,
        "flush_every_n_rows": 100,
    }

    # NOTE: DualDeviceNav() uses hardcoded defaults from eve_bench/eve_bench/dualdevicenav.py:
    #   - rotation_yzx_deg = [90, -90, 0]  (line 36)
    #   - fluoroscopy_rot_zx = [20, 5]     (line 82)
    # These match the preprocessing in the original DualDeviceNav data (model 0105)
    intervention = DualDeviceNav()
    env_train_unwrapped = EnvClass(intervention=intervention, mode="train", visualisation=False)
    env_train = env_train_unwrapped  # May be wrapped below

    # Wrap with action curriculum if requested
    if args.curriculum:
        from util.action_curriculum import ActionCurriculumWrapper
        env_train = ActionCurriculumWrapper(
            env_train_unwrapped,
            stage1_steps=args.curriculum_stage1,
            stage2_steps=args.curriculum_stage2,
        )
        print(f"Action curriculum enabled: Stage1={args.curriculum_stage1}, Stage2={args.curriculum_stage2}")

    intervention_eval = DualDeviceNav()
    env_eval = EnvClass(
        intervention=intervention_eval, mode="eval", visualisation=False
    )
    agent = BenchAgentSynchron(
        trainer_device,
        worker_device,
        lr,
        LR_END_FACTOR,
        LR_LINEAR_END_STEPS,
        hidden_layers,
        embedder_nodes,
        embedder_layers,
        GAMMA,
        BATCH_SIZE,
        REWARD_SCALING,
        REPLAY_BUFFER_SIZE,
        env_train,
        env_eval,
        CONSECUTIVE_ACTION_STEPS,
        n_worker,
        stochastic_eval,
        False,
        diagnostics_config=diagnostics_config,
    )

    # Save config from unwrapped env (ActionCurriculumWrapper has no save_config)
    env_train_config = os.path.join(config_folder, "env_train.yml")
    env_train_unwrapped.save_config(env_train_config)
    env_eval_config = os.path.join(config_folder, "env_eval.yml")
    env_eval.save_config(env_eval_config)
    infos = list(env_eval.info.info.keys())
    runner = Runner(
        agent=agent,
        heatup_action_low=[[-10.0, -1.5], [-10.0, -1.5]],
        heatup_action_high=[[30.0, 1.5], [30.0, 1.5]],
        agent_parameter_for_result_file=custom_parameters,
        checkpoint_folder=checkpoint_folder,
        results_file=results_file,
        info_results=infos,
        quality_info="success",
        diagnostics_folder=diagnostics_folder,
        policy_snapshot_every_steps=10000,
    )
    runner_config = os.path.join(config_folder, "runner.yml")
    runner.save_config(runner_config)

    # Optional: seed replay buffer with heuristic episodes
    # Collects episodes using a centerline-following heuristic and pushes
    # them into the agent's replay buffer BEFORE SAC training starts.
    # This avoids the early "collapse to masked no-op" failure mode.
    # Uses parallel workers for fast seeding (same infrastructure as explore).
    if args.heuristic_seeding > 0 and env_version in (4, 5):
        from eve_rl.replaybuffer import EpisodeReplay
        from eve_rl.util.experience_cache import save_episodes_npz, load_episodes_npz

        # Try to load from cache first
        if args.heuristic_cache_file and os.path.isfile(args.heuristic_cache_file):
            print(f"Loading heuristic cache from {args.heuristic_cache_file}...")
            episodes_tuples, _ = load_episodes_npz(args.heuristic_cache_file)
            n_pushed = 0
            for ep_tuple in episodes_tuples:
                flat_obs, actions, rewards, terminals = ep_tuple
                replay_ep = EpisodeReplay(
                    flat_obs=list(flat_obs),
                    actions=list(actions),
                    rewards=list(rewards),
                    terminals=list(terminals),
                )
                agent.replay_buffer.push(replay_ep)
                n_pushed += 1
            print(f"Heuristic cache loaded: {n_pushed} episodes pushed to replay buffer.")
        else:
            # Parallel heuristic seeding with minimum success rate guarantee
            from util.heuristic_policy import HeuristicActionFunctionFactory
            import math

            N = args.heuristic_seeding
            min_successes = math.ceil(args.min_success_rate * N)
            max_total = N * args.max_seeding_multiplier

            factory = HeuristicActionFunctionFactory(
                noise_std=0.0,
                normalize_output=True,
            )

            def _is_success(ep):
                """Check episode success via info dict (clearer contract than terminals)."""
                if ep.infos and "success" in ep.infos[-1]:
                    return bool(ep.infos[-1]["success"])
                # Fallback to terminals if info not available
                return bool(ep.terminals[-1]) if ep.terminals else False

            all_episodes = []
            batch_num = 0
            seed_offset = 0
            # Dedicated RNG for failure sampling — derived from base seed
            # so the final selected set is fully reproducible.
            selection_rng = np.random.default_rng(42 + 999)

            while True:
                batch_num += 1
                if batch_num == 1:
                    batch_size = N
                else:
                    # Deficit-based: estimate how many more episodes needed
                    n_success_so_far = sum(1 for ep in all_episodes if _is_success(ep))
                    needed = min_successes - n_success_so_far
                    observed_rate = max(n_success_so_far / len(all_episodes), 0.05)
                    batch_size = math.ceil(needed / observed_rate * 1.5)
                    batch_size = min(batch_size, max_total - len(all_episodes))

                if batch_size <= 0:
                    break

                schedule = build_episode_schedule(
                    batch_size, TARGET_BRANCHES, base_seed=42 + seed_offset, heuristic_mode=True
                )
                seed_offset += batch_size

                batch_episodes = agent.heuristic_seed(
                    episodes=batch_size,
                    heuristic_factory=factory,
                    episode_schedule=schedule,
                    push_to_buffer=False,
                )
                all_episodes.extend(batch_episodes)

                n_success = sum(1 for ep in all_episodes if _is_success(ep))
                batch_success = sum(1 for ep in batch_episodes if _is_success(ep))
                print(f"  Batch {batch_num}: {len(batch_episodes)} episodes, "
                      f"{batch_success} successes | "
                      f"Total: {len(all_episodes)} episodes, {n_success} successes "
                      f"({100*n_success/len(all_episodes):.1f}%)")

                if n_success >= min_successes:
                    break
                if len(all_episodes) >= max_total:
                    print(f"  WARNING: hit max seeding cap ({max_total} episodes) "
                          f"with only {n_success} successes")
                    break

            # Filter: keep all successes + enough failures for healthy mix
            successes = [ep for ep in all_episodes if _is_success(ep)]
            failures = [ep for ep in all_episodes if not _is_success(ep)]

            n_success = len(successes)
            # Ensure success ratio <= 70%: pad with failures if needed
            min_failures = math.ceil(n_success / 0.7) - n_success
            # Also fill up to at least N total
            n_failures_needed = max(min_failures, N - n_success)
            n_failures_needed = max(n_failures_needed, 0)

            if n_failures_needed > 0 and len(failures) > 0:
                n_sample = min(n_failures_needed, len(failures))
                sampled_failures = list(
                    selection_rng.choice(failures, size=n_sample, replace=False)
                )
            else:
                sampled_failures = []

            to_push = successes + sampled_failures
            for ep in to_push:
                agent.replay_buffer.push(ep)

            print(f"Heuristic seeding complete: pushing {n_success} successes + "
                  f"{len(sampled_failures)} failures = {len(to_push)} episodes "
                  f"({100*n_success/len(to_push):.1f}% success rate)")

            # Save cache — saves the final selected set, not all attempts
            if args.save_heuristic_cache and to_push:
                episodes_to_save = []
                for ep in to_push:
                    episodes_to_save.append((
                        np.array(ep.flat_obs),
                        np.array(ep.actions),
                        np.array(ep.rewards),
                        np.array(ep.terminals),
                    ))
                os.makedirs(os.path.dirname(args.save_heuristic_cache) or ".", exist_ok=True)
                save_episodes_npz(args.save_heuristic_cache, episodes_to_save)
                print(f"Saved heuristic cache to {args.save_heuristic_cache}")

    # Load heatup cache if provided (skips heatup phase)
    heatup_steps_effective = HEATUP_STEPS
    if args.heatup_cache_file and os.path.isfile(args.heatup_cache_file):
        from eve_rl.replaybuffer import EpisodeReplay
        from eve_rl.util.experience_cache import load_episodes_npz

        print(f"Loading heatup cache from {args.heatup_cache_file}...")
        episodes_tuples, _ = load_episodes_npz(args.heatup_cache_file)
        n_pushed = 0
        for ep_tuple in episodes_tuples:
            flat_obs, actions, rewards, terminals = ep_tuple
            replay_ep = EpisodeReplay(
                flat_obs=list(flat_obs),
                actions=list(actions),
                rewards=list(rewards),
                terminals=list(terminals),
            )
            agent.replay_buffer.push(replay_ep)
            n_pushed += 1
        print(f"Heatup cache loaded: {n_pushed} episodes. Skipping heatup phase.")
        heatup_steps_effective = 0

    # Determine heatup cache save path (only if not loading from cache)
    heatup_cache_save = args.save_heatup_cache if not args.heatup_cache_file else None

    reward, success = runner.training_run(
        heatup_steps_effective,
        TRAINING_STEPS,
        EXPLORE_STEPS_BTW_EVAL,
        CONSECUTIVE_EXPLORE_EPISODES,
        UPDATE_PER_EXPLORE_STEP,
        eval_seeds=EVAL_SEEDS,
        heatup_cache_save_path=heatup_cache_save,
    )
    agent.close()
