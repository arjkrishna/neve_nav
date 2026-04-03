#!/usr/bin/env python
"""
Visualize a trained policy on the DualDeviceNav mesh.

Usage:
  # Using full checkpoint:
  python visualize_policy.py --checkpoint <path_to_checkpoint.everl> [--episodes 5] [--seed 42]
  python visualize_policy.py --name diag_debug_test9  # Auto-finds best checkpoint

  # Using policy snapshot (smaller, faster to load):
  python visualize_policy.py --snapshot <path_to_policy_XXXXX.pt> --checkpoint <any_checkpoint.everl>
  python visualize_policy.py --name diag_debug_test9 --snapshot-step 1000000  # Use specific snapshot

Requires X11 display forwarding in Docker:
  docker run -e DISPLAY=10.113.123.221:0 ...
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import eve
import eve_rl
from eve.visualisation import SofaPygame
from eve_bench import DualDeviceNav


def find_checkpoint(name: str, results_base: str = "training_scripts/results/eve_paper/neurovascular/full/mesh_ben") -> Path:
    """Find the best checkpoint for an experiment by name."""
    base_dir = Path(results_base).expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Results base dir does not exist: {base_dir}")
    
    matching = [d for d in base_dir.iterdir() if d.is_dir() and name in d.name]
    if not matching:
        raise FileNotFoundError(f"No experiment folder matching '{name}' found in {base_dir}")
    
    # Sort by modification time (newest first)
    matching.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    exp_dir = matching[0]
    
    # Look for best_checkpoint first, then latest checkpoint
    checkpoint_dir = exp_dir / "checkpoints"
    best = checkpoint_dir / "best_checkpoint.everl"
    if best.exists():
        return best
    
    # Find latest checkpoint by number
    checkpoints = list(checkpoint_dir.glob("checkpoint*.everl"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    checkpoints.sort(key=lambda p: int(p.stem.replace("checkpoint", "")), reverse=True)
    return checkpoints[0]


def find_snapshot(exp_dir: Path, step: int = None) -> Path:
    """Find a policy snapshot in the experiment's diagnostics folder."""
    snapshots_dir = exp_dir / "diagnostics" / "policy_snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"No policy_snapshots folder found in {exp_dir}")
    
    snapshots = list(snapshots_dir.glob("policy_*.pt"))
    if not snapshots:
        raise FileNotFoundError(f"No policy snapshots found in {snapshots_dir}")
    
    if step is not None:
        # Find specific step
        target = snapshots_dir / f"policy_{step}.pt"
        if target.exists():
            return target
        raise FileNotFoundError(f"Snapshot for step {step} not found: {target}")
    
    # Find latest snapshot
    snapshots.sort(key=lambda p: int(p.stem.replace("policy_", "")), reverse=True)
    return snapshots[0]


def load_algo_from_snapshot(snapshot_path: Path, checkpoint_path: Path):
    """
    Load algorithm from a lightweight policy snapshot.
    
    The checkpoint is needed to get the model architecture config.
    The snapshot provides the actual policy weights (potentially from a different training step).
    """
    # Load checkpoint to get model architecture
    algo = eve_rl.algo.AlgoPlayOnly.from_checkpoint(str(checkpoint_path))
    
    # Load snapshot weights
    snapshot = torch.load(str(snapshot_path), map_location="cpu")
    
    # Replace policy weights with snapshot
    if "policy" in snapshot:
        algo.model.policy.load_state_dict(snapshot["policy"])
        print(f"  Loaded policy weights from snapshot (step {snapshot.get('explore_step', '?')})")
    else:
        raise ValueError(f"Snapshot does not contain 'policy' key: {list(snapshot.keys())}")
    
    return algo


def main():
    parser = argparse.ArgumentParser(description="Visualize trained policy on DualDeviceNav mesh")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file (.everl)")
    parser.add_argument("--name", type=str, help="Experiment name (auto-finds best checkpoint)")
    parser.add_argument("--snapshot", type=str, help="Path to policy snapshot file (.pt)")
    parser.add_argument("--snapshot-step", type=int, help="Use snapshot from this explore step (requires --name)")
    parser.add_argument("--results-base", type=str, 
                        default="training_scripts/results/eve_paper/neurovascular/full/mesh_ben",
                        help="Base results directory for --name search")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=42, help="Starting seed for evaluation")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions (vs deterministic)")
    args = parser.parse_args()
    
    # Resolve paths
    exp_dir = None
    checkpoint_path = None
    snapshot_path = None
    
    if args.name:
        # Find experiment directory
        base_dir = Path(args.results_base).expanduser().resolve()
        matching = [d for d in base_dir.iterdir() if d.is_dir() and args.name in d.name]
        if not matching:
            raise FileNotFoundError(f"No experiment matching '{args.name}' found")
        matching.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        exp_dir = matching[0]
        print(f"Found experiment: {exp_dir}")
        
        # Find checkpoint (always needed for model architecture)
        checkpoint_path = find_checkpoint(args.name, args.results_base)
        
        # Find snapshot if requested
        if args.snapshot_step:
            snapshot_path = find_snapshot(exp_dir, args.snapshot_step)
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    
    if args.snapshot:
        snapshot_path = Path(args.snapshot).expanduser().resolve()
    
    if not checkpoint_path:
        parser.error("Either --checkpoint or --name is required")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load algorithm
    if snapshot_path:
        print(f"Loading model architecture from: {checkpoint_path}")
        print(f"Loading policy weights from: {snapshot_path}")
        algo = load_algo_from_snapshot(snapshot_path, checkpoint_path)
    else:
        print(f"Loading full checkpoint: {checkpoint_path}")
        algo = eve_rl.algo.AlgoPlayOnly.from_checkpoint(str(checkpoint_path))
    
    # Load environment from checkpoint (preserves exact config - BenchEnv/BenchEnv2/BenchEnv3)
    env = eve_rl.util.get_env_from_checkpoint(str(checkpoint_path), "eval")
    env.intervention.make_non_mp()  # Required for visualization
    
    # Override max steps if specified
    if args.max_steps:
        env.truncation.max_steps = args.max_steps
    
    # Enable visualization
    visu = SofaPygame(env.intervention)
    env.visualisation = visu
    
    print(f"\nRunning {args.episodes} episodes with {'stochastic' if args.stochastic else 'deterministic'} actions")
    print(f"Starting seed: {args.seed}, max steps: {args.max_steps}")
    print("-" * 60)
    
    successes = 0
    total_rewards = []
    
    seed = args.seed
    for ep in range(args.episodes):
        algo.reset()
        # Gymnasium returns (observation, info) tuple, older gym returns just observation
        reset_result = env.reset(seed=seed)
        if isinstance(reset_result, tuple):
            obs, reset_info = reset_result
        else:
            obs = reset_result
        
        obs_flat, _ = eve_rl.util.flatten_obs(obs)
        
        episode_reward = 0.0
        steps = 0
        
        while True:
            if args.stochastic:
                action = algo.get_exploration_action(obs_flat)
            else:
                action = algo.get_eval_action(obs_flat)
            
            obs, reward, terminal, trunc, info = env.step(action)
            obs_flat, _ = eve_rl.util.flatten_obs(obs)
            episode_reward += reward
            steps += 1
            
            # Render visualization
            env.render()
            
            if terminal or trunc:
                break
        
        success = info.get("success", False)
        if success:
            successes += 1
        total_rewards.append(episode_reward)
        
        status = "SUCCESS" if success else "FAIL"
        print(f"Episode {ep + 1}/{args.episodes} (seed={seed}): {status}, reward={episode_reward:.4f}, steps={steps}")
        
        seed += 1
    
    print("-" * 60)
    print(f"Results: {successes}/{args.episodes} successes ({100*successes/args.episodes:.1f}%)")
    print(f"Average reward: {sum(total_rewards)/len(total_rewards):.4f}")
    
    algo.close()
    env.close()


if __name__ == "__main__":
    main()
