"""Heuristic policy wrapper for parallel heuristic seeding.

This module provides a way to use the CenterlineFollowerHeuristic
with the eve_rl worker infrastructure, enabling parallel heuristic
seeding instead of sequential.

Usage:
    from util.heuristic_policy import create_heuristic_action_function

    # In heatup, use heuristic instead of random:
    action_fn = create_heuristic_action_function(env)
    episode, steps = agent._play_episode(
        env=env,
        action_function=action_fn,
        consecutive_actions=1,
    )
"""

import numpy as np
from typing import Callable, Optional


class HeuristicActionFunction:
    """Wraps CenterlineFollowerHeuristic as an action function for workers.

    This class creates a heuristic controller from an environment and
    provides a callable interface compatible with _play_episode().

    The action function returns normalized actions in [-1, 1] when
    normalize_actions=True is used by the agent.
    """

    def __init__(
        self,
        env,
        noise_std: float = 0.0,
        normalize_output: bool = True,
    ):
        """Initialize heuristic action function.

        Args:
            env: The environment (must have .pathfinder and .intervention)
            noise_std: Standard deviation of Gaussian noise to add (default 0)
            normalize_output: If True, return actions in [-1, 1] range
        """
        from .heuristic_controller import CenterlineFollowerHeuristic

        # Unwrap gymnasium wrappers (e.g. ActionCurriculumWrapper) to access
        # pathfinder/intervention on the base BenchEnv4/5.
        base_env = getattr(env, 'unwrapped', env)

        self.env = env
        self.noise_std = noise_std
        self.normalize_output = normalize_output
        self._rng = np.random.default_rng()
        self._needs_reset = True  # Lazy reset flag

        # Create heuristic controller from the unwrapped environment
        self.heuristic = CenterlineFollowerHeuristic(
            pathfinder=base_env.pathfinder,
            intervention=base_env.intervention,
        )

        # Get action space bounds for normalization (same on wrapper and base)
        self.action_low = env.action_space.low.flatten().astype(np.float64)
        self.action_high = env.action_space.high.flatten().astype(np.float64)
        self.action_range = self.action_high - self.action_low

    def reset(self):
        """Mark for lazy reset — actual heuristic.reset() happens on the
        first __call__ AFTER _play_episode has called env.reset(), so the
        pathfinder already has fresh path data."""
        self._needs_reset = True

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Get action from heuristic controller.

        Args:
            obs: Observation (ignored - heuristic uses pathfinder directly)

        Returns:
            Action as flat numpy array, normalized to [-1, 1] if normalize_output=True
        """
        # Lazy reset: env.reset() has already been called by _play_episode,
        # so pathfinder.path_points_vessel_cs is current.
        if self._needs_reset:
            self.heuristic.reset()
            self._needs_reset = False

        # Get raw action from heuristic (in environment action space)
        raw_action = self.heuristic.get_action(self._rng)

        # Ensure it's a flat numpy array
        action = np.asarray(raw_action).flatten().astype(np.float64)

        # Add noise if requested
        if self.noise_std > 0:
            noise = self._rng.normal(0, self.noise_std, size=action.shape)
            action = action + noise
            # Clip to valid range
            action = np.clip(action, self.action_low, self.action_high)

        # Normalize to [-1, 1] if requested
        if self.normalize_output:
            action = 2 * (action - self.action_low) / self.action_range - 1
            action = np.clip(action, -1.0, 1.0)

        return action.astype(np.float32)


def create_heuristic_action_function(
    env,
    noise_std: float = 0.0,
    normalize_output: bool = True,
) -> HeuristicActionFunction:
    """Factory function to create a heuristic action function for an environment.

    Args:
        env: The environment (must have .pathfinder and .intervention)
        noise_std: Standard deviation of Gaussian noise to add
        normalize_output: If True, return actions in [-1, 1] range

    Returns:
        HeuristicActionFunction instance
    """
    return HeuristicActionFunction(
        env=env,
        noise_std=noise_std,
        normalize_output=normalize_output,
    )


# For pickling support - workers need to recreate the heuristic
class HeuristicActionFunctionFactory:
    """Pickleable factory that creates HeuristicActionFunction in worker process.

    Since HeuristicActionFunction contains references to the environment's
    pathfinder and intervention, it can't be pickled directly. This factory
    is pickled instead and creates the action function in the worker.
    """

    def __init__(self, noise_std: float = 0.0, normalize_output: bool = True):
        self.noise_std = noise_std
        self.normalize_output = normalize_output

    def create(self, env) -> HeuristicActionFunction:
        """Create HeuristicActionFunction for the given environment."""
        return HeuristicActionFunction(
            env=env,
            noise_std=self.noise_std,
            normalize_output=self.normalize_output,
        )
