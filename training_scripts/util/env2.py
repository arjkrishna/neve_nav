"""
env2.py - Enhanced BenchEnv with centerline-aware observation and reward

This is a duplicate of env.py with the following enhancements:
1. Centerlines2D observation: exposes vessel centerlines to the policy
2. CenterlineProgress reward: rewards progress along the root→target path
3. Uses DijkstraPathfinder for weighted shortest path computation

Changes from env.py:
- Added Centerlines2D observation in ObsDict
- Added CenterlineProgress reward in Combination
- Uses DijkstraPathfinder instead of BruteForceBFS for path computation
"""

import eve
import eve.visualisation
import time
import logging
import sys
import os


def setup_step_logger(name="step_logger"):
    """Create a logger that flushes immediately after each log message."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create a handler that writes to a dedicated file
    # Use the current working directory's results folder if available
    log_dir = os.environ.get("STEP_LOG_DIR", "/tmp")
    log_file = os.path.join(log_dir, "step_by_step.log")
    
    try:
        handler = logging.FileHandler(log_file, mode='a')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except Exception as e:
        # Fallback to stderr if file creation fails
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Also add stderr handler for immediate visibility
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)  # Less verbose on stderr
    stderr_formatter = logging.Formatter('STEP: %(message)s')
    stderr_handler.setFormatter(stderr_formatter)
    logger.addHandler(stderr_handler)
    
    return logger


class BenchEnv2(eve.Env):
    """
    Enhanced BenchEnv with centerline-aware observation and waypoint-based reward.
    
    Adds:
    - Centerlines2D observation: downsampled centerline points in 2D + path mask
    - CenterlineWaypointProgress reward: positive reward for passing waypoints
    - Uses DijkstraPathfinder for proper weighted shortest path
    
    Waypoint rewards:
    - Dynamic spacing (2-4 forward actions per waypoint)
    - Increasing reward within branches (0.1, 0.13, 0.16...)
    - Jump when entering next branch (0.2, 0.24...)
    - Target reached = 1.0 (from TargetReached)
    """
    
    def __init__(
        self,
        intervention: eve.intervention.SimulatedIntervention,
        mode: str = "train",
        visualisation: bool = False,
        n_max_steps=1000,
        # Parameters for centerline observation
        n_points_per_branch: int = 10,  # Downsample each branch to this many points
        # Parameters for waypoint reward
        waypoint_spacing_mm: float = 10.0,    # Fixed 10mm spacing between waypoints
        branch_base_increment: float = 0.1,   # Score jump at branch change
        wrong_branch_penalty: float = 0.01,   # Not used directly (for API compat)
    ) -> None:
        self.mode = mode
        self.visualisation = visualisation
        
        # Step-level logging
        self._step_logger = setup_step_logger(f"step_logger_{mode}_{id(self)}")
        self._step_count = 0
        self._episode_count = 0
        self._episode_step_count = 0
        self._last_step_time = None
        self._episode_start_time = None
        self._episode_total_reward = 0.0
        self._step_logger.info(f"=== BenchEnv2 initialized (mode={mode}, visualisation={visualisation}) ===")
        sys.stderr.flush()
        
        start = eve.start.InsertionPoint(intervention)
        
        # Use FixedPathfinder - computes path from insertion point to target ONCE at reset
        # Much more efficient than DijkstraPathfinder which recomputes every step
        # The waypoint reward only needs the fixed path, not dynamic updates
        pathfinder = eve.pathfinder.FixedPathfinder(intervention=intervention)
        
        # ============ Observation ============
        
        # Standard tracking observation (device position)
        tracking = eve.observation.Tracking2D(intervention, n_points=3, resolution=2)
        tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(
            tracking, intervention
        )
        tracking = eve.observation.wrapper.Memory(
            tracking, 2, eve.observation.wrapper.MemoryResetMode.FILL
        )
        
        # Standard target observation
        target_state = eve.observation.Target2D(intervention)
        target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
            target_state, intervention
        )
        
        # Last action observation
        last_action = eve.observation.LastAction(intervention)
        last_action = eve.observation.wrapper.Normalize(last_action)
        
        # NEW: Centerlines observation - exposes vessel geometry to policy
        centerlines = eve.observation.Centerlines2D(
            intervention=intervention,
            pathfinder=pathfinder,
            n_points_per_branch=n_points_per_branch,
        )
        # Normalize centerlines to same scale as tracking
        centerlines = eve.observation.wrapper.NormalizeCenterlines2DEpisode(
            centerlines, intervention
        )

        observation = eve.observation.ObsDict(
            {
                "tracking": tracking,
                "target": target_state,
                "last_action": last_action,
                "centerlines": centerlines,  # NEW: centerline points + path mask
            }
        )

        # ============ Reward ============
        
        # Target reached reward (factor=1.0 for reaching target)
        target_reward = eve.reward.TargetReached(
            intervention,
            factor=1.0,
            final_only_after_all_interim=False,
        )
        
        # Step penalty (small constant penalty per step for urgency)
        step_reward = eve.reward.Step(factor=-0.005)
        
        # Waypoint progress reward:
        # - Positive reward for passing waypoints along the path
        # - Increasing reward within branches (0.1, 0.13, 0.16...)
        # - Jump at branch change (0.2, 0.24...)
        # - Negative for going backward or wrong branch
        waypoint_progress = eve.reward.CenterlineWaypointProgress(
            intervention=intervention,
            pathfinder=pathfinder,
            waypoint_spacing_mm=waypoint_spacing_mm,
            branch_base_increment=branch_base_increment,
            wrong_branch_penalty=wrong_branch_penalty,
        )
        
        # No PathLengthDelta - replaced by waypoint-based reward
        reward = eve.reward.Combination([
            target_reward,
            step_reward,
            waypoint_progress,
        ])

        # ============ Terminal and Truncation ============
        terminal = eve.terminal.TargetReached(intervention)

        max_steps = eve.truncation.MaxSteps(n_max_steps)
        vessel_end = eve.truncation.VesselEnd(intervention)
        sim_error = eve.truncation.SimError(intervention)

        if mode == "train":
            truncation = eve.truncation.Combination([max_steps, vessel_end, sim_error])
        else:
            truncation = max_steps

        # ============ Info ============
        target_reached = eve.info.TargetReached(intervention, name="success")
        path_ratio = eve.info.PathRatio(pathfinder)
        steps = eve.info.Steps()
        trans_speed = eve.info.AverageTranslationSpeed(intervention)
        trajectory_length = eve.info.TrajectoryLength(intervention)
        info = eve.info.Combination(
            [target_reached, path_ratio, steps, trans_speed, trajectory_length]
        )

        if visualisation:
            intervention.make_non_mp()
            visu = eve.visualisation.SofaPygame(intervention)
        else:
            intervention.make_non_mp()
            visu = None
            
        super().__init__(
            intervention,
            observation,
            reward,
            terminal,
            truncation=truncation,
            start=start,
            pathfinder=pathfinder,
            visualisation=visu,
            info=info,
            interim_target=None,
        )

    def __setstate__(self, state):
        """Re-initialize step logger after unpickling in a worker process.

        When env_train is pickled and sent to a worker Process, __init__ is not
        re-called. File handlers from the parent process are invalid in the child.
        __setstate__ is called by pickle on unpickling, so we re-create the logger
        here using the STEP_LOG_DIR env var which is inherited from the parent.
        """
        self.__dict__.update(state)
        # Re-create logger with a fresh file handler in this process
        self._step_logger = setup_step_logger(f"step_logger_{self.mode}_{id(self)}")

    def reset(self, seed=None, options=None):
        """Override reset to log episode boundaries."""
        # Log end of previous episode if applicable
        if self._episode_count > 0 and self._episode_start_time is not None:
            episode_duration = time.time() - self._episode_start_time
            self._step_logger.info(
                f"EPISODE_END | ep={self._episode_count} | steps={self._episode_step_count} | "
                f"total_reward={self._episode_total_reward:.4f} | duration={episode_duration:.2f}s | "
                f"avg_step_time={episode_duration/max(1,self._episode_step_count):.3f}s"
            )
            sys.stderr.flush()
        
        # Reset episode counters
        self._episode_count += 1
        self._episode_step_count = 0
        self._episode_total_reward = 0.0
        self._episode_start_time = time.time()
        self._last_step_time = time.time()
        
        self._step_logger.info(f"EPISODE_START | ep={self._episode_count} | global_steps={self._step_count}")
        sys.stderr.flush()
        
        # Call parent reset
        return super().reset(seed=seed, options=options)
    
    def step(self, action):
        """Override step to log detailed step information."""
        step_start_time = time.time()
        
        # Time since last step
        if self._last_step_time is not None:
            time_since_last = step_start_time - self._last_step_time
        else:
            time_since_last = 0.0
        
        # Call parent step
        try:
            obs, reward, terminated, truncated, info = super().step(action)
            step_success = True
            step_error = None
        except Exception as e:
            step_success = False
            step_error = str(e)
            self._step_logger.error(f"STEP_ERROR | ep={self._episode_count} | step={self._episode_step_count} | error={step_error}")
            sys.stderr.flush()
            raise
        
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        
        # Update counters
        self._step_count += 1
        self._episode_step_count += 1
        self._episode_total_reward += reward
        self._last_step_time = step_end_time
        
        # Get SOFA-specific info if available
        sofa_info = ""
        try:
            if hasattr(self.intervention, 'simulation'):
                sim = self.intervention.simulation
                if hasattr(sim, 'inserted_lengths'):
                    inserted = sim.inserted_lengths
                    sofa_info = f"inserted=[{inserted[0]:.2f},{inserted[1]:.2f}]"
        except Exception:
            pass
        
        # Log every step with immediate flush
        log_msg = (
            f"STEP | ep={self._episode_count} | ep_step={self._episode_step_count} | "
            f"global={self._step_count} | reward={reward:.4f} | cum_reward={self._episode_total_reward:.4f} | "
            f"step_time={step_duration:.3f}s | gap_time={time_since_last:.3f}s | "
            f"term={terminated} | trunc={truncated} | {sofa_info}"
        )
        
        # Always log to file
        self._step_logger.debug(log_msg)
        
        # Log to stderr less frequently (every 50 steps)
        if self._episode_step_count % 50 == 0 or terminated or truncated:
            self._step_logger.info(log_msg)
        
        # Force flush
        for handler in self._step_logger.handlers:
            handler.flush()
        sys.stderr.flush()
        
        return obs, reward, terminated, truncated, info
