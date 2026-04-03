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
    log_file = os.path.join(log_dir, f"worker_{os.getpid()}.log")
    
    try:
        handler = logging.FileHandler(log_file, mode='a')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s,%(msecs)03d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except Exception as e:
        # Fallback to stderr if file creation fails
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s,%(msecs)03d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Also add stderr handler for immediate visibility
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)  # Less verbose on stderr
    stderr_formatter = logging.Formatter('STEP: %(message)s')
    stderr_handler.setFormatter(stderr_formatter)
    logger.addHandler(stderr_handler)
    
    return logger


class BenchEnv(eve.Env):
    def __init__(
        self,
        intervention: eve.intervention.SimulatedIntervention,
        mode: str = "train",
        visualisation: bool = False,
        n_max_steps=1000,
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
        # NEW: Track previous insertion for delta computation
        self._prev_inserted = [0.0, 0.0]
        self._step_logger.info(f"=== BenchEnv initialized (mode={mode}, visualisation={visualisation}) ===")
        sys.stderr.flush()

        start = eve.start.InsertionPoint(intervention)
        pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)
        # Observation

        tracking = eve.observation.Tracking2D(intervention, n_points=3, resolution=2)
        tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(
            tracking, intervention
        )
        tracking = eve.observation.wrapper.Memory(
            tracking, 2, eve.observation.wrapper.MemoryResetMode.FILL
        )
        target_state = eve.observation.Target2D(intervention)
        target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
            target_state, intervention
        )
        last_action = eve.observation.LastAction(intervention)
        last_action = eve.observation.wrapper.Normalize(last_action)

        observation = eve.observation.ObsDict(
            {
                "tracking": tracking,
                "target": target_state,
                "last_action": last_action,
            }
        )

        # Reward
        target_reward = eve.reward.TargetReached(
            intervention,
            factor=1.0,
            final_only_after_all_interim=False,
        )
        step_reward = eve.reward.Step(factor=-0.005)
        path_delta = eve.reward.PathLengthDelta(pathfinder, 0.001)
        reward = eve.reward.Combination([target_reward, path_delta, step_reward])

        # Terminal and Truncation
        terminal = eve.terminal.TargetReached(intervention)

        max_steps = eve.truncation.MaxSteps(n_max_steps)
        vessel_end = eve.truncation.VesselEnd(intervention)
        sim_error = eve.truncation.SimError(intervention)

        if mode == "train":
            truncation = eve.truncation.Combination([max_steps, vessel_end, sim_error])
        else:
            truncation = max_steps

        # Info
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
            # intervention.make_mp()
            # CRITICAL: Use make_non_mp() to avoid nested multiprocessing!
            # BenchAgentSynchron already spawns worker processes.
            # If SOFA also spawns subprocesses (make_mp), we get:
            #   Main → 4 workers → 4 SOFA subprocesses = 9 processes
            # This exhausts Docker's /dev/shm and breaks queue timeouts.
            # With make_non_mp(), SOFA runs in the worker process directly.
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
                f"avg_step_time={episode_duration/max(1,self._episode_step_count):.3f}s | "
                f"wall_time={time.time():.6f} | pid={os.getpid()}"
            )
            sys.stderr.flush()
        
        # Reset episode counters
        self._episode_count += 1
        self._episode_step_count = 0
        self._episode_total_reward = 0.0
        self._episode_start_time = time.time()
        self._last_step_time = time.time()
        # NEW: Reset previous insertion tracking
        self._prev_inserted = [0.0, 0.0]
        
        self._step_logger.info(
            f"EPISODE_START | ep={self._episode_count} | global_steps={self._step_count} | "
            f"wall_time={time.time():.6f} | pid={os.getpid()}"
        )
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
        delta_ins = [0.0, 0.0]
        inserted = [0.0, 0.0]
        try:
            if hasattr(self.intervention, 'simulation'):
                sim = self.intervention.simulation
                if hasattr(sim, 'inserted_lengths'):
                    inserted = list(sim.inserted_lengths)
                    # NEW: Compute delta insertion
                    delta_ins = [
                        inserted[0] - self._prev_inserted[0],
                        inserted[1] - self._prev_inserted[1]
                    ]
                    self._prev_inserted = inserted.copy()
                    sofa_info = f"inserted=[{inserted[0]:.2f},{inserted[1]:.2f}]"
        except Exception:
            pass
        
        # NEW: Determine termination reason
        term_reason = "none"
        if terminated:
            term_reason = "target_reached"
        elif truncated:
            # Try to determine truncation reason
            if self._episode_step_count >= getattr(self, '_n_max_steps', 1000):
                term_reason = "max_steps"
            else:
                # Could be vessel_end or sim_error - check if we can tell
                term_reason = "truncated"  # Generic truncation
        
        # NEW: Format action for logging (handle numpy arrays)
        try:
            if hasattr(action, 'tolist'):
                action_str = f"[{','.join(f'{a:.3f}' for a in action.flatten())}]"
            elif hasattr(action, '__iter__'):
                action_str = f"[{','.join(f'{a:.3f}' for a in action)}]"
            else:
                action_str = f"{action:.3f}"
        except Exception:
            action_str = str(action)
        
        # NEW: Get executed action and mask reason from intervention (if available)
        exec_action_str = ""
        mask_reason_str = "none"
        try:
            if hasattr(self.intervention, 'last_exec_action'):
                exec_action = self.intervention.last_exec_action
                if hasattr(exec_action, 'flatten'):
                    exec_action_str = f"[{','.join(f'{a:.3f}' for a in exec_action.flatten())}]"
                else:
                    exec_action_str = str(exec_action)
            if hasattr(self.intervention, 'last_mask_reasons'):
                reasons = self.intervention.last_mask_reasons
                # Report first non-"none" reason, or "none" if all are none
                for r in reasons:
                    if r != "none":
                        mask_reason_str = r
                        break
        except Exception:
            pass
        
        # Log every step with immediate flush
        # ENHANCED: Added action, exec_action, mask_reason, delta_ins, term_reason
        log_msg = (
            f"STEP | ep={self._episode_count} | ep_step={self._episode_step_count} | "
            f"global={self._step_count} | wall_time={time.time():.6f} | pid={os.getpid()} | "
            f"cmd_action={action_str} | "
        )
        # Add exec_action only if available (backward compatible)
        if exec_action_str:
            log_msg += f"exec_action={exec_action_str} | mask_reason={mask_reason_str} | "
        log_msg += (
            f"reward={reward:.4f} | cum_reward={self._episode_total_reward:.4f} | "
            f"step_time={step_duration:.3f}s | gap_time={time_since_last:.3f}s | "
            f"term={terminated} | trunc={truncated} | term_reason={term_reason} | "
            f"{sofa_info} | delta_ins=[{delta_ins[0]:.2f},{delta_ins[1]:.2f}]"
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