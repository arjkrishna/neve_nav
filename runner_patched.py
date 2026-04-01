from typing import List, Optional
from math import inf
import csv
import json
import logging
import os
import time
import numpy as np
from ..util import EveRLObject
from ..agent.agent import Agent, StepCounter, EpisodeCounter


class Runner(EveRLObject):
    def __init__(
        self,
        agent: Agent,
        heatup_action_low: List[float],
        heatup_action_high: List[float],
        agent_parameter_for_result_file: dict,
        checkpoint_folder: str,
        results_file: str,
        quality_info: Optional[str] = None,
        info_results: Optional[List[str]] = None,
        diagnostics_folder: Optional[str] = None,
        policy_snapshot_every_steps: int = 10000,
        n_probe_episodes: int = 3,
        n_probe_near_start_steps: int = 10,
    ) -> None:
        self.agent = agent
        self.heatup_action_low = heatup_action_low
        self.heatup_action_high = heatup_action_high
        self.agent_parameter_for_result_file = agent_parameter_for_result_file
        self.checkpoint_folder = checkpoint_folder
        self.results_file = results_file
        self.quality_info = quality_info
        self.info_results = info_results or []
        self.logger = logging.getLogger(self.__module__)
        
        # Diagnostics settings
        self.diagnostics_folder = diagnostics_folder
        # Ensure worker step logs land under diagnostics so diagnose_collapse3.py can find them
        if self.diagnostics_folder is not None:
            logs_subprocesses = os.path.join(self.diagnostics_folder, "logs_subprocesses")
            os.makedirs(logs_subprocesses, exist_ok=True)
            # Inherit into subprocesses if they are spawned after Runner init
            os.environ.setdefault("STEP_LOG_DIR", logs_subprocesses)

        self.policy_snapshot_every_steps = policy_snapshot_every_steps
        self.n_probe_episodes = n_probe_episodes
        self.n_probe_near_start_steps = n_probe_near_start_steps
        self._next_snapshot_step = policy_snapshot_every_steps
        self._probe_states_set = False
        
        # Episode summary logger
        self._episode_summary_file = None
        self._episode_summary_logger = None
        if self.diagnostics_folder is not None:
            csv_folder = os.path.join(self.diagnostics_folder, "csv")
            os.makedirs(csv_folder, exist_ok=True)
            self._episode_summary_file = os.path.join(csv_folder, "episode_summary.jsonl")
            self.logger.info(f"Episode summary will be logged to {self._episode_summary_file}")

        self._results = {
            "episodes explore": 0,
            "steps explore": 0,
        }
        self._results["quality"] = 0.0
        for info_result in self.info_results:
            self._results[info_result] = 0.0
        self._results["reward"] = 0.0
        self._results["best quality"] = 0.0
        self._results["best explore steps"] = 0.0

        file_existed = os.path.isfile(results_file)

        with open(results_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_existed:
                writer.writerow(
                    list(self._results.keys())
                    + [" "]
                    + list(agent_parameter_for_result_file.keys())
                )
                writer.writerow(
                    [" "] * (len(self._results.values()) + 1)
                    + list(agent_parameter_for_result_file.values())
                )

        self.best_eval = {"steps": 0, "quality": -inf}

    @property
    def step_counter(self) -> StepCounter:
        return self.agent.step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self.agent.episode_counter

    def heatup(self, steps: int):
        episodes = self.agent.heatup(
            steps=steps,
            custom_action_low=self.heatup_action_low,
            custom_action_high=self.heatup_action_high,
        )
        return episodes

    def explore(self, n_episodes: int):
        self.agent.explore(episodes=n_episodes)

    def update(self, n_steps: int):
        self.agent.update(n_steps)

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[List[int]] = None
    ):
        explore_steps = self.step_counter.exploration
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{explore_steps}.everl"
        )
        result_episodes = self.agent.evaluate(episodes=episodes, seeds=seeds)
        qualities, rewards = [], []
        results_for_info = {
            info_result_name: [] for info_result_name in self.info_results
        }
        eval_results = {"episodes": []}
        for episode in result_episodes:
            reward = episode.episode_reward
            quality = (
                episode.infos[-1][self.quality_info]
                if self.quality_info is not None
                else reward
            )
            qualities.append(quality)
            rewards.append(reward)
            for info_result_name in self.info_results:
                info = episode.infos[-1][info_result_name]
                results_for_info[info_result_name].append(info)
            eval_results["episodes"].append(
                {
                    "seed": episode.seed,
                    "options": episode.options,
                    "quality": episode.infos[-1][self.quality_info],
                }
            )

        reward = sum(rewards) / len(rewards)
        quality = sum(qualities) / len(qualities)
        for info_result_name, results in results_for_info.items():
            result = sum(results) / len(results)
            self._results[info_result_name] = round(result, 3)
        save_best = False
        if quality > self.best_eval["quality"]:
            save_best = True
            self.best_eval["quality"] = quality
            self.best_eval["steps"] = explore_steps

        self._results["episodes explore"] = self.episode_counter.exploration
        self._results["steps explore"] = explore_steps
        self._results["reward"] = round(reward, 3)
        self._results["quality"] = round(quality, 3)
        self._results["best quality"] = self.best_eval["quality"]
        self._results["best explore steps"] = self.best_eval["steps"]

        eval_results.update(self._results)
        eval_results.pop("best quality")
        eval_results.pop("best explore steps")

        self.agent.save_checkpoint(checkpoint_file, eval_results)
        if save_best:
            checkpoint_file = os.path.join(
                self.checkpoint_folder, "best_checkpoint.everl"
            )
            self.agent.save_checkpoint(checkpoint_file, eval_results)

        log_info = (
            f"Quality: {quality}, Reward: {reward}, Exploration steps: {explore_steps}"
        )
        self.logger.info(log_info)
        with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(self._results.values())

        return quality, reward

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: int = None,
        explore_steps_limit: int = None,
    ):
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        while self.step_counter.exploration < explore_steps_limit:
            # Total experience = heatup + exploration (both contribute to replay buffer)
            total_experience_steps = (
                self.step_counter.heatup + self.step_counter.exploration
            )
            update_steps = (
                total_experience_steps * update_steps_per_explore_step
                - self.step_counter.update
            )
            result = self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )
            
            # Log episode summaries if we got results
            if result is not None:
                episodes, _ = result if isinstance(result, tuple) else (result, None)
                if episodes:
                    self._log_episode_summaries(episodes)

    def training_run(
        self,
        heatup_steps: int,
        training_steps: int,
        explore_steps_between_eval: int,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
    ):
        # TODO: Log Training Run Infos
        heatup_episodes = self.heatup(heatup_steps)
        
        # Capture and set probe states after heatup
        self._capture_and_set_probe_states(heatup_episodes)
        
        next_eval_step_limt = (
            self.agent.step_counter.exploration + explore_steps_between_eval
        )
        while self.agent.step_counter.exploration < training_steps:
            self.explore_and_update(
                explore_episodes_between_updates,
                update_steps_per_explore_step,
                explore_steps_limit=next_eval_step_limt,
            )
            
            # Periodic policy snapshots
            self._maybe_save_policy_snapshot()
            
            quality, reward = self.eval(episodes=eval_episodes, seeds=eval_seeds)
            next_eval_step_limt += explore_steps_between_eval

        return quality, reward

    def _capture_and_set_probe_states(self, episodes: List):
        """Capture probe states from heatup episodes and set them in trainer."""
        if not episodes or self._probe_states_set:
            return
            
        try:
            probe_states = []
            n_episodes = min(len(episodes), self.n_probe_episodes)
            
            for ep_idx in range(n_episodes):
                episode = episodes[ep_idx]
                if hasattr(episode, 'flat_obs') and len(episode.flat_obs) > 0:
                    # Collect start state
                    probe_states.append(episode.flat_obs[0])
                    
                    # Collect near-start states
                    n_states = min(self.n_probe_near_start_steps, len(episode.flat_obs) - 1)
                    for step_idx in range(1, n_states + 1):
                        probe_states.append(episode.flat_obs[step_idx])
                        
            if probe_states:
                probe_states_array = np.array(probe_states)
                
                # Set probe states in agent (if it supports it)
                if hasattr(self.agent, 'set_probe_states'):
                    self.agent.set_probe_states(probe_states_array.tolist())
                    self.logger.info(f"Set {len(probe_states)} probe states for diagnostics")
                    
                # Save probe states to disk
                if self.diagnostics_folder is not None:
                    probes_dir = os.path.join(self.diagnostics_folder, "probes")
                    os.makedirs(probes_dir, exist_ok=True)
                    probe_path = os.path.join(probes_dir, "probe_states.npz")
                    np.savez(probe_path, probe_states=probe_states_array)
                    self.logger.info(f"Saved probe states to {probe_path}")
                    
                self._probe_states_set = True
                
        except Exception as e:
            self.logger.warning(f"Failed to capture probe states: {e}")

    def _maybe_save_policy_snapshot(self):
        """
        Save policy snapshot if we've reached the next snapshot step.
        
        FIXED: Uses a while loop to catch up on all missed snapshots when the
        eval cycle spans multiple snapshot intervals.
        """
        if self.diagnostics_folder is None:
            return
            
        current_step = self.step_counter.exploration
        
        # Save all missed snapshots (in case eval cycle spans multiple snapshot intervals)
        while current_step >= self._next_snapshot_step:
            try:
                snapshots_dir = os.path.join(self.diagnostics_folder, "policy_snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)
                
                # Use the scheduled snapshot step, not current step, for consistent naming
                snapshot_step = self._next_snapshot_step
                snapshot_path = os.path.join(snapshots_dir, f"policy_{snapshot_step}.pt")
                
                if hasattr(self.agent, 'save_policy_snapshot'):
                    self.agent.save_policy_snapshot(snapshot_path)
                    self.logger.info(f"Saved policy snapshot at scheduled step {snapshot_step} (current={current_step})")
                    
            except Exception as e:
                self.logger.warning(f"Failed to save policy snapshot at step {self._next_snapshot_step}: {e}")
                
            self._next_snapshot_step += self.policy_snapshot_every_steps
        
        # Flush diagnostics logs after all snapshots are saved
        if hasattr(self.agent, 'flush_diagnostics'):
            try:
                self.agent.flush_diagnostics()
            except Exception as e:
                self.logger.warning(f"Failed to flush diagnostics: {e}")

    def _log_episode_summaries(self, episodes: List):
        """
        Log episode summaries to episode_summary.jsonl with explore_step and update_step.
        
        This directly solves the timestamp correlation problem by providing exact
        step counters at episode completion time.
        """
        if self._episode_summary_file is None:
            return
            
        try:
            # Capture current step counters
            explore_step = self.step_counter.exploration
            update_step = self.step_counter.update
            
            with open(self._episode_summary_file, "a", encoding="utf-8") as f:
                for i, episode in enumerate(episodes):
                    try:
                        wall_time = time.time()
                        # Extract episode data
                        total_reward = getattr(episode, 'episode_reward', 0.0)
                        steps = len(getattr(episode, 'rewards', []))
                        terminated = getattr(episode, 'terminals', [False])[-1] if getattr(episode, 'terminals', []) else False
                        truncated = getattr(episode, 'truncations', [False])[-1] if getattr(episode, 'truncations', []) else False
                        
                        # Try to get max_insertion from episode info if available
                        max_insertion = None
                        infos = getattr(episode, 'infos', [])
                        if infos:
                            last_info = infos[-1] if infos else {}
                            # Look for insertion depth in common info keys
                            for key in ['max_insertion', 'inserted_length', 'insertion_depth']:
                                if key in last_info:
                                    val = last_info[key]
                                    if hasattr(val, '__iter__') and not isinstance(val, str):
                                        max_insertion = float(max(val))
                                    else:
                                        max_insertion = float(val)
                                    break
                        
                        summary = {
                            "wall_time": wall_time,
                            "explore_step": explore_step,
                            "update_step": update_step,
                            "episode_id": self.episode_counter.exploration - len(episodes) + i,
                            "total_reward": float(total_reward),
                            "steps": steps,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                        if max_insertion is not None:
                            summary["max_insertion"] = max_insertion
                        
                        f.write(json.dumps(summary) + "\n")
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to log episode {i} summary: {e}")
                        
                f.flush()
                
        except Exception as e:
            self.logger.warning(f"Failed to log episode summaries: {e}")
