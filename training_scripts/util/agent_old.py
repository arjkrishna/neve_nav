from copy import deepcopy
import torch
from torch.optim.lr_scheduler import LinearLR
from eve_rl.algo import SAC
from eve_rl.model import SACModel
from eve_rl.agent import Synchron
from eve_rl.replaybuffer import VanillaReplayBuffer
import gymnasium as gym


class BenchAgentSynchron(Synchron):
    def __init__(
        self,
        trainer_device: torch.device,
        worker_device: torch.device,
        lr: float,
        lr_end_factor: float,
        lr_linear_end_steps: float,
        hidden_layers: list,
        embedder_nodes: int,
        embedder_layers: int,
        gamma: float,
        batch_size: int,
        reward_scaling: float,
        replay_buffer_size: float,
        env_train: gym.Env,
        env_eval: gym.Env,
        consecutive_action_steps: int,
        n_worker: int,
        stochastic_eval: bool,
        normalize_actions: bool = True,
    ):
        (
            trainer_algo,
            trainer_model,
            trainer_replay_buffer,
        ) = self._create_algo_model_replay(
            env_train=env_train,
            batch_size=batch_size,
            gamma=gamma,
            reward_scaling=reward_scaling,
            replay_buffer_size=replay_buffer_size,
            hidden_layers=hidden_layers,
            embedder_nodes=embedder_nodes,
            embedder_layers=embedder_layers,
            lr=lr,
            lr_end_factor=lr_end_factor,
            lr_linear_end_steps=lr_linear_end_steps,
            stochastic_eval=stochastic_eval,
        )
        (
            worker_algo,
            worker_model,
            worker_replay_buffer,
        ) = self._create_algo_model_replay(
            env_train=env_train,
            batch_size=batch_size,
            gamma=gamma,
            reward_scaling=reward_scaling,
            replay_buffer_size=replay_buffer_size,
            hidden_layers=hidden_layers,
            embedder_nodes=embedder_nodes,
            embedder_layers=embedder_layers,
            lr=lr,
            lr_end_factor=lr_end_factor,
            lr_linear_end_steps=lr_linear_end_steps,
            stochastic_eval=stochastic_eval,
        )
        super().__init__(
            trainer_algo=trainer_algo,
            worker_algo=worker_algo,
            trainer_env_train=deepcopy(env_train),
            worker_env_train=deepcopy(env_train),
            env_eval=deepcopy(env_eval),
            trainer_replay_buffer=trainer_replay_buffer,
            worker_replay_buffer=worker_replay_buffer,
            trainer_device=trainer_device,
            worker_device=worker_device,
            consecutive_action_steps=consecutive_action_steps,
            normalize_actions=normalize_actions,
            n_worker=n_worker,
        )

    def _create_algo_model_replay(
        self,
        env_train,
        batch_size,
        gamma,
        reward_scaling,
        replay_buffer_size,
        hidden_layers,
        embedder_nodes,
        embedder_layers,
        lr,
        lr_end_factor,
        lr_linear_end_steps,
        stochastic_eval,
    ):
        model = SACModel(
            obs_space=env_train.observation_space,
            n_actions=env_train.action_space.shape[0] * env_train.action_space.shape[1],
            embedder_nodes=embedder_nodes,
            embedder_layers=embedder_layers,
            hidden_layers=hidden_layers,
            lr=lr,
            lr_alpha=0.001,
        )
        policy_scheduler = LinearLR(
            model.policy_optimizer,
            start_factor=1,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )
        q1_scheduler = LinearLR(
            model.q1_optimizer,
            start_factor=1,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )
        q2_scheduler = LinearLR(
            model.q2_optimizer,
            start_factor=1,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )
        model.set_schedulers(policy_scheduler, q1_scheduler, q2_scheduler)
        algo = SAC(
            model=model,
            n_actions=env_train.action_space.shape[0] * env_train.action_space.shape[1],
            gamma=gamma,
            reward_scaling=reward_scaling,
            stochastic_eval=stochastic_eval,
        )
        replay_buffer = VanillaReplayBuffer(
            max_size=replay_buffer_size,
            batch_size=batch_size,
        )

        return algo, model, replay_buffer
