"""Action-space curriculum wrapper for DualDeviceNav.

Simplifies the dual-device coordination problem by staging:
  Stage 1: Only guidewire acts; catheter follows with a simple rule.
  Stage 2: Catheter action enabled but scaled down (×0.1).
  Stage 3: Full control of both devices.

Usage:
    env = BenchEnv4(intervention, mode="train")
    env = ActionCurriculumWrapper(env, stage1_steps=200_000, stage2_steps=500_000)
"""

import numpy as np
import gymnasium as gym


class ActionCurriculumWrapper(gym.Wrapper):
    """Curriculum that gradually enables catheter control.

    The wrapped environment must have a 4-dimensional action space:
    ``[gw_translation, gw_rotation, cath_translation, cath_rotation]``.

    Args:
        env: The base environment.
        stage1_steps: Number of steps in Stage 1 (guidewire-only).
        stage2_steps: Cumulative step count at which Stage 2 ends and
            Stage 3 (full control) begins.
        catheter_follow_ratio: In Stage 1, catheter translation is set to
            ``gw_translation * catheter_follow_ratio``.
        stage2_action_scale: In Stage 2, catheter actions are multiplied
            by this factor.
    """

    def __init__(
        self,
        env: gym.Env,
        stage1_steps: int = 200_000,
        stage2_steps: int = 500_000,
        catheter_follow_ratio: float = 0.8,
        stage2_action_scale: float = 0.1,
    ) -> None:
        super().__init__(env)
        self.stage1_steps = stage1_steps
        self.stage2_steps = stage2_steps
        self.catheter_follow_ratio = catheter_follow_ratio
        self.stage2_action_scale = stage2_action_scale
        self._total_steps = 0

    @property
    def current_stage(self) -> int:
        if self._total_steps < self.stage1_steps:
            return 1
        elif self._total_steps < self.stage2_steps:
            return 2
        return 3

    def step(self, action):
        stage = self.current_stage

        if stage == 1:
            # Catheter follows guidewire with a simple rule
            action = np.asarray(action, dtype=np.float64).copy()
            gw_trans = action.flat[0]
            action.flat[2] = gw_trans * self.catheter_follow_ratio  # cath translation
            action.flat[3] = 0.0  # cath rotation = 0
        elif stage == 2:
            # Catheter actions scaled down
            action = np.asarray(action, dtype=np.float64).copy()
            action.flat[2] *= self.stage2_action_scale
            action.flat[3] *= self.stage2_action_scale
        # Stage 3: pass through unchanged — no copy needed

        self._total_steps += 1
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
