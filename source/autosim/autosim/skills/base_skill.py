import torch

from autosim.core.skill import Skill
from autosim.core.types import SkillGoal, SkillOutput, WorldState


class GripperSkillBase(Skill):
    """Base class for gripper skills open/close skills."""

    def __init__(self, extra_cfg: dict = {}, gripper_value: float = 0.0, duration: int = 10) -> None:
        super().__init__(extra_cfg)
        self._gripper_value = extra_cfg.get("gripper_value", gripper_value)
        self._duration = extra_cfg.get("duration", duration)
        self._step_count = 0
        self._target_object_name = None

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        self._target_object_name = goal.target_object
        self._step_count = 0
        return True

    def step(self, state: WorldState) -> SkillOutput:
        done = self._step_count >= self._duration
        self._step_count += 1

        return SkillOutput(
            action=torch.tensor([self._gripper_value], device=state.device),
            done=done,
            success=done,
            info={"step": self._step_count, "target_object": self._target_object_name},
        )


class CuroboSkillBase(Skill):
    """Base class for skills dependent on curobo."""

    def __init__(self, extra_cfg: dict = {}) -> None:
        super().__init__(extra_cfg)

    def _init_modules(self) -> None:
        # curobo related modules
        pass
