import torch

from autosim import register_skill
from autosim.core.types import SkillGoal, SkillOutput, WorldState

from .reach import ReachSkill


@register_skill(name="push", description="Push end-effector forward (target: 'forward')", required_modules=["curobo"])
class PushSkill(ReachSkill):
    """Skill to push end-effector forward"""

    def __init__(self, extra_cfg: dict = {}) -> None:
        super().__init__(extra_cfg)

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        return True

    def step(self, state: WorldState) -> SkillOutput:
        return SkillOutput(action=torch.zeros(6), done=True, success=True)
