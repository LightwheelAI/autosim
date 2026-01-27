import torch

from autosim import register_skill
from autosim.core.types import SkillGoal, SkillOutput, WorldState

from .reach import ReachSkill


@register_skill(name="pull", description="Pull end-effector backward (target: 'backward')", required_modules=["curobo"])
class PullSkill(ReachSkill):
    """Skill to pull end-effector backward"""

    def __init__(self, extra_cfg: dict = {}) -> None:
        super().__init__(extra_cfg)

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        return True

    def step(self, state: WorldState) -> SkillOutput:
        return SkillOutput(action=torch.zeros(6), done=True, success=True)
