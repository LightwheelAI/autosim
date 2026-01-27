import torch

from autosim import register_skill
from autosim.core.types import SkillGoal, SkillOutput, WorldState

from .base_skill import CuroboSkillBase


@register_skill(
    name="rotate", description="Rotate action (for rotating objects or operating knobs)", required_modules=["curobo"]
)
class RotateSkill(CuroboSkillBase):
    """Skill to reach to a target object or location"""

    def __init__(self, extra_cfg: dict = {}) -> None:
        super().__init__(extra_cfg)

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        return True

    def step(self, state: WorldState) -> SkillOutput:
        return SkillOutput(action=torch.zeros(6), done=True, success=True)
