import torch

from autosim import register_skill
from autosim.core.types import SkillGoal, SkillOutput, WorldState

from .base_skill import CuroboSkillBase


@register_skill(
    name="press", description="Press action (for buttons and interactive elements).", required_modules=["curobo"]
)
class PressSkill(CuroboSkillBase):
    """Skill to press buttons or interactive elements"""

    def __init__(self, extra_cfg: dict = {}) -> None:
        super().__init__(extra_cfg)

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        return True

    def step(self, state: WorldState) -> SkillOutput:
        return SkillOutput(action=torch.zeros(6), done=True, success=True)
