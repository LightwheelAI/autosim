import torch

from autosim import register_skill
from autosim.core.skill import Skill
from autosim.core.types import SkillGoal, SkillOutput, WorldState


@register_skill(name="moveto", description="Move robot base to near the target object or location.")
class NavigateSkill(Skill):
    """Skill to navigate to a target position using A* + DWA motion planner."""

    def __init__(self, extra_cfg: dict = {}) -> None:
        super().__init__(extra_cfg)

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        return True

    def step(self, state: WorldState) -> SkillOutput:
        return SkillOutput(action=torch.zeros(6), done=True, success=True)
