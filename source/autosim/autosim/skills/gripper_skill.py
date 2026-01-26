from autosim import register_skill

from .base_skill import GripperSkillBase


@register_skill(name="grasp", description="Grasp object (close gripper)")
class GraspSkill(GripperSkillBase):
    def __init__(self, extra_cfg: dict = {}) -> None:
        """default configuration: close gripper[-1.0] for 10 steps"""
        super().__init__(extra_cfg, gripper_value=-1.0, duration=10)


@register_skill(name="ungrasp", description="Release object (open gripper)")
class UngraspSkill(GripperSkillBase):
    def __init__(self, extra_cfg: dict = {}) -> None:
        """default configuration: open gripper[1.0] for 10 steps"""
        super().__init__(extra_cfg, gripper_value=1.0, duration=10)
