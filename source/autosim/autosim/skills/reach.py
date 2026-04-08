from __future__ import annotations

import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from autosim import register_skill
from autosim.core.logger import AutoSimLogger
from autosim.core.skill import SkillCfg
from autosim.core.types import (
    EnvExtraInfo,
    SkillGoal,
    SkillInfo,
    SkillOutput,
    WorldState,
)

from .base_skill import CuroboSkillBase, CuroboSkillExtraCfg


@configclass
class ReachSkillCfg(SkillCfg):
    """Configuration for the reach skill."""

    extra_cfg: CuroboSkillExtraCfg = CuroboSkillExtraCfg()
    """Extra configuration for the reach skill."""


@register_skill(
    name="reach",
    cfg_type=ReachSkillCfg,
    description="Extend robot arm to target position (for approaching objects or placement locations)",
)
class ReachSkill(CuroboSkillBase):
    """Skill to reach to a target object or location"""

    def __init__(self, extra_cfg: CuroboSkillExtraCfg) -> None:
        super().__init__(extra_cfg)

        self._logger = AutoSimLogger("ReachSkill")

        # variables for the skill execution
        self._trajectory = None
        self._step_idx = 0

        # corrective reach state
        self._corrective_reach_done: bool = False
        self._saved_env: ManagerBasedEnv | None = None
        self._saved_robot_name: str | None = None
        self._saved_target_object: str | None = None
        self._saved_reach_offset: torch.Tensor | None = None    # [7] in object frame
        self._saved_extra_offsets: dict[str, torch.Tensor] | None = None  # {ee_name: [7]} in object frame

    def _compute_goal_from_offset(
        self,
        env: ManagerBasedEnv,
        robot_name: str,
        target_object: str,
        reach_offset: torch.Tensor,
        extra_offsets: dict[str, torch.Tensor] | None,
    ) -> SkillGoal | None:
        """Compute SkillGoal by applying object-frame offsets to the object's current pose.

        Args:
            reach_offset: [7] tensor (pos + quat) in object frame for the primary EE.
            extra_offsets: per-EE offsets in object frame, or None.

        Returns:
            SkillGoal with target poses expressed in robot root frame, or None on failure.
        """

        try:
            object_pose_in_env = env.scene[target_object].data.root_pose_w
        except Exception:
            self._logger.warning(f"could not read pose for '{target_object}', skipping")
            return None

        object_pos_in_env = object_pose_in_env[:, :3]
        object_quat_in_env = object_pose_in_env[:, 3:]

        offset = reach_offset.to(env.device).unsqueeze(0)
        reach_target_pos_in_env, reach_target_quat_in_env = PoseUtils.combine_frame_transforms(
            object_pos_in_env, object_quat_in_env, offset[:, :3], offset[:, 3:]
        )
        self._logger.info(f"Reach target position in environment: {reach_target_pos_in_env}")
        self._logger.info(f"Reach target quaternion in environment: {reach_target_quat_in_env}")
        self._target_poses["target_pose"] = torch.cat((reach_target_pos_in_env, reach_target_quat_in_env), dim=-1)

        robot = env.scene[robot_name]
        robot_root_pos_in_env = robot.data.root_pose_w[:, :3]
        robot_root_quat_in_env = robot.data.root_pose_w[:, 3:]

        reach_target_pos_in_root, reach_target_quat_in_root = PoseUtils.subtract_frame_transforms(
            robot_root_pos_in_env, robot_root_quat_in_env, reach_target_pos_in_env, reach_target_quat_in_env
        )
        target_pose = torch.cat((reach_target_pos_in_root, reach_target_quat_in_root), dim=-1).squeeze(0)

        extra_target_poses: dict[str, torch.Tensor] | None = None
        if extra_offsets:
            extra_target_poses = {}
            for ee_name, ee_offset in extra_offsets.items():
                ee_off = ee_offset.to(env.device).unsqueeze(0)
                ee_pos_in_env, ee_quat_in_env = PoseUtils.combine_frame_transforms(
                    object_pos_in_env, object_quat_in_env, ee_off[:, :3], ee_off[:, 3:]
                )
                self._logger.info(f"Extra target position for {ee_name} in environment: {ee_pos_in_env}")
                self._logger.info(f"Extra target quaternion for {ee_name} in environment: {ee_quat_in_env}")
                ee_pos_in_root, ee_quat_in_root = PoseUtils.subtract_frame_transforms(
                    robot_root_pos_in_env, robot_root_quat_in_env, ee_pos_in_env, ee_quat_in_env
                )
                extra_target_poses[ee_name] = torch.cat((ee_pos_in_root, ee_quat_in_root), dim=-1).squeeze(0)

        return SkillGoal(target_object=target_object, target_pose=target_pose, extra_target_poses=extra_target_poses)

    def extract_goal_from_info(
        self, skill_info: SkillInfo, env: ManagerBasedEnv, env_extra_info: EnvExtraInfo
    ) -> SkillGoal:
        """Return the target pose[x, y, z, qw, qx, qy, qz] in the robot root frame.
        IMPORTANT: the robot root frame is not the same as the robot base frame.
        """

        target_object = skill_info.target_object

        reach_offset = env_extra_info.get_next_reach_target_pose(target_object).to(env.device)

        extra_offsets: dict[str, torch.Tensor] | None = None
        if target_object in env_extra_info.object_extra_reach_target_poses:
            extra_offsets = {
                ee_name: torch.as_tensor(
                    env_extra_info.get_next_extra_reach_target_pose(target_object, ee_name), device=env.device
                )
                for ee_name in env_extra_info.object_extra_reach_target_poses[target_object]
            }

        # Save state needed for corrective reach re-planning
        self._saved_env = env
        self._saved_robot_name = env_extra_info.robot_name
        self._saved_target_object = target_object
        self._saved_reach_offset = reach_offset
        self._saved_extra_offsets = extra_offsets

        return self._compute_goal_from_offset(env, env_extra_info.robot_name, target_object, reach_offset, extra_offsets)

    def _compute_corrective_goal(self) -> SkillGoal | None:
        """Re-compute reach goal using the object's current actual pose.

        This is called after the first trajectory finishes. The same relative offset (in object
        frame) is re-applied to the object's current pose, so if the object was nudged during
        approach the robot corrects for it.
        """

        if self._saved_env is None or self._saved_target_object is None or self._saved_reach_offset is None:
            return None

        goal = self._compute_goal_from_offset(
            self._saved_env,
            self._saved_robot_name,
            self._saved_target_object,
            self._saved_reach_offset,
            self._saved_extra_offsets,
        )
        if goal is not None:
            self._logger.info("corrective_reach: recomputed target from current object pose")
        return goal

    def execute_plan(self, state: WorldState, goal: SkillGoal) -> bool:
        """Execute the plan of the reach skill."""

        self._logger.info(f"Reach from pose in environment: {state.robot_ee_pose}")

        target_pose = goal.target_pose  # target pose in the robot root frame
        target_pos, target_quat = target_pose[:3], target_pose[3:]

        full_sim_joint_names = state.sim_joint_names
        full_sim_q = state.robot_joint_pos
        full_sim_qd = state.robot_joint_vel
        planner_activate_joints = self._planner.target_joint_names

        activate_q, activate_qd = [], []
        for joint_name in planner_activate_joints:
            if joint_name in full_sim_joint_names:
                activate_q.append(full_sim_q[full_sim_joint_names.index(joint_name)])
                activate_qd.append(full_sim_qd[full_sim_joint_names.index(joint_name)])
            else:
                raise ValueError(
                    f"Joint {joint_name} in planner activate joints is not in the full simulation joint names."
                )
        activate_q = torch.stack(activate_q, dim=0)
        activate_qd = torch.stack(activate_qd, dim=0)
        self._trajectory = self._planner.plan_motion(
            target_pos,
            target_quat,
            activate_q,
            activate_qd,
            link_goals=goal.extra_target_poses,
        )

        return self._trajectory is not None

    def step(self, state: WorldState) -> SkillOutput:
        """Step the reach skill.

        Args:
            state: The current state of the world.

        Returns:
            The output of the skill execution.
                action: The action to be applied to the environment. [joint_positions with isaaclab joint order]
        """

        self.visualize_debug_target_pose()

        traj_positions = self._trajectory.position
        if self._step_idx >= len(self._trajectory.position):
            traj_pos = traj_positions[-1]
            done = True
        else:
            traj_pos = traj_positions[self._step_idx]
            done = False
            self._step_idx += 1

        # Corrective reach: when the first trajectory finishes, re-plan using the object's
        # actual current position in case it was nudged during approach.
        if done and not self._corrective_reach_done and self.cfg.extra_cfg.corrective_reach:
            self._corrective_reach_done = True  # prevent infinite loop
            new_goal = self._compute_corrective_goal()
            if new_goal is not None:
                self._logger.info("corrective_reach: re-planning to corrected object pose")
                self._step_idx = 0
                plan_success = self.execute_plan(state, new_goal)
                if plan_success:
                    done = False  # continue with corrective trajectory

        curobo_joint_names = self._trajectory.joint_names
        sim_joint_names = state.sim_joint_names
        joint_pos = state.robot_joint_pos.clone()
        for curobo_idx, curobo_joint_name in enumerate(curobo_joint_names):
            sim_idx = sim_joint_names.index(curobo_joint_name)
            joint_pos[sim_idx] = traj_pos[curobo_idx]

        return SkillOutput(
            action=joint_pos,
            done=done,
            success=True,
            info={},
        )

    def reset(self) -> None:
        """Reset the reach skill."""

        super().reset()
        self._step_idx = 0
        self._trajectory = None
        self._corrective_reach_done = False
        self._saved_env = None
        self._saved_robot_name = None
        self._saved_target_object = None
        self._saved_reach_offset = None
        self._saved_extra_offsets = None
