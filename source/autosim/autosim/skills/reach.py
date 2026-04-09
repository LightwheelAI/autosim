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

    def _build_activate_joint_state(
        self, full_sim_joint_names: list[str], full_sim_q: torch.Tensor, full_sim_qd: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract the planner's active joint state from full simulation joint state.

        cuRobo typically plans over a subset of "active joints" (`self._planner.target_joint_names`).
        This helper slices the full simulation joint vectors into that active subset, ordered exactly
        as the planner expects, returning `q` and (optionally) `qd`.

        Args:
            full_sim_joint_names: Joint name list from simulation (index-aligned with `full_sim_q`/`full_sim_qd`).
            full_sim_q: Full simulation joint positions, shape `[num_sim_joints]`.
            full_sim_qd: Optional full simulation joint velocities, shape `[num_sim_joints]`.

        Returns:
            A tuple `(activate_q, activate_qd)` where:
            - `activate_q` is ordered by `self._planner.target_joint_names`, shape `[num_active_joints]`.
            - `activate_qd` is the corresponding velocities if `full_sim_qd` is provided; otherwise `None`.

        Raises:
            ValueError: If any planner target joint is missing from `full_sim_joint_names`.
        """

        activate_q, activate_qd = [], [] if full_sim_qd is not None else None
        for joint_name in self._planner.target_joint_names:
            if joint_name not in full_sim_joint_names:
                raise ValueError(
                    f"Joint {joint_name} in planner activate joints is not in the full simulation joint names."
                )
            sim_joint_idx = full_sim_joint_names.index(joint_name)
            activate_q.append(full_sim_q[sim_joint_idx])
            if full_sim_qd is not None and activate_qd is not None:
                activate_qd.append(full_sim_qd[sim_joint_idx])

        activate_q_tensor = torch.stack(activate_q, dim=0)
        if activate_qd is None:
            return activate_q_tensor, None
        return activate_q_tensor, torch.stack(activate_qd, dim=0)

    def _build_extra_target_poses(self, activate_q: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """Build link-level extra target poses based on configuration.

        This is the dispatcher for `extra_target_mode`. It returns a dict mapping link names to pose
        tensors in `[x, y, z, qw, qx, qy, qz]` (single-sample), used as additional link goals/constraints
        during planning.
        """

        if self.cfg.extra_cfg.extra_target_mode == "keep_current":
            return self._build_keep_current_extra_target_poses(activate_q)
        raise ValueError(f"Unsupported extra_target_mode: {self.cfg.extra_cfg.extra_target_mode}")

    def _build_keep_current_extra_target_poses(self, activate_q: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """Build "keep current pose" extra targets for configured links.

        In `keep_current` mode, this computes FK for each link in `extra_target_link_names` and uses
        its current pose as the planning target, effectively constraining those links to remain fixed.
        """

        extra_target_link_names = self.cfg.extra_cfg.extra_target_link_names
        if not extra_target_link_names:
            return None

        extra_target_poses = {}
        for link_name, pose in self._planner.get_link_poses(activate_q, extra_target_link_names).items():
            extra_target_poses[link_name] = torch.cat((pose.position, pose.quaternion), dim=-1).squeeze(0)

        return extra_target_poses

    def extract_goal_from_info(
        self, skill_info: SkillInfo, env: ManagerBasedEnv, env_extra_info: EnvExtraInfo
    ) -> SkillGoal:
        """Return the target pose[x, y, z, qw, qx, qy, qz] in the robot root frame.
        IMPORTANT: the robot root frame is not the same as the robot base frame.
        """

        target_object = skill_info.target_object
        robot = env.scene[env_extra_info.robot_name]

        object_pose_in_env = env.scene[target_object].data.root_pose_w
        object_pos_in_env, object_quat_in_env = object_pose_in_env[:, :3], object_pose_in_env[:, 3:]

        reach_target_pose = env_extra_info.get_next_reach_target_pose(target_object)
        reach_target_pose = reach_target_pose.to(env.device)
        reach_target_pose_in_object = reach_target_pose.unsqueeze(0)
        reach_target_pos_in_object, reach_target_quat_in_object = (
            reach_target_pose_in_object[:, :3],
            reach_target_pose_in_object[:, 3:],
        )

        reach_target_pos_in_env, reach_target_quat_in_env = PoseUtils.combine_frame_transforms(
            object_pos_in_env, object_quat_in_env, reach_target_pos_in_object, reach_target_quat_in_object
        )
        self._logger.info(f"Reach target position in environment: {reach_target_pos_in_env}")
        self._logger.info(f"Reach target quaternion in environment: {reach_target_quat_in_env}")
        self._target_poses["target_pose"] = torch.cat((reach_target_pos_in_env, reach_target_quat_in_env), dim=-1)

        robot_root_pose_in_env = robot.data.root_pose_w
        robot_root_pos_in_env, robot_root_quat_in_env = robot_root_pose_in_env[:, :3], robot_root_pose_in_env[:, 3:]

        reach_target_pos_in_robot_root, reach_target_quat_in_robot_root = PoseUtils.subtract_frame_transforms(
            robot_root_pos_in_env, robot_root_quat_in_env, reach_target_pos_in_env, reach_target_quat_in_env
        )

        target_pose = torch.cat((reach_target_pos_in_robot_root, reach_target_quat_in_robot_root), dim=-1).squeeze(0)
        activate_q, _ = self._build_activate_joint_state(
            robot.data.joint_names, robot.data.joint_pos[0], robot.data.joint_vel[0]
        )
        extra_target_poses = self._build_extra_target_poses(activate_q)

        return SkillGoal(target_object=target_object, target_pose=target_pose, extra_target_poses=extra_target_poses)

    def execute_plan(self, state: WorldState, goal: SkillGoal) -> bool:
        """Execute the plan of the reach skill."""

        self._logger.info(f"Reach from pose in environment: {state.robot_ee_pose}")

        target_pose = goal.target_pose  # target pose in the robot root frame
        target_pos, target_quat = target_pose[:3], target_pose[3:]

        activate_q, activate_qd = self._build_activate_joint_state(
            state.sim_joint_names, state.robot_joint_pos, state.robot_joint_vel
        )
        if activate_qd is None:
            raise ValueError("activate_qd should not be None when planning reach trajectories.")

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
