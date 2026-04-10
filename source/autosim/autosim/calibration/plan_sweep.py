import time
from dataclasses import dataclass, field
from typing import Any

import isaaclab.utils.math as PoseUtils
import torch

from autosim.core.pipeline import AutoSimPipeline
from autosim.core.registration import SkillRegistry
from autosim.skills.reach import ReachSkill

from .pose_sampler import OffsetSampler, PoseSampler


@dataclass
class ReachPlanSweepCfg:
    reach_skill_index: int = 0
    """Which reach skill to sweep at (0-based, globally across all subtasks)."""
    sampling: PoseSampler = field(default_factory=OffsetSampler)
    """Sampler used to generate candidate poses around the base pose."""
    top_k: int = 10
    """Number of top poses to print."""
    ik_only: bool = False
    """If True, use IK-only solving instead of full motion planning. Much faster for reachability checking; does not produce trajectories."""


def _tensor_to_list(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().cpu().flatten().tolist()]


def _fmt_pose(vals: list[float]) -> str:
    """Format a 7-value pose as a Python list literal, ready to copy-paste into code."""
    return "[" + ", ".join(f"{v:.4f}" for v in vals) + "]"


def _build_extra_target_link_goals(
    reach_skill: ReachSkill,
    activate_q: torch.Tensor,
    target_poses_r: torch.Tensor,
    env_extra_info,
) -> dict[str, torch.Tensor] | None:
    """Build batched extra link goals by reusing ReachSkill logic."""
    if not reach_skill.cfg.extra_cfg.extra_target_link_names:
        return None

    goal_list = []
    for target_pose in target_poses_r:
        goal_list.append(reach_skill._build_extra_target_poses(activate_q, target_pose, env_extra_info))  # noqa: SLF001

    first_goal = goal_list[0]
    if first_goal is None:
        return None

    return {link_name: torch.stack([goal[link_name] for goal in goal_list], dim=0) for link_name in first_goal.keys()}


def reach_plan_sweep(pipeline: AutoSimPipeline, cfg: ReachPlanSweepCfg) -> list[dict[str, Any]]:
    """
    Execute the pipeline step by step. When the reach_skill_index-th reach skill
    is encountered, capture the live robot and object state and sweep around the
    reach target pose using cuRobo batch planning.

    All skills before the target reach skill are executed normally, so the
    environment reflects the actual state at the point of interest.

    Returns:
        Top-k result rows sorted by plan quality. Each row contains:
            "pose_oe": list[float]  — main EE pose in object frame [x,y,z,qw,qx,qy,qz]
            "plan_success": bool
            "traj_len": int | None  — trajectory length (full planning only)
            "position_error": float | None  — IK position error (IK-only mode)

    Typical multi-reach workflow (each step is a separate script invocation):
        # Step 1: sweep reach 0, note the best pose_oe from the printout
        python reach_plan_sweep.py --reach_skill_index 0 ...

        # Manually update object_reach_target_poses[obj][0] in the pipeline code

        # Step 2: sweep reach 1; reach 0 now runs with the updated hard-coded pose
        python reach_plan_sweep.py --reach_skill_index 1 ...
    """

    pipeline.initialize()

    decompose_result = pipeline.decompose()
    pipeline._check_skill_extra_cfg()
    pipeline.reset_env()

    reach_skill_counter = 0
    reach_count_per_object: dict[str, int] = {}

    for subtask in decompose_result.subtasks:
        for skill_info in subtask.skills:
            skill = SkillRegistry.create(
                skill_info.skill_type,
                pipeline.cfg.skills.get(skill_info.skill_type).extra_cfg,
            )

            if pipeline._action_adapter.should_skip_apply(skill):
                continue

            is_reach = skill_info.skill_type == "reach"

            if is_reach and reach_skill_counter == cfg.reach_skill_index:
                obj_name = skill_info.target_object
                obj_pose_idx = reach_count_per_object.get(obj_name, 0)
                return _sweep(pipeline, cfg, obj_name, obj_pose_idx, skill)

            goal = skill.extract_goal_from_info(skill_info, pipeline._env, pipeline._env_extra_info)
            if is_reach:
                reach_count_per_object[skill_info.target_object] = (
                    reach_count_per_object.get(skill_info.target_object, 0) + 1
                )
                reach_skill_counter += 1

            success, _ = pipeline._execute_single_skill(skill, goal)
            if not success:
                raise ValueError(
                    f"Skill '{skill_info.skill_type}' (step {skill_info.step}) failed before reaching target reach"
                    f" skill (index {cfg.reach_skill_index})."
                )

    raise ValueError(
        f"reach_skill_index={cfg.reach_skill_index} is out of range: only {reach_skill_counter} reach skill(s) found in"
        " the decompose result."
    )


def _sweep(
    pipeline: AutoSimPipeline,
    cfg: ReachPlanSweepCfg,
    obj_name: str,
    obj_pose_idx: int,
    reach_skill: ReachSkill,
) -> list[dict[str, Any]]:
    """
    Core sweep logic. Called once the environment is in the correct pre-reach state.

    Samples K candidate poses around the base reach target (object frame), transforms them to
    robot root frame, then batch-plans with cuRobo. When configured, extra link goals are
    generated from the live joint state using the same extra-target strategy as
    `ReachSkill.extract_goal_from_info()`.
    """

    env = pipeline._env
    env_id = pipeline._env_id
    env_extra_info = pipeline._env_extra_info
    planner = pipeline._motion_planner
    robot = pipeline._robot

    pose_list = env_extra_info.object_reach_target_poses[obj_name]
    if not (0 <= obj_pose_idx < len(pose_list)):
        raise ValueError(f"pose index {obj_pose_idx} out of range for object '{obj_name}' ({len(pose_list)} poses).")

    base_pose_oe = torch.as_tensor(pose_list[obj_pose_idx], device=env.device, dtype=torch.float32).view(7)
    poses_oe = cfg.sampling.sample(base_pose_oe)
    k = int(poses_oe.shape[0])

    obj_pose_w = env.scene[obj_name].data.root_pose_w[env_id]
    obj_pos_w = obj_pose_w[:3].view(1, 3).repeat(k, 1)
    obj_quat_w = obj_pose_w[3:].view(1, 4).repeat(k, 1)

    target_pos_w, target_quat_w = PoseUtils.combine_frame_transforms(
        obj_pos_w, obj_quat_w, poses_oe[:, :3], poses_oe[:, 3:]
    )
    robot_root_pose_w = robot.data.root_pose_w[env_id]
    rr_pos_w = robot_root_pose_w[:3].view(1, 3).repeat(k, 1)
    rr_quat_w = robot_root_pose_w[3:].view(1, 4).repeat(k, 1)
    target_pos_r, target_quat_r = PoseUtils.subtract_frame_transforms(rr_pos_w, rr_quat_w, target_pos_w, target_quat_w)

    state = pipeline._build_world_state()
    activate_q, activate_qd = reach_skill._build_activate_joint_state(  # noqa: SLF001
        state.sim_joint_names, state.robot_joint_pos, state.robot_joint_vel
    )
    if activate_qd is None:
        raise ValueError("activate_qd should not be None when sweep planning reach trajectories.")

    target_poses_r = torch.cat((target_pos_r, target_quat_r), dim=-1)
    link_goals = _build_extra_target_link_goals(reach_skill, activate_q, target_poses_r, env_extra_info)

    t0 = time.time()
    if cfg.ik_only:
        result = planner.solve_ik_batch(target_pos_r, target_quat_r, link_goals=link_goals)
    else:
        result = planner.plan_motion_batch(target_pos_r, target_quat_r, activate_q, activate_qd, link_goals=link_goals)
    dt_ms = (time.time() - t0) * 1000.0

    success = (
        result.success.detach().cpu().bool() if result.success is not None else torch.zeros((k,), dtype=torch.bool)
    )
    pos_err = result.position_error.detach().cpu() if result.position_error is not None else None
    traj_last = (
        result.path_buffer_last_tstep if (not cfg.ik_only and result.path_buffer_last_tstep is not None) else None
    )

    rows = []
    for i in range(k):
        rows.append({
            "pose_oe": _tensor_to_list(poses_oe[i]),
            "plan_success": bool(success[i].item()),
            "traj_len": int(traj_last[i]) if traj_last is not None else None,
            "position_error": float(pos_err[i].item()) if pos_err is not None else None,
        })

    def _sort_key(r):
        if not r["plan_success"]:
            return (10**9, 10**9)
        return (r["traj_len"] or 10**8, r["position_error"] or 10**8)

    top_k = sorted(rows, key=_sort_key)[: cfg.top_k]

    success_count = int(success.sum().item())
    _SEP = "─" * 80
    print(_SEP)
    print(f"  reach_plan_sweep  │  object='{obj_name}'  reach_skill_index={cfg.reach_skill_index}")
    print(f"  success  {success_count}/{k} ({success_count / k:.1%})  │  time  {dt_ms:.0f} ms")
    print(_SEP)
    print(f"  top {len(top_k)} poses  (object frame  [x, y, z, qw, qx, qy, qz])")
    print()
    for rank, r in enumerate(top_k):
        mark = "✓" if r["plan_success"] else "✗"
        metric = f"traj_len={r['traj_len']}" if r["traj_len"] is not None else f"pos_err={r['position_error']:.4f}"
        print(f"  [{rank}] {mark}  {_fmt_pose(r['pose_oe'])}  # {metric}")
    print(_SEP)

    return top_k
