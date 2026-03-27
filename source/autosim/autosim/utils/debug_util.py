import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms

markers: dict[str, VisualizationMarkers] = {}


def create_marker(marker_name: str):
    if marker_name in markers.keys():
        return
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg = frame_marker_cfg.replace(prim_path=f"/World/Visuals/replay_marker_{marker_name}")
    marker = VisualizationMarkers(marker_cfg)
    markers[marker_name] = marker


def visualize_marker(marker_name: str, pose: torch.Tensor):
    pos, quat = pose[:, :3], pose[:, 3:]
    markers[marker_name].visualize(translations=pos, orientations=quat, marker_indices=[0] * pos.shape[0])


def _collect_world_poses(obj_poses: dict[str, list[torch.Tensor]], env: ManagerBasedEnv) -> torch.Tensor | None:
    """Transform a dict of object-frame poses to world frame and stack them.

    Args:
        obj_poses: dict mapping object name -> list of [7] tensors in object frame.
        env: The Isaac Lab environment (used to look up live object poses).

    Returns:
        Stacked world-frame poses of shape [N, 7], or None if there are no poses.
    """
    world_poses = []
    for obj_name, pose_list in obj_poses.items():
        obj_pose_w = env.scene[obj_name].data.root_pose_w[0]  # [7]
        obj_pos_w = obj_pose_w[:3].unsqueeze(0)  # [1, 3]
        obj_quat_w = obj_pose_w[3:].unsqueeze(0)  # [1, 4]
        for pose in pose_list:
            p = pose.unsqueeze(0)  # [1, 7]
            pos_w, quat_w = combine_frame_transforms(obj_pos_w, obj_quat_w, p[:, :3], p[:, 3:])
            world_poses.append(torch.cat([pos_w, quat_w], dim=-1))  # [1, 7]
    if not world_poses:
        return None
    return torch.cat(world_poses, dim=0)  # [N, 7]


def visualize_reach_target_poses(env_extra_info, env: ManagerBasedEnv) -> None:
    """Visualize all reach target poses from env_extra_info as frame markers.

    Creates markers for:
    - ``env_extra_info.object_reach_target_poses`` under the marker name
      ``"reach_target_poses"``.
    - Each extra EE in ``env_extra_info.object_extra_reach_target_poses`` under
      ``"reach_target_poses_{ee_name}"``.

    Must be called after the environment has been reset so that object poses are
    at their initial positions.
    """
    # Primary reach target poses
    primary_poses_w = _collect_world_poses(env_extra_info.object_reach_target_poses, env)
    if primary_poses_w is not None:
        create_marker("reach_target_poses")
        visualize_marker("reach_target_poses", primary_poses_w)

    # Extra EE reach target poses (multi-arm)
    # object_extra_reach_target_poses: dict[obj_name, dict[ee_name, list[Tensor]]]
    ee_pose_lists: dict[str, dict[str, list[torch.Tensor]]] = {}
    for obj_name, ee_dict in env_extra_info.object_extra_reach_target_poses.items():
        for ee_name, pose_list in ee_dict.items():
            if ee_name not in ee_pose_lists:
                ee_pose_lists[ee_name] = {}
            ee_pose_lists[ee_name][obj_name] = pose_list

    for ee_name, obj_poses in ee_pose_lists.items():
        extra_poses_w = _collect_world_poses(obj_poses, env)
        if extra_poses_w is not None:
            marker_name = f"reach_target_poses_{ee_name}"
            create_marker(marker_name)
            visualize_marker(marker_name, extra_poses_w)
