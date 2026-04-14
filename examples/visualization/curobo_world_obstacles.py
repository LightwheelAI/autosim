"""Visualize cuRobo world obstacles in the Isaac Sim viewport.

This script reads the obstacle world currently held inside cuRobo's collision checker
(`planner.motion_gen.world_coll_checker.world_model`) and draws simple wireframes.
Before drawing, it synchronizes dynamic rigid object poses so the visualization matches
the obstacle poses currently used by cuRobo for collision checking.

Usage
-----
Run with Isaac Sim UI enabled (do NOT use ``--headless`` if you want to see the viewport):

    python examples/visualization/curobo_world_obstacles.py --pipeline_id <PIPELINE_ID>

Notes
-----
* This script reads from cuRobo's live collision world instead of rebuilding from USD,
  so reset-time pose changes for dynamic rigid objects are reflected in the visualization.
* Wireframe drawing uses debug lines; keep the app running to inspect the scene.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize cuRobo collision world obstacles.")
parser.add_argument("--pipeline_id", type=str, default=None, help="Name of the autosim pipeline.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import isaaclab.utils.math as PoseUtils

import autosim_examples  # noqa: F401
from autosim import make_pipeline
from autosim.utils.debug_util import clear_debug_drawing, draw_line


@dataclass(frozen=True)
class _Pose7:
    pos: torch.Tensor  # (3,)
    quat: torch.Tensor  # (4,) wxyz


def _as_pose7(pose7: Sequence[float] | torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> _Pose7:
    t = torch.as_tensor(pose7, device=device, dtype=dtype).view(7)
    return _Pose7(pos=t[:3], quat=t[3:])


def _quat_to_rotmat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz) to rotation matrix. q shape (4,). Returns (3,3)."""
    q = q / (q.norm(p=2) + 1e-12)
    w, x, y, z = q
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return torch.stack(
        [
            torch.stack([ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=0),
            torch.stack([2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)], dim=0),
            torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz], dim=0),
        ],
        dim=0,
    )


def _transform_points(pose: _Pose7, pts: torch.Tensor) -> torch.Tensor:
    """Apply pose (pos, quat) to points. pts shape (...,3) in local frame."""
    r = _quat_to_rotmat_wxyz(pose.quat)
    return (pts @ r.T) + pose.pos


def _draw_oriented_box(*, pose_w: _Pose7, half_dims_xyz: torch.Tensor, color, thickness: float, z_lift: float) -> None:
    hx, hy, hz = float(half_dims_xyz[0]), float(half_dims_xyz[1]), float(half_dims_xyz[2])
    corners_l = torch.tensor(
        [
            [-hx, -hy, -hz],
            [-hx, -hy, +hz],
            [-hx, +hy, -hz],
            [-hx, +hy, +hz],
            [+hx, -hy, -hz],
            [+hx, -hy, +hz],
            [+hx, +hy, -hz],
            [+hx, +hy, +hz],
        ],
        device=pose_w.pos.device,
        dtype=pose_w.pos.dtype,
    )
    corners_w = _transform_points(pose_w, corners_l).detach().cpu()
    if z_lift != 0.0:
        corners_w[:, 2] += float(z_lift)

    # 12 edges by index pairs
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges:
        pa = tuple(float(v) for v in corners_w[a].tolist())
        pb = tuple(float(v) for v in corners_w[b].tolist())
        draw_line(pa, pb, color=color, size=thickness)


def _compose_robot_to_world(*, robot_root_pose_w: torch.Tensor, pose_r: _Pose7) -> _Pose7:
    """Compose pose in robot root frame to world frame."""
    rr_pos_w = robot_root_pose_w[:3].view(1, 3)
    rr_quat_w = robot_root_pose_w[3:].view(1, 4)
    pos_r = pose_r.pos.view(1, 3)
    quat_r = pose_r.quat.view(1, 4)
    pos_w, quat_w = PoseUtils.combine_frame_transforms(rr_pos_w, rr_quat_w, pos_r, quat_r)
    return _Pose7(pos=pos_w.view(3), quat=quat_w.view(4))


def _visualize_world_obstacles(
    *,
    pipeline,
    env_id: int = 0,
    color=(0.95, 0.2, 0.25, 1.0),
    thickness: float = 2.0,
    z_lift: float = 0.0,
    max_mesh_vertices: int = 200000,
):
    planner = pipeline._motion_planner
    planner.sync_dynamic_objects()

    world_model = planner.motion_gen.world_coll_checker.world_model
    if isinstance(world_model, list):
        world_model = world_model[env_id]
    if world_model is None:
        raise RuntimeError("cuRobo collision checker has no world model loaded.")

    robot_root_pose_w = pipeline._robot.data.root_pose_w[env_id].detach()
    device = robot_root_pose_w.device
    dtype = robot_root_pose_w.dtype

    clear_debug_drawing()

    # NOTE: poses in the world_model are in the robot-root frame

    # Cuboids (OBB pose + dims)
    for cub in world_model.cuboid:
        pose_r = _as_pose7(cub.pose, device=device, dtype=dtype)
        pose_w = _compose_robot_to_world(robot_root_pose_w=robot_root_pose_w, pose_r=pose_r)
        dims = torch.as_tensor(cub.dims, device=device, dtype=dtype).view(3)
        half_dims = dims * 0.5
        _draw_oriented_box(pose_w=pose_w, half_dims_xyz=half_dims, color=color, thickness=thickness, z_lift=z_lift)

    # Spheres (pose center + radius) -> draw as cube approximation
    for sph in world_model.sphere:
        pose_r = _as_pose7(sph.pose, device=device, dtype=dtype)
        pose_w = _compose_robot_to_world(robot_root_pose_w=robot_root_pose_w, pose_r=pose_r)
        r = float(sph.radius)
        half_dims = torch.tensor([r, r, r], device=device, dtype=dtype)
        _draw_oriented_box(pose_w=pose_w, half_dims_xyz=half_dims, color=color, thickness=thickness, z_lift=z_lift)

    # Cylinders (pose + radius + height) -> draw as oriented bounding box approximation
    for cyl in world_model.cylinder:
        pose_r = _as_pose7(cyl.pose, device=device, dtype=dtype)
        pose_w = _compose_robot_to_world(robot_root_pose_w=robot_root_pose_w, pose_r=pose_r)
        half_dims = torch.tensor(
            [float(cyl.radius), float(cyl.radius), float(cyl.height) * 0.5], device=device, dtype=dtype
        )
        _draw_oriented_box(pose_w=pose_w, half_dims_xyz=half_dims, color=color, thickness=thickness, z_lift=z_lift)

    # Capsules (pose + radius + base/tip) -> draw as oriented bounding box approximation
    for cap in world_model.capsule:
        pose_r = _as_pose7(cap.pose, device=device, dtype=dtype)
        pose_w = _compose_robot_to_world(robot_root_pose_w=robot_root_pose_w, pose_r=pose_r)
        base = torch.as_tensor(cap.base, device=device, dtype=dtype).view(3)
        tip = torch.as_tensor(cap.tip, device=device, dtype=dtype).view(3)
        height = float((tip - base).norm().item())
        half_dims = torch.tensor([float(cap.radius), float(cap.radius), height * 0.5], device=device, dtype=dtype)
        _draw_oriented_box(pose_w=pose_w, half_dims_xyz=half_dims, color=color, thickness=thickness, z_lift=z_lift)

    # Mesh obstacles: prefer a tight OBB via cuRobo's `Mesh.get_cuboid()`.
    for mesh in world_model.mesh:
        verts = getattr(mesh, "vertices", None)
        if verts is not None and len(verts) > int(max_mesh_vertices):
            continue

        try:
            cub = mesh.get_cuboid()
        except Exception:
            # Fallback: if OBB conversion fails, skip visualization for this mesh.
            print(f"Failed to convert mesh {mesh.name} to OBB")
            continue

        pose_r = _as_pose7(cub.pose, device=device, dtype=dtype)
        pose_w = _compose_robot_to_world(robot_root_pose_w=robot_root_pose_w, pose_r=pose_r)
        dims = torch.as_tensor(cub.dims, device=device, dtype=dtype).view(3)
        half_dims = dims * 0.5
        _draw_oriented_box(pose_w=pose_w, half_dims_xyz=half_dims, color=color, thickness=thickness, z_lift=z_lift)


def main():
    pipeline = make_pipeline(args_cli.pipeline_id)
    pipeline.initialize()
    pipeline.reset_env()

    _visualize_world_obstacles(pipeline=pipeline)

    while simulation_app.is_running():
        pipeline._env.sim.render()


if __name__ == "__main__":
    main()
    simulation_app.close()
