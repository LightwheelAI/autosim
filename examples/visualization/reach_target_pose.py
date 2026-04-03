"""Visualize all reach target poses for an autosim pipeline.

Usage
-----
1) Start the visualization (it will export a debug JSON once after `pipeline.reset_env()`):
   python examples/visualization/reach_target_pose.py --pipeline_id <PIPELINE_ID> \
     --debug_poses_path /abs/path/reach_target_poses_debug.json

2) Edit and save that JSON file. The script polls its mtime and reloads markers automatically.

`--debug_poses_path` JSON format
---------------------------------
The script expects this payload:
{
  "object_reach_target_poses": {
    "<object_name>": [
      [x, y, z, qw, qx, qy, qz],
      ...
    ],
    ...
  },
  "object_extra_reach_target_poses": {
    "<object_name>": {
      "<ee_name>": [
        [x, y, z, qw, qx, qy, qz],
        ...
      ],
      ...
    },
    ...
  }
}

Notes
-----
* Poses are in the object frame: [x, y, z, qw, qx, qy, qz].
* `--live_poll_interval_s` controls how often the file is checked (default: 0.2s).
"""

import argparse
import json
import os
import time

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize reach target poses for an autosim pipeline.")
parser.add_argument("--pipeline_id", type=str, default=None, help="Name of the autosim pipeline.")
parser.add_argument(
    "--debug_poses_path",
    type=str,
    default="reach_target_poses_debug.json",
    help=(
        "If provided, the script will export the current `object_reach_target_poses` "
        "(and `object_extra_reach_target_poses`) to this JSON file after `reset_env()`, "
        "and reload it on every file change."
    ),
)
parser.add_argument(
    "--live_poll_interval_s",
    type=float,
    default=0.2,
    help="Polling interval (seconds) for checking `--debug_poses_path` updates.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app


import autosim_examples  # noqa: F401
from autosim import make_pipeline
from autosim.utils.debug_util import visualize_reach_target_poses


def _load_env_extra_poses_json(
    path: str,
) -> tuple[dict[str, list[list[float]]], dict[str, dict[str, list[list[float]]]]]:
    """Load env extra poses from the exported debug JSON.

    We expect the payload format produced by this script:
    {
      "object_reach_target_poses": { "<obj>": [[7], ...] },
      "object_extra_reach_target_poses": { "<obj>": { "<ee>": [[7], ...] } }
    }
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    object_reach_target_poses: dict[str, list[list[float]]] = {}
    object_extra_reach_target_poses: dict[str, dict[str, list[list[float]]]] = {}

    if not isinstance(data, dict):
        raise ValueError("Debug JSON root must be an object.")

    reach = data.get("object_reach_target_poses", {})
    if not isinstance(reach, dict):
        raise ValueError("`object_reach_target_poses` must be an object mapping.")
    for obj_name, pose_list in reach.items():
        if not isinstance(obj_name, str):
            raise ValueError("Reach: object names must be strings.")
        if not isinstance(pose_list, list):
            raise ValueError(f"Reach: `{obj_name}` must map to a list of poses.")
        normalized: list[list[float]] = []
        for pose in pose_list:
            if not (isinstance(pose, list) and len(pose) == 7):
                raise ValueError(f"Reach: each pose for `{obj_name}` must be list length 7.")
            normalized.append([float(v) for v in pose])
        object_reach_target_poses[obj_name] = normalized

    extra = data.get("object_extra_reach_target_poses", {})
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise ValueError("`object_extra_reach_target_poses` must be an object mapping.")
    for obj_name, ee_dict in extra.items():
        if not isinstance(obj_name, str):
            raise ValueError("Extra: object names must be strings.")
        if not isinstance(ee_dict, dict):
            raise ValueError(f"Extra: `{obj_name}` must map ee_name -> pose_list.")
        object_extra_reach_target_poses[obj_name] = {}
        for ee_name, pose_list in ee_dict.items():
            if not isinstance(ee_name, str):
                raise ValueError("Extra: ee_name must be a string.")
            if not isinstance(pose_list, list):
                raise ValueError(f"Extra: `{obj_name}.{ee_name}` must be a list.")
            normalized: list[list[float]] = []
            for pose in pose_list:
                if not (isinstance(pose, list) and len(pose) == 7):
                    raise ValueError(f"Extra: each pose for `{obj_name}.{ee_name}` must be list length 7.")
                normalized.append([float(v) for v in pose])
            object_extra_reach_target_poses[obj_name][ee_name] = normalized

    return object_reach_target_poses, object_extra_reach_target_poses


def _apply_live_poses(*, poses_path: str, pipeline) -> None:
    """Update `pipeline._env_extra_info.*` from JSON."""
    env = pipeline._env
    env_extra_info = pipeline._env_extra_info
    object_reach_target_poses, object_extra_reach_target_poses = _load_env_extra_poses_json(poses_path)

    # Replace dicts entirely so removed keys in the JSON disappear from visualization.
    env_extra_info.object_reach_target_poses = {}
    env_extra_info.object_extra_reach_target_poses = {}

    # Update primary reach targets
    for obj_name, pose_list in object_reach_target_poses.items():
        if obj_name not in env.scene.keys():
            continue
        obj_pose_w = env.scene[obj_name].data.root_pose_w[0]  # [7]
        device = obj_pose_w.device
        dtype = obj_pose_w.dtype
        env_extra_info.object_reach_target_poses[obj_name] = [
            torch.tensor(pose, device=device, dtype=dtype) for pose in pose_list
        ]

    # Update extra EE reach targets (if present)
    for obj_name, ee_dict in object_extra_reach_target_poses.items():
        if obj_name not in env.scene.keys():
            continue
        obj_pose_w = env.scene[obj_name].data.root_pose_w[0]  # [7]
        device = obj_pose_w.device
        dtype = obj_pose_w.dtype
        if obj_name not in env_extra_info.object_extra_reach_target_poses:
            env_extra_info.object_extra_reach_target_poses[obj_name] = {}
        for ee_name, pose_list in ee_dict.items():
            env_extra_info.object_extra_reach_target_poses[obj_name][ee_name] = [
                torch.tensor(pose, device=device, dtype=dtype) for pose in pose_list
            ]


def _export_env_extra_poses_to_json(*, out_path: str, pipeline) -> None:
    """Export current env_extra_info reach targets to JSON."""
    env_extra_info = pipeline._env_extra_info

    def _tensor_pose_to_list(p: list) -> list[float]:
        # p is torch.Tensor(7,) but we want plain python floats.
        return [float(x) for x in p]

    object_reach_target_poses: dict[str, list[list[float]]] = {}
    for obj_name, pose_list in env_extra_info.object_reach_target_poses.items():
        object_reach_target_poses[obj_name] = [_tensor_pose_to_list(pose.tolist()) for pose in pose_list]

    object_extra_reach_target_poses: dict[str, dict[str, list[list[float]]]] = {}
    if hasattr(env_extra_info, "object_extra_reach_target_poses"):
        for obj_name, ee_dict in env_extra_info.object_extra_reach_target_poses.items():
            object_extra_reach_target_poses[obj_name] = {}
            for ee_name, pose_list in ee_dict.items():
                object_extra_reach_target_poses[obj_name][ee_name] = [
                    _tensor_pose_to_list(pose.tolist()) for pose in pose_list
                ]

    payload = {
        "object_reach_target_poses": object_reach_target_poses,
        "object_extra_reach_target_poses": object_extra_reach_target_poses,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    pipeline = make_pipeline(args_cli.pipeline_id)
    pipeline.initialize()
    pipeline.reset_env()

    debug_path = os.path.abspath(args_cli.debug_poses_path)

    # Always export first so user can edit immediately after startup.
    _export_env_extra_poses_to_json(out_path=debug_path, pipeline=pipeline)
    print(f"[reach_target_pose] Exported debug poses to: {debug_path}")

    # Apply once (in case user edited the file between process start and export completion).
    try:
        _apply_live_poses(poses_path=debug_path, pipeline=pipeline)
    except Exception as e:
        print(f"[reach_target_pose] Failed to apply exported debug poses: {e}")
    visualize_reach_target_poses(pipeline._env_extra_info, pipeline._env)

    last_mtime = os.path.getmtime(debug_path)
    last_poll_t = 0.0

    while simulation_app.is_running():
        pipeline._env.sim.render()

        now = time.time()
        if now - last_poll_t < args_cli.live_poll_interval_s:
            continue
        last_poll_t = now

        try:
            mtime = os.path.getmtime(debug_path)
        except OSError:
            continue

        if mtime > last_mtime:
            last_mtime = mtime
            try:
                _apply_live_poses(poses_path=debug_path, pipeline=pipeline)
                visualize_reach_target_poses(pipeline._env_extra_info, pipeline._env)
                print(f"[reach_target_pose] Reloaded markers from: {debug_path}")
            except Exception as e:
                print(f"[reach_target_pose] Failed to reload poses: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()
