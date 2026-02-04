import torch
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

markers: list[VisualizationMarkers] = []


def create_replay_marker(marker_id: str | int = 0):
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg = frame_marker_cfg.replace(prim_path=f"/World/Visuals/replay_marker_{marker_id}")
    marker = VisualizationMarkers(marker_cfg)
    markers.append(marker)


def marker_visualize(marker_idx: int, pos: torch.Tensor, quat: torch.Tensor):
    markers[marker_idx].visualize(translations=pos, orientations=quat, marker_indices=[0])
