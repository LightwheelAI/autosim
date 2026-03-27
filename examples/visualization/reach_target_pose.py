"""Visualize all reach target poses for an autosim pipeline."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize reach target poses for an autosim pipeline.")
parser.add_argument("--pipeline_id", type=str, default=None, help="Name of the autosim pipeline.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app


import autosim_examples  # noqa: F401
from autosim import make_pipeline
from autosim.utils.debug_util import visualize_reach_target_poses


def main():
    pipeline = make_pipeline(args_cli.pipeline_id)
    pipeline.initialize()
    pipeline.reset_env()
    visualize_reach_target_poses(pipeline._env_extra_info, pipeline._env)

    while simulation_app.is_running():
        pipeline._env.sim.render()


if __name__ == "__main__":
    main()
    simulation_app.close()
