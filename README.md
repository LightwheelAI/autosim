# autosim

**autosim: An automated data generation pipeline based on NVIDIA Isaac Lab**

An automatic simulation data generation pipeline built on Isaac Lab, integrating LLM-based task decomposition, motion planning, and navigation capabilities.

## Overview

`autosim` provides an extensible automated data generation pipeline for NVIDIA Isaac Lab environments:

- Starting from **task code and scene information**, it uses an LLM to **decompose high-level tasks** automatically.
- Maps the decomposition result into a sequence of **atomic skills**.
- Invokes **motion planning (based on cuRobo)** and **navigation** to execute these skills in simulation.
- Produces unified **action sequences / trajectory data** for downstream robot learning and research.

> In short: `autosim` helps you automatically turn “a task in Isaac Lab” into an executable sequence of skills, and generates data that can be used for training or evaluation.

## Installation

Below is a typical setup workflow. `autosim` can be installed as a submodule inside an environment that already contains Isaac Lab.

```bash
conda create -n autosim python=3.11

conda activate autosim

git clone https://github.com/LightwheelAI/autosim.git

cd autosim

git submodule update --init --recursive
```

### IsaacLab Installation

`autosim` depends on Isaac Lab. You can follow the official installation guide [here](https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/pip_installation.html), or use the commands below. If you already have an environment with Isaac Lab installed, you can reuse it and skip this step.

```bash
# Install CUDA toolkit
conda install -c "nvidia/label/cuda-12.8.1" cuda-toolkit

# Install PyTorch
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install IsaacSim
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com


# Install IsaacLab
sudo apt install cmake build-essential

cd dependencies/IsaacLab
./isaaclab.sh --install
```

> ⚠️ The installation of Isaac Lab is relatively involved. Please follow the official documentation carefully. This repository only works on top of a correctly configured Isaac Lab environment.

### cuRobo Installation

Some skills in `autosim` depend on cuRobo. You can follow the official [documentation](https://curobo.org/get_started/1_install_instructions.html), or use the commands below:

```bash
cd dependencies/curobo

pip install -e . --no-build-isolation
```

### autosim Installation

Finally, install `autosim` into your environment:

```bash
pip install -e source/autosim
```

## Quick Start

### Run the Example Pipeline (Franka Cube Lift)

After completing the installation and configuration steps above, you can directly run the built-in example.

First, install the `autosim` example package:

```bash
pip install -e source/autosim_examples
```

Then launch the example with:

```bash
cd autosim

python examples/run_autosim_example.py \
  --pipeline_id AutoSimPipeline-FrankaCubeLift-v0
```

After launching, you will see in the Isaac Sim UI that the manipulator automatically executes the Cube Lift task.

## Defining Your Own Pipeline

For a task that has already been defined in Isaac Lab, you can use `autosim` to define a custom `AutoSimPipeline` for it. A typical workflow looks like this:

1. **Implement a config class**

   Inherit from `AutoSimPipelineCfg` and adjust components as needed:

   - `decomposer` (e.g., using a different LLM or prompt template)
   - `motion_planner` (e.g., different robot model or planner parameters)
   - `skills` / `action_adapter`, etc.

2. **Implement the pipeline class**

   Inherit from `AutoSimPipeline` and implement:

   - `load_env(self) -> ManagerBasedEnv`: load the environment based on Isaac Lab / Gymnasium; this environment should correspond to the pre-defined Task.
   - `get_env_extra_info(self) -> EnvExtraInfo`: provide the task name, robot name, end-effector link, and reach targets expressed as poses relative to objects, etc.

3. **Register the pipeline in the package’s `__init__.py`**

   For example:

   ```python
   from autosim import register_pipeline

   register_pipeline(
       id="MyPipeline-v0",
       entry_point="my_package.pipelines.my_pipeline:MyPipeline",
       cfg_entry_point="my_package.pipelines.my_pipeline:MyPipelineCfg",
   )
   ```

4. **Use it from a script or external project**

   ```python
   from autosim import make_pipeline

   pipeline = make_pipeline("MyPipeline-v0")
   output = pipeline.run()
   ```

> You can refer to `source/autosim_examples/autosim_examples/autosim/pipelines/franka_lift_cube.py` for a minimal working example.

## Contributing

Issues, feature requests, and pull requests are all welcome!

Before submitting contributions, we recommend:

- First verify that the example pipeline runs correctly in your local environment.
- Follow the existing code style in this repository (Black + isort, see the root `pyproject.toml`).
- Whenever possible, add tests or minimal reproducible examples for new features or bug fixes.

## Acknowledgements

`autosim` is built on top of the following excellent open-source projects:

- **cuRobo**: GPU-accelerated robot motion planning.
- **Isaac Lab**: NVIDIA’s framework for robot simulation and reinforcement learning.
- And other dependencies and upstream projects used in this repository.

We sincerely thank the authors and communities behind these projects.
