from abc import ABC, abstractmethod

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from .decomposer import Decomposer, DecomposerCfg
from .types import DecomposeResult, EnvExtraInfo, PipelineOutput


@configclass
class AutoSimPipelineCfg:
    """Configuration for the AutoSim pipeline."""

    decomposer: DecomposerCfg = DecomposerCfg()
    """The decomposer for the AutoSim pipeline."""


class AutoSimPipeline(ABC):
    def __init__(self, cfg: AutoSimPipelineCfg) -> None:
        self.cfg = cfg

        self._decomposer: Decomposer = self.cfg.decomposer.class_type(self.cfg.decomposer)

        # load the environment and extra information
        self._env: ManagerBasedEnv = self.load_env()
        self._env_extra_info: EnvExtraInfo = self.get_env_extra_info()

    def run(self) -> PipelineOutput:
        """Run the AutoSim pipeline."""

        # decompose the task with cache hit check
        if self._decomposer.is_cache_hit(self._env_extra_info.task_name):
            decompose_result: DecomposeResult = self._decomposer.read_cache(self._env_extra_info.task_name)
        else:
            decompose_result: DecomposeResult = self._decomposer.decompose(self._env_extra_info)
            self._decomposer.write_cache(self._env_extra_info.task_name, decompose_result)

        # execute the pipeline

        return PipelineOutput(success=True)

    @abstractmethod
    def load_env(self) -> ManagerBasedEnv:
        """Load the environment in isaaclab."""
        raise NotImplementedError(f"{self.__class__.__name__}.load_env() must be implemented.")

    @abstractmethod
    def get_env_extra_info(self) -> EnvExtraInfo:
        """Get the extra information from the environment."""
        raise NotImplementedError(f"{self.__class__.__name__}.get_env_extra_info() must be implemented.")
