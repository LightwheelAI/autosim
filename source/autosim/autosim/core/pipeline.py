from abc import ABC, abstractmethod

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from .types import PipelineOutput


@configclass
class AutoSimPipelineCfg:
    """Configuration for the AutoSim pipeline."""

    place_holder: str = "placeholder for now"


class AutoSimPipeline(ABC):
    def __init__(self, cfg: AutoSimPipelineCfg) -> None:
        self.cfg = cfg

    def run(self) -> PipelineOutput:
        """Run the pipeline."""
        return PipelineOutput(success=True)

    @abstractmethod
    def load_env(self) -> ManagerBasedEnv:
        """Load the environment in isaaclab."""
        raise NotImplementedError(f"{self.__class__.__name__}.load_env() must be implemented.")
