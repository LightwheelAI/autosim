import os
from dataclasses import MISSING

from isaaclab.utils import configclass

from autosim.core.decomposer import DecomposerCfg

from .llm_decomposer import LLMDecomposer


@configclass
class LLMDecomposerCfg(DecomposerCfg):
    """Configuration for the LLM decomposer."""

    class_type: type = LLMDecomposer
    """The class type of the LLM decomposer."""

    api_key: str = MISSING
    """The API key for the LLM."""

    base_url: str = "https://api.chatanywhere.org/v1"  # TODO: change here.
    """The base URL for the LLM API."""

    model: str = "gpt-3.5-turbo"
    """The model name for the LLM."""

    temperature: float = 0.3
    """The temperature for the LLM."""

    max_tokens: int = 4000
    """The maximum number of tokens to generate."""

    def __post_init__(self) -> None:
        super().__post_init__()
        api_key = os.environ.get("AUTOSIM_LLM_API_KEY")
        if api_key is None:
            raise ValueError(
                "Please set the AUTOSIM_LLM_API_KEY environment variable when using the LLMDecomposer, e.g. export"
                " AUTOSIM_LLM_API_KEY=your_api_key"
            )
        self.api_key = api_key
