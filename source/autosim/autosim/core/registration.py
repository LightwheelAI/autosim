"""AutoSim registration system."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from autosim.core.pipeline import AutoSimPipeline as Pipeline


# ============================================================================
# Pipeline Registration System
# ============================================================================
# This section provides the registration and instantiation system for
# pipelines. Pipelines can be registered with an ID and entry points, then
# created using make_pipeline().
#
# Usage:
#     # 1. Register a pipeline
#     register_pipeline(
#         id="MyPipeline-v0",
#         entry_point="autosim.pipelines:MyPipeline",
#         cfg_entry_point="autosim.pipelines:MyPipelineCfg",
#     )
#
#     # 2. Create a pipeline instance
#     pipeline = make_pipeline("MyPipeline-v0")
#     pipeline.run()
#
#     # 3. List all registered pipelines
#     pipeline_ids = list_pipelines()
#
#     # 4. Unregister a pipeline
#     unregister_pipeline("MyPipeline-v0")
# ============================================================================


class PipelineCreator(Protocol):
    """Function that creates a pipeline instance."""

    def __call__(self, **kwargs: Any) -> Pipeline: ...


class ConfigCreator(Protocol):
    """Function that creates a configuration instance."""

    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass
class PipelineEntry:
    """Entry for a pipeline in the registry.

    Attributes:
        id: Unique identifier for the pipeline (e.g., "MyPipeline-v0").
        entry_point: String pointing to the Pipeline class or a callable that creates a pipeline instance. Format: "module.path:ClassName".
        cfg_entry_point: String pointing to the configuration class or a callable that creates a config instance. Format: "module.path:ConfigClass".
    """

    id: str
    entry_point: PipelineCreator | str | None = field(default=None)
    cfg_entry_point: ConfigCreator | str | None = field(default=None)


# Global registry for pipelines
pipeline_registry: dict[str, PipelineEntry] = {}


def register_pipeline(
    id: str,
    entry_point: PipelineCreator | str | None = None,
    cfg_entry_point: ConfigCreator | str | None = None,
) -> None:
    """Register a pipeline in the global registry."""
    assert entry_point is not None, "Entry point must be provided."
    assert cfg_entry_point is not None, "Configuration entry point must be provided."

    if id in pipeline_registry:
        raise ValueError(
            f"Pipeline with id '{id}' is already registered. To register a new version, use a different id (e.g.,"
            f" '{id}-v1')."
        )

    entry = PipelineEntry(
        id=id,
        entry_point=entry_point,
        cfg_entry_point=cfg_entry_point,
    )
    pipeline_registry[entry.id] = entry


def _load_entry_point(entry_point: str) -> Any:
    """Load a class or function from an entry point string."""
    try:
        mod_name, attr_name = entry_point.split(":")
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr_name)
        return obj
    except (ValueError, ModuleNotFoundError, AttributeError) as e:
        raise ValueError(
            f"Could not resolve entry point '{entry_point}'. Expected format: 'module.path:ClassName'. Error: {e}"
        ) from e


def _load_creator(creator: str | PipelineCreator | ConfigCreator) -> PipelineCreator | ConfigCreator:
    if isinstance(creator, str):
        return _load_entry_point(creator)
    else:
        return creator


def make_pipeline(
    id: str,
) -> Pipeline:
    """Create a pipeline instance from the registry."""
    if id not in pipeline_registry:
        raise ValueError(
            f"Pipeline '{id}' not found in registry. You can list all registered pipelines with list_pipelines()."
        )

    entry = pipeline_registry[id]

    pipeline_creator = _load_creator(entry.entry_point)
    cfg_creator = _load_creator(entry.cfg_entry_point)

    # Instantiate the pipeline
    try:
        cfg = cfg_creator()
        pipeline = pipeline_creator(cfg=cfg)
    except TypeError as e:
        entry_point_str = entry.entry_point if isinstance(entry.entry_point, str) else str(entry.entry_point)
        raise TypeError(
            f"Failed to instantiate pipeline '{id}' with entry point '{entry_point_str}'. Error: {e}"
        ) from e

    return pipeline


def list_pipelines() -> list[str]:
    """List all registered pipeline IDs."""
    return sorted(pipeline_registry.keys())


def unregister_pipeline(id: str) -> None:
    """Unregister a pipeline from the registry."""
    if id not in pipeline_registry:
        raise ValueError(f"Pipeline '{id}' not found in registry.")
    del pipeline_registry[id]


# ============================================================================
# Skill Registration System
# ============================================================================
# This section will provide the registration and instantiation system for
# skills. (To be implemented)
# ============================================================================
