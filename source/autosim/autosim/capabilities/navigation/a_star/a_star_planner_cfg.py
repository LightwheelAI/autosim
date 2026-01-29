from isaaclab.utils import configclass

from .a_star_planner import AStarPlanner


@configclass
class AStarPlannerCfg:
    """Configuration for the A* planner."""

    class_type: type = AStarPlanner
    """The class type of the A* planner."""

    safety_distance: float = 0.5
    """The safety distance from the obstacle, start penalizing when closer than this distance (meters)."""
    proximity_weight: float = 1.0
    """The weight of the proximity penalty. More high, more avoid the obstacle."""
    angle_threshold: float = 0.1
    """The threshold of the angle to simplify the path."""
    goal_tolerance: float = 1.5
    """The tolerance of the goal, in grid cells."""
