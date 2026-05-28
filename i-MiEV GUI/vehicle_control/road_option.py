from __future__ import annotations

from enum import IntEnum
from typing import Any, Iterable, Sequence, Tuple


class RoadOption(IntEnum):
    """CARLA-compatible high-level navigation command.

    The values intentionally match CARLA 0.9.15's
    agents.navigation.local_planner.RoadOption. Real-vehicle code must not
    import CARLA directly, so this lightweight mirror keeps the model input
    format compatible without introducing a simulator dependency.
    """

    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


def normalize_road_option(value: Any, default: RoadOption = RoadOption.LANEFOLLOW) -> RoadOption:
    """Convert int/enum/string command to a RoadOption."""
    if isinstance(value, RoadOption):
        return value

    if isinstance(value, str):
        text = value.strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "void": RoadOption.VOID,
            "none": RoadOption.VOID,
            "idle": RoadOption.LANEFOLLOW,
            "start": RoadOption.LANEFOLLOW,
            "lane_follow": RoadOption.LANEFOLLOW,
            "lanefollow": RoadOption.LANEFOLLOW,
            "follow_lane": RoadOption.LANEFOLLOW,
            "intersection": RoadOption.STRAIGHT,
            "straight": RoadOption.STRAIGHT,
            "go_straight": RoadOption.STRAIGHT,
            "forward": RoadOption.STRAIGHT,
            "turn_left": RoadOption.LEFT,
            "left": RoadOption.LEFT,
            "turn_right": RoadOption.RIGHT,
            "right": RoadOption.RIGHT,
            "change_lane_left": RoadOption.CHANGELANELEFT,
            "changelaneleft": RoadOption.CHANGELANELEFT,
            "lane_change_left": RoadOption.CHANGELANELEFT,
            "change_lane_right": RoadOption.CHANGELANERIGHT,
            "changelaneright": RoadOption.CHANGELANERIGHT,
            "lane_change_right": RoadOption.CHANGELANERIGHT,
            "slow": RoadOption.LANEFOLLOW,
            "stop": RoadOption.LANEFOLLOW,
            "goal": RoadOption.LANEFOLLOW,
            "pose_lost": RoadOption.VOID,
            "no_mission": RoadOption.VOID,
        }
        if text in aliases:
            return aliases[text]
        try:
            return RoadOption(int(text))
        except Exception:
            return default

    try:
        return RoadOption(int(value))
    except Exception:
        return default


def nav_command_to_road_option(command: Any) -> RoadOption:
    """Map internal NavCommand/strings to CARLA-compatible RoadOption."""
    return normalize_road_option(command, default=RoadOption.LANEFOLLOW)


def road_option_to_one_hot(option: Any) -> Tuple[float, float, float, float, float, float]:
    """Return the same 6-class one-hot command vector used by CARLA datasets.

    Class order is: LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGELANELEFT,
    CHANGELANERIGHT. VOID and unknown values fall back to LANEFOLLOW, matching
    lead.data_loader.carla_dataset_utils.command_to_one_hot behavior.
    """
    road_option = normalize_road_option(option, default=RoadOption.LANEFOLLOW)
    command = int(road_option)
    if command < 0:
        command = int(RoadOption.LANEFOLLOW)
    index = command - 1
    if index not in (0, 1, 2, 3, 4, 5):
        index = int(RoadOption.LANEFOLLOW) - 1
    values = [0.0] * 6
    values[index] = 1.0
    return tuple(values)  # type: ignore[return-value]


def road_option_name(option: Any) -> str:
    return normalize_road_option(option, default=RoadOption.LANEFOLLOW).name


def goal_command_payload(goal: Any) -> dict:
    """Build a model-ready command payload from LocalNavigationGoal-like data.

    The returned dict is intentionally dependency-free. A PyTorch adapter can
    convert `command_one_hot` and `next_command_one_hot` to tensors with shape
    (1, 6).
    """
    road_option = normalize_road_option(getattr(goal, "road_option", RoadOption.LANEFOLLOW))
    next_option = normalize_road_option(getattr(goal, "next_road_option", road_option))
    return {
        "road_option": int(road_option),
        "road_option_name": road_option.name,
        "command_one_hot": road_option_to_one_hot(road_option),
        "next_road_option": int(next_option),
        "next_road_option_name": next_option.name,
        "next_command_one_hot": road_option_to_one_hot(next_option),
    }
