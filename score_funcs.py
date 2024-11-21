from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from misc_utils import check, is_list_of_type, lookat_matrix
from scannet_utils import ObjInstance
from scope_env import GlobalState

SCORE_FUNCTIONS: dict[str, type[ScoreFuncBase]] = {}


def register_score_func(score_func_class: type[ScoreFuncBase]):
    assert hasattr(score_func_class, "NEED_ANCHOR")
    assert isinstance(score_func_class.NEED_ANCHOR, bool)

    assert hasattr(score_func_class, "NAME")

    if isinstance(score_func_class.NAME, str):
        assert score_func_class.NAME not in SCORE_FUNCTIONS
        SCORE_FUNCTIONS[score_func_class.NAME] = score_func_class

    elif is_list_of_type(score_func_class.NAME, str):
        for name in score_func_class.NAME:
            assert name not in SCORE_FUNCTIONS
            SCORE_FUNCTIONS[name] = score_func_class

    else:
        raise SystemError(f"invalid score_func NAME: {score_func_class.NAME}")


class ScoreFuncBase(ABC):
    NAME: str | None = None
    NEED_ANCHOR: bool | None = None

    @staticmethod
    @abstractmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        """compute the score for each instance. anchor is optional."""


@register_score_func
class ScoreDistance(ScoreFuncBase):
    NAME = "distance"
    NEED_ANCHOR = True

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        return [
            np.linalg.norm(x.bbox.center - anchor.bbox.center)
            for x in candidate_instances
        ]


@register_score_func
class ScoreSizeX(ScoreFuncBase):
    NAME = "size-x"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        return [x.bbox.size[0] for x in candidate_instances]


@register_score_func
class ScoreSizeY(ScoreFuncBase):
    NAME = "size-y"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        return [x.bbox.size[1] for x in candidate_instances]


@register_score_func
class ScoreSizeZ(ScoreFuncBase):
    NAME = "size-z"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        return [x.bbox.size[2] for x in candidate_instances]


@register_score_func
class ScoreMaxSize(ScoreFuncBase):
    NAME = "size"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        return [x.bbox.max_extent for x in candidate_instances]


@register_score_func
class ScorePositionZ(ScoreFuncBase):
    NAME = "position-z"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        return [x.bbox.center[2] for x in candidate_instances]


@register_score_func
class ScoreLeft(ScoreFuncBase):
    NAME = "left"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        cand_center = np.mean([x.bbox.center for x in candidate_instances], axis=0)
        # look at the center of all candidate instances from the room center
        world_to_local = lookat_matrix(eye=cand_center, target=GlobalState.room_center)

        return [
            -(world_to_local @ np.hstack([x.bbox.center, 1]))[0]
            for x in candidate_instances
        ]


@register_score_func
class ScoreRight(ScoreFuncBase):
    NAME = "right"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        cand_center = np.mean([x.bbox.center for x in candidate_instances], axis=0)
        # look at the center of all candidate instances from the room center
        world_to_local = lookat_matrix(eye=cand_center, target=GlobalState.room_center)

        return [
            (world_to_local @ np.hstack([x.bbox.center, 1]))[0]
            for x in candidate_instances
        ]


@register_score_func
class ScoreFront(ScoreFuncBase):
    NAME = "front"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        cand_center = np.mean([x.bbox.center for x in candidate_instances], axis=0)
        # look at the center of all candidate instances from the room center
        world_to_local = lookat_matrix(eye=cand_center, target=GlobalState.room_center)

        # the larger the z-coord value, the nearer the instance is to the room center, i.e. "to the front"
        return [
            (world_to_local @ np.hstack([x.bbox.center, 1]))[2]
            for x in candidate_instances
        ]


@register_score_func
class ScoreCenter(ScoreFuncBase):
    NAME = ["distance-to-center", "distance-to-middle"]
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        center = np.mean([x.bbox.center for x in candidate_instances], axis=0)
        return [np.linalg.norm(x.bbox.center - center) for x in candidate_instances]
