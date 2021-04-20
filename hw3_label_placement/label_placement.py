from __future__ import annotations

import itertools
import typing as tp
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from satisfy import TwoSATSolver


@dataclass
class Point:
    x: float
    y: float


class LabelBoundingBox:
    def __init__(self, xy: tuple[float, float], width: float, height: float) -> None:
        self.bottom_left = Point(*xy)
        self.top_right = Point(xy[0] + width, xy[1] + height)

    @property
    def xy(self) -> tuple[float, float]:
        return self.bottom_left.x, self.bottom_left.y

    @property
    def width(self) -> float:
        return self.top_right.x - self.bottom_left.x

    @property
    def height(self) -> float:
        return self.top_right.y - self.bottom_left.y

    def intersects(self, other: LabelBoundingBox) -> bool:
        """
        Checks whether self and other bounding boxes do intersect.
        """
        return not (
            self.top_right.x < other.bottom_left.x
            or self.bottom_left.x > other.top_right.x
            or self.top_right.y < other.bottom_left.y
            or self.bottom_left.y > other.top_right.y
        )


class LabelPlacementSolver:
    """
    Solves simplified case of label placement problem.
    Accepts positions of dots to label and optional label sizes (if not specified,
    all labels are assumed to be of width 10 and height 5). Assumes that every
    label has only two available positions - on the top of its dit left or right from it.
    I.e.:

            +-------------+
            |             |
            +-------------+.
    or:
             +-------------+
             |             |
            .+-------------+

    Method `solve` runs the solver and returns a list of LabelBoundingBox objects,
    one for each dot. They give proposed positions for the labels. In case, specified
    task has no solution (i.e. in the following case:

            +-------------+                         +-------------+
            |           +-+-----------+ +-----------+-+           |
            +-----------+-+.          | |          .+-+-----------+
                        +-------------+.+-------------+

    )
    method returns None.
    """

    def __init__(
        self,
        points: list[tuple[float, float]],
        bbox_sizes: tp.Optional[list[tuple[float, float]]] = None,
    ) -> None:
        if bbox_sizes is None:
            bbox_sizes = [(10.0, 5.0)] * len(points)

        self.points = points
        self.bbox_sizes = bbox_sizes

    def get_point_bboxes(self, point_index: int) -> tuple[LabelBoundingBox, LabelBoundingBox]:
        """
        Returns tuple with LabelBoundingBox objects for required point.
        """
        point = self.points[point_index]
        width, height = self.bbox_sizes[point_index]
        first_bbox = LabelBoundingBox(point, width, height)
        point = (point[0] - width, point[1])
        second_bbox = LabelBoundingBox(point, width, height)
        return first_bbox, second_bbox

    def find_intersetions(
        self, first_point_index: int, second_point_index: int
    ) -> tp.Iterable[tuple[int, int]]:
        first_point_bboxes = self.get_point_bboxes(first_point_index)
        second_point_bboxes = self.get_point_bboxes(second_point_index)
        for first_bbox_index, first_bbox in enumerate(first_point_bboxes):
            for second_bbox_index, second_bbox in enumerate(second_point_bboxes):
                if first_bbox.intersects(second_bbox):
                    yield first_bbox_index, second_bbox_index

    def solve(self) -> tp.Optional[list[LabelBoundingBox]]:
        satisfier = TwoSATSolver(len(self.points))

        # Iterate over all pairs of points and find all intersecting bounding boxes.
        point_index_pairs = itertools.combinations(range(len(self.points)), 2)
        for first_point_index, second_point_index in point_index_pairs:
            intersections = self.find_intersetions(first_point_index, second_point_index)
            for first_bbox_index, second_bbox_index in intersections:
                # Bounding box indices can only be 0 or 1, hence easily translatable to booleans.
                # As these tow bboxes intersect, we have to deal with implication of the form
                # (+-first_point_index -> -+second_point_index) where signs of both terms is
                # defined by its bbox index via ((-1) ** bbox_index), which is 1 iif bbox_index
                # equals 0 and (-1) iif bbox_index is 1. Note that we need to negate second point
                # term, as we need to avoid collision, not cause one.
                first_variable = satisfier.index_and_value_to_variable(
                    first_point_index, bool(first_bbox_index)
                )
                second_variable = satisfier.index_and_value_to_variable(
                    second_point_index, not bool(second_bbox_index)
                )

                # Now we have implication (first_variable -> second_variable).
                # Implication can be rewritten as disjunction (-first_variable || second_variable)
                # and passed to the 2-SAT solver.
                satisfier.add_disjunction(-first_variable, second_variable)

        # Now we try to satisfy the resulting formula. If it is not possible, we return None.
        satisfier_result = satisfier.solve()
        if satisfier_result is None:
            return None

        # If formula is satisfiable, we obtain list with boolean values for each point.
        # Booleans are easily translatable to indices in the bbox tuples via int(value)
        return [
            self.get_point_bboxes(point_index)[int(value)]
            for point_index, value in enumerate(satisfier_result)
        ]


def draw_labeled_points(
    points: list[tuple[float, float]],
    bbox_sizes: tp.Optional[list[tuple[float, float]]] = None,
    ax: tp.Optional[mpl.axes.Axes] = None,
) -> None:
    label_placement = LabelPlacementSolver(points, bbox_sizes).solve()
    if label_placement is None:
        raise ValueError("Failed to draw points. Label placement task is unsolvable.")

    if ax is None:
        ax = plt.gca()

    x, y = list(zip(*points))
    ax.scatter(x, y)

    for bbox in label_placement:
        ax.add_patch(patches.Rectangle(bbox.xy, bbox.width, bbox.height))
