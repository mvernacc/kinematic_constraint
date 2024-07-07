"""Unit tests for the geometry module"""

import numpy as np
from pytest import approx

from kinematic_constraint.geometry import Line3


class TestShortestDistanceToLine:
    def test_parallel(self):
        a = Line3((0, 0, 0), (0, 1, 0))
        b = Line3((1, 0, 0), (0, 1, 0))

        assert a.shortest_distance_to_line(b) == approx(1.0)

    def test_skew(self):
        a = Line3((1, 1, -1), (-1, 0, 1))
        b = Line3((-1, -1, -1), (1, 0, 1))

        # Closest approach should be 2 apart, between points (0, 1, 0) and (0, -1, 0)
        assert a.shortest_distance_to_line(b) == approx(2.0)

    def test_intersecting(self):
        a = Line3((1, 0, 0), (-1, 1, 0))
        b = Line3((-1, 0, 0), (1, 1, 0))

        # The lines intersect at (0, 1, 0)
        assert a.shortest_distance_to_line(b) == approx(0.0)


class TestClosestPoints:
    def test_parallel(self):
        a = Line3((0, 0, 0), (0, 1, 0))
        b = Line3((1, 0, 0), (0, 1, 0))

        closest_a, closest_b = a.closest_points(b)
        assert np.linalg.norm(closest_a - closest_b) == approx(1.0)

    def test_skew(self):
        a = Line3((1, 1, -1), (-1, 0, 1))
        b = Line3((-1, -1, -1), (1, 0, 1))

        # Closest approach should be 2 apart, between points (0, 1, 0) and (0, -1, 0)
        closest_a, closest_b = a.closest_points(b)
        assert closest_a == approx(np.array([0, 1, 0]))
        assert closest_b == approx(np.array([0, -1, 0]))

    def test_intersecting(self):
        a = Line3((1, 0, 0), (-1, 1, 0))
        b = Line3((-1, 0, 0), (1, 1, 0))

        # The lines intersect at (0, 1, 0)
        closest_a, closest_b = a.closest_points(b)
        assert closest_a == approx(np.array([0, 1, 0]))
        assert closest_b == approx(np.array([0, 1, 0]))
