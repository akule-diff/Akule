import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common_strict_safety import evaluate, smd_native_success


def test_swept_pair_catches_a_swap_missed_by_samples():
    path = np.array([[[-.5, 0], [.5, 0]], [[.5, 0], [-.5, 0]]])
    result = evaluate(path, path[0], path[-1])
    assert result["sampled_pair_clearance_m"] > 0
    assert result["swept_pair_clearance_m"] < 0
    assert not result["success"]


def test_exact_circle_and_box_sweeps():
    path = np.array([[[-.4, 0]], [[.4, 0]]])
    circle = {"kind": "circles", "items": [{"center": [0, 0], "radius": .1}]}
    box = {"kind": "boxes", "centers": [[0, 0]], "sizes": [[.2, .2]]}
    for obstacles in (circle, box):
        result = evaluate(path, path[0], path[-1], obstacles)
        assert result["sampled_obstacle_clearance_m"] > 0
        assert result["swept_obstacle_clearance_m"] < 0
        assert not result["success"]


def test_endpoint_and_workspace_are_part_of_contract():
    path = np.array([[[0., 0.]], [[1.01, 0.]]])
    result = evaluate(path, path[0], path[-1])
    assert result["workspace_margin_m"] < 0
    assert not result["success"]
    path[-1, 0, 0] = .9
    result = evaluate(path, path[0], np.array([[.8, 0.]]))
    assert result["goal_error_m"] > 0
    assert not result["success"]


def test_native_smd_tolerance_is_distinct_from_strict_point_one_meter():
    path = np.array([[[0., 0.], [.097, 0.]], [[0., 0.], [.097, 0.]]])
    circles = {"kind": "circles", "items": []}
    assert smd_native_success(path, circles)
    assert not evaluate(path, path[0], path[-1], circles)["success"]
