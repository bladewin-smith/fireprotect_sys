import numpy as np

from fire_rgbd.depth import CameraIntrinsics, deproject_pixel, estimate_box_depth


INTRINSICS = CameraIntrinsics(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0)


def test_deproject_center_pixel() -> None:
    assert deproject_pixel((50, 40), 2000.0, INTRINSICS) == (0.0, 0.0, 2.0)


def test_estimate_box_depth_rejects_outlier() -> None:
    depth = np.zeros((80, 100), dtype=np.float32)
    depth[40:70, 30:70] = 2000.0
    depth[56, 48] = 9000.0
    estimate = estimate_box_depth(
        depth,
        (25, 20, 75, 75),
        INTRINSICS,
        roi_mode="base",
        min_valid_pixels=5,
    )
    assert estimate is not None
    assert estimate.depth_mm == 2000.0
    assert estimate.xyz_m[2] == 2.0


def test_estimate_box_depth_returns_none_without_enough_valid_pixels() -> None:
    depth = np.zeros((80, 100), dtype=np.float32)
    estimate = estimate_box_depth(
        depth,
        (10, 10, 30, 30),
        INTRINSICS,
        min_valid_pixels=3,
    )
    assert estimate is None

