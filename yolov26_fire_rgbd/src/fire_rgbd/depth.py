from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for the color image onto which depth is aligned."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class DepthEstimate:
    """Robust depth estimate and the corresponding camera-frame location."""

    depth_mm: float
    xyz_m: tuple[float, float, float]
    sample_pixel: tuple[int, int]
    valid_ratio: float
    valid_count: int
    roi_xyxy: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "depth_mm": round(self.depth_mm, 1),
            "xyz_m": [round(value, 4) for value in self.xyz_m],
            "sample_pixel": list(self.sample_pixel),
            "valid_ratio": round(self.valid_ratio, 4),
            "valid_count": self.valid_count,
            "roi_xyxy": list(self.roi_xyxy),
        }


def deproject_pixel(
    pixel: tuple[float, float], depth_mm: float, intrinsics: CameraIntrinsics
) -> tuple[float, float, float]:
    """Convert an aligned color pixel and Z depth to camera-frame XYZ in metres."""

    u, v = pixel
    z_m = float(depth_mm) / 1000.0
    x_m = (float(u) - intrinsics.cx) * z_m / intrinsics.fx
    y_m = (float(v) - intrinsics.cy) * z_m / intrinsics.fy
    return x_m, y_m, z_m


def _sampling_roi(
    bbox_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    mode: Literal["base", "center", "box"],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    box_w = max(x2 - x1, 2.0)
    box_h = max(y2 - y1, 2.0)

    if mode == "base":
        # The flame itself often has no trustworthy active-stereo return. Sampling
        # near the base is more likely to observe the burning object/support.
        rx1, rx2 = x1 + 0.30 * box_w, x1 + 0.70 * box_w
        ry1, ry2 = y1 + 0.70 * box_h, y1 + 0.95 * box_h
    elif mode == "center":
        rx1, rx2 = x1 + 0.30 * box_w, x1 + 0.70 * box_w
        ry1, ry2 = y1 + 0.30 * box_h, y1 + 0.70 * box_h
    elif mode == "box":
        rx1, rx2, ry1, ry2 = x1, x2, y1, y2
    else:
        raise ValueError(f"Unsupported depth ROI mode: {mode}")

    ix1 = int(np.clip(np.floor(rx1), 0, max(image_width - 1, 0)))
    iy1 = int(np.clip(np.floor(ry1), 0, max(image_height - 1, 0)))
    ix2 = int(np.clip(np.ceil(rx2), ix1 + 1, image_width))
    iy2 = int(np.clip(np.ceil(ry2), iy1 + 1, image_height))
    return ix1, iy1, ix2, iy2


def estimate_box_depth(
    depth_mm: np.ndarray,
    bbox_xyxy: Sequence[float],
    intrinsics: CameraIntrinsics,
    *,
    roi_mode: Literal["base", "center", "box"] = "base",
    min_depth_mm: float = 150.0,
    max_depth_mm: float = 10000.0,
    min_valid_pixels: int = 12,
) -> DepthEstimate | None:
    """Estimate an object's range using median/MAD filtering in a box sub-region.

    ``depth_mm`` must already be aligned to the color frame. Zero, NaN and values
    outside the configured range are rejected. The median is robust to isolated
    depth speckles; an additional MAD gate removes strong foreground/background
    outliers when enough samples are available.
    """

    if depth_mm.ndim != 2:
        raise ValueError("depth_mm must be a HxW array")
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        raise ValueError("Camera focal lengths must be positive")

    height, width = depth_mm.shape
    roi = _sampling_roi(bbox_xyxy, width, height, roi_mode)
    x1, y1, x2, y2 = roi
    values = depth_mm[y1:y2, x1:x2].astype(np.float32, copy=False).reshape(-1)
    valid_mask = (
        np.isfinite(values)
        & (values >= float(min_depth_mm))
        & (values <= float(max_depth_mm))
    )
    valid = values[valid_mask]
    valid_ratio = float(valid.size / max(values.size, 1))
    if valid.size < min_valid_pixels:
        return None

    median = float(np.median(valid))
    absolute_deviation = np.abs(valid - median)
    mad = float(np.median(absolute_deviation))
    if mad > 0.0:
        # 1.4826 converts MAD to a robust standard-deviation estimate.
        inliers = valid[absolute_deviation <= 3.0 * 1.4826 * mad]
        if inliers.size >= min_valid_pixels:
            valid = inliers
            median = float(np.median(valid))

    sample_u = int(round((x1 + x2 - 1) / 2.0))
    sample_v = int(round((y1 + y2 - 1) / 2.0))
    xyz_m = deproject_pixel((sample_u, sample_v), median, intrinsics)
    return DepthEstimate(
        depth_mm=median,
        xyz_m=xyz_m,
        sample_pixel=(sample_u, sample_v),
        valid_ratio=valid_ratio,
        valid_count=int(valid.size),
        roi_xyxy=roi,
    )

