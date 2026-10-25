"""YOLO26 + Orbbec RGB-D fire detection utilities."""

from .depth import CameraIntrinsics, DepthEstimate, estimate_box_depth

__all__ = ["CameraIntrinsics", "DepthEstimate", "estimate_box_depth"]

