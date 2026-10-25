from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from .depth import CameraIntrinsics


@dataclass(frozen=True)
class RGBDFrame:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    intrinsics: CameraIntrinsics
    captured_at: str
    alignment: str


def _load_sdk() -> dict[str, Any]:
    try:
        from pyorbbecsdk import (  # type: ignore[import-not-found]
            AlignFilter,
            Config,
            Context,
            OBAlignMode,
            OBFormat,
            OBFrameAggregateOutputMode,
            OBSensorType,
            OBStreamType,
            Pipeline,
        )
    except ImportError as exc:
        raise RuntimeError(
            "未找到 Orbbec SDK。请在当前虚拟环境执行：pip install pyorbbecsdk2"
        ) from exc
    return locals()


def color_frame_to_bgr(frame: Any, ob_format: Any) -> np.ndarray | None:
    """Convert common Orbbec color formats to an OpenCV BGR image."""

    width, height = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    raw = np.frombuffer(frame.get_data(), dtype=np.uint8)
    try:
        if fmt == ob_format.RGB:
            return cv2.cvtColor(raw.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        if fmt == ob_format.BGR:
            return raw.reshape(height, width, 3).copy()
        if fmt == ob_format.MJPG:
            return cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if hasattr(ob_format, "YUYV") and fmt == ob_format.YUYV:
            return cv2.cvtColor(raw.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
        if hasattr(ob_format, "UYVY") and fmt == ob_format.UYVY:
            return cv2.cvtColor(raw.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY)
        if hasattr(ob_format, "NV12") and fmt == ob_format.NV12:
            return cv2.cvtColor(raw.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)
        if hasattr(ob_format, "NV21") and fmt == ob_format.NV21:
            return cv2.cvtColor(raw.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV21)
    except (ValueError, cv2.error):
        return None
    return None


class OrbbecRGBDCamera:
    """Gemini RGB-D source with depth aligned onto the color image."""

    def __init__(self, *, prefer_hardware_align: bool = True, timeout_ms: int = 1000):
        self.prefer_hardware_align = prefer_hardware_align
        self.timeout_ms = timeout_ms
        self._sdk: dict[str, Any] | None = None
        self._pipeline: Any = None
        self._align_filter: Any = None
        self._intrinsics: CameraIntrinsics | None = None
        self._alignment = "not-started"
        self._started = False

    @property
    def alignment(self) -> str:
        return self._alignment

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise RuntimeError("Camera has not started")
        return self._intrinsics

    def _hardware_config(self, pipeline: Any) -> Any | None:
        assert self._sdk is not None
        config = self._sdk["Config"]()
        profile_list = pipeline.get_stream_profile_list(self._sdk["OBSensorType"].COLOR_SENSOR)
        for index in range(len(profile_list)):
            color_profile = profile_list[index]
            # RGB is easiest to consume and mirrors the official alignment example.
            if color_profile.get_format() != self._sdk["OBFormat"].RGB:
                continue
            depth_profiles = pipeline.get_d2c_depth_profile_list(
                color_profile, self._sdk["OBAlignMode"].HW_MODE
            )
            if len(depth_profiles) == 0:
                continue
            config.enable_stream(depth_profiles[0])
            config.enable_stream(color_profile)
            config.set_align_mode(self._sdk["OBAlignMode"].HW_MODE)
            config.set_frame_aggregate_output_mode(
                self._sdk["OBFrameAggregateOutputMode"].FULL_FRAME_REQUIRE
            )
            return config
        return None

    def _software_config(self, pipeline: Any) -> Any:
        assert self._sdk is not None
        config = self._sdk["Config"]()
        color_profiles = pipeline.get_stream_profile_list(
            self._sdk["OBSensorType"].COLOR_SENSOR
        )
        try:
            color_profile = color_profiles.get_video_stream_profile(
                0, 0, self._sdk["OBFormat"].RGB, 0
            )
        except Exception:
            color_profile = color_profiles.get_default_video_stream_profile()
        depth_profiles = pipeline.get_stream_profile_list(
            self._sdk["OBSensorType"].DEPTH_SENSOR
        )
        config.enable_stream(color_profile)
        config.enable_stream(depth_profiles.get_default_video_stream_profile())
        config.set_frame_aggregate_output_mode(
            self._sdk["OBFrameAggregateOutputMode"].FULL_FRAME_REQUIRE
        )
        return config

    def start(self) -> "OrbbecRGBDCamera":
        if self._started:
            return self
        self._sdk = _load_sdk()
        context = self._sdk["Context"]()
        if context.query_devices().get_count() == 0:
            raise RuntimeError("未检测到 Orbbec 相机，请检查 USB 3.x 连接、驱动和供电。")

        self._pipeline = self._sdk["Pipeline"]()
        config = None
        if self.prefer_hardware_align:
            try:
                self._pipeline.enable_frame_sync()
                config = self._hardware_config(self._pipeline)
            except Exception:
                config = None
        if config is not None:
            self._alignment = "hardware-d2c"
        else:
            config = self._software_config(self._pipeline)
            self._align_filter = self._sdk["AlignFilter"](
                align_to_stream=self._sdk["OBStreamType"].COLOR_STREAM
            )
            self._alignment = "software-d2c"

        self._pipeline.start(config)
        self._started = True
        try:
            # A delivered frame lets the SDK resolve the active profile pair.
            warmup_frame = None
            for _ in range(30):
                warmup_frame = self._pipeline.wait_for_frames(self.timeout_ms)
                if warmup_frame is not None:
                    break
            if warmup_frame is None:
                raise RuntimeError("相机已启动但未收到 RGB-D 帧。")

            camera_param = self._pipeline.get_camera_param()
            intr = camera_param.rgb_intrinsic
            self._intrinsics = CameraIntrinsics(
                width=int(intr.width),
                height=int(intr.height),
                fx=float(intr.fx),
                fy=float(intr.fy),
                cx=float(intr.cx),
                cy=float(intr.cy),
            )
        except BaseException:
            self.stop()
            raise
        return self

    def read(self) -> RGBDFrame | None:
        if not self._started or self._pipeline is None or self._sdk is None:
            raise RuntimeError("Camera has not started")
        frames = self._pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return None
        if self._align_filter is not None:
            frames = self._align_filter.process(frames)
            if frames is None:
                return None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None
        color_bgr = color_frame_to_bgr(color_frame, self._sdk["OBFormat"])
        if color_bgr is None:
            raise RuntimeError(f"不支持的彩色帧格式：{color_frame.get_format()}")

        try:
            depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                depth_frame.get_height(), depth_frame.get_width()
            )
        except ValueError:
            return None
        depth_mm = depth_raw.astype(np.float32) * float(depth_frame.get_depth_scale())
        depth_mm[(depth_mm < 1.0) | ~np.isfinite(depth_mm)] = 0.0

        if depth_mm.shape != color_bgr.shape[:2]:
            # D2C should already match. A nearest-neighbour resize keeps distance
            # samples discrete and makes a profile mismatch explicit but usable.
            depth_mm = cv2.resize(
                depth_mm,
                (color_bgr.shape[1], color_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        intrinsics = self.intrinsics
        if (intrinsics.width, intrinsics.height) != (
            color_bgr.shape[1],
            color_bgr.shape[0],
        ):
            sx = color_bgr.shape[1] / intrinsics.width
            sy = color_bgr.shape[0] / intrinsics.height
            intrinsics = CameraIntrinsics(
                width=color_bgr.shape[1],
                height=color_bgr.shape[0],
                fx=intrinsics.fx * sx,
                fy=intrinsics.fy * sy,
                cx=intrinsics.cx * sx,
                cy=intrinsics.cy * sy,
            )

        return RGBDFrame(
            color_bgr=color_bgr,
            depth_mm=depth_mm,
            intrinsics=intrinsics,
            captured_at=datetime.now(timezone.utc).isoformat(),
            alignment=self._alignment,
        )

    def stop(self) -> None:
        if self._pipeline is not None and self._started:
            try:
                self._pipeline.stop()
            finally:
                self._started = False

    def __enter__(self) -> "OrbbecRGBDCamera":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

