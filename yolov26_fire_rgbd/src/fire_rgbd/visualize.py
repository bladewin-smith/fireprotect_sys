from __future__ import annotations

import cv2
import numpy as np


def colorize_depth(depth_mm: np.ndarray, max_depth_mm: float = 8000.0) -> np.ndarray:
    clipped = np.clip(depth_mm, 0.0, max_depth_mm)
    normalized = (clipped / max_depth_mm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[depth_mm <= 0.0] = 0
    return colored


def put_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - height - baseline - 6)
    cv2.rectangle(image, (x, top), (x + width + 6, y), color, -1)
    cv2.putText(image, text, (x + 3, y - baseline - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

