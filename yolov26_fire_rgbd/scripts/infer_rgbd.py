from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..src.fire_rgbd.depth import estimate_box_depth
from ..src.fire_rgbd.orbbec import OrbbecRGBDCamera
from ..src.fire_rgbd.visualize import colorize_depth, put_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 + Gemini 336L 实时火焰检测和深度定位")
    parser.add_argument("--model", type=Path, required=True, help="best.pt / ONNX / TensorRT engine")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth-roi", choices=("base", "center", "box"), default="base")
    parser.add_argument("--min-depth", type=float, default=150.0)
    parser.add_argument("--max-depth", type=float, default=10000.0)
    parser.add_argument("--min-depth-pixels", type=int, default=12)
    parser.add_argument("--min-consecutive", type=int, default=3)
    parser.add_argument("--software-align", action="store_true")
    parser.add_argument("--jsonl", type=Path, default=Path("runs/infer/events.jsonl"))
    parser.add_argument("--record", type=Path, default=None, help="可选的标注视频输出 .mp4")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0=持续运行")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("未安装 ultralytics。") from exc
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"模型不存在：{model_path}")
    if not 0.0 < args.conf <= 1.0:
        raise SystemExit("--conf 必须在 (0,1]")

    model = YOLO(str(model_path))
    log_path = args.jsonl.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record_path = args.record.resolve() if args.record else None
    if record_path:
        record_path.parent.mkdir(parents=True, exist_ok=True)

    consecutive = 0
    frame_number = 0
    smoothed_fps = 0.0
    previous_time = time.perf_counter()
    writer: cv2.VideoWriter | None = None

    with log_path.open("a", encoding="utf-8") as event_log:
        with OrbbecRGBDCamera(prefer_hardware_align=not args.software_align) as camera:
            print(f"模型: {model_path}")
            print(f"D2C 对齐: {camera.alignment}")
            intr = camera.intrinsics
            print(f"RGB 内参: {intr.width}x{intr.height}, fx={intr.fx:.2f}, fy={intr.fy:.2f}")
            print("相机坐标: +X 向右, +Y 向下, +Z 向前；火焰深度仅为底部邻域代理值。")
            while True:
                frame = camera.read()
                if frame is None:
                    continue
                frame_number += 1
                results = model.predict(
                    source=frame.color_bgr,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    half=args.half,
                    verbose=False,
                )
                result = results[0]
                annotated = frame.color_bgr.copy()
                detections: list[dict[str, object]] = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    scores = result.boxes.conf.detach().cpu().numpy()
                    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                    for bbox, score, class_id in zip(boxes, scores, classes, strict=True):
                        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
                        class_name = str(result.names.get(class_id, class_id))
                        depth = estimate_box_depth(
                            frame.depth_mm,
                            bbox,
                            frame.intrinsics,
                            roi_mode=args.depth_roi,
                            min_depth_mm=args.min_depth,
                            max_depth_mm=args.max_depth,
                            min_valid_pixels=args.min_depth_pixels,
                        )
                        detection: dict[str, object] = {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(float(score), 4),
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "depth_is_proxy": True,
                        }
                        label = f"{class_name} {score:.2f}"
                        if depth is not None:
                            detection["location"] = depth.to_dict()
                            label += f" Z={depth.xyz_m[2]:.2f}m"
                            rx1, ry1, rx2, ry2 = depth.roi_xyxy
                            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 180, 0), 1)
                            cv2.circle(annotated, depth.sample_pixel, 3, (255, 180, 0), -1)
                        else:
                            detection["location"] = None
                            label += " Z=invalid"
                        detections.append(detection)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        put_label(annotated, label, (max(x1, 0), max(y1, 22)))

                consecutive = consecutive + 1 if detections else 0
                alarm_active = consecutive >= args.min_consecutive
                current_time = time.perf_counter()
                instant_fps = 1.0 / max(current_time - previous_time, 1e-6)
                previous_time = current_time
                smoothed_fps = (
                    instant_fps
                    if smoothed_fps == 0.0
                    else 0.9 * smoothed_fps + 0.1 * instant_fps
                )
                alarm_text = "ON" if alarm_active else "OFF"
                status = f"FPS={smoothed_fps:.1f} DET={len(detections)} ALARM={alarm_text}"
                cv2.putText(
                    annotated,
                    status,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255) if alarm_active else (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                event = {
                    "captured_at_utc": frame.captured_at,
                    "frame_number": frame_number,
                    "alignment": frame.alignment,
                    "alarm_active": alarm_active,
                    "consecutive_positive_frames": consecutive,
                    "detections": detections,
                }
                event_log.write(json.dumps(event, ensure_ascii=False) + "\n")
                event_log.flush()

                if record_path is not None:
                    if writer is None:
                        height, width = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            str(record_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height)
                        )
                        if not writer.isOpened():
                            raise OSError(f"无法创建视频：{record_path}")
                    writer.write(annotated)

                if not args.no_show:
                    depth_view = colorize_depth(frame.depth_mm, args.max_depth)
                    view = np.hstack((annotated, depth_view))
                    cv2.imshow("YOLO26 fire RGB-D localization", view)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                if args.max_frames > 0 and frame_number >= args.max_frames:
                    break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"事件日志: {log_path}")


if __name__ == "__main__":
    main()

