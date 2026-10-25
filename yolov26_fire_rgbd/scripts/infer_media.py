from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对图片/视频/普通摄像头做二维火焰检测")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", required=True, help="图片、视频、目录，或 webcam 编号 0")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--project", type=Path, default=Path("runs/predict"))
    parser.add_argument("--name", default="fire")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("未安装 ultralytics。") from exc
    source: str | int = int(args.source) if args.source.isdigit() else args.source
    YOLO(str(args.model.resolve())).predict(
        source=source,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=True,
        show=args.show,
        project=str(args.project.resolve()),
        name=args.name,
    )


if __name__ == "__main__":
    main()

