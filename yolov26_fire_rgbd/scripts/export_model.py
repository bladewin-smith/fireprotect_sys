from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 YOLO26 火焰模型")
    parser.add_argument("--model", type=Path, required=True, help="best.pt 路径")
    parser.add_argument("--format", choices=("onnx", "engine"), default="engine")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--quantize", choices=("16", "32", "8"), default="16")
    parser.add_argument("--data", type=Path, default=Path("data/fire/dataset.yaml"))
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT 工作区 GiB")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument(
        "--nms",
        choices=("default", "embedded", "end-to-end"),
        default="default",
        help="default=原始头；embedded=导出内置 NMS；end-to-end=YOLO26 one-to-one 头",
    )
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

    nms = {"default": None, "embedded": True, "end-to-end": False}[args.nms]
    export_args: dict[str, object] = {
        "format": args.format,
        "imgsz": args.imgsz,
        "device": args.device,
        "quantize": int(args.quantize),
        "dynamic": args.dynamic,
        "nms": nms,
        "batch": 1,
        "simplify": True,
    }
    if args.format == "engine":
        export_args["workspace"] = args.workspace
    if args.quantize == "8":
        data_path = args.data.resolve()
        if not data_path.is_file():
            raise SystemExit("INT8 导出需要有效的 --data 校准数据集")
        export_args["data"] = str(data_path)

    output = YOLO(str(model_path)).export(**export_args)
    print(f"导出完成: {output}")


if __name__ == "__main__":
    main()

