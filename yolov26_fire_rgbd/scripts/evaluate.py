from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在冻结的 val/test 划分上评估 YOLO26 检测模型")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/fire/dataset.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--project", type=Path, default=Path("runs/eval"))
    parser.add_argument("--name", default="fire_yolo26_test")
    return parser.parse_args()


def json_compatible(value):
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data_path = args.data.resolve()
    if not model_path.is_file():
        raise SystemExit(f"模型不存在：{model_path}")
    if not data_path.is_file():
        raise SystemExit(f"数据配置不存在：{data_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("未安装 ultralytics，请先安装 requirements.txt。") from exc

    project = args.project.resolve()
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        project=str(project),
        name=args.name,
        plots=True,
        save_json=True,
    )
    save_dir = Path(metrics.save_dir)
    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "results": json_compatible(getattr(metrics, "results_dict", {})),
        "speed_ms_per_image": json_compatible(getattr(metrics, "speed", {})),
        "save_dir": str(save_dir),
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = save_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"评估摘要: {summary_path}")


if __name__ == "__main__":
    main()
