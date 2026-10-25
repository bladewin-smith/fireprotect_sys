from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = PROJECT_ROOT.parent
DEFAULT_YOLOV5_DIR = SYSTEM_ROOT / "smoke" / "yolov5-6.0-fire_smoke"
DEFAULT_WEIGHTS = DEFAULT_YOLOV5_DIR / "runs" / "train" / "fire_smoke-v5m" / "weights" / "best.pt"


@dataclass(frozen=True)
class Prediction:
    source_class_id: int
    target_class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float

    def yolo_line(self) -> str:
        return (
            f"{self.target_class_id} {self.x_center:.8f} {self.y_center:.8f} "
            f"{self.width:.8f} {self.height:.8f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "调用本地 YOLOv5 教师模型，为按场景采集的 RGB-D 数据生成安全隔离的自动标签"
        )
    )
    parser.add_argument("--source", type=Path, default=Path("data/raw"))
    parser.add_argument("--yolov5-dir", type=Path, default=DEFAULT_YOLOV5_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--python",
        dest="python_executable",
        type=Path,
        default=Path(sys.executable),
        help="能够运行旧 YOLOv5 的 Python；默认使用当前解释器",
    )
    parser.add_argument("--scene", action="append", default=[], help="只处理指定场景，可重复")
    parser.add_argument("--imgsz", type=int, default=416, help="旧模型训练尺寸为 416")
    parser.add_argument("--conf-thres", type=float, default=0.35)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default="0", help="YOLOv5 设备，例如 0 或 cpu")
    parser.add_argument(
        "--class-map",
        nargs="+",
        default=["0:0"],
        metavar="SRC:DST",
        help="旧模型到新数据集的类别映射；默认只保留 0:0（fire）",
    )
    parser.add_argument("--half", action="store_true", help="CUDA 推理使用 FP16")
    parser.add_argument("--no-preview", action="store_true", help="不保存带框预览图")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "pseudo_label_yolov5",
        help="运行清单保存目录",
    )
    parser.add_argument(
        "--overwrite-auto",
        action="store_true",
        help="允许覆盖 labels_auto/predictions_auto/review_auto；不会覆盖正式 labels",
    )
    parser.add_argument(
        "--include-labeled",
        action="store_true",
        help="也为已有正式 labels/ 的图片生成自动结果；默认跳过人工已标注图片",
    )
    parser.add_argument("--dry-run", action="store_true", help="只校验路径并打印将执行的命令")
    return parser.parse_args()


def parse_class_map(values: list[str]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for value in values:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError(f"类别映射格式错误：{value!r}，应为 SRC:DST")
        try:
            source_id, target_id = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"类别映射必须使用整数：{value!r}") from exc
        if source_id < 0 or target_id < 0:
            raise ValueError(f"类别 ID 不能为负数：{value!r}")
        if source_id in mapping:
            raise ValueError(f"源类别 {source_id} 被重复映射")
        mapping[source_id] = target_id
    if not mapping:
        raise ValueError("至少需要一个类别映射")
    return mapping


def discover_scenes(source: Path, selected: set[str] | None = None) -> list[Path]:
    if (source / "images").is_dir():
        candidates = [source]
    elif source.is_dir():
        candidates = [
            path
            for path in source.iterdir()
            if path.is_dir() and (path / "images").is_dir()
        ]
    else:
        candidates = []
    scenes = sorted(candidates, key=lambda path: path.name)
    if selected:
        scenes = [path for path in scenes if path.name in selected]
        missing = selected - {path.name for path in scenes}
        if missing:
            raise FileNotFoundError(f"找不到指定场景：{', '.join(sorted(missing))}")
    return scenes


def image_files(images_dir: Path) -> list[Path]:
    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    stems = [path.stem for path in images]
    duplicates = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{images_dir} 存在不同扩展名但同 stem 的图片，会造成标签冲突：{duplicates[:5]}"
        )
    return images


def parse_prediction_file(path: Path, class_map: dict[int, int]) -> list[Prediction]:
    if not path.is_file():
        return []
    predictions: list[Prediction] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 6:
            raise ValueError(f"{path}:{line_number} 应有 6 列（含置信度），实际 {len(parts)} 列")
        try:
            source_class_id = int(float(parts[0]))
            x_center, y_center, width, height, confidence = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} 含非数值字段") from exc
        if source_class_id not in class_map:
            continue
        values = (x_center, y_center, width, height)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{path}:{line_number} 坐标未归一化到 [0,1]")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"{path}:{line_number} 框宽高必须大于 0")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{path}:{line_number} 置信度必须在 [0,1]")
        predictions.append(
            Prediction(
                source_class_id=source_class_id,
                target_class_id=class_map[source_class_id],
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                confidence=confidence,
            )
        )
    return predictions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transfer(source: Path, destination: Path, overwrite: bool) -> bool:
    if destination.exists() and not overwrite:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def write_text(path: Path, text: str, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def build_detect_command(
    python_executable: Path,
    yolov5_dir: Path,
    weights: Path,
    images_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    class_map: dict[int, int],
) -> list[str]:
    command = [
        str(python_executable),
        str(yolov5_dir / "detect.py"),
        "--weights",
        str(weights),
        "--source",
        str(images_dir),
        "--imgsz",
        str(args.imgsz),
        "--conf-thres",
        str(args.conf_thres),
        "--iou-thres",
        str(args.iou_thres),
        "--max-det",
        str(args.max_det),
        "--device",
        str(args.device),
        "--save-txt",
        "--save-conf",
        "--project",
        str(output_dir),
        "--name",
        "pred",
        "--exist-ok",
        "--classes",
        *(str(class_id) for class_id in sorted(class_map)),
    ]
    if args.half:
        command.append("--half")
    if args.no_preview:
        command.append("--nosave")
    return command


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, dict[int, int]]:
    source = args.source.resolve()
    yolov5_dir = args.yolov5_dir.resolve()
    weights = args.weights.resolve()
    python_executable = args.python_executable.resolve()
    class_map = parse_class_map(args.class_map)
    if not source.is_dir():
        raise FileNotFoundError(f"RGB-D 采集目录不存在：{source}")
    if not (yolov5_dir / "detect.py").is_file():
        raise FileNotFoundError(f"不是有效 YOLOv5 仓库，缺少 detect.py：{yolov5_dir}")
    if not weights.is_file():
        raise FileNotFoundError(f"教师模型权重不存在：{weights}")
    if not python_executable.is_file():
        raise FileNotFoundError(f"Python 解释器不存在：{python_executable}")
    if args.imgsz <= 0 or args.max_det <= 0:
        raise ValueError("imgsz 和 max-det 必须大于 0")
    if not 0.0 <= args.conf_thres <= 1.0 or not 0.0 <= args.iou_thres <= 1.0:
        raise ValueError("conf-thres 和 iou-thres 必须在 [0,1]")
    return source, yolov5_dir, weights, python_executable, class_map


def main() -> None:
    args = parse_args()
    try:
        source, yolov5_dir, weights, python_executable, class_map = validate_args(args)
        scenes = discover_scenes(source, set(args.scene) if args.scene else None)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not scenes:
        raise SystemExit(f"没有找到场景目录；期望 {source}/<scene>/images/ 或 {source}/images/")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    manifest_records: list[dict[str, object]] = []
    totals = {
        "images": 0,
        "positive_images": 0,
        "negative_images": 0,
        "detections": 0,
        "written_labels": 0,
        "skipped_existing_labels": 0,
        "skipped_formal_labels": 0,
    }

    print(f"教师权重: {weights}")
    print(f"类别映射: {class_map}（旧模型 0=fire, 1=smoke）")
    print("自动标签写入 labels_auto/；正式 labels/ 不会被修改。")
    weights_hash = "dry-run" if args.dry_run else sha256_file(weights)
    staging_root = PROJECT_ROOT / "runs" / ".pseudo_label_tmp"
    if not args.dry_run:
        # Keep staging under the workspace. On managed Windows installations,
        # sandbox-specific %TEMP% paths may not be visible to the legacy Python
        # interpreter launched for YOLOv5.
        staging_root.mkdir(parents=True, exist_ok=True)

    for scene_dir in scenes:
        images_dir = scene_dir / "images"
        images = image_files(images_dir)
        if not images:
            print(f"跳过空场景: {scene_dir.name}")
            continue
        totals["images"] += len(images)
        with tempfile.TemporaryDirectory(
            prefix=f"fire_pseudo_{scene_dir.name}_", dir=staging_root if not args.dry_run else None
        ) as temp_name:
            temp_root = Path(temp_name)
            command = build_detect_command(
                python_executable,
                yolov5_dir,
                weights,
                images_dir,
                temp_root,
                args,
                class_map,
            )
            print(f"\n[{scene_dir.name}] {len(images)} 张图片")
            print(subprocess.list2cmdline(command))
            if args.dry_run:
                continue
            try:
                subprocess.run(command, cwd=yolov5_dir, check=True)
            except subprocess.CalledProcessError as exc:
                raise SystemExit(
                    f"YOLOv5 推理失败（退出码 {exc.returncode}）。"
                    "请确认 --python 指向已安装旧仓库 requirements.txt 的环境。"
                ) from exc

            result_dir = temp_root / "pred"
            result_labels_dir = result_dir / "labels"
            for image_path in images:
                auto_label_path = scene_dir / "labels_auto" / f"{image_path.stem}.txt"
                final_label_path = scene_dir / "labels" / f"{image_path.stem}.txt"
                if final_label_path.exists() and not args.include_labeled:
                    totals["skipped_formal_labels"] += 1
                    manifest_records.append(
                        {
                            "scene": scene_dir.name,
                            "image": str(image_path),
                            "formal_label": str(final_label_path),
                            "detection_count": None,
                            "review_status": "formal_label_preserved",
                        }
                    )
                    continue
                if auto_label_path.exists() and not args.overwrite_auto:
                    totals["skipped_existing_labels"] += 1
                    manifest_records.append(
                        {
                            "scene": scene_dir.name,
                            "image": str(image_path),
                            "auto_label": str(auto_label_path),
                            "detection_count": None,
                            "review_status": "existing_auto_preserved",
                        }
                    )
                    continue
                raw_prediction_path = result_labels_dir / f"{image_path.stem}.txt"
                predictions = parse_prediction_file(raw_prediction_path, class_map)
                if predictions:
                    totals["positive_images"] += 1
                else:
                    totals["negative_images"] += 1
                totals["detections"] += len(predictions)

                label_text = "".join(f"{prediction.yolo_line()}\n" for prediction in predictions)
                write_text(auto_label_path, label_text, overwrite=True)
                totals["written_labels"] += 1

                prediction_path = scene_dir / "predictions_auto" / f"{image_path.stem}.json"
                prediction_payload = {
                    "image": image_path.name,
                    "teacher_weights": str(weights),
                    "teacher_weights_sha256": weights_hash,
                    "needs_human_review": True,
                    "predictions": [asdict(prediction) for prediction in predictions],
                }
                write_text(
                    prediction_path,
                    json.dumps(prediction_payload, ensure_ascii=False, indent=2),
                    overwrite=True,
                )

                preview_source = result_dir / image_path.name
                preview_destination = scene_dir / "review_auto" / image_path.name
                preview_written = False
                if not args.no_preview and preview_source.is_file():
                    preview_written = transfer(
                        preview_source, preview_destination, overwrite=True
                    )

                manifest_records.append(
                    {
                        "scene": scene_dir.name,
                        "image": str(image_path),
                        "auto_label": str(auto_label_path),
                        "prediction_json": str(prediction_path),
                        "preview": str(preview_destination) if not args.no_preview else None,
                        "preview_written": preview_written,
                        "detection_count": len(predictions),
                        "confidence_min": min(
                            (prediction.confidence for prediction in predictions), default=None
                        ),
                        "confidence_max": max(
                            (prediction.confidence for prediction in predictions), default=None
                        ),
                        "review_status": "needs_review",
                    }
                )

    if args.dry_run:
        print("\nDry run 完成：未运行模型，也未写入自动标签。")
        return

    run_dir = args.manifest_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / f"{stamp}.json"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "yolov5_dir": str(yolov5_dir),
        "weights": str(weights),
        "weights_sha256": weights_hash,
        "teacher_class_names": {"0": "fire", "1": "smoke"},
        "class_map": {str(key): value for key, value in class_map.items()},
        "imgsz": args.imgsz,
        "conf_thres": args.conf_thres,
        "iou_thres": args.iou_thres,
        "device": args.device,
        "totals": totals,
        "records": manifest_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n自动标注完成：")
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    print(f"运行清单: {manifest_path}")
    print("下一步必须运行 review_pseudo_labels.py；未复核标签不会进入 labels/。")


if __name__ == "__main__":
    # Keep this wrapper independent from legacy YOLOv5 imports. Inference runs in
    # the explicitly selected interpreter through a subprocess.
    main()
