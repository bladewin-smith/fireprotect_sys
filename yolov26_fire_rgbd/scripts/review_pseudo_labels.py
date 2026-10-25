from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐张复核 YOLOv5 自动标签并提升为正式标签")
    parser.add_argument("--source", type=Path, default=Path("data/raw"))
    parser.add_argument("--scene", action="append", default=[], help="只复核指定场景，可重复")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["fire"],
        help="目标类别名，顺序必须与目标类别 ID 一致",
    )
    parser.add_argument(
        "--order",
        choices=("risk", "name"),
        default="risk",
        help="risk 优先显示自动判为负样本和低置信度样本",
    )
    parser.add_argument("--include-reviewed", action="store_true")
    parser.add_argument(
        "--overwrite-final",
        action="store_true",
        help="允许 A/N 覆盖 labels/ 中已有正式标签；默认严格保护人工标签",
    )
    parser.add_argument("--list-only", action="store_true", help="只打印待复核统计，不打开窗口")
    return parser.parse_args()


def discover_scenes(source: Path, selected: set[str] | None = None) -> list[Path]:
    if (source / "images").is_dir():
        candidates = [source]
    elif source.is_dir():
        candidates = [
            path for path in source.iterdir() if path.is_dir() and (path / "images").is_dir()
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


def load_state(scene_dir: Path) -> dict[str, dict[str, str]]:
    state_path = scene_dir / "review_state.json"
    if not state_path.is_file():
        return {}
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    records = payload.get("records", {})
    if not isinstance(records, dict):
        raise ValueError(f"复核状态格式错误：{state_path}")
    return records


def save_state(scene_dir: Path, records: dict[str, dict[str, str]]) -> None:
    state_path = scene_dir / "review_state.json"
    temporary_path = state_path.with_suffix(".json.tmp")
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(state_path)


def read_predictions(scene_dir: Path, stem: str) -> list[dict[str, float | int]]:
    path = scene_dir / "predictions_auto" / f"{stem}.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions", [])
    return predictions if isinstance(predictions, list) else []


def risk_key(scene_dir: Path, image_path: Path) -> tuple[int, float, str]:
    predictions = read_predictions(scene_dir, image_path.stem)
    if not predictions:
        return (0, 0.0, image_path.name)
    confidences = [float(prediction.get("confidence", 0.0)) for prediction in predictions]
    return (1, min(confidences, default=0.0), image_path.name)


def scene_images(scene_dir: Path, order: str) -> list[Path]:
    images = sorted(
        path
        for path in (scene_dir / "images").iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and (scene_dir / "labels_auto" / f"{path.stem}.txt").is_file()
    )
    if order == "risk":
        images.sort(key=lambda image_path: risk_key(scene_dir, image_path))
    return images


def review_summary(
    scenes: list[Path], order: str
) -> tuple[list[tuple[Path, Path]], dict[str, int]]:
    queue: list[tuple[Path, Path]] = []
    summary = {"auto_labeled": 0, "reviewed": 0, "pending": 0, "final_labels": 0}
    for scene_dir in scenes:
        state = load_state(scene_dir)
        images = scene_images(scene_dir, order)
        summary["auto_labeled"] += len(images)
        summary["reviewed"] += sum(
            1 for image_path in images if image_path.name in state
        )
        summary["final_labels"] += sum(
            1
            for image_path in images
            if (scene_dir / "labels" / f"{image_path.stem}.txt").is_file()
        )
        queue.extend((scene_dir, image_path) for image_path in images)
    summary["pending"] = summary["auto_labeled"] - summary["reviewed"]
    return queue, summary


def draw_predictions(image, predictions, class_names: list[str]):
    import cv2

    height, width = image.shape[:2]
    for prediction in predictions:
        class_id = int(prediction["target_class_id"])
        x_center = float(prediction["x_center"]) * width
        y_center = float(prediction["y_center"]) * height
        box_width = float(prediction["width"]) * width
        box_height = float(prediction["height"]) * height
        x1 = max(0, round(x_center - box_width / 2))
        y1 = max(0, round(y_center - box_height / 2))
        x2 = min(width - 1, round(x_center + box_width / 2))
        y2 = min(height - 1, round(y_center + box_height / 2))
        confidence = float(prediction.get("confidence", 0.0))
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), (20, 220, 20), 2)
        cv2.putText(
            image,
            f"{class_name} {confidence:.2f}",
            (x1, max(22, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 220, 20),
            2,
            cv2.LINE_AA,
        )
    return image


def fit_for_display(image, max_width: int = 1600, max_height: int = 900):
    import cv2

    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)))


def promote_label(scene_dir: Path, image_path: Path, negative: bool, overwrite: bool) -> bool:
    destination = scene_dir / "labels" / f"{image_path.stem}.txt"
    if destination.exists() and not overwrite:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if negative:
        destination.write_text("", encoding="utf-8")
    else:
        source = scene_dir / "labels_auto" / f"{image_path.stem}.txt"
        shutil.copy2(source, destination)
    return True


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    try:
        scenes = discover_scenes(source, set(args.scene) if args.scene else None)
        queue, summary = review_summary(scenes, args.order)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    if not scenes:
        raise SystemExit(f"没有找到场景：{source}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.list_only:
        return
    if not queue:
        raise SystemExit("没有找到 labels_auto/ 与 images/ 的同名样本。请先运行自动标注脚本。")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("缺少 OpenCV，请安装 requirements.txt 后重试。") from exc

    states = {scene_dir: load_state(scene_dir) for scene_dir in scenes}
    if not args.include_reviewed:
        queue = [
            (scene_dir, image_path)
            for scene_dir, image_path in queue
            if image_path.name not in states[scene_dir]
            and not (scene_dir / "labels" / f"{image_path.stem}.txt").is_file()
        ]
    if not queue:
        raise SystemExit("所有自动标签都已复核。使用 --include-reviewed 可重新查看。")

    print(
        "按键：A=接受自动框，N=确认为无火负样本，D=丢弃/不入训练，"
        "S=暂跳过，B=上一张，Q=保存并退出"
    )
    index = 0
    window_name = "YOLOv5 pseudo-label review"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while 0 <= index < len(queue):
        scene_dir, image_path = queue[index]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取，跳过：{image_path}")
            index += 1
            continue
        predictions = read_predictions(scene_dir, image_path.stem)
        draw_predictions(image, predictions, args.class_names)
        existing_label = (scene_dir / "labels" / f"{image_path.stem}.txt").is_file()
        status = states[scene_dir].get(image_path.name, {}).get("status", "pending")
        final_label_text = "YES" if existing_label else "NO"
        lines = [
            f"{index + 1}/{len(queue)}  scene={scene_dir.name}  file={image_path.name}",
            f"boxes={len(predictions)}  review={status}  final_label={final_label_text}",
            "A accept | N negative | D discard | S skip | B back | Q quit",
        ]
        for line_index, text in enumerate(lines):
            cv2.putText(
                image,
                text,
                (15, 30 + line_index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                text,
                (15, 30 + line_index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(window_name, fit_for_display(image))
        key = cv2.waitKey(0) & 0xFF
        now = datetime.now(timezone.utc).isoformat()
        if key in (ord("q"), 27):
            break
        if key == ord("b"):
            index = max(0, index - 1)
            continue
        if key == ord("s"):
            index += 1
            continue
        if key == ord("d"):
            states[scene_dir][image_path.name] = {"status": "discarded", "reviewed_at": now}
            save_state(scene_dir, states[scene_dir])
            index += 1
            continue
        if key in (ord("a"), ord("n")):
            negative = key == ord("n")
            if promote_label(scene_dir, image_path, negative, args.overwrite_final):
                status = "negative" if negative else "accepted"
                states[scene_dir][image_path.name] = {"status": status, "reviewed_at": now}
                save_state(scene_dir, states[scene_dir])
                print(f"{status:>9}: {scene_dir.name}/{image_path.name}")
                index += 1
            else:
                print(
                    f"保护已有正式标签，未覆盖：{scene_dir.name}/{image_path.stem}.txt；"
                    "如确需覆盖请使用 --overwrite-final"
                )

    for scene_dir, records in states.items():
        save_state(scene_dir, records)
    cv2.destroyAllWindows()
    print("复核状态已保存。A/N 样本已进入 labels/；D/S 样本不会进入训练集。")


if __name__ == "__main__":
    main()
