from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按场景划分 YOLO 火焰检测数据集")
    parser.add_argument("--source", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/fire"))
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-name",
        default="fire",
        help="单类别兼容参数；未指定 --class-names 时使用",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="按类别 ID 顺序给出名称，例如 --class-names fire smoke",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="尝试创建硬链接以节省空间；失败时自动复制",
    )
    return parser.parse_args()


def split_scene_names(
    scene_names: list[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[str]]:
    if not scene_names:
        raise ValueError("没有找到可用场景")
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(value < 0 for value in ratios) or not 0.999 <= sum(ratios) <= 1.001:
        raise ValueError("train/val/test 比例必须非负且总和为 1")

    names = sorted(scene_names)
    random.Random(seed).shuffle(names)
    total = len(names)
    counts = [int(total * value) for value in ratios]
    remainder = total - sum(counts)
    for index in sorted(range(3), key=lambda i: ratios[i] * total - counts[i], reverse=True):
        if remainder <= 0:
            break
        counts[index] += 1
        remainder -= 1

    nonzero_splits = [i for i, ratio in enumerate(ratios) if ratio > 0]
    if total >= len(nonzero_splits):
        for index in nonzero_splits:
            if counts[index] == 0:
                donor = max(nonzero_splits, key=lambda i: counts[i])
                counts[donor] -= 1
                counts[index] += 1

    train_end = counts[0]
    val_end = train_end + counts[1]
    return {
        "train": names[:train_end],
        "val": names[train_end:val_end],
        "test": names[val_end:],
    }


def resolve_class_names(class_name: str, class_names: list[str] | None) -> dict[int, str]:
    names = class_names if class_names is not None else [class_name]
    cleaned = [name.strip() for name in names]
    if not cleaned or any(not name for name in cleaned):
        raise ValueError("类别名不能为空")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("类别名不能重复")
    return {index: name for index, name in enumerate(cleaned)}


def scene_samples(scene_dir: Path) -> tuple[list[tuple[Path, Path]], int]:
    images_dir = scene_dir / "images"
    labels_dir = scene_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return [], 0
    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            pairs.append((image_path, label_path))
        else:
            missing += 1
    return pairs, missing


def ensure_empty_output(output: Path) -> None:
    existing = (
        []
        if not output.exists()
        else [path for path in output.iterdir() if path.name != ".gitkeep"]
    )
    if existing:
        raise FileExistsError(
            f"输出目录非空：{output}\n为防止覆盖数据，请改用新的 --output 或手动确认后清空。"
        )
    output.mkdir(parents=True, exist_ok=True)


def transfer(source: Path, destination: Path, link: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    try:
        class_names = resolve_class_names(args.class_name, args.class_names)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    scene_data: dict[str, list[tuple[Path, Path]]] = {}
    missing_labels: dict[str, int] = {}

    if not source.is_dir():
        raise SystemExit(f"采集目录不存在：{source}")
    for scene_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        pairs, missing = scene_samples(scene_dir)
        if pairs:
            scene_data[scene_dir.name] = pairs
        if missing:
            missing_labels[scene_dir.name] = missing
    if not scene_data:
        raise SystemExit("没有找到已标注图片。需要 images/a.jpg 与 labels/a.txt 同名配对。")

    splits = split_scene_names(
        list(scene_data), args.train, args.val, args.test, args.seed
    )
    ensure_empty_output(output)
    sample_counts: Counter[str] = Counter()
    for split, scene_names in splits.items():
        for scene_name in scene_names:
            for image_path, label_path in scene_data[scene_name]:
                safe_stem = f"{scene_name}__{image_path.stem}"
                transfer(
                    image_path,
                    output / "images" / split / f"{safe_stem}{image_path.suffix.lower()}",
                    args.link,
                )
                transfer(label_path, output / "labels" / split / f"{safe_stem}.txt", args.link)
                sample_counts[split] += 1

    dataset_yaml = {
        "path": output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "seed": args.seed,
        "scene_splits": splits,
        "sample_counts": dict(sample_counts),
        "images_without_labels_skipped": missing_labels,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"数据配置: {output / 'dataset.yaml'}")


if __name__ == "__main__":
    main()

