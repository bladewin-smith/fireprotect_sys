from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 YOLO 检测标签和数据集完整性")
    parser.add_argument("--data", type=Path, default=Path("data/fire/dataset.yaml"))
    return parser.parse_args()


def resolve_root(config_path: Path, configured_path: str) -> Path:
    root = Path(configured_path)
    if not root.is_absolute():
        root = config_path.parent / root
    return root.resolve()


def main() -> None:
    args = parse_args()
    config_path = args.data.resolve()
    if not config_path.is_file():
        raise SystemExit(f"配置文件不存在：{config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = resolve_root(config_path, str(config.get("path", ".")))
    names = config.get("names", {})
    class_count = len(names)
    errors: list[str] = []
    class_instances: Counter[int] = Counter()

    print(f"数据根目录: {root}")
    for split in ("train", "val", "test"):
        relative = config.get(split)
        if not relative:
            continue
        image_dir = root / str(relative)
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            errors.append(f"{split}: 图片目录不存在 {image_dir}")
            continue
        images = [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        negatives = 0
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                errors.append(f"缺少标签文件: {label_path}")
                continue
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                negatives += 1
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label_path}:{line_number} 应为 5 列，实际 {len(parts)} 列")
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append(f"{label_path}:{line_number} 含非数值字段")
                    continue
                if not 0 <= class_id < class_count:
                    errors.append(f"{label_path}:{line_number} 类别 ID {class_id} 越界")
                if any(not 0.0 <= value <= 1.0 for value in values):
                    errors.append(f"{label_path}:{line_number} 坐标必须归一化到 [0,1]")
                if values[2] <= 0 or values[3] <= 0:
                    errors.append(f"{label_path}:{line_number} 宽高必须大于 0")
                class_instances[class_id] += 1
        print(f"{split:>5}: {len(images)} 张图片，{negatives} 张负样本")

    print(f"实例数: {dict(sorted(class_instances.items()))}")
    if errors:
        print(f"发现 {len(errors)} 个问题：")
        for error in errors[:100]:
            print(f"  - {error}")
        if len(errors) > 100:
            print(f"  ... 另有 {len(errors) - 100} 个")
        raise SystemExit(1)
    print("检查通过。")


if __name__ == "__main__":
    main()

