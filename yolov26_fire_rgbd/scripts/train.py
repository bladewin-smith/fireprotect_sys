from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微调 YOLO26 火焰检测模型")
    parser.add_argument("--data", type=Path, default=Path("data/fire/dataset.yaml"))
    parser.add_argument("--model", default="yolo26s.pt", help="预训练权重或模型 YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="RTX 3060 12GB 建议先用 8；-1=自动")
    parser.add_argument("--device", default="0", help="CUDA 设备，例如 0；CPU 使用 cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=None, help="不设置时沿用 YOLO26 推荐策略")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--freeze", type=int, default=0, help="冻结前 N 层；0=全部微调")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", default="fire_yolo26s")
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    parser.add_argument("--resume", type=Path, default=None, help="从 last.pt 完整续训")
    parser.add_argument("--cos-lr", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("依赖未安装。请先按 README 创建环境并安装 requirements.txt。") from exc

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.resume is not None:
        checkpoint = args.resume.resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"续训权重不存在：{checkpoint}")
        YOLO(str(checkpoint)).train(resume=True)
        return

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise SystemExit(f"数据配置不存在：{data_path}")
    cache: bool | str = False if args.cache == "false" else args.cache
    train_args: dict[str, object] = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "freeze": args.freeze,
        "seed": args.seed,
        "deterministic": True,
        "project": str(args.project.resolve()),
        "name": args.name,
        "cache": cache,
        "amp": True,
        "plots": True,
        "val": True,
        "cos_lr": args.cos_lr,
        # Conservative fire-specific augmentation. Vertical flips are physically
        # implausible; strong mixup/copy-paste can destroy flame boundaries.
        "hsv_h": 0.015,
        "hsv_s": 0.55,
        "hsv_v": 0.35,
        "degrees": 5.0,
        "translate": 0.10,
        "scale": 0.40,
        "shear": 2.0,
        "perspective": 0.0002,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.7,
        "close_mosaic": 10,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }
    if args.lr0 is not None:
        train_args["lr0"] = args.lr0
    if args.weight_decay is not None:
        train_args["weight_decay"] = args.weight_decay

    model = YOLO(args.model)
    model.train(**train_args)


if __name__ == "__main__":
    # Required on Windows because Ultralytics DataLoader uses multiprocessing.
    main()

