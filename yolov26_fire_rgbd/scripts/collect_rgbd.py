from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from fire_rgbd.orbbec import OrbbecRGBDCamera, RGBDFrame
from fire_rgbd.visualize import colorize_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="采集 Gemini 336L 对齐后的彩色图、16 位深度图和相机内参"
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--scene", required=True, help="场景 ID，例如 lab_day_01")
    parser.add_argument("--interval", type=float, default=0.5, help="自动采集间隔（秒）")
    parser.add_argument("--auto", action="store_true", help="启动后立即自动采集")
    parser.add_argument("--max-samples", type=int, default=0, help="达到数量后退出，0=不限")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-view-depth", type=float, default=8000.0)
    parser.add_argument("--software-align", action="store_true", help="禁用硬件 D2C，改用软件对齐")
    parser.add_argument("--headless", action="store_true", help="不打开窗口；必须同时使用 --auto")
    return parser.parse_args()


def save_sample(scene_dir: Path, frame: RGBDFrame, index: int, jpeg_quality: int) -> str:
    local_time = datetime.now().astimezone()
    stem = f"{local_time:%Y%m%d_%H%M%S_%f}_{index:06d}"
    image_path = scene_dir / "images" / f"{stem}.jpg"
    depth_path = scene_dir / "depth" / f"{stem}.png"
    metadata_path = scene_dir / "meta" / f"{stem}.json"

    depth_u16 = np.clip(np.rint(frame.depth_mm), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if not cv2.imwrite(str(image_path), frame.color_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
        raise OSError(f"无法写入 {image_path}")
    if not cv2.imwrite(str(depth_path), depth_u16):
        raise OSError(f"无法写入 {depth_path}")

    metadata = {
        "sample_id": stem,
        "captured_at_utc": frame.captured_at,
        "captured_at_local": local_time.isoformat(),
        "alignment": frame.alignment,
        "color_file": image_path.name,
        "depth_file": depth_path.name,
        "color_shape_hw": list(frame.color_bgr.shape[:2]),
        "depth_shape_hw": list(frame.depth_mm.shape),
        "depth_unit": "millimetre",
        "invalid_depth_value": 0,
        "color_intrinsics": frame.intrinsics.to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return stem


def main() -> None:
    args = parse_args()
    if args.headless and not args.auto:
        raise SystemExit("--headless 必须与 --auto 一起使用")
    if args.interval <= 0:
        raise SystemExit("--interval 必须大于 0")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality 必须在 1..100")

    scene_dir = (args.output / args.scene).resolve()
    for directory in ("images", "depth", "meta", "labels"):
        (scene_dir / directory).mkdir(parents=True, exist_ok=True)

    session = {
        "scene": args.scene,
        "started_at": datetime.now().astimezone().isoformat(),
        "note": "labels/ 中的同名 .txt 由标注工具生成；负样本需创建空 txt。",
    }
    (scene_dir / "session.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"保存目录: {scene_dir}")
    if not args.headless:
        print("按键: S=保存一帧, A=切换自动采集, Q/ESC=退出")
    auto_capture = args.auto
    saved_count = 0
    last_saved_at = 0.0

    with OrbbecRGBDCamera(prefer_hardware_align=not args.software_align) as camera:
        print(f"D2C 对齐方式: {camera.alignment}")
        intr = camera.intrinsics
        print(f"RGB 内参: {intr.width}x{intr.height}, fx={intr.fx:.2f}, fy={intr.fy:.2f}")
        while True:
            frame = camera.read()
            if frame is None:
                continue
            now = time.monotonic()
            should_save = auto_capture and now - last_saved_at >= args.interval

            key = -1
            if not args.headless:
                depth_view = colorize_depth(frame.depth_mm, args.max_view_depth)
                view = np.hstack((frame.color_bgr, depth_view))
                state = f"AUTO={'ON' if auto_capture else 'OFF'} SAVED={saved_count} ALIGN={camera.alignment}"
                cv2.putText(
                    view,
                    state,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("RGB | aligned depth", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("a"):
                    auto_capture = not auto_capture
                    print(f"自动采集: {'开启' if auto_capture else '关闭'}")
                if key == ord("s"):
                    should_save = True

            if should_save:
                saved_count += 1
                sample_id = save_sample(scene_dir, frame, saved_count, args.jpeg_quality)
                last_saved_at = now
                print(f"[{saved_count:06d}] {sample_id}")
                if args.max_samples > 0 and saved_count >= args.max_samples:
                    break

    cv2.destroyAllWindows()
    print(f"采集结束，共保存 {saved_count} 组 RGB-D 样本。")


if __name__ == "__main__":
    main()

