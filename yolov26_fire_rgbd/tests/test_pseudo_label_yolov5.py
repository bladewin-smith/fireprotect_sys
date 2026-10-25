from argparse import Namespace
from pathlib import Path

import pytest

from scripts.pseudo_label_yolov5 import (
    build_detect_command,
    discover_scenes,
    parse_class_map,
    parse_prediction_file,
)


def test_parse_class_map() -> None:
    assert parse_class_map(["0:0", "1:1"]) == {0: 0, 1: 1}
    with pytest.raises(ValueError, match="重复"):
        parse_class_map(["0:0", "0:1"])
    with pytest.raises(ValueError, match="SRC:DST"):
        parse_class_map(["0"])


def test_parse_prediction_file_filters_and_maps_classes(tmp_path: Path) -> None:
    prediction_file = tmp_path / "sample.txt"
    prediction_file.write_text(
        "0 0.5 0.4 0.2 0.3 0.91\n1 0.2 0.2 0.1 0.1 0.75\n",
        encoding="utf-8",
    )
    predictions = parse_prediction_file(prediction_file, {0: 0})
    assert len(predictions) == 1
    assert predictions[0].source_class_id == 0
    assert predictions[0].target_class_id == 0
    assert predictions[0].confidence == pytest.approx(0.91)
    assert predictions[0].yolo_line() == "0 0.50000000 0.40000000 0.20000000 0.30000000"


def test_discover_scenes_supports_root_and_single_scene(tmp_path: Path) -> None:
    scene_a = tmp_path / "scene_a"
    scene_b = tmp_path / "scene_b"
    (scene_a / "images").mkdir(parents=True)
    (scene_b / "images").mkdir(parents=True)
    assert discover_scenes(tmp_path) == [scene_a, scene_b]
    assert discover_scenes(scene_a) == [scene_a]
    assert discover_scenes(tmp_path, {"scene_b"}) == [scene_b]


def test_build_detect_command_passes_only_selected_teacher_classes(tmp_path: Path) -> None:
    args = Namespace(
        imgsz=416,
        conf_thres=0.35,
        iou_thres=0.45,
        max_det=100,
        device="0",
        half=True,
        no_preview=False,
    )
    command = build_detect_command(
        Path("python.exe"),
        tmp_path / "yolov5",
        tmp_path / "best.pt",
        tmp_path / "images",
        tmp_path / "output",
        args,
        {1: 0, 0: 0},
    )
    classes_index = command.index("--classes")
    assert command[classes_index + 1 : classes_index + 3] == ["0", "1"]
    assert "--save-conf" in command
    assert "--half" in command
    assert "--nosave" not in command
