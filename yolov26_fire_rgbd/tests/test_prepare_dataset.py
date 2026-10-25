import pytest

from scripts.prepare_dataset import resolve_class_names, split_scene_names


def test_scene_split_is_disjoint_and_complete() -> None:
    scenes = [f"scene_{index}" for index in range(10)]
    split = split_scene_names(scenes, 0.7, 0.2, 0.1, seed=42)
    all_assigned = split["train"] + split["val"] + split["test"]
    assert set(all_assigned) == set(scenes)
    assert len(all_assigned) == len(set(all_assigned))
    assert len(split["train"]) == 7
    assert len(split["val"]) == 2
    assert len(split["test"]) == 1


def test_small_scene_split_keeps_each_requested_split() -> None:
    split = split_scene_names(["a", "b", "c"], 0.7, 0.2, 0.1, seed=1)
    assert all(len(split[name]) == 1 for name in ("train", "val", "test"))


def test_resolve_class_names_supports_single_and_multiple_classes() -> None:
    assert resolve_class_names("fire", None) == {0: "fire"}
    assert resolve_class_names("ignored", ["fire", "smoke"]) == {0: "fire", 1: "smoke"}


def test_resolve_class_names_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="不能重复"):
        resolve_class_names("fire", ["fire", "fire"])

