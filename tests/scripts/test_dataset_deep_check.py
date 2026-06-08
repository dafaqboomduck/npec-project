"""Tests for scripts/dataset_deep_check.py — pure helper functions."""

from __future__ import annotations

import io
from pathlib import Path

import dataset_deep_check as ddc
from PIL import Image


def _write_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.new("L", (16, 16), 0).save(buf, format="PNG")
    path.write_bytes(buf.getvalue())
    return path


class TestTryRead:
    def test_returns_pil_for_valid_png(self, tmp_path: Path) -> None:
        p = _write_png(tmp_path / "x.png")
        ok, reader = ddc.try_read(p)
        assert ok is True
        assert reader == "PIL"

    def test_returns_none_for_garbage(self, tmp_path: Path) -> None:
        bad = tmp_path / "x.tif"
        bad.write_bytes(b"not a tiff")
        ok, _ = ddc.try_read(bad)
        # Either PIL or tifffile may swallow; with garbage neither succeeds.
        assert ok is False


class TestParseMaskName:
    def test_root_mask(self) -> None:
        out = ddc.parse_mask_name(Path("plant_001_root_mask.tif"))
        assert out == ("root", "plant_001")

    def test_shoot_mask(self) -> None:
        out = ddc.parse_mask_name(Path("plant_X_shoot_mask.tif"))
        assert out == ("shoot", "plant_X")

    def test_occluded_root_mask(self) -> None:
        out = ddc.parse_mask_name(Path("p_occluded_root_mask.tif"))
        assert out == ("occluded_root", "p")

    def test_non_matching_returns_none(self) -> None:
        assert ddc.parse_mask_name(Path("not_a_mask.png")) is None


class TestCohortAndAnnotator:
    def test_cohort_of_y2b(self, tmp_path: Path) -> None:
        cohort_root = tmp_path
        target = cohort_root / "Y2B_24" / "images" / "x.png"
        target.parent.mkdir(parents=True)
        target.touch()
        assert ddc.cohort_of(target, cohort_root) == "Y2B_24"

    def test_cohort_fallback_to_first_part(self, tmp_path: Path) -> None:
        target = tmp_path / "miscellaneous" / "x.png"
        target.parent.mkdir(parents=True)
        target.touch()
        assert ddc.cohort_of(target, tmp_path) == "miscellaneous"

    def test_annotator_of_finds_named_folder(self, tmp_path: Path) -> None:
        target = tmp_path / "masks" / "alice" / "x_root_mask.tif"
        target.parent.mkdir(parents=True)
        target.touch()
        assert ddc.annotator_of(target, tmp_path) == "alice"

    def test_annotator_of_flat(self, tmp_path: Path) -> None:
        target = tmp_path / "masks" / "x_root_mask.tif"
        target.parent.mkdir(parents=True)
        target.touch()
        assert ddc.annotator_of(target, tmp_path) == "(flat)"


class TestWalk:
    def test_separates_images_and_masks(self, tmp_path: Path) -> None:
        _write_png(tmp_path / "img1.png")
        _write_png(tmp_path / "img1_root_mask.tif")
        _write_png(tmp_path / "other.txt")
        images, masks = ddc.walk(tmp_path)
        assert len(images) == 1
        assert len(masks) == 1


class TestCheckUnreadable:
    def test_all_readable(self, tmp_path: Path) -> None:
        paths = [_write_png(tmp_path / f"m{i}.png") for i in range(3)]
        out = ddc.check_unreadable(paths)
        assert out["pil_ok"] == 3
        assert out["pil_fail_tifffile_fail"] == []
