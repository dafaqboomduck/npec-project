"""Tests for scripts/inventory_dataset.py — pure helper functions."""

from __future__ import annotations

import io
from pathlib import Path

import inventory_dataset as inv
import numpy as np
import pytest
from PIL import Image


def _write_png(path: Path, width: int = 32, height: int = 32, value: int = 128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.new("L", (width, height), value).save(buf, format="PNG")
    path.write_bytes(buf.getvalue())
    return path


class TestIsMaskPath:
    @pytest.mark.parametrize(
        "path",
        [
            Path("data/masks/x_mask.png"),
            Path("data/x_label.png"),
            Path("annotations/foo.png"),
            Path("data/SEGMENTation_x.png"),
        ],
    )
    def test_recognises_mask_hints(self, path: Path) -> None:
        assert inv.is_mask_path(path) is True

    def test_rejects_plain_image(self) -> None:
        assert inv.is_mask_path(Path("data/images/plant.png")) is False


class TestLoadImage:
    def test_returns_array_for_valid_png(self, tmp_path: Path) -> None:
        p = _write_png(tmp_path / "a.png")
        out = inv.load_image(p)
        assert isinstance(out, np.ndarray)
        assert out.shape == (32, 32)

    def test_returns_none_for_invalid_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not really a png")
        assert inv.load_image(bad) is None


class TestSummariseArray:
    def test_2d_array(self) -> None:
        arr = np.zeros((10, 10), dtype=np.uint8)
        out = inv.summarise_array(arr)
        assert out["shape"] == [10, 10]
        assert out["dtype"] == "uint8"
        assert out["channels"] == 1

    def test_3d_array(self) -> None:
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        out = inv.summarise_array(arr)
        assert out["channels"] == 3


class TestAnalysePairing:
    def test_no_images(self) -> None:
        out = inv.analyse_pairing([], [Path("a_mask.png")])
        assert out["num_images"] == 0
        assert out["ratio_masks_per_image"] == 0

    def test_perfect_pairs(self) -> None:
        images = [Path("img1.png"), Path("img2.png")]
        masks = [Path("img1_mask.png"), Path("img2_mask.png")]
        out = inv.analyse_pairing(images, masks)
        assert out["matched_pairs"] == 2
        assert out["images_with_no_mask_count"] == 0
        assert out["masks_with_no_image_count"] == 0

    def test_orphan_mask_recorded(self) -> None:
        images = [Path("img1.png")]
        masks = [Path("orphan_mask.png")]
        out = inv.analyse_pairing(images, masks)
        assert out["masks_with_no_image_count"] == 1
        assert out["images_with_no_mask_count"] == 1

    def test_multiple_masks_per_image(self) -> None:
        images = [Path("img1.png")]
        masks = [Path("img1_root_mask.png"), Path("img1_shoot_mask.png")]
        out = inv.analyse_pairing(images, masks)
        # Only the suffix-pattern (e.g. _mask) is peeled. After that the iterative
        # underscore peel finds img1 as a prefix in both, so both match.
        assert out["matched_pairs"] == 1
        assert "2_masks" in out["masks_per_image_distribution"]


class TestWalkDataset:
    def test_walks_real_tree(self, tmp_path: Path) -> None:
        _write_png(tmp_path / "imgs" / "a.png")
        _write_png(tmp_path / "masks" / "a_mask.png")
        (tmp_path / "junk.txt").write_text("hi")
        out = inv.walk_dataset(tmp_path)
        assert ".png" in out["ext_counts"]
        assert ".txt" in out["ext_counts"]
        assert len(out["image_paths"]) == 1
        assert len(out["mask_paths"]) == 1
        assert out["total_bytes"] > 0


class TestSampleAndInspect:
    def test_empty(self) -> None:
        out = inv.sample_and_inspect([], n=5, label="images")
        assert out["count"] == 0
        assert out["samples"] == []

    def test_real_sample(self, tmp_path: Path) -> None:
        paths = [_write_png(tmp_path / f"i{i}.png") for i in range(3)]
        out = inv.sample_and_inspect(paths, n=2, label="images")
        assert out["count"] == 3
        assert out["sampled"] == 2

    def test_mask_label_captures_unique_values(self, tmp_path: Path) -> None:
        paths = [_write_png(tmp_path / "m.png", value=0)]
        out = inv.sample_and_inspect(paths, n=1, label="masks")
        # Pure-zero mask: unique values include 0
        assert 0 in out["unique_values_across_samples"]

    def test_unreadable_files_reported(self, tmp_path: Path) -> None:
        good = _write_png(tmp_path / "ok.png")
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"garbage")
        out = inv.sample_and_inspect([good, bad], n=2, label="images")
        # Unreadable shows up in the report
        assert any("bad.png" in u for u in out["unreadable"])


class TestFormatReport:
    def test_report_includes_required_sections(self, tmp_path: Path) -> None:
        _write_png(tmp_path / "img.png")
        _write_png(tmp_path / "img_mask.png")
        walk = inv.walk_dataset(tmp_path)
        report = inv.format_report(
            {
                "walk": walk,
                "image_report": inv.sample_and_inspect(walk["image_paths"], 1, "images"),
                "mask_report": inv.sample_and_inspect(walk["mask_paths"], 1, "masks"),
                "pairing": inv.analyse_pairing(walk["image_paths"], walk["mask_paths"]),
            },
            tmp_path,
        )
        assert "Dataset inventory report" in report
        assert "File counts by extension" in report
        assert "Image ↔ mask pairing" in report
