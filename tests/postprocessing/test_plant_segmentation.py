"""Tests for npec_cv.core.plant_segmentation.PlantSegmentation.

Strategy: build small synthetic anchor / root masks where geometry is
known by construction, and exercise the public surface plus the helpers
that account for most of the file's red lines (segment_roots, the
zone-assignment branches, and the various filtering paths).
"""

from __future__ import annotations

import numpy as np
import pytest

from npec_cv.core.plant_segmentation import PlantSegmentation
from npec_cv.exceptions import ConfigurationError
from npec_cv.models import Anchor, SegmentationResult


class TestPlantSegmentationInit:
    """Construction-time validation."""

    def test_none_config_raises(self):
        """A None config is rejected with ConfigurationError."""
        with pytest.raises(ConfigurationError):
            PlantSegmentation(None)

    def test_empty_plant_zones_raises(self, seg_config):
        """A config with no plant zones is rejected."""
        seg_config.plant_zones = {}
        with pytest.raises(ConfigurationError):
            PlantSegmentation(seg_config)

    def test_valid_config_initializes(self, seg_config):
        """A valid config produces a usable instance."""
        seg = PlantSegmentation(seg_config)
        assert seg.config is seg_config


class TestDetectAnchorComponents:
    """Internal anchor detection — area + y-position filtering."""

    def test_returns_empty_for_empty_mask(self, seg_config, empty_anchor_mask):
        """No components → empty list."""
        seg = PlantSegmentation(seg_config)
        assert seg._detect_anchor_components(empty_anchor_mask) == []

    def test_finds_three_anchors(self, seg_config, three_zone_anchor_mask):
        """Three valid components → three Anchor objects."""
        seg = PlantSegmentation(seg_config)
        anchors = seg._detect_anchor_components(three_zone_anchor_mask)
        assert len(anchors) == 3
        assert all(isinstance(a, Anchor) for a in anchors)
        assert all(a.area >= seg_config.min_anchor_area for a in anchors)

    def test_filters_out_components_below_y_cutoff(self, seg_config, too_low_anchor_mask):
        """Components below y_top_ratio cutoff are discarded."""
        seg = PlantSegmentation(seg_config)
        assert seg._detect_anchor_components(too_low_anchor_mask) == []

    def test_filters_out_components_below_min_area(self, seg_config, too_small_anchor_mask):
        """Components smaller than min_anchor_area are discarded."""
        seg = PlantSegmentation(seg_config)
        assert seg._detect_anchor_components(too_small_anchor_mask) == []


class TestAssignAnchorsToZones:
    """Anchor → zone_id resolution and largest-per-zone rule."""

    def test_anchors_get_assigned_to_correct_zones(self, seg_config, three_zone_anchor_mask):
        """Three anchors at x=100, 200, 300 land in zones 0, 1, 2."""
        seg = PlantSegmentation(seg_config)
        anchors = seg._detect_anchor_components(three_zone_anchor_mask)
        assigned = seg._assign_anchors_to_zones(anchors)
        zone_ids = sorted(a.zone_id for a in assigned)
        assert zone_ids == [0, 1, 2]

    def test_keeps_only_largest_anchor_per_zone(self, seg_config):
        """When two components fall in one zone, only the larger survives."""
        seg = PlantSegmentation(seg_config)
        small = Anchor(
            label_id=1,
            centroid=(100, 50),
            area=50,
            mask=np.zeros((10, 10), dtype=np.uint8),
        )
        large = Anchor(
            label_id=2,
            centroid=(110, 50),
            area=400,
            mask=np.zeros((10, 10), dtype=np.uint8),
        )
        result = seg._assign_anchors_to_zones([small, large])
        assert len(result) == 1
        assert result[0].label_id == 2

    def test_anchor_outside_any_zone_is_dropped(self, seg_config):
        """Anchor with x outside all configured zone ranges has zone_id=None and is dropped."""
        seg = PlantSegmentation(seg_config)
        outside = Anchor(
            label_id=1,
            centroid=(10, 50),
            area=100,  # x=10, before zone 0's start (50)
            mask=np.zeros((10, 10), dtype=np.uint8),
        )
        assert seg._assign_anchors_to_zones([outside]) == []


class TestCreateAnchorBboxes:
    """Bbox computation, including image-edge clamping."""

    def test_invalid_img_shape_raises(self, seg_config):
        """img_shape with <2 dims is rejected."""
        seg = PlantSegmentation(seg_config)
        with pytest.raises(ValueError):
            seg.create_anchor_bboxes([], img_shape=(100,))

    def test_bbox_is_centered_on_centroid_with_offset(self, seg_config):
        """Bbox is symmetric around centroid horizontally, shifted down by offset."""
        seg = PlantSegmentation(seg_config)
        anchor = Anchor(
            label_id=1,
            centroid=(200, 100),
            area=100,
            mask=np.zeros((10, 10), dtype=np.uint8),
        )
        result = seg.create_anchor_bboxes([anchor], img_shape=(300, 400))
        x_min, y_min, x_max, y_max = result[0].bbox
        half = seg_config.anchor_bbox_size // 2  # 30
        assert (x_max - x_min) == 2 * half
        assert x_min == 200 - half and x_max == 200 + half
        assert y_min == 100 - half + seg_config.vertical_offset

    def test_bbox_clamps_at_image_edges(self, seg_config):
        """Bbox bounds are clamped to image dimensions."""
        seg = PlantSegmentation(seg_config)
        # Anchor near top-left corner — should clamp left/top to 0
        anchor = Anchor(
            label_id=1,
            centroid=(5, 5),
            area=100,
            mask=np.zeros((10, 10), dtype=np.uint8),
        )
        result = seg.create_anchor_bboxes([anchor], img_shape=(300, 400))
        x_min, y_min, _, _ = result[0].bbox
        assert x_min == 0 and y_min == 0


class TestExtractAnchors:
    """End-to-end anchor extraction pipeline."""

    def test_full_pipeline_produces_zoned_anchors(self, seg_config, three_zone_anchor_mask):
        """extract_anchors detects, assigns zones — bbox is created separately."""
        seg = PlantSegmentation(seg_config)
        anchors = seg.extract_anchors(three_zone_anchor_mask)
        assert len(anchors) == 3
        assert all(a.zone_id is not None for a in anchors)


class TestSegmentRoots:
    """Root segmentation — the 50%-coverage gap in this file."""

    def test_skips_anchors_with_no_zone(self, seg_config, root_mask_one_zone):
        """An anchor without zone_id or bbox is skipped silently."""
        seg = PlantSegmentation(seg_config)
        result = seg.segment_roots(root_mask_one_zone, anchors=[])
        assert isinstance(result, SegmentationResult)
        assert len(result.zone_masks) == 0
        assert result.filtered_mask.sum() == 0

    def test_assigns_root_to_matching_zone(self, seg_config, root_mask_one_zone, anchors_at_zone0):
        """A root component originating under an anchor's bbox is assigned to that zone."""
        seg = PlantSegmentation(seg_config)
        result = seg.segment_roots(root_mask_one_zone, anchors=anchors_at_zone0)
        assert 0 in result.zone_masks
        assert result.zone_masks[0].any()
        # Labels array tags zone 0 pixels with 1 (zone_id + 1)
        assert (result.labels == 1).any()
        assert (result.filtered_mask == 255).any()

    def test_filters_out_below_min_area(self, seg_config, anchors_at_zone0):
        """Root components below min_root_area are skipped."""
        seg = PlantSegmentation(seg_config)
        # tiny 2x2 component at x≈100 — only 4 pixels, below min_root_area=10
        tiny_mask = np.zeros((300, 400), dtype=np.uint8)
        tiny_mask[70:72, 99:101] = 255
        result = seg.segment_roots(tiny_mask, anchors=anchors_at_zone0)
        assert len(result.zone_masks) == 0

    def test_root_outside_bbox_is_not_assigned(self, seg_config, anchors_at_zone0):
        """A root component whose top is outside the bbox + tolerance is dropped."""
        seg = PlantSegmentation(seg_config)
        # vertical component at x=300 (zone 2 territory) — anchor is at zone 0
        far_mask = np.zeros((300, 400), dtype=np.uint8)
        far_mask[70:200, 298:303] = 255
        result = seg.segment_roots(far_mask, anchors=anchors_at_zone0)
        assert 0 not in result.zone_masks

    def test_merges_nearby_components(self, seg_config, anchors_at_zone0):
        """Two root fragments within merge_distance fuse into one zone mask.

        Both fragments must have their *top* inside the anchor bbox
        (anchor_zone0 bbox = (70, 25, 130, 85), tolerance=10 → top_y must
        be ≤ 105 and top_x in [60, 140]). Once both pass entry filtering,
        the merge step bridges them via x-gap (their bbox edges sit 10px
        apart in x, well under merge_distance=15).
        """
        seg = PlantSegmentation(seg_config)
        mask = np.zeros((300, 400), dtype=np.uint8)
        # Two parallel vertical stripes — both start at y=70 (in bbox),
        # separated horizontally by 10 px (< merge_distance=15).
        mask[70:220, 90:95] = 255  # left fragment, top at (70, ~92)
        mask[70:220, 105:110] = 255  # right fragment, top at (70, ~107)
        result = seg.segment_roots(mask, anchors=anchors_at_zone0)
        assert 0 in result.zone_masks
        # Both fragments should now live in zone 0's mask
        assert result.zone_masks[0][70:220, 90:95].all()
        assert result.zone_masks[0][70:220, 105:110].all()
