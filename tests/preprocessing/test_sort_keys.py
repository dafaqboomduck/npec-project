"""Tests for npec_cv.helpers.sort_keys.extract_sort_keys.

Module under test: src/npec_cv/helpers/sort_keys.py
Function:          extract_sort_keys(plant_id) -> tuple[int, int]

Plant-ID format:   "<image_name>_plant_<plant_num>"
                   image_name typically follows "smth_im<N>" with N un-padded
                   (e.g. "exp_im3", not "exp_im03").

Test coverage
-------------
  TestExtractSortKeysHappyPath
    - standard format parses image and plant numbers correctly
    - typical project naming convention (smth_imN_plant_M)
    - plant_id values: 0, large numbers
  TestExtractSortKeysFallbacks
    - no digits in image name → image_num falls back to 0
    - last digit group is used when multiple groups exist (LOCKED behavior)
  TestExtractSortKeysSorting
    - sorts a realistic batch into the expected order (the actual contract)
  TestExtractSortKeysErrors
    - missing "_plant_" separator raises (LOCKED behavior — IndexError)
    - non-numeric plant_num raises (LOCKED behavior — ValueError)
"""

from __future__ import annotations

import pytest

from src.npec_cv.helpers.sort_keys import extract_sort_keys


class TestExtractSortKeysHappyPath:
    """Standard parsing of well-formed plant_id strings."""

    def test_standard_format(self):
        """'image_005_plant_2' parses to (5, 2)."""
        assert extract_sort_keys("image_005_plant_2") == (5, 2)

    def test_project_naming_convention(self):
        """'experiment_im3_plant_2' parses to (3, 2) — matches actual project format."""
        assert extract_sort_keys("experiment_im3_plant_2") == (3, 2)

    def test_plant_zero(self):
        """plant_num of 0 is returned as int 0, not falsy-coerced."""
        result = extract_sort_keys("exp_im1_plant_0")
        assert result[1] == 0
        assert isinstance(result[1], int)

    def test_large_image_and_plant_numbers(self):
        """Multi-digit numbers parse correctly without truncation."""
        assert extract_sort_keys("exp_im142_plant_57") == (142, 57)


class TestExtractSortKeysFallbacks:
    """Edge-case parsing behavior."""

    def test_no_digits_in_image_name_falls_back_to_zero(self):
        """When image_name has no digits, image_num defaults to 0."""
        assert extract_sort_keys("foo_plant_4") == (0, 4)

    def test_last_digit_group_is_used(self):
        """LOCKED BEHAVIOR: when image_name has multiple digit groups, the last wins.

        This is a documented quirk of the regex \\d+ + [-1] indexing.
        Project naming convention puts the image number at the end so this
        works in practice, but the test pins the behavior so a future
        refactor (e.g. switching to "first group") is a deliberate choice.
        """
        # groups in 'plate_v2_im7' are '2' and '7' — regex finds both, [-1] picks '7'
        assert extract_sort_keys("plate_v2_im7_plant_3") == (7, 3)


class TestExtractSortKeysSorting:
    """The real-world contract: sort a batch correctly."""

    def test_sorts_mixed_batch_in_image_then_plant_order(self):
        """A list of plant_ids sorts by (image_num, plant_num) primary/secondary."""
        unsorted = [
            "exp_im2_plant_3",
            "exp_im1_plant_5",
            "exp_im2_plant_1",
            "exp_im1_plant_2",
        ]
        expected = [
            "exp_im1_plant_2",
            "exp_im1_plant_5",
            "exp_im2_plant_1",
            "exp_im2_plant_3",
        ]
        assert sorted(unsorted, key=extract_sort_keys) == expected


class TestExtractSortKeysErrors:
    """Failure modes — currently implicit. Locked in for now."""

    def test_missing_plant_separator_raises(self):
        """Plant IDs without '_plant_' raise IndexError (split returns 1 part).

        LOCKED: current behavior is implicit — could later be wrapped in a
        domain-specific exception. This test surfaces any such change.
        """
        with pytest.raises(IndexError):
            extract_sort_keys("no_separator_here")

    def test_non_numeric_plant_num_raises(self):
        """Plant IDs with non-numeric plant_num raise ValueError."""
        with pytest.raises(ValueError):
            extract_sort_keys("img_3_plant_abc")
