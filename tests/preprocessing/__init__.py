"""Preprocessing test suite for npec_cv.

Suite 1 of 3 — covers all data-preprocessing concerns:
  - helpers/sort_keys.py
  - helpers/padder.py
  - helpers/patchify.py (light coverage of vendored code)
  - helpers/extract_petri_dish.py
  - inference/inference_manager.py::patch_image (preprocessing entry point)
  - inference/inference_manager.py::restore_mask (preprocessing inverse)

Synthesized images at runtime — no checked-in fixtures required.
"""
