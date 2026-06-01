"""Azure ML managed online endpoint scoring script — root detection pipeline.

How the two models are loaded
-----------------------------
Both models are bundled inside the deployment's registered model artifact
('root-detection-serving'), so AZUREML_MODEL_DIR contains BOTH:
    root_det_model_512px.keras
    anchor_det_model_512px.keras

No runtime download is needed.

Request body (JSON):
    {
      "image_b64": "<base64-encoded image bytes>",
      "filename":  "image.png"           # optional, used in response metadata
    }

Response body follows the IO specification response envelope (docs/IO
specifications.md §3.6). The segmentation mask is inlined as a base64-encoded
PNG in the 'segmentation_mask_b64' field.

Environment variables (set by deploy_endpoint.py):
    MODEL_VERSION  Human-readable version string in response metadata.
"""

import base64
import json
import logging
import os
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# ── Make repo packages importable ────────────────────────────────────────────
# Azure ML uploads the full repo as code; __file__ lives at
# {code_root}/scripts/score.py so parent.parent is the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC))

# Prevent npec_cv.__init__ from executing — it pulls in pandas + tf_setup that
# we don't need here and would slow container startup.
if "npec_cv" not in sys.modules:
    _stub = types.ModuleType("npec_cv")
    _stub.__path__ = [str(_SRC / "npec_cv")]
    _stub.__package__ = "npec_cv"
    sys.modules["npec_cv"] = _stub

# ── Keras 2/3 mixed-version compat ───────────────────────────────────────────
# The two registered .keras files were saved with DIFFERENT Keras versions:
#   - root model:   Keras 2 (uses paths like keras.src.engine.functional,
#                   Conv2DTranspose accepts `groups=1`, etc.)
#   - anchor model: Keras 3 (uses paths like keras.src.models.functional,
#                   DTypePolicy, batch_shape, kwargs-style inbound_nodes).
# Neither loader can read the other's format. We try Keras 3 first, fall back
# to tf_keras (Keras 2 backward-compat package) if the modern loader can't
# deserialize. compile=False skips loading the saved optimizer/loss/metrics
# (which we never invoke at inference time).
import keras  # noqa: E402
import npec_cv.inference.inference_manager as _inference_manager  # noqa: E402
import tf_keras  # noqa: E402

logger = logging.getLogger(__name__)


def _load_model_compat(checkpoint, custom_objects=None):
    """Drop-in load_model: try Keras 3 first, fall back to Keras 2."""
    path = str(checkpoint)
    try:
        return keras.models.load_model(path, custom_objects=custom_objects, compile=False)
    except Exception as exc_k3:
        logger.info(
            f"Keras 3 load_model failed for {path} ({exc_k3.__class__.__name__}); "
            f"falling back to tf_keras (Keras 2)"
        )
        return tf_keras.models.load_model(path, custom_objects=custom_objects, compile=False)


_inference_manager.load_model = _load_model_compat

from npec_cv.configs.inference import CUSTOM_OBJECTS  # noqa: E402
from npec_cv.configs.lengths import LENGTH_CALCULATOR_CONFIG, NUM_PLANTS  # noqa: E402
from npec_cv.configs.segmentation import SEGMENTATION_CONFIG  # noqa: E402
from npec_cv.configs.thresholds import ADAPTIVE_THRESHOLD_CONFIG  # noqa: E402
from npec_cv.core.plant_master import PlantMaster  # noqa: E402
from npec_cv.models.inference import ModelConfig  # noqa: E402

# With tf_keras + compile=False the saved loss/metrics aren't deserialized,
# so we no longer need stubs for combined_loss / F1Score. CUSTOM_OBJECTS still
# carries f1/iou/dice_bce_loss in case the model file references them
# elsewhere via Lambda layers or @register_keras_serializable.
CUSTOM_OBJECTS_SCORING = dict(CUSTOM_OBJECTS)

# ── Constants (IO spec §2.3, §2.7, §6) ───────────────────────────────────────
SCALE_FACTOR_MM_PER_PX = 1 / 150  # 150 px/mm → mm per px
ANCHOR_MODEL_FILENAME = "anchor_det_model_512px.keras"
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
MIN_RESOLUTION = 512
MAX_WIDTH = 5120
MAX_HEIGHT = 4096
MIN_PIXEL_STD = 1e-4
MIN_MEAN_INTENSITY = 20
MAX_MEAN_INTENSITY = 235
TIP_CONFIDENCE_RADIUS = 5

# Globals populated once in init()
plant_master: "PlantMaster | None" = None
model_version: str = "unknown"


# ── Lifecycle ─────────────────────────────────────────────────────────────────


def init() -> None:
    """Load both models and build PlantMaster. Called once at container startup."""
    global plant_master, model_version

    # Azure ML's inference server pre-attaches a root logger handler at
    # WARNING level, so a plain basicConfig is a no-op. force=True replaces
    # those handlers; we also pin our module logger to INFO defensively so
    # the DIAGNOSTIC / Calibrated / Injected lines reach the container log
    # stream (the only place we can see them via get_logs.py).
    logging.basicConfig(level=logging.INFO, force=True)
    logger.setLevel(logging.INFO)

    # Both models live inside AZUREML_MODEL_DIR (combined serving artifact
    # registered by deploy_endpoint.py via the combine job).
    model_dir = Path(os.environ.get("AZUREML_MODEL_DIR", "."))
    root_model_path = _find_file(
        model_dir, "root_det_model_512px.keras", legacy_names=("root_model.keras", "root_model")
    )
    anchor_model_path = _find_file(
        model_dir, ANCHOR_MODEL_FILENAME, legacy_names=("anchor_model.keras", "anchor_model")
    )
    logger.info(f"Root model   : {root_model_path}")
    logger.info(f"Anchor model : {anchor_model_path}")

    root_cfg = ModelConfig(
        checkpoint=str(root_model_path),
        threshold=0.01,
        patch_size=512,
        batch_size=4,
        custom_objects=CUSTOM_OBJECTS_SCORING,
    )
    anchor_cfg = ModelConfig(
        checkpoint=str(anchor_model_path),
        threshold=0.001,
        patch_size=512,
        batch_size=4,
        custom_objects=CUSTOM_OBJECTS_SCORING,
    )

    plant_master = PlantMaster(
        inference_config=(root_cfg, anchor_cfg),
        segmentation_config=SEGMENTATION_CONFIG,
        calculator_config=LENGTH_CALCULATOR_CONFIG,
        thresholds_config=ADAPTIVE_THRESHOLD_CONFIG,
    )

    model_version = os.environ.get("MODEL_VERSION", model_dir.name)
    _install_corner_padding()  # match training padding (fixes left-bias)
    _install_synthetic_anchor_fallback()
    if os.environ.get("DISABLE_DISH_CROP", "").lower() in ("1", "true", "yes"):
        _disable_dish_cropping()
    if os.environ.get("DISABLE_FLIP_MERGE", "").lower() not in ("1", "true", "yes"):
        _install_horizontal_flip_merge()
    _disable_shoot_penalty()
    _install_permissive_endpoint_matcher()
    _install_zone_isolated_segmenter()
    _install_fixed_low_threshold()
    logger.info(f"PlantMaster ready — model version: {model_version}")


def _install_fixed_low_threshold() -> None:
    """Skip the adaptive threshold sweep — always use a low fixed threshold.

    The adaptive logic iterates from 0.50 down to 0.05 and returns the FIRST
    threshold where all plants are detected. With the permissive matcher, a
    plant counts as "detected" as soon as any tiny component above
    min_root_area exists in its zone, so the search stops at 0.50 — leaving
    plants with weaker predictions (typically the leftmost and rightmost)
    represented by only a fragment of their real root.

    Diagnostic from a real request:
        zone 0: pixels>0.5=752    → reported root length 137 px (tiny fragment)
        zone 4: pixels>0.5=276    → reported root length 52 px  (tiny fragment)
        zones 1-3: pixels>0.5=10k+ → realistic lengths ~1000 px

    Forcing threshold=0.05 means every plant's mask includes everything the
    model thought was a root with even mild confidence — producing complete,
    realistic root traces.
    """
    import npec_cv.core.plant_master as _pm_mod

    FIXED_THRESHOLD = 0.05

    def fixed_threshold_detect(self, root_probs, anchors):
        results = self._find_endpoints_at_threshold(root_probs, anchors, FIXED_THRESHOLD)
        found = sum(1 for r in results.values() if r["detected"])
        all_found = found == self.num_plants
        logger.info(f"DIAGNOSTIC: Fixed-threshold {FIXED_THRESHOLD} run — {found}/{self.num_plants} detected")
        return FIXED_THRESHOLD, results, all_found

    _pm_mod.PlantMaster.detect_endpoints_adaptive = fixed_threshold_detect
    logger.info(f"DIAGNOSTIC: Fixed low threshold Installed — adaptive sweep skipped (threshold={FIXED_THRESHOLD})")


def _install_permissive_endpoint_matcher() -> None:
    """Replace _find_endpoints_at_threshold with a permissive zone-only matcher.

    The original matcher rejects a root component unless its TOP pixel lies
    inside the anchor's bbox window (in both x and y). We've already widened
    the bbox to span the full plant zone in x and the top half in y, but
    diagnostics show that even with that fix, some zones with thousands of
    activated pixels never produce a detection.

    Symptoms observed (test_image_02.png, 4202x3006):
        zone 3 max=1.0000  pixels>0.5=13258  detected=False  ← bug
        zone 4 max=0.9999  pixels>0.5=276    detected=True

    Root cause is murky — possibly the bbox widening isn't being picked up
    by the running container, or the original function has another constraint
    we haven't identified. Rather than chase it, this replacement skips the
    bbox check entirely and just picks the largest connected component within
    the zone's x-range. The zone constraint alone is enough to disambiguate
    plants because zones are mutually exclusive in x.

    Behaviour:
        - For each zone, build a zone_mask restricted to plant_zones[zid] x range.
        - Find connected components in zone_mask.
        - Filter out components below min_root_area.
        - Pick the largest by area.
        - start = (top_y, top_x) of that component.
        - end = (bottom_y, bottom_x) of that component (max y, mean x there).
    """
    import npec_cv.core.plant_master as _pm_mod

    def permissive_find_endpoints(self, probs, anchors, threshold):
        results = {zid: {"start": None, "end": None, "detected": False} for zid in self.plant_zones.keys()}
        binary_mask = (probs > threshold).astype(np.uint8) * 255
        min_area = self.segmentation_config.min_root_area

        for zid, (zx_min, zx_max) in self.plant_zones.items():
            zone_mask = np.zeros_like(binary_mask, dtype=np.uint8)
            zone_mask[:, zx_min:zx_max] = binary_mask[:, zx_min:zx_max]

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(zone_mask, connectivity=8)

            best_lid = None
            best_area = 0
            for lid in range(1, num_labels):
                area = int(stats[lid, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                if area > best_area:
                    best_area = area
                    best_lid = lid

            if best_lid is None:
                continue

            comp_mask = labels == best_lid
            ys, xs = np.where(comp_mask)
            top_y = int(ys.min())
            top_x = int(np.mean(xs[ys == top_y]))

            # Bottom of root — used by adaptive threshold convergence check.
            y_range = ys.max() - ys.min()
            bottom_thresh = ys.min() + y_range * (1 - self.segmentation_config.y_bot_ratio)
            bottom_ys = ys[ys >= bottom_thresh]
            bottom_xs = xs[ys >= bottom_thresh]
            end_y = int(bottom_ys.max())
            end_x = int(np.mean(bottom_xs[bottom_ys == end_y]))

            results[zid] = {
                "start": (top_y, top_x),
                "end": (end_y, end_x),
                "detected": True,
                "component_mask": comp_mask,
            }

        return results

    _pm_mod.PlantMaster._find_endpoints_at_threshold = permissive_find_endpoints
    logger.info("DIAGNOSTIC: Permissive endpoint matcher Installed — zone-only, no bbox y check")


def _install_zone_isolated_segmenter() -> None:
    """Replace _segment_roots with a version that doesn't let zones eat each other.

    The original _segment_roots, for each zone, dilates the zone's merged
    component and absorbs any other connected component that touches it AND
    intersects the zone's x-range. Then it assigns `zid + 1` to every pixel
    in the merged mask. The next zone does the same. When two adjacent plants'
    roots form a single connected blob in the binary mask (common at low
    adaptive thresholds with branching roots), the blob is absorbed into BOTH
    zones' merges. Zone N labels it N+1, then zone N+1 overwrites those same
    pixels to N+2. analyze_plant_lengths then sees zone N+1's label missing
    from the array and reports plant N+1 as 'not detected', even though
    _find_endpoints_at_threshold correctly identified it.

    This replacement:
      1. Uses the component_mask written by the permissive matcher directly
         (already restricted to the zone's x-range — no cross-zone absorption).
      2. Refuses to overwrite pixels that an earlier zone already claimed.
      3. Sets `end_point` for downstream length calculation.
    """
    import npec_cv.core.plant_master as _pm_mod

    def zone_isolated_segment(self, root_probs, endpoint_info, optimal_threshold):
        root_mask = (root_probs > optimal_threshold).astype(np.uint8) * 255
        filtered_mask = np.zeros_like(root_mask)
        final_labels = np.zeros(root_mask.shape, dtype=np.int32)

        for zid, ep_data in endpoint_info.items():
            if not ep_data.get("detected"):
                continue
            comp_mask = ep_data.get("component_mask")
            if comp_mask is None:
                continue

            # Bool mask of pixels not yet claimed by an earlier zone.
            unlabeled = final_labels == 0
            target = comp_mask & unlabeled
            if not target.any():
                logger.info(f"DIAGNOSTIC: zone {zid} component fully overlaps earlier zone — skipping")
                continue

            filtered_mask[target] = 255
            final_labels[target] = zid + 1

            ys, xs = np.where(target)
            end_y = int(ys.max())
            end_x = int(np.mean(xs[ys == end_y]))
            endpoint_info[zid]["end"] = (end_y, end_x)
            endpoint_info[zid]["threshold_used"] = optimal_threshold

        return filtered_mask, final_labels

    _pm_mod.PlantMaster._segment_roots = zone_isolated_segment
    logger.info("DIAGNOSTIC: Zone-isolated segmenter Installed — no cross-zone label theft")


def _disable_shoot_penalty() -> None:
    """Make _apply_shoot_penalty a passthrough.

    The anchor model runs at threshold=0.001 (so faint shoots still register
    as anchors), but the same broad mask is fed into _apply_shoot_penalty
    which zeroes any root pixel under it with prob < 0.5 and subtracts 0.4
    from those ≥ 0.5. With such a low anchor threshold the mask is so wide
    that legitimate root predictions for plants with fainter roots get
    crushed before the adaptive threshold step ever sees them.

    The pipeline doesn't actually need this penalty: the zone+anchor-bbox
    matching in _find_endpoints_at_threshold already discards root activations
    outside plant zones, and the leaf/shoot area sits above the root start so
    it doesn't compete for the "top of root component" position.

    Set DISABLE_SHOOT_PENALTY=0 to restore the original behaviour.
    """
    if os.environ.get("DISABLE_SHOOT_PENALTY", "").lower() in ("0", "false", "no"):
        return
    plant_master._apply_shoot_penalty = lambda root_probs, anchor_mask: root_probs
    logger.info("DIAGNOSTIC: Shoot penalty Disabled — root_probs passed through unchanged")


def _install_corner_padding() -> None:
    """Replace the centred padder with a top-left-corner padder to match training.

    Training (scripts/train_root_detection.py::_pad_to_multiple) puts the
    image in the top-left and pads the bottom+right with zeros. Inference
    (src/npec_cv/helpers/padder.py::padder) splits padding evenly on all
    sides. That mismatch means inference patches at the LEFT and TOP edges
    contain zero-pad rows the model never saw during training, which causes
    the model to output near-zero on those edges. We restore the training
    behaviour at inference time.
    """

    def _corner_padder(image, patch_size):
        h, w = image.shape[:2]
        ph = ((h + patch_size - 1) // patch_size) * patch_size
        pw = ((w + patch_size - 1) // patch_size) * patch_size
        if image.ndim == 2:
            out = np.zeros((ph, pw), dtype=image.dtype)
        else:
            out = np.zeros((ph, pw, image.shape[2]), dtype=image.dtype)
        out[:h, :w] = image
        return out

    # The `from ... import padder` in inference_manager.py captured a local
    # reference to the original function, so patching the helpers module
    # alone is not enough — we must rebind the reference inside
    # inference_manager itself.
    import npec_cv.helpers.padder as _padder_module
    import npec_cv.inference.inference_manager as _im

    _padder_module.padder = _corner_padder
    _im.padder = _corner_padder
    logger.info("Padding aligned with training (top-left corner, not centred)")


def _install_horizontal_flip_merge() -> None:
    """Symmetrize the biased models by running inference twice.

    The root + anchor models have a strong right-side bias (the diagnostics
    showed ~6x higher activation on the right vs the left for the same image).
    This wrapper runs PlantMaster._run_inference once on the original image
    and once on a horizontally-flipped copy, un-flips the latter, and merges
    via elementwise max. The effective predictions become left/right
    symmetric.

    Cost: ~2x inference time (~60-70s per request instead of ~30s). Stays
    well under the 180s request_timeout_ms.

    Disable by setting DISABLE_FLIP_MERGE=1.
    """
    if getattr(plant_master, "_flip_merge_installed", False):
        return

    original_run = plant_master._run_inference  # captured bound method

    def run_inference_with_flip(img):
        # Pass 1: original orientation
        root_p, anchor_p = original_run(img)
        # Pass 2: horizontally flipped
        flipped = np.ascontiguousarray(np.fliplr(img))
        root_p_flip, anchor_p_flip = original_run(flipped)
        # Un-flip the second result and merge with elementwise max
        merged_root = np.maximum(root_p, np.fliplr(root_p_flip))
        merged_anchor = np.maximum(anchor_p, np.fliplr(anchor_p_flip))
        return merged_root, merged_anchor

    plant_master._run_inference = run_inference_with_flip
    plant_master._flip_merge_installed = True
    logger.info("Horizontal-flip merge inference installed (2x inference per request)")


def _disable_dish_cropping() -> None:
    """Replace extract_petri_dish with a no-op that returns full-image bounds.

    Enabled by env var DISABLE_DISH_CROP=1. Useful when the petri-dish
    detection is mis-cropping (e.g. excluding half the image) and the input
    image is already framed correctly.
    """
    import npec_cv.core.plant_master as _pm
    import npec_cv.helpers.extract_petri_dish as _eh
    import npec_cv.inference.inference_manager as _im

    def _full_image_crop(img, padding=30, return_coords=False):
        h, w = img.shape[:2]
        if return_coords:
            return (0, 0, w, h)
        return img.copy()

    _eh.extract_petri_dish = _full_image_crop
    _im.extract_petri_dish = _full_image_crop
    _pm.extract_petri_dish = _full_image_crop
    logger.info("Dish cropping disabled — using full image bounds")


def _log_pipeline_diagnostics(result) -> None:
    """Log spatial stats so we can tell whether the root model fired evenly."""
    if result.root_probs is None:
        logger.info("DIAGNOSTIC: root_probs is None")
        return
    rp = result.root_probs
    h, w = rp.shape[:2]
    # Per-quadrant max + mean to spot left/right asymmetry
    half_w = w // 2
    left, right = rp[:, :half_w], rp[:, half_w:]
    logger.info(
        f"DIAGNOSTIC: root_probs shape={rp.shape} "
        f"left:  max={left.max():.4f}  mean={left.mean():.5f}  "
        f"right: max={right.max():.4f}  mean={right.mean():.5f}"
    )
    if result.anchor_mask is not None:
        am = result.anchor_mask
        logger.info(
            f"DIAGNOSTIC: anchor_mask nonzero_left={int((am[:, :half_w] > 0).sum())}  "
            f"nonzero_right={int((am[:, half_w:] > 0).sum())}"
        )
    logger.info(f"DIAGNOSTIC: anchors detected at " f"{[(a.zone_id, a.centroid) for a in result.anchors]}")
    # Per-zone activation stats — tells us if a missing plant is because the
    # model didn't fire in that zone vs. matching threw it away.
    for zid in sorted(plant_master.plant_zones.keys()):
        zx_min, zx_max = plant_master.plant_zones[zid]
        zone_probs = rp[:, zx_min:zx_max]
        if zone_probs.size == 0:
            continue
        n_above_005 = int((zone_probs > 0.05).sum())
        n_above_050 = int((zone_probs > 0.50).sum())
        detected = zid in result.plant_results and result.plant_results[zid].detected
        logger.info(
            f"DIAGNOSTIC: zone {zid} x=[{zx_min},{zx_max}] "
            f"max={float(zone_probs.max()):.4f}  mean={float(zone_probs.mean()):.5f}  "
            f"pixels>0.05={n_above_005}  pixels>0.5={n_above_050}  "
            f"detected={detected}"
        )


def _calibrate_for_image(img_bgr: np.ndarray) -> None:
    """Adapt plant zones + anchor sensitivity to the actual image.

    The default plant_zones in configs/segmentation.py are hardcoded pixel
    ranges (890-3400) calibrated for a specific HADES reference size. When the
    image dimensions differ, plants outside those ranges never get matched to
    a zone and their roots silently drop out of the output mask.

    This helper:
      1. Locates the petri dish via extract_petri_dish.
      2. Distributes 5 evenly-spaced plant zones across the detected dish width.
      3. Lowers min_anchor_area so small shoots aren't filtered out.

    Mutates the global PlantMaster's segmentation config in place. Safe because
    Azure ML deployments run with max_concurrent_requests_per_instance=1.
    """
    h, w = img_bgr.shape[:2]

    # When DISABLE_DISH_CROP is on, calibrate against the full image. The
    # monkey-patch in _disable_dish_cropping can't intercept this code path
    # because npec_cv/helpers/__init__.py re-exports the function under the
    # same name as the submodule, shadowing it in attribute lookups.
    disable_env = os.environ.get("DISABLE_DISH_CROP", "")
    if disable_env.lower() in ("1", "true", "yes"):
        logger.info(f"DIAGNOSTIC: dish cropping bypassed via DISABLE_DISH_CROP={disable_env!r}")
        x1, x2, dish_w = 0, w, w
    else:
        logger.info(f"DIAGNOSTIC: dish cropping path active (DISABLE_DISH_CROP={disable_env!r})")
        from npec_cv.helpers.extract_petri_dish import extract_petri_dish

        try:
            ex1, ey1, ex2, ey2 = extract_petri_dish(img_bgr, return_coords=True)
            logger.info(
                f"DIAGNOSTIC: extract_petri_dish returned ({ex1},{ey1},{ex2},{ey2}) "
                f"for image {w}x{h} — crop_w={ex2-ex1}, crop_h={ey2-ey1}, "
                f"crop_area_pct={100*(ex2-ex1)*(ey2-ey1)/(w*h):.1f}%"
            )
            x1, x2 = ex1, ex2
            dish_w = max(1, x2 - x1)
        except Exception as exc:
            # Fallback: use ~90% of image width if dish detection fails
            logger.warning(f"DIAGNOSTIC: extract_petri_dish raised {exc!r} — using 5-95% fallback")
            x1, x2, dish_w = int(w * 0.05), int(w * 0.95), int(w * 0.90)

    # Inset 5% on each side so zones don't bleed onto the dish rim
    inset = int(dish_w * 0.05)
    span_start = x1 + inset
    span_end = x2 - inset
    zone_w = (span_end - span_start) // 5

    new_zones = {i: (span_start + i * zone_w, span_start + (i + 1) * zone_w) for i in range(5)}

    # Mutate in place — both PlantMaster.plant_zones and segmentor.config.plant_zones
    # reference the same dict in normal construction, but mutate both defensively.
    seg_cfg = plant_master.segmentation_config
    seg_cfg.plant_zones.clear()
    seg_cfg.plant_zones.update(new_zones)
    plant_master.plant_zones.clear()
    plant_master.plant_zones.update(new_zones)

    # Smaller / fainter shoots in the source images get filtered out at the
    # default min_anchor_area=100 / min_root_area=50. Lower both so the
    # leftmost plants — where the root model fires more sparsely — still
    # register as connected components.
    seg_cfg.min_anchor_area = 30
    seg_cfg.min_root_area = 20

    logger.info(
        f"Calibrated to image {w}x{h}: dish=[{x1},{x2}] (w={dish_w}px), "
        f"zone_w={zone_w}px, min_anchor_area={seg_cfg.min_anchor_area}, "
        f"min_root_area={seg_cfg.min_root_area}"
    )


def _install_synthetic_anchor_fallback() -> None:
    """Monkey-patch extract_anchors to synthesize anchors for zones the model misses.

    The anchor (shoot) model trained on the team's data sometimes fails to fire
    on small/faint shoots, which makes the root pipeline drop those plants
    entirely. This wrapper, after the real anchor model runs, fills in any
    zone that lacks an anchor with a synthetic anchor placed at the zone
    centre, ~15 % from the top of the image. The root model can then segment
    roots inside that zone as usual.

    Caveat: this also produces synthetic anchors for *truly empty* zones, so
    you'll get phantom 'detections' for slots that have no plant. Accept it
    as a trade-off vs. losing real plants.
    """
    from npec_cv.models import Anchor

    if getattr(plant_master.segmentor, "_synthetic_fallback_installed", False):
        return

    original_extract = plant_master.segmentor.extract_anchors
    original_create_bboxes = plant_master.segmentor.create_anchor_bboxes

    def extract_with_synthetic_fallback(anchor_mask):
        anchors = original_extract(anchor_mask)
        detected_zones = {a.zone_id for a in anchors if a.zone_id is not None}
        h_img, w_img = anchor_mask.shape[:2]
        synthetic_y = int(h_img * 0.15)  # roughly where shoots sit
        for zid, (zx_min, zx_max) in plant_master.plant_zones.items():
            if zid in detected_zones:
                continue
            cx = (zx_min + zx_max) // 2
            anchors.append(
                Anchor(
                    label_id=-1 - zid,  # negative => synthetic
                    centroid=(cx, synthetic_y),
                    area=100,
                    mask=np.zeros_like(anchor_mask, dtype=np.uint8),
                    zone_id=zid,
                    bbox=None,  # filled by create_anchor_bboxes
                )
            )
            logger.info(f"Injected synthetic anchor for zone {zid} at ({cx},{synthetic_y})")
        return anchors

    def create_bboxes_with_wide_synthetic(anchors, img_shape, bbox_size=None, vertical_offset=None):
        # Widen EVERY anchor's bbox to span the entire plant zone in x AND the
        # top half of the image in y. The 150-px default window was designed
        # to disambiguate multiple shoots in one zone, but the pipeline keeps
        # only one anchor per zone anyway — so any root tip inside the zone
        # is a valid match. The narrow bbox just causes plants to be missed
        # when:
        #   - the synthetic anchor (placed at y=0.15*H) sits 100+ px below
        #     where the real shoot/root junction actually is, so the matcher
        #     in _find_endpoints_at_threshold rejects the root component
        #     because its top pixel falls outside [y_min-bbox_tol, y_max+2*bbox_tol].
        #   - a real anchor lands slightly above/below the root start.
        # The score = area / (dist + 1) ranking later picks the best component
        # per zone, so widening can't cause cross-matching between zones.
        h_img = img_shape[0]
        result = original_create_bboxes(anchors, img_shape, bbox_size, vertical_offset)
        for a in result:
            if a.zone_id is None:
                continue
            zx_min, zx_max = plant_master.plant_zones[a.zone_id]
            a.bbox = (zx_min, 0, zx_max, h_img // 2)
        return result

    plant_master.segmentor.extract_anchors = extract_with_synthetic_fallback
    plant_master.segmentor.create_anchor_bboxes = create_bboxes_with_wide_synthetic
    plant_master.segmentor._synthetic_fallback_installed = True


def _find_file(model_dir: Path, canonical: str, legacy_names: tuple = ()) -> Path:
    """Find a .keras file inside AZUREML_MODEL_DIR.

    Tries canonical and legacy names, first as direct children then with a
    recursive search (Azure sometimes mounts files inside an INPUT_* subdir).
    """
    candidates = (canonical, *legacy_names)
    for name in candidates:
        direct = model_dir / name
        if direct.exists():
            return direct
    for name in candidates:
        matches = list(model_dir.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Model file not found under {model_dir}. Expected one of: {', '.join(candidates)}")


# ── Request handler ───────────────────────────────────────────────────────────


def run(raw_data: str) -> str:
    """Process one inference request — returns JSON per IO spec §3.6."""
    t_start = time.perf_counter()

    try:
        request = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        return _error("INVALID_REQUEST", f"Request body is not valid JSON: {exc}")

    image_b64 = request.get("image_b64", "")
    filename = request.get("filename", "image.png")

    if not image_b64:
        return _error("INVALID_REQUEST", "Field 'image_b64' is required.")

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return _error("INVALID_REQUEST", "Field 'image_b64' is not valid base64.")

    # ── Input validation (IO spec §6) ────────────────────────────────────────

    ext = Path(filename).suffix.lower()
    if ext not in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
        return _error("UNSUPPORTED_FORMAT", f"File extension '{ext}' is not supported.")

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(image_bytes) / (1024 * 1024)
        return _error("FILE_TOO_LARGE", f"File size {size_mb:.1f} MB exceeds maximum allowed 50 MB.")

    img = _decode_image(image_bytes)
    if img is None:
        return _error("CORRUPT_FILE", "File could not be decoded. Ensure it is a valid PNG, TIFF, or JPEG.")

    h, w = img.shape[:2]
    if not (MIN_RESOLUTION <= w <= MAX_WIDTH and MIN_RESOLUTION <= h <= MAX_HEIGHT):
        return _error(
            "INVALID_RESOLUTION",
            f"Image resolution {w}x{h} is outside the allowed range "
            f"width [{MIN_RESOLUTION}, {MAX_WIDTH}] x height [{MIN_RESOLUTION}, {MAX_HEIGHT}] px.",
        )

    if img.std() < MIN_PIXEL_STD:
        return _error("EMPTY_IMAGE", "Input image has no content (pixel variance ≈ 0). Check imaging system.")

    warnings = []
    mean_intensity = float(img.mean())
    if not (MIN_MEAN_INTENSITY <= mean_intensity <= MAX_MEAN_INTENSITY):
        warnings.append(f"LOW_CONTRAST: mean_intensity={mean_intensity:.1f} — segmentation quality may be reduced")

    input_channels = 1 if (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)) else 3
    bgr_img = _to_bgr(img)

    # The training pipeline (train_root_detection.py:131 and
    # evaluate_root_detection.py:244/267) converts BGR -> RGB before feeding
    # the model, so we must do the same here or the R and B channels are
    # silently swapped at inference time.
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # ── Run pipeline ─────────────────────────────────────────────────────────
    _calibrate_for_image(bgr_img)
    try:
        result = plant_master.process_image(rgb_img, image_name=filename)
    except Exception as exc:
        logger.exception("PlantMaster.process_image failed")
        return _error("INFERENCE_ERROR", f"Inference failed: {exc}")

    _log_pipeline_diagnostics(result)

    processing_ms = int((time.perf_counter() - t_start) * 1000)

    # ── Build plants array (IO spec §3.2, §3.3, §3.4) ────────────────────────
    plants = []
    confidences = []

    for pid in range(NUM_PLANTS):
        pr = result.plant_results.get(pid)
        if pr is None or not pr.detected:
            warnings.append(f"PLANT_NOT_DETECTED: plant_id={pid + 1}")
            continue

        tip_y, tip_x = pr.end_point if pr.end_point else (0, 0)
        confidence = _tip_confidence(result.root_probs, tip_y, tip_x)
        length_px = float(pr.length) if pr.length else 0.0
        length_mm = round(length_px * SCALE_FACTOR_MM_PER_PX, 3)

        plants.append(
            {
                "plant_id": pid + 1,  # 1-indexed per IO spec §3.2
                "tip_x": int(tip_x),
                "tip_y": int(tip_y),
                "confidence": round(confidence, 4),
                "root_length_px": round(length_px, 2),
                "root_length_mm": length_mm,
            }
        )
        confidences.append(confidence)

    if not plants:
        warnings.append("NO_PLANTS_DETECTED")

    overall_confidence = round(float(np.mean(confidences)), 4) if confidences else 0.0

    if 0 < overall_confidence < 0.5:
        warnings.append(f"LOW_OVERALL_CONFIDENCE: {overall_confidence} — flag for manual review")

    # ── Encode mask as base64 PNG (IO spec §3.1) ─────────────────────────────
    mask_b64 = _encode_mask(result.filtered_mask, target_shape=(h, w))

    # ── Metadata (IO spec §3.5) ──────────────────────────────────────────────
    patch_count = _estimate_patch_count(result.root_probs, patch_size=512)

    response = {
        "status": "ok",
        "metadata": {
            "model_version": model_version,
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": processing_ms,
            "patch_count": patch_count,
            "input_resolution": [w, h],
            "input_channels": input_channels,
            "normalisation_applied": True,
            "scale_factor_mm_per_px": round(SCALE_FACTOR_MM_PER_PX, 5),
        },
        "segmentation_mask_b64": mask_b64,
        "plants": plants,
        "overall_confidence": overall_confidence,
        "warnings": warnings,
    }

    return json.dumps(response)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _decode_image(image_bytes: bytes) -> "np.ndarray | None":
    """Decode raw image bytes. Strips alpha channel if present."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure image is BGR 3-channel as expected by PlantMaster."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img.squeeze(-1), cv2.COLOR_GRAY2BGR)
    return img


def _tip_confidence(root_probs: "np.ndarray | None", y: int, x: int) -> float:
    """Mean root probability in a small window around the detected tip."""
    if root_probs is None:
        return 0.0
    h, w = root_probs.shape[:2]
    r = TIP_CONFIDENCE_RADIUS
    patch = root_probs[
        max(0, y - r) : min(h, y + r + 1),
        max(0, x - r) : min(w, x + r + 1),
    ]
    return float(patch.mean()) if patch.size > 0 else 0.0


def _encode_mask(mask: "np.ndarray | None", target_shape: tuple) -> str:
    """Encode segmentation mask as a base64 PNG (single channel, 8-bit, 0/255).

    target_shape is (H, W) of the original input image — the mask is resized
    to match it if PlantMaster returned a different shape (defensive only).
    """
    if mask is None:
        return ""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    if mask_uint8.shape[:2] != target_shape:
        mask_uint8 = cv2.resize(mask_uint8, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    _, buf = cv2.imencode(".png", mask_uint8)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _estimate_patch_count(root_probs: "np.ndarray | None", patch_size: int) -> int:
    """Estimate how many patches were processed based on the probability map."""
    if root_probs is None:
        return 0
    h, w = root_probs.shape[:2]
    return ((h + patch_size - 1) // patch_size) * ((w + patch_size - 1) // patch_size)


def _error(error_code: str, message: str) -> str:
    """Return a structured JSON error response per IO spec edge case format."""
    return json.dumps(
        {
            "status": "error",
            "error_code": error_code,
            "message": message,
            "plants": [],
            "overall_confidence": 0.0,
        }
    )
