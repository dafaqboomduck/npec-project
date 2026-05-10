Architecture and Design Rationale
==================================

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

NPEC CV is split into two cooperating Python packages:

- **npec_cv** — the computer vision core (segmentation model, image
  processing, CLI, and FastAPI service).
- **npec_robotics** — the robotics control layer that depends on
  ``npec_cv`` for real-time feedback during automated phenotyping runs.

This separation lets the CV pipeline be installed and used independently
(e.g. on a data-analysis workstation) without pulling in robotics
dependencies.

Package layout (``src/npec_cv``)
---------------------------------

.. code-block:: text

   npec_cv/
   ├── __init__.py          # Exposes CVPipeline at the package level
   ├── artifacts/           # Bundled pre-trained Keras models (.keras)
   ├── configs/             # Dataclass-based configuration objects
   ├── core/                # Image pre-processing and post-processing
   ├── exceptions/          # Custom exception hierarchy
   ├── helpers/             # Vendored patchify + miscellaneous utilities
   ├── inference/           # Model loading, patch inference, mask stitching
   └── models/              # Keras model definitions / factory functions

The CLI entry point (``npec-cv``) lives in ``npec_cv_cli/`` — a sibling
package that imports ``npec_cv`` only inside command bodies. This avoids
loading TensorFlow on every invocation when only ``--help`` is needed.

Why TensorFlow and Keras?
--------------------------

The segmentation model is a U-Net variant trained with Keras 3. TF/Keras
was chosen because:

- The existing pre-trained checkpoints were authored in TF.
- ``keras~=3.x`` supports both TF and JAX backends, giving a migration
  path to JAX for faster inference in future.
- The NPEC lab already operated TF-based pipelines, minimising retraining
  effort.

Patch-based inference
----------------------

Plant root images are typically large (2000 x 2000 px or more). Running the
U-Net on the full image would exceed GPU memory on common hardware. The
``helpers.patchify`` module (vendored from an upstream library that had
stale NumPy < 2 pins) divides each image into overlapping tiles, runs
inference on each tile independently, and reassembles the mask with an
overlap-average blend to suppress boundary artefacts.

The patchify helper was vendored rather than monkey-patched upstream so
that NumPy 2.x compatibility is maintained without forking the dependency.

Post-processing graph
----------------------

After the U-Net produces a probability map, the following steps convert
it to a structured measurement:

- **Threshold** — binarise at a configurable confidence cutoff.
- **Skeletonise** — reduce the binary mask to a 1-pixel-wide skeleton
  (``skimage.morphology.skeletonize``).
- **Graph extraction** — model the skeleton as a NetworkX graph where
  nodes are branch points and tips, and edges carry pixel-length weights.
- **Measurement** — derive total root length, number of lateral roots,
  tip count, and average internode distance from the graph.

Using a graph representation makes it straightforward to add new
morphological metrics without changing earlier pipeline stages.

Configuration via dataclasses
-------------------------------

All tuneable parameters (tile size, overlap, threshold, model path) are
declared as Python dataclasses in ``npec_cv.configs``. This gives:

- Type checking at construction time (via ``__post_init__`` validators).
- A single source of truth that is shared between the CLI, the API, and
  the Python API.
- Easy serialisation to / from JSON for experiment reproducibility.

Two-package distribution strategy
-----------------------------------

Both ``npec_cv`` and ``npec_robotics`` are published as separate wheels.
This means a user who only needs CV analysis does not download robotics
drivers, and a robotics deployment does not need the full development
toolchain included in ``npec_cv[dev]``.

Database persistence
---------------------

The FastAPI service uses SQLAlchemy + Alembic to persist inference
results in PostgreSQL. The schema is intentionally minimal: each
prediction run stores the image hash, timestamp, and the measurements
JSON blob. No pixel data is written to the database; it is always
re-derivable from the source image.

This design prioritises storage efficiency and keeps the database schema
stable across model updates (measurements JSON is schema-flexible).
