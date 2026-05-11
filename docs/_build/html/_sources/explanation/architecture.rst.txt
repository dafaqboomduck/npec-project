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
without pulling in robotics dependencies.

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
   └── models/              # Data models and result types

The CLI entry point (``npec-cv``) lives in ``npec_cv_cli/`` — a sibling
package that imports ``npec_cv`` only inside command bodies. This avoids
loading TensorFlow on every invocation when only ``--help`` is needed.

Why TensorFlow and Keras?
--------------------------

- The existing pre-trained checkpoints were authored in TF.
- ``keras~=3.x`` supports both TF and JAX backends, giving a migration
  path to JAX for faster inference in future.
- The NPEC lab already operated TF-based pipelines, minimising retraining
  effort.

Patch-based inference
----------------------

Plant root images are typically large (2000 x 2000 px or more). Running the
U-Net on the full image would exceed GPU memory on common hardware. The
``helpers.patchify`` module divides each image into overlapping tiles, runs
inference on each tile independently, and reassembles the mask with an
overlap-average blend to suppress boundary artefacts.

The patchify helper was vendored from an upstream library to maintain
NumPy 2.x compatibility without forking the dependency.

Post-processing graph
----------------------

After the U-Net produces a probability map:

- **Threshold** — binarise at a configurable confidence cutoff.
- **Skeletonise** — reduce the binary mask to a 1-pixel-wide skeleton.
- **Graph extraction** — model the skeleton as a NetworkX graph where
  nodes are branch points and tips, edges carry pixel-length weights.
- **Measurement** — derive total root length, lateral root count,
  tip count, and average internode distance from the graph.

Configuration via dataclasses
-------------------------------

All tuneable parameters are declared as Python dataclasses in
``npec_cv.configs``. This gives type checking at construction time,
a single source of truth shared between CLI, API, and Python API,
and easy serialisation to JSON for experiment reproducibility.

Database persistence
---------------------

The FastAPI service uses SQLAlchemy + Alembic to persist inference
results in PostgreSQL. Each prediction run stores the image hash,
timestamp, and measurements JSON blob. No pixel data is written to
the database — it is always re-derivable from the source image.
