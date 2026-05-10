Getting Started
===============

This tutorial walks you through installing NPEC CV and running your first
root segmentation on a sample image. By the end you will have a working
pipeline that produces a segmented mask and exports key morphological
measurements.

Prerequisites
-------------

- Python 3.10 or newer
- A CUDA-capable GPU is recommended for training; CPU-only inference is
  supported but will be slower.
- ``git`` installed on your machine.

Step 1 — Clone the repository
------------------------------

.. code-block:: bash

   git clone https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5.git
   cd 2025-26d-fai2-adsai-group-computervision5

Step 2 — Install NPEC CV
-------------------------

Install the core package in editable mode so that local changes are
immediately reflected:

.. code-block:: bash

   pip install -e src/npec_cv

For NVIDIA GPU support, install the CUDA extras instead:

.. code-block:: bash

   pip install -e src/npec_cv[cuda]

Step 3 — Verify the installation
---------------------------------

Run the CLI health check to confirm the package and its dependencies are
correctly installed:

.. code-block:: bash

   npec-cv --help

You should see the top-level help text listing available sub-commands
(``predict``, ``version``, etc.).

Step 4 — Run inference on a sample image
-----------------------------------------

NPEC CV ships with a pre-trained Keras model in ``src/npec_cv/artifacts/``.
Use the ``predict`` command to segment a plant root image:

.. code-block:: bash

   npec-cv predict --image path/to/root_image.png --output results/

The pipeline will:

- Load and pre-process the image (resize, normalise).
- Tile the image into overlapping patches using the built-in patchify helper.
- Run each patch through the segmentation model.
- Stitch the predictions back into a full-resolution mask.
- Post-process the mask (skeletonise, extract graph topology).
- Write the binary mask and a CSV of measurements to ``results/``.

Step 5 — Inspect the outputs
-----------------------------

After a successful run, ``results/`` contains:

.. code-block:: text

   results/
   ├── mask.png          # Binary segmentation mask
   └── measurements.csv  # Root length, branching points, tip count, …

Open ``measurements.csv`` in any spreadsheet application or with pandas:

.. code-block:: python

   import pandas as pd
   df = pd.read_csv("results/measurements.csv")
   print(df.head())

Next steps
----------

- :doc:`/how-to/common-tasks` — learn how to batch-process directories,
  adjust model thresholds, and export results in different formats.
- :doc:`/explanation/architecture` — understand how the pipeline is
  structured and why key design decisions were made.
