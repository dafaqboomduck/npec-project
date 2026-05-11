Getting Started
===============

This tutorial walks you through installing NPEC CV and running your first
root segmentation on a sample image. By the end you will have a working
pipeline that produces a segmented mask and exports key morphological
measurements.

Prerequisites
-------------

- Python 3.10 or newer
- A CUDA-capable GPU is recommended; CPU-only inference is supported but slower.
- ``git`` installed on your machine.

Step 1 — Clone the repository
------------------------------

.. code-block:: bash

   git clone https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5.git
   cd 2025-26d-fai2-adsai-group-computervision5

Step 2 — Install NPEC CV
-------------------------

Install the core package in editable mode:

.. code-block:: bash

   pip install -e src/npec_cv

For NVIDIA GPU support:

.. code-block:: bash

   pip install -e src/npec_cv[cuda]

Step 3 — Verify the installation
---------------------------------

.. code-block:: bash

   npec-cv --help

You should see the top-level help text listing available sub-commands.

Step 4 — Run inference on a sample image
-----------------------------------------

.. code-block:: bash

   npec-cv predict --image path/to/root_image.png --output results/

The pipeline will:

- Load and pre-process the image (resize, normalise).
- Tile the image into overlapping patches.
- Run each patch through the segmentation model.
- Stitch the predictions back into a full-resolution mask.
- Post-process the mask (skeletonise, extract graph topology).
- Write the binary mask and a CSV of measurements to ``results/``.

Step 5 — Inspect the outputs
-----------------------------

.. code-block:: text

   results/
   ├── mask.png          # Binary segmentation mask
   └── measurements.csv  # Root length, branching points, tip count

.. code-block:: python

   import pandas as pd
   df = pd.read_csv("results/measurements.csv")
   print(df.head())

Next steps
----------

- :doc:`/how-to/common-tasks` — batch processing, thresholds, export formats.
- :doc:`/explanation/architecture` — design decisions and pipeline structure.
