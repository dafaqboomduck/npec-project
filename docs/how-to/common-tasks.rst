Common Tasks
============

.. contents:: On this page
   :local:
   :depth: 2

Batch-process a directory of images
-------------------------------------

Use a shell glob or a small Python script to run ``npec-cv predict`` over
every image in a folder:

.. code-block:: bash

   for img in data/images/*.png; do
       npec-cv predict --image "$img" --output results/
   done

Or from Python using the public API:

.. code-block:: python

   from pathlib import Path
   from npec_cv.inference.pipeline import CVPipeline

   pipeline = CVPipeline()
   for img_path in Path("data/images").glob("*.png"):
       result = pipeline.run(str(img_path))
       result.save(f"results/{img_path.stem}")

Adjust the segmentation threshold
-----------------------------------

The default binarisation threshold is chosen for the bundled model. To
tune it for your imaging conditions, pass ``--threshold``:

.. code-block:: bash

   npec-cv predict --image root.png --output results/ --threshold 0.4

Lower values keep more pixels (higher recall); higher values discard
borderline predictions (higher precision).

Use the FastAPI inference endpoint
------------------------------------

The repository ships a FastAPI service for programmatic access. Start it with:

.. code-block:: bash

   uvicorn api.main:app --host 0.0.0.0 --port 8000

Then send an image via HTTP:

.. code-block:: bash

   curl -X POST http://localhost:8000/predict \
        -F "file=@root_image.png" \
        --output mask.png

The ``/health`` endpoint returns ``{"status": "ok"}`` when the model is
loaded and ready.

Run the service with Docker Compose
--------------------------------------

A ``docker-compose.yml`` is provided at the repository root:

.. code-block:: bash

   docker-compose up --build

This starts the API service together with the PostgreSQL database used
for persisting inference results.

Load a custom model
---------------------

To substitute your own Keras model, set the ``MODEL_PATH`` environment
variable before starting the CLI or API:

.. code-block:: bash

   MODEL_PATH=/path/to/my_model.keras npec-cv predict --image root.png --output out/

The model must accept ``(batch, H, W, 1)`` float32 inputs and output a
``(batch, H, W, 1)`` sigmoid mask.

Export measurements to JSON
-----------------------------

By default measurements are written as CSV. Use ``--format json`` for
structured output that integrates easily with downstream services:

.. code-block:: bash

   npec-cv predict --image root.png --output results/ --format json

Enable verbose logging
------------------------

Set the ``LOG_LEVEL`` environment variable to ``DEBUG`` to see per-patch
timing and model confidence statistics:

.. code-block:: bash

   LOG_LEVEL=DEBUG npec-cv predict --image root.png --output results/
