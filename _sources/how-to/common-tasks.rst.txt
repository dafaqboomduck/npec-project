Common Tasks
============

.. contents:: On this page
   :local:
   :depth: 2

Batch-process a directory of images
-------------------------------------

.. code-block:: bash

   for img in data/images/*.png; do
       npec-cv predict --image "$img" --output results/
   done

Or from Python:

.. code-block:: python

   from pathlib import Path
   from npec_cv.pipeline import CVPipeline

   pipeline = CVPipeline()
   for img_path in Path("data/images").glob("*.png"):
       result = pipeline.run(str(img_path))
       result.save(f"results/{img_path.stem}")

Adjust the segmentation threshold
-----------------------------------

.. code-block:: bash

   npec-cv predict --image root.png --output results/ --threshold 0.4

Lower values keep more pixels (higher recall); higher values discard
borderline predictions (higher precision).

Use the FastAPI inference endpoint
------------------------------------

Start the service:

.. code-block:: bash

   uvicorn api.main:app --host 0.0.0.0 --port 8000

Send an image:

.. code-block:: bash

   curl -X POST http://localhost:8000/predict \
        -F "file=@root_image.png" \
        --output mask.png

The ``/health`` endpoint returns ``{"status": "ok"}`` when ready.

Run the service with Docker Compose
--------------------------------------

.. code-block:: bash

   docker-compose up --build

This starts the API service together with the PostgreSQL database.

Load a custom model
---------------------

.. code-block:: bash

   MODEL_PATH=/path/to/my_model.keras npec-cv predict --image root.png --output out/

The model must accept ``(batch, H, W, 1)`` float32 inputs and output a
``(batch, H, W, 1)`` sigmoid mask.

Export measurements to JSON
-----------------------------

.. code-block:: bash

   npec-cv predict --image root.png --output results/ --format json

Enable verbose logging
------------------------

.. code-block:: bash

   LOG_LEVEL=DEBUG npec-cv predict --image root.png --output results/
