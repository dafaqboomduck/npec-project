# ILO 10.3: Observability, Traceability, and Frontend

**Requirement**: Present deployed application outcomes through a monitoring dashboard and a user-friendly frontend with visualisations that convey business insights to stakeholders.

## Evidence in the Project

### 1. Stakeholder-Facing Frontend Dashboard
The project includes a robust React/Vite/TypeScript frontend application (`frontend/`) designed to interact with the backend API:
- **Image Uploader**: Provides an intuitive drag-and-drop interface for users to submit Petri dish images.
- **Interactive Visualisations**: Features an interactive segmentation overlay allowing stakeholders to toggle the model's predictions over the original image.
- **Business Insights**: Displays historical prediction tables and active job statuses.

### 2. Monitoring and Traceability
- **Database-Backed Audit Logging**: Every inference request is logged in the `inference_audit_log` table via SQLAlchemy. This tracks latency, environment, and the specific model version used to serve the request.
- **Lineage Traceability CLI**: The `lineage/traceability.py` module provides strict auditing capabilities. By querying a prediction UUID, the system traces the exact lineage path: from the deployed model version, back to the Git commit hash, the specific training job ID, and the dataset versions used (`cv5_train:1.0.0`).

### Summary
The system provides both the user-friendly interface required for stakeholder interaction and the rigorous backend observability necessary for production environments. The frontend effectively visualizes the outcomes of the computer vision models, while the lineage tools ensure full accountability.
