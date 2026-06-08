# ILO 4.4: Cloud Architecture and Financial Cost Evaluation

**Requirement**: Design and plan for a cloud-based ML application architecture and data pipeline. Evaluation of financial costs with cost-effective resource management.

## Evidence in the Project

### 1. System Architecture Design
The platform architecture is fully documented and designed as a modular, decoupled set of services:
- **Visual Evidence**: [Project Diagram.drawio](../Project%20Diagram.drawio) and [docs/architecture/data_pipeline.drawio](../docs/architecture/data_pipeline.drawio) provide visual maps of the system.
- **Frontend**: A React (Vite + TS) application ([frontend/](../frontend)).
- **Backend API**: A FastAPI application ([api/](../api)) that handles REST requests and database interactions.
- **Database**: PostgreSQL database ([db/](../db)) with SQLAlchemy ORM for metadata, user accounts, and tracking.
- **Lineage Core**: A dedicated module ([lineage/](../lineage)) for tracking model provenance and inference audit logs.
- **ML Deployment**: Azure ML Workspace for model registry and a Kubernetes Endpoint for inference.

### 2. Data Pipeline Planning
The data pipeline is designed to handle large Petri dish scans (up to 5120 × 4096 px). The architecture breaks this down into:
- Image Cropping -> Patch Division (512x512) -> Batch Inference -> Mask Stitching -> Post-Processing -> Morphological Measurement.
- This is implemented within the [src/npec_cv](../src/npec_cv) package and orchestrated via [scripts/](../scripts).

### 3. Financial Cost Evaluation
The team performed a significant evaluation of cloud costs and implemented a highly cost-effective deployment strategy:
- **Inference (CPU-Bound)**: Instead of using standard Azure Managed Online Endpoints (~€150/month), the team deployed a `KubernetesOnlineEndpoint` on the local lab server (`lambda-4`) connected via Azure Arc. This results in **zero incremental Azure compute costs** for inference.
- **Training (GPU-Bound)**: Training utilizes Azure ML compute instances with temporary GPU resources. Command jobs are scheduled and run only when retraining is triggered, minimizing active GPU runtime and associated costs.
- **Resource Limits**: Concurrency per instance is pinned to 1 with queueing mechanisms to prevent resource exhaustion.

### Summary
The project demonstrates a thorough understanding of cloud architecture, clearly separating concerns between storage, serving, and frontend. The strategic decision to use Arc-connected local Kubernetes for inference shows a deep consideration for real-world financial constraints.
