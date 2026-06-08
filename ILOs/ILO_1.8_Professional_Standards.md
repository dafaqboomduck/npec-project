# ILO 1.8: Professional Standards

**Requirement**: Work meets professional standards across all formats: code, documentation, presentations, and peer reviews. Consistent polish, audience-awareness, and attention to detail.

## Evidence in the Project

### 1. Code Quality and Structure
The repository is highly structured and professional, following industry-standard Python project layouts:
- **`src/` layout**: The core computer vision package `npec_cv` and the `api` are properly separated.
- **Linting & Formatting**: Enforced via `ruff` and configured in `pyproject.toml`. Pre-commit hooks (`.pre-commit-config.yaml`) ensure that all commits meet formatting standards before they are pushed.
- **Continuous Integration**: `.github/workflows/` contains pipelines that automatically run linting and tests on every Pull Request, enforcing professional code standards.

### 2. Documentation
- The project includes a comprehensive `README.md` that serves as a central index.
- A dedicated `docs/` folder built with Sphinx provides detailed tutorials, how-to guides, and architecture explanations.
- API endpoints are documented with OpenAPI specifications and automatically generated interactive docs (FastAPI Swagger UI).

### 3. Work Logs and Peer Reviews
- **Azure DevOps Boards**: The team utilizes Azure DevOps for agile project management, estimating tasks, and managing sprints. This acts as the central work log.
- **Learning Logs & Self-Assessments**: Maintained individually by team members as required by the assessment criteria.
- **Peer Reviews**: Conducted during and at the end of sprints to ensure code quality and team cohesion.

### Summary
The consistent use of typing, comprehensive docstrings (Google Style), structured project directories, and automated code checks demonstrate a high level of professionalism and attention to detail.
