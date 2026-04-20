# Frontend — Folder Structure Guide

This document explains the purpose of every folder and key file inside `frontend/`.
The frontend is built with **React + Vite + TypeScript** and communicates with the
Python backend (FastAPI) via REST and WebSocket.

---

## Top-Level Files

| File             | Purpose                                                              |
| ---------------- | -------------------------------------------------------------------- |
| `index.html`     | Single HTML entry point — Vite injects the JS bundle here            |
| `vite.config.ts` | Vite build & dev-server configuration (port, proxy to backend, etc.) |
| `tsconfig.json`  | TypeScript compiler options                                          |
| `package.json`   | NPM dependencies and scripts (`dev`, `build`, `preview`)             |
| `STRUCTURE.md`   | **This file** — living documentation of the project layout           |

---

## `public/`

Static assets that are served **as-is** (not processed by Vite).

| File          | Purpose          |
| ------------- | ---------------- |
| `favicon.ico` | Browser tab icon |

> Put files here if they need a fixed, predictable URL (e.g. `robots.txt`, static JSON).

---

## `src/` — Application Source

All application code lives here. Vite compiles everything under `src/`.

---

### `src/api/`

All HTTP and WebSocket calls to the Python backend.
**No fetch/axios calls should exist outside this folder.**

| File          | Purpose                                                                       |
| ------------- | ----------------------------------------------------------------------------- |
| `pipeline.ts` | `runPipeline()`, `getResults()`, `getRobotStatus()` — wraps backend endpoints |

> If the API grows, split by domain: `robotics.ts`, `cv.ts`, etc.

---

### `src/assets/`

Static media imported directly by components (processed by Vite — hashed filenames, optimized).

Store here: logos, background images, SVG icons.

---

### `src/components/`

**Reusable UI building blocks.** Each component should do one thing well and be
usable on multiple pages.

| File                   | Purpose                                                               |
| ---------------------- | --------------------------------------------------------------------- |
| `ImageUploader.tsx`    | Drag-and-drop / file-picker for uploading Petri dish images           |
| `PlantCard.tsx`        | Displays per-plant detection result (length, tip coordinates, status) |
| `RobotStatus.tsx`      | Live OT-2 robot state indicator (position, connectivity, mode)        |
| `SegmentationView.tsx` | Renders the source image with segmentation masks overlaid             |

> Rule: if a UI element is used in more than one page, it belongs here.

---

### `src/pages/`

**Route-level views** — one file per page/screen of the app.
Pages compose components together; they should contain minimal logic themselves.

| File            | Purpose                                                                 |
| --------------- | ----------------------------------------------------------------------- |
| `Dashboard.tsx` | Main monitoring view — pipeline status, robot live feed, latest results |
| `Upload.tsx`    | Submit a new Petri dish image to the CV pipeline                        |
| `Results.tsx`   | Browse and inspect historical pipeline run results                      |

---

### `src/hooks/`

Custom **React hooks** — reusable stateful logic extracted from components.

| File                   | Purpose                                                |
| ---------------------- | ------------------------------------------------------ |
| `usePipelineStatus.ts` | Polls or subscribes (WebSocket) to pipeline run status |

> Naming convention: always prefix with `use` (React requirement).

---

### `src/styles/`

Global CSS files. Component-specific styles should live in the component file
(CSS modules or inline) to keep things co-located.

| File            | Purpose                                                                     |
| --------------- | --------------------------------------------------------------------------- |
| `global.css`    | CSS resets, base typography, body defaults                                  |
| `variables.css` | Design tokens — colours, spacing scale, font sizes as CSS custom properties |

---

### `src/types/`

Shared **TypeScript interfaces and types** used across the app.
Keeps components and API layer in sync with the backend response shapes.

| File          | Purpose                                                                     |
| ------------- | --------------------------------------------------------------------------- |
| `pipeline.ts` | `PlantResult`, `PipelineRun`, `RobotState`, `SegmentationResult` interfaces |

---

### `src/utils/`

Pure helper functions with **no React/UI dependency** — safe to unit-test in isolation.

| File         | Purpose                                                        |
| ------------ | -------------------------------------------------------------- |
| `helpers.ts` | Pixel-to-mm conversion, result formatting, date/time utilities |

---

### `src/main.tsx`

Application entry point. Mounts the React app into `index.html` and sets up the router.

---

## Data Flow Summary

```
Browser (React)
    │
    │  HTTP / WebSocket
    ▼
Python Backend  (FastAPI — not in this folder)
    │
    │  imports
    ▼
src/  (CVPipeline, RoboticsPipeline)
```

The frontend **never** imports from `../src` (the Python source).
All communication goes through the API layer (`src/api/`).

---

## Adding New Features — Checklist

1. **New API endpoint?** → add a function in `src/api/`
2. **New data shape from backend?** → add an interface in `src/types/pipeline.ts`
3. **New reusable UI element?** → create a component in `src/components/`
4. **New page/screen?** → create a file in `src/pages/` and register a route in `main.tsx`
5. **Shared stateful logic?** → extract a custom hook in `src/hooks/`
6. **Pure utility function?** → add to `src/utils/helpers.ts`
