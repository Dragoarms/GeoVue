# GeoVue

Program for comprehensive capture, extraction, and visualisation of chip tray compartments with plotting, stitching, and register maintaining capabilities.

## Overview

- **Version:** 2.7.0 (see `pyproject.toml`)
- **License:** MIT (see [LICENSE](LICENSE))

## Structure

- `src/` — Main application (GUI, processing, utils)
- `ml_detection/` — YOLO detection and cell classification (see [ml_detection/README.md](ml_detection/README.md))
- `ml_pipeline/` — ML pipeline utilities
- `docs/` — Documentation and plans

## Setup

```bash
pip install -e .
# or from project root: pip install -e ".[dev]" if extras exist
```

See `install.bat` for Windows, and `ml_detection/requirements.txt` for ML dependencies.

## Running

- **GUI:** `python launcher.py` (or use the built executable from PyInstaller specs)
- **CLI:** `chiptray` (if installed as package)

## Repository

- **GitHub:** https://github.com/Dragoarms/GeoVue
