# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GeoVue is a geological chip tray photo processing application built with Python and Tkinter. It provides tools for processing drill core samples, managing geological data, and performing QA/QC on core photos with ArUco marker detection and depth validation.

## Build and Run Commands

### Development
```bash
# Run the application directly
python launcher.py

# Or run main module
cd src
python main.py
```

### Building Executable
```bash
# Build using batch file (Windows) - provides interactive menu
"Rebuild App.bat"

# Or use PyInstaller directly:
# One-file build (slower startup, single exe)
pyinstaller GeoVue.spec

# One-folder build (faster startup)
pyinstaller GeoVue-OneDir.spec

# Debug build with console
pyinstaller GeoVue-Debug.spec
```

### Installation
```bash
# Install dependencies from pyproject.toml
pip install -e .

# Or install specific requirements
pip install -r src/requirements.txt
```

## Architecture Overview

The application follows a modular architecture with clear separation of concerns:

### Entry Points
- **launcher.py**: Production launcher with splash screen management and error handling
- **src/main.py**: Main application initialization and component orchestration

### Core Components (`src/core/`)
- **config_manager.py**: Centralized configuration management
- **file_manager.py**: File operations and directory management
- **repo_updater.py**: Repository update functionality
- **tesseract_manager.py**: OCR integration for text extraction
- **translator.py**: Multi-language support system
- **visualization_manager.py**: Visualization generation and management

### GUI Layer (`src/gui/`)
- **main_gui.py**: Primary application interface
- **gui_manager.py**: GUI state and component management
- **splash_screen.py**: Startup splash screen
- **widgets/**: Custom Tkinter widgets (collapsible frames, modern buttons, themed components)

### Processing Pipeline (`src/processing/`)
- **ArucoMarkersAndBlurDetectionStep/**: ArUco marker detection and blur analysis
- **LoggingReviewStep/**: Drill hole data management and visualization
  - Color mapping system for geological attributes
  - Drill hole trace generation
- **MachineLearningStep/**: Embedding training for ML features

### Utilities (`src/utils/`)
- **cloud_sync_manager.py**: Cloud synchronization features
- **image_pan_zoom_handler.py**: Image navigation controls
- **json_register_manager.py**: JSON-based data persistence
- **register_synchronizer.py**: Excel register synchronization
- **image_processing_depth_validation.py**: Depth validation logic

### Resources (`src/resources/`)
- ArUco marker templates (SVG format)
- Color preset configurations (JSON)
- Application icons and logos
- Excel template for chip tray registration
- Translation CSV files

## Key Features

1. **Image Processing Pipeline**: Processes geological core photos with ArUco marker detection for orientation and scale
2. **Depth Validation**: Validates depth intervals in core photos against expected values
3. **QA/QC Management**: Quality assurance and control workflow for geological data
4. **Visualization**: Generates drill hole traces and color-coded geological visualizations
5. **Multi-language Support**: Internationalization through CSV-based translations
6. **Excel Integration**: Synchronizes with Excel registers for data management

## Testing

Currently no automated tests are configured. we need to create and implement a series of tests - specifically around the json register methods.

To run tests when pytest is installed:
```bash
pytest src/gui/tests/
```

## Important Notes

- The application uses Tkinter for GUI, requiring proper display configuration
- ArUco markers are used for image calibration and orientation
- Excel integration requires pywin32 on Windows
- The application supports both frozen (PyInstaller) and development environments
- Splash screen coordination between launcher.py and main application is handled via sys.modules injection