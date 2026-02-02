# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GeoVue is a comprehensive geological chip tray photo processing application built with Python and Tkinter. It provides tools for processing chip tray photos of RC drilling samples, managing geological data, and performing QA/QC on chip photos with ArUco marker detection and depth validation. The application is designed for geologists and mining professionals to efficiently process and analyze core sample imagery.

## Build and Run Commands

### Development
```bash
# Run the application directly with splash screen
python launcher.py

# Or run main module directly (no splash)
cd src
python main.py

# Run with virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -e .
python launcher.py
```

### Building Executable
```bash
# Build using batch file (Windows) - provides interactive menu
"Rebuild App.bat"

# Or use PyInstaller directly:
# One-file build (slower startup, single exe, ~150MB)
pyinstaller GeoVue.spec

# One-folder build (faster startup, multiple files)
pyinstaller GeoVue-OneDir.spec

# Debug build with console output
pyinstaller GeoVue-Debug.spec
```

### Installation
```bash
# Install from pyproject.toml (development mode)
pip install -e .

# Install specific requirements
pip install -r src/requirements.txt

# Core dependencies only
pip install numpy opencv-python pillow pandas openpyxl piexif requests pywin32 psutil
```

## Architecture Overview

The application follows a modular architecture with clear separation of concerns:

### Entry Points
- **launcher.py**: Production launcher with splash screen management, error handling, and sys.modules injection
- **src/main.py**: Main application initialization, component orchestration, and GUI setup

### Core Components (`src/core/`)
- **config_manager.py**: Centralized configuration management, settings persistence, default values
- **file_manager.py**: File operations, directory management, path validation, file watching
- **repo_updater.py**: Repository update functionality, version checking, auto-update features
- **tesseract_manager.py**: OCR integration for text extraction from images, Tesseract wrapper
- **translator.py**: Multi-language support system, CSV-based translations, locale management
- **visualization_manager.py**: Visualization generation, chart creation, export functionality

### GUI Layer (`src/gui/`)
- **main_gui.py**: Primary application interface, main window management, menu system
- **gui_manager.py**: GUI state management, component coordination, event handling
- **splash_screen.py**: Startup splash screen, loading progress, version display
- **qaqc_manager.py**: Quality assurance/control workflow, validation interface
- **boundary_manager.py**: Compartment boundary detection and management
- **duplicate_handler.py**: Duplicate image detection and resolution
- **dialog_helper.py**: Common dialog utilities and helpers
- **widgets/**: Custom Tkinter widgets
  - collapsible_frame.py: Expandable/collapsible UI sections
  - modern_button.py: Styled button components
  - themed_combobox.py: Themed dropdown components
  - three_state_toggle.py: Tri-state toggle switches
  - multiselect_review_dialog.py: Multi-selection review interface

#### Specialized Dialogs
- **compartment_registration_dialog.py**: Compartment registration and management
- **image_alignment_dialog.py**: Image alignment and transformation tools
- **logging_review_dialog.py**: Geological logging review interface
- **color_map_config_dialog.py**: Color mapping configuration for visualizations
- **drillhole_trace_designer.py**: Drill hole trace design and editing
- **embedding_training_dialog.py**: Machine learning embedding training interface
- **first_run_dialog.py**: Initial setup and configuration wizard
- **progress_dialog.py**: Progress tracking for long-running operations

### Processing Pipeline (`src/processing/`)
- **ArucoMarkersAndBlurDetectionStep/**
  - **aruco_manager.py**: ArUco marker detection, validation, coordinate extraction
  - **blur_detector.py**: Image blur analysis, quality assessment
- **LoggingReviewStep/**
  - **color_map_manager.py**: Color mapping for geological attributes
  - **drillhole_data_manager.py**: Drill hole data management and storage
  - **drillhole_data_visualizer.py**: Visualization generation for drill data
  - **drillhole_trace_generator.py**: Trace generation from drill hole data
- **MachineLearningStep/**
  - **embedding_trainer.py**: ML model training for feature extraction
- **visualization_drawer.py**: Common visualization drawing utilities

### Utilities (`src/utils/`)
- **cloud_sync_manager.py**: Cloud synchronization, backup features
- **image_pan_zoom_handler.py**: Image navigation, zoom/pan controls
- **json_register_manager.py**: JSON-based data persistence, register management
- **register_synchronizer.py**: Excel register synchronization, data import/export
- **image_processing_depth_validation.py**: Depth interval validation logic

### Resources (`src/resources/`)
- **ArUco Markers/**: 25 ArUco marker templates (4x4_1000 dictionary, SVG format)
- **color_presets/**: Geological attribute color mappings (JSON)
  - fe_grade.json: Iron grade color schemes
  - lithology.json: Rock type color schemes
  - sio2_grade.json: Silica grade color schemes
  - BIFf.json: Banded Iron Formation schemes
  - Normative Mineralogy.json: Mineral composition schemes
- **Icons/**: Application and folder icons (ICO/PNG formats)
- **Register Template File/**: Excel template (Chip_Tray_Register.xltx)
- **translations.csv**: Multi-language support data
- **logo.ico**, **full_logo.png**: Application branding

### GeoVue Capture Module (`src/GeoVue_Capture/`)
- **geovue_capture_app.py**: Image capture application for field use
- **RaspberryPi/stepper_control_script.py**: Hardware control for automated capture

## Key Features

1. **Image Processing Pipeline**
   - ArUco marker detection for automatic orientation and scale calibration
   - Compartment boundary detection and extraction
   - Blur detection and image quality assessment
   - Batch processing capabilities

2. **Depth Validation**
   - Validates depth intervals against expected ranges
   - Automatic depth calculation from markers
   - Depth consistency checking across samples

3. **QA/QC Management**
   - Comprehensive quality assurance workflow
   - Duplicate detection and resolution
   - Data validation and error reporting
   - Audit trail and logging

4. **Visualization**
   - Drill hole trace generation
   - Color-coded geological attribute mapping
   - Customizable visualization templates
   - Export to multiple formats

5. **Data Management**
   - JSON-based register system with enhanced metadata
   - Excel integration for data import/export
   - Cloud synchronization support
   - Multi-language support

6. **Machine Learning**
   - Embedding training for feature extraction
   - Pattern recognition capabilities
   - Automated classification support

## Testing

### Test Structure
```
tests/
├── README_enhancements.md          # Comprehensive test documentation
├── test_json_register_manager_enhancements.py  # Core enhancement tests
├── test_png_uid_fix.py            # PNG UID consistency tests
├── test_file_manager.py           # File management tests
├── test_integration.py            # Integration tests
├── run_enhancement_tests.py       # Test runner with dependency checking
├── validate_test_structure.py     # Test structure validation
└── test_enhancement_summary.py    # Test result aggregation
```

### Running Tests
```bash
# Run all tests with pytest
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m uid          # UID-specific tests

# Run enhancement tests with runner
python tests/run_enhancement_tests.py

# Validate test structure
python tests/validate_test_structure.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Configuration (pytest.ini)
- Test discovery: `test_*.py` files
- Markers: unit, integration, uid, register, slow
- Verbose output with color and short tracebacks

## Validation Tools

### GeoVue Validation Tool
A standalone GUI tool for validating compartment extraction accuracy:

```bash
# Run validation tool
python validation_tool.py
# or
run_validation_tool.bat
```

Features:
- Visual overlay of detected compartments
- Corner coordinate verification
- Register data validation
- Image quality assessment

## Dependencies

### Core Requirements
- Python 3.7+
- numpy: Numerical operations
- opencv-python (4.10.0.84): Image processing
- opencv-contrib-python (4.11.0.86): Extended OpenCV features
- Pillow: Image manipulation
- pandas: Data analysis
- openpyxl (3.1.5): Excel file handling
- piexif (1.1.3): EXIF data management

### Platform-Specific
- pywin32 (308): Windows COM integration for Excel
- pytesseract (0.3.13): OCR functionality

### Optional
- pillow_heif (0.22.0): HEIF/HEIC image support
- python_ternary (1.0.8): Ternary plot visualizations
- tomli (2.2.1): TOML configuration parsing

## Build Configuration

### PyInstaller Specs
- **GeoVue.spec**: Single-file executable, includes all resources
- **GeoVue-OneDir.spec**: Folder distribution, faster startup
- **GeoVue-Debug.spec**: Debug build with console output

### Resource Bundling
- All resources copied to `_internal/` directory
- ArUco markers, color presets, icons included
- Config.json packaged with executable
- Recursive copy of all resource subdirectories

## Important Implementation Notes

1. **Thread Safety**: JSONRegisterManager implements file locking for concurrent access
2. **UID Management**: PNG files use custom UID embedding for tracking
3. **Error Handling**: Comprehensive error handling with user-friendly messages
4. **Performance**: Batch processing optimized for large image sets
5. **Memory Management**: Image processing uses lazy loading and cleanup
6. **Splash Screen**: Coordination via sys.modules injection between launcher and main
7. **Path Handling**: Supports both frozen (PyInstaller) and development environments
8. **Display Requirements**: Tkinter requires proper display configuration on Linux
9. **Excel Integration**: Requires Windows COM support via pywin32
10. **Backward Compatibility**: Maintains compatibility with existing register data

## Development Guidelines

1. **Code Style**: Follow PEP 8 guidelines
2. **Documentation**: Use docstrings for all public methods
3. **Testing**: Write tests for new features
4. **Error Messages**: Provide clear, actionable error messages
5. **Logging**: Use structured logging for debugging
6. **Configuration**: Use config_manager for all settings
7. **Resources**: Place all assets in appropriate resource directories
8. **Translations**: Update translations.csv for new UI text

## Common Tasks

### Adding a New Color Preset
1. Create JSON file in `src/resources/color_presets/`
2. Define color mappings with ranges and RGB values
3. Update color_map_manager.py to load new preset

### Adding a New Dialog
1. Create dialog class in `src/gui/`
2. Inherit from appropriate base class
3. Register with gui_manager
4. Add to main_gui menu if needed

### Adding a New Processing Step
1. Create module in `src/processing/`
2. Implement process() method
3. Add to processing pipeline
4. Update progress tracking

### Adding Translations
1. Edit `src/resources/translations.csv`
2. Add new language column if needed
3. Update translator.py language list
4. Test with language switching