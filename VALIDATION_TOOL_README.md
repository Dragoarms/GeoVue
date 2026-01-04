# GeoVue Validation Tool

A GUI-based validation tool for verifying the accuracy of compartment extraction from geological core tray images.

## Purpose

This tool allows you to visually verify that:
- Compartments are being extracted from the correct locations in original images
- Corner detection and boundary calculations are accurate
- The processing pipeline is working as expected
- Register data matches the actual compartment locations

## Features

### Visual Validation
- **Image Display**: View original images with compartment boundary overlays
- **Color-coded Overlays**: Each compartment is outlined in a different color
- **Corner Markers**: Shows exact corner detection points
- **Numbered Labels**: Compartments are labeled with numbers for easy identification

### Navigation & Controls
- **Image Selection**: Browse through all processed images
- **Previous/Next Navigation**: Step through images sequentially
- **Zoom & Pan**: Examine details with mouse wheel zoom and click-drag pan
- **Fit to Window**: Automatically scale image to fit the display area
- **Toggle Overlays**: Show/hide compartment boundaries and labels

### Information Panel
- **Image Metadata**: File path, dimensions, and processing details
- **Register Data**: Hole ID, depth ranges, and processing status
- **Corner Coordinates**: Exact pixel coordinates for each compartment corner
- **Related Files**: Lists corresponding compartment images

## How to Run

### Method 1: Batch File (Recommended)
1. Double-click `run_validation_tool.bat`
2. The script will check dependencies and launch the tool

### Method 2: Python Command
1. Open command prompt in the GeoVue folder
2. Run: `python validation_tool.py`

## Setup Instructions

### 1. Folder Selection
When the tool opens, you need to select three folders:

**Original Images Folder**
- Contains the processed original tray images
- Usually named something like "Processed Original Images"
- These are the full tray images with embedded metadata

**Compartment Images Folder** 
- Contains the extracted individual compartment images
- Usually under "Extracted Compartment Images"
- These are the individual core compartments

**Register Data Folder**
- Contains the JSON register files
- Usually named "Register Data (Do not edit)"
- Contains corner coordinates and processing metadata

### 2. Load Data
After selecting all three folders, click "Load Data" to:
- Parse the register JSON files
- Match original images with their register data
- Build the image list for validation

## Using the Tool

### Image Navigation
- **Dropdown Menu**: Select any image directly from the list
- **Previous/Next Buttons**: Navigate sequentially through images
- **Status Bar**: Shows current image position and total count

### Viewing Controls
- **Mouse Wheel**: Zoom in/out on the image
- **Click & Drag**: Pan around when zoomed in
- **Fit to Window**: Scale image to fit the display area
- **Actual Size**: Reset to 100% zoom level

### Overlay Controls
- **Show Overlays**: Toggle compartment boundary display on/off
- **Show Labels**: Toggle compartment number labels on/off

### Information Panel
The right panel shows detailed information about the current image:
- Image file details and path
- Register data (Hole ID, depths, processing info)
- Compartment corner coordinates
- List of related compartment files

## Validation Workflow

### 1. Initial Setup
1. Launch the validation tool
2. Select your three data folders
3. Click "Load Data"

### 2. Visual Inspection
For each image, verify:
- **Boundary Accuracy**: Do the colored outlines match the actual compartment boundaries?
- **Corner Placement**: Are the corner dots positioned correctly at compartment corners?
- **Compartment Count**: Does the number of overlays match expected compartments?
- **Alignment**: Are boundaries properly aligned with visible compartment edges?

### 3. Problem Identification
Look for these common issues:
- **Misaligned boundaries**: Overlays don't match visible compartment edges
- **Missing compartments**: Fewer overlays than expected compartments
- **Extra boundaries**: More overlays than actual compartments
- **Skewed detection**: Boundaries appear rotated or distorted

### 4. Data Verification
Use the information panel to:
- Confirm hole ID and depth ranges match expectations
- Verify corner coordinates are reasonable
- Check that related compartment images exist

## Data Format Details

### Register Data Structure
The tool expects JSON register files with corner coordinate data in this format:
```json
{
  "HoleID": "ABC123",
  "Original_Filename": "image.jpg",
  "Corner_1_X": 100, "Corner_1_Y": 200,
  "Corner_2_X": 300, "Corner_2_Y": 200,
  "Corner_3_X": 300, "Corner_3_Y": 400,
  "Corner_4_X": 100, "Corner_4_Y": 400
}
```

Corner order: 1=Top-Left, 2=Top-Right, 3=Bottom-Right, 4=Bottom-Left

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- BMP (.bmp)

## Troubleshooting

### Common Issues

**"No valid register data found"**
- Check that register JSON files contain corner coordinate data
- Verify files are in the correct JSON format
- Ensure the register data folder is correct

**"No matching images found"**
- Verify original images folder contains processed images
- Check that image filenames match those in register data
- Ensure image file extensions are supported

**"Failed to load image"**
- Verify image files are not corrupted
- Check that images are in supported formats
- Ensure sufficient memory for large images

**Tool won't start**
- Run `pip install Pillow pandas` to install dependencies
- Verify Python 3.7+ is installed
- Check that tkinter is available (usually included with Python)

### Performance Tips

**For Large Images**
- Use "Fit to Window" for overview, then zoom to examine details
- Close and reopen tool if memory usage becomes high
- Process smaller batches of images if performance is slow

**For Many Images**
- Sort images by hole ID or date for systematic validation
- Use the dropdown to jump to specific images of interest
- Take notes of problematic images for follow-up processing

## Expected Output

After validation, you should have confidence that:
1. **Compartment boundaries are accurate** - Overlays align with visible compartment edges
2. **Corner detection is precise** - Corner markers are positioned correctly
3. **Data integrity is maintained** - Register data matches visual observations
4. **Processing pipeline is working** - Consistent results across similar images

## Next Steps

Based on validation results:
- **If boundaries are accurate**: Processing pipeline is working correctly
- **If boundaries are misaligned**: Review ArUco marker detection parameters
- **If compartments are missing**: Check compartment detection thresholds
- **If data is inconsistent**: Review register synchronization process

The validation tool provides the visual feedback needed to tune and optimize the GeoVue processing pipeline for your specific core tray formats and image conditions.