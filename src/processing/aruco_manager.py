# processing/aruco_manager.py

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import threading
import queue



class ArucoManager:
    """
    Handles ArUco marker detection, compartment extraction, and image correction.
    Provides systematic logging and detection of ArUco markers for chip tray processing.
    """
    
    
    
    
    def __init__(self, config: Dict[str, Any], progress_queue=None, app=None):
        """
        Initialize the ArUco manager with configuration settings.
        
        Args:
            config: Configuration dictionary with ArUco settings
            progress_queue: Optional queue for reporting progress
            app: Optional reference to the main application for thread-safe dialogs
        """
        self.config = config
        self.progress_queue = progress_queue
        self.app = app  # Store reference to the main app
        self.logger = logging.getLogger(__name__)
        
        # Initialize detector
        self.initialize_detector()
        
        # Cache for detected markers
        self.cached_markers = {}
        

    def initialize_detector(self):
        """Initialize ArUco detector with configuration settings."""
        try:
            # Get ArUco dictionary type from config
            aruco_dict_type = self.config.get('aruco_dict_type', 'DICT_4X4_1000')
            
            # Lookup dictionary constant from cv2.aruco
            dict_const = getattr(cv2.aruco, aruco_dict_type, None)
            if dict_const is None:
                self.logger.error(f"Invalid ArUco dictionary type: {aruco_dict_type}")
                # Fallback to a standard dictionary
                dict_const = cv2.aruco.DICT_4X4_1000
            
            # Create ArUco detector
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_const)
            self.aruco_params = cv2.aruco.DetectorParameters()
            
            # Set detector parameters for better detection
            # Increase adaptiveThreshWinSizeMax for better marker detection in various lighting
            if hasattr(self.aruco_params, 'adaptiveThreshWinSizeMax'):
                self.aruco_params.adaptiveThreshWinSizeMax = 23
                
            # Set corner refinement method to CORNER_REFINE_SUBPIX for more accurate corners
            if hasattr(self.aruco_params, 'cornerRefinementMethod'):
                self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                
            # Create detector
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            
            self.logger.debug(f"Initialized ArUco detector with dictionary {aruco_dict_type}")
            
        except Exception as e:
            self.logger.error(f"Error initializing ArUco detector: {str(e)}")
            # Create fallback detector with default parameters
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
    
    # ArUco Marker Detection Processes

    def detect_markers(self, image: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Detect ArUco markers in an image with improved robustness.
        Uses preprocessing to enhance marker detection in challenging conditions.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary mapping marker IDs to corner coordinates
        """
        if image is None:
            self.logger.error("Cannot detect markers - image is None")
            return {}
            
        # Check image dimensions
        if len(image.shape) != 3 or image.shape[2] != 3:
            self.logger.warning("Image is not a 3-channel color image, converting")
            if len(image.shape) == 2:
                # Convert grayscale to color
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # Convert RGBA to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                
        # Initialize result dictionary
        markers_dict = {}
        
        try:
            # First try with initial input image
            self.logger.info("ArUcoManager - Detecting markers in provided image")
            corners, ids, rejected = self.aruco_detector.detectMarkers(image)
            
            if ids is not None and len(ids) > 0:
                # Convert detection results to dictionary
                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    marker_corners = corners[i][0]
                    markers_dict[marker_id] = marker_corners
                
                self.logger.info(f"Detected {len(markers_dict)} markers in original image")
            else:
                self.logger.warning("No markers detected in original image, trying preprocessing")
                
                # Try with different preprocessed versions if no markers found
                preprocessing_methods = [
                    ("grayscale", lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
                    ("adaptive_threshold", lambda img: cv2.adaptiveThreshold(
                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                    )),
                    ("contrast_enhanced", lambda img: cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
                ]
                
                for method_name, preprocess_func in preprocessing_methods:
                    try:
                        # Preprocess image
                        processed = preprocess_func(image)
                        
                        # Detect markers in preprocessed image
                        corners, ids, rejected = self.aruco_detector.detectMarkers(processed)
                        
                        if ids is not None and len(ids) > 0:
                            # Convert detection results to dictionary
                            for i in range(len(ids)):
                                marker_id = ids[i][0]
                                marker_corners = corners[i][0]
                                markers_dict[marker_id] = marker_corners
                            
                            # self.logger.info(f"Detected {len(markers_dict)} markers using {method_name}")
                            break
                    except Exception as e:
                        self.logger.error(f"Error with {method_name} preprocessing: {str(e)}")
            
            # Cache the detected markers
            self.cached_markers = markers_dict.copy()
            
            return markers_dict
            
        except Exception as e:
            self.logger.error(f"Error detecting markers: {str(e)}")
            return {}
    
    def improve_marker_detection(self, image: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Attempt to improve ArUco marker detection using various image preprocessing techniques.
        
        Args:
            image: Input image as numpy array
                
        Returns:
            Dict mapping marker IDs to corner coordinates
        """
        # Try original detection first
        markers = self.detect_markers(image)
        
        initial_marker_count = len(markers)
        
        # Log initial detection
        self.logger.info(f"Initial detection found {initial_marker_count} ArUco markers")
        
        # Check if we need to improve detection
        corner_ids = self.config.get('corner_marker_ids', [0, 1, 2, 3])
        compartment_ids = self.config.get('compartment_marker_ids', list(range(4, 24)))
        metadata_ids = self.config.get('metadata_marker_ids', [24])
        expected_markers = len(corner_ids) + len(compartment_ids) + len(metadata_ids)
        
        if initial_marker_count >= expected_markers:
            # All markers detected, no need for improvement
            return markers
        
        # Store the best result
        best_markers = markers
        best_count = initial_marker_count
        
        # Initialize a combined marker dictionary
        combined_markers = markers.copy()
        
        # Try different preprocessing methods
        preprocessing_methods = [
            ("Adaptive thresholding", self._preprocess_adaptive_threshold),
            ("Histogram equalization", self._preprocess_histogram_equalization),
            ("Contrast enhancement", self._preprocess_contrast_enhancement),
            ("Edge enhancement", self._preprocess_edge_enhancement)
        ]
        
        for method_name, preprocess_func in preprocessing_methods:
            # Apply preprocessing
            processed_image = preprocess_func(image)
            
            # Detect markers on processed image
            new_markers = self.detect_markers(processed_image)
            
            # Log results
            # self.logger.info(f"{method_name}: detected {len(new_markers)} ArUco markers")
            
            # Check if this method found any new markers
            new_marker_ids = set(new_markers.keys()) - set(combined_markers.keys())
            if new_marker_ids:
                # self.logger.info(f"{method_name} found {len(new_marker_ids)} new markers: {sorted(new_marker_ids)}")
                
                # Add new markers to combined set
                for marker_id in new_marker_ids:
                    combined_markers[marker_id] = new_markers[marker_id]
            
            # Update best result if this method found more markers
            if len(new_markers) > best_count:
                best_markers = new_markers
                best_count = len(new_markers)
        
            # Log final results
            # self.logger.info(f"Best single method found {best_count} markers")
            self.logger.info(f"Combined methods found {len(combined_markers)} markers")
            
            # Determine which markers to use and store in cache
            final_markers = combined_markers if len(combined_markers) > best_count else best_markers
            
            # Store the markers in visualization_cache if app is available
            if hasattr(self, 'app') and hasattr(self.app, 'visualization_cache'):
                self.app.visualization_cache.setdefault('current_processing', {})['all_markers'] = final_markers
                print(f"DEBUG: Stored {len(final_markers)} markers in visualization_cache")
            
            # Use combined markers if better than any single method
            if len(combined_markers) > best_count:
                return combined_markers
            else:
                return best_markers
            
    def _preprocess_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding to improve marker detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    def _preprocess_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to improve contrast."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        return cv2.equalizeHist(gray)

    def _preprocess_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _preprocess_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges to improve marker detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect potential markers
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        return dilated
    
   # Compartment Detection and Extraction Processes with ArUco markers
    def analyze_compartment_boundaries(self, image: np.ndarray, markers: Dict[int, np.ndarray], 
                                     compartment_count: int = 20, smart_cropping: bool = True,
                                     metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze and extract compartment boundary data without any UI interaction.
        
        This method performs all the data processing for compartment boundaries but
        returns the results as a dictionary for the caller to handle UI aspects.
        
        Args:
            image: Input image
            markers: Detected ArUco markers dictionary
            compartment_count: Number of expected compartments
            smart_cropping: Whether to use smart cropping for gaps
            metadata: Optional metadata dictionary containing hole_id, depth_from, depth_to
            
        Returns:
            Dictionary containing:
                - boundaries: List of detected compartment boundaries
                - missing_marker_ids: List of missing marker IDs
                - vertical_constraints: Tuple of (top_y, bottom_y) for compartment placement
                - marker_to_compartment: Mapping of marker IDs to compartment numbers
                - visualization: Initial visualization image
                - corner_markers: Dictionary of corner marker positions
                - avg_compartment_width: Average width of compartments
                - viz_steps: Dictionary of visualization steps
        """
        print(f"DEBUG: Starting analyze_compartment_boundaries with {len(markers) if markers else 0} markers")
        
        result = {
            'boundaries': [],
            'missing_marker_ids': [],
            'vertical_constraints': None,
            'marker_to_compartment': {},
            'visualization': None,
            'corner_markers': {},
            'avg_compartment_width': 80,  # Default
            'viz_steps': {},
            'boundary_to_marker': {}
        }
        
        if image is None or not markers:
            print("DEBUG: Cannot analyze boundaries - no image or markers")
            self.logger.warning("Cannot analyze boundaries - no image or markers")
            result['visualization'] = image.copy() if image is not None else None
            return result

        # Create visualization image
        viz = image.copy()
        print("DEBUG: Created visualization image copy")
        
        # Store all visualization steps
        self.viz_steps = {}
        
        # STEP 1: Get marker ID ranges from config
        corner_ids = self.config.get('corner_marker_ids', [0, 1, 2, 3])
        compartment_ids = self.config.get('compartment_marker_ids', list(range(4, 24)))
        metadata_ids = self.config.get('metadata_marker_ids', [24])
        
        # Get compartment interval from config or metadata
        compartment_interval = metadata.get('compartment_interval', 
                                        self.config.get('compartment_interval', 1.0))
        compartment_interval = float(compartment_interval)
        
        # Get the configured compartment count or use default
        expected_compartment_count = compartment_count or self.config.get('compartment_count', 20)

        # STEP 2: Define ROI using corner markers for vertical boundaries
        corner_markers = {mid: corners for mid, corners in markers.items() if mid in corner_ids}
        result['corner_markers'] = corner_markers
        
        if not corner_markers:
            print("DEBUG: No corner markers detected, using image boundaries")
            self.logger.warning("No corner markers detected, using image boundaries")
            top_y, bottom_y = 0, image.shape[0]
        else:
            # Top corners are typically IDs 0 and 1, bottom corners are 2 and 3
            top_corner_ids = [0, 1]
            bottom_corner_ids = [2, 3]
            
            # Initialize to handle cases where not all markers are detected
            top_y_values = []
            bottom_y_values = []
            
            # Find the most extreme Y values for each corner marker
            for mid, corners in corner_markers.items():
                if mid in top_corner_ids:
                    top_y_values.append(np.min(corners[:, 1]))
                elif mid in bottom_corner_ids:
                    bottom_y_values.append(np.max(corners[:, 1]))
            
            # Set boundaries based on detected corners
            if top_y_values:
                top_y = int(np.min(top_y_values))
            else:
                all_corner_points = np.vstack([corners for corners in corner_markers.values()])
                top_y = int(np.min(all_corner_points[:, 1]))
            
            if bottom_y_values:
                bottom_y = int(np.max(bottom_y_values))
            else:
                all_corner_points = np.vstack([corners for corners in corner_markers.values()])
                bottom_y = int(np.max(all_corner_points[:, 1]))
            
            # No margin - i want to show exactly what will be extracted.
            # TODO - Add in a method here or fix this def to respect a new config entry - compartment_height this is SET to 4.5cm or ~830pixels.
            margin = 0
            top_y = max(0, top_y - margin)
            bottom_y = min(image.shape[0], bottom_y + margin)
            
            # Draw horizontal lines on visualization
            cv2.line(viz, (0, top_y), (image.shape[1], top_y), (0, 255, 255), 2)
            cv2.line(viz, (0, bottom_y), (image.shape[1], bottom_y), (0, 255, 255), 2)
        
        result['vertical_constraints'] = (top_y, bottom_y)
        
        # STEP 3: Extract and sort compartment markers
        comp_markers = {}
        for mid in markers:
            if mid in compartment_ids and mid not in corner_ids and mid not in metadata_ids:
                comp_markers[mid] = markers[mid]
        
        print(f"DEBUG: Found {len(comp_markers)} compartment markers: {sorted(comp_markers.keys())}")
        
        if not comp_markers:
            print("DEBUG: No compartment markers detected")
            self.logger.warning("No compartment markers detected")
            result['visualization'] = viz
            return result

        # STEP 4: Calculate marker centers and dimensions
        marker_data = []
        for mid, corners in comp_markers.items():
            # Calculate marker dimensions
            left_x = int(corners[:, 0].min())
            right_x = int(corners[:, 0].max())
            width = right_x - left_x
            center_x = int(corners[:, 0].mean())
            center_y = int(corners[:, 1].mean())
            
            marker_data.append({
                'id': mid,
                'left': left_x,
                'right': right_x,
                'width': width,
                'center_x': center_x,
                'center_y': center_y
            })
        
        # STEP 5: Sort markers by center_x
        marker_data.sort(key=lambda m: m['center_x'])
        
        # Draw markers on visualization
        for marker in marker_data:
            cv2.circle(viz, (marker['center_x'], marker['center_y']), 8, (255, 0, 0), -1)
            cv2.putText(viz, str(marker['id']), 
                    (marker['center_x'] - 10, marker['center_y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Create mapping between marker IDs and compartment numbers
        marker_to_compartment = {
            4+i: int((i+1) * compartment_interval) for i in range(expected_compartment_count)
        }
        result['marker_to_compartment'] = marker_to_compartment

        # STEP 6: Identify missing markers
        expected_marker_ids = list(range(4, 4 + expected_compartment_count))
        detected_marker_ids = [marker['id'] for marker in marker_data]
        missing_marker_ids = [marker_id for marker_id in expected_marker_ids if marker_id not in detected_marker_ids]
        
        # Check if metadata marker (ID 24) is missing
        if 24 not in markers and self.config.get('enable_ocr', True):
            missing_marker_ids.append(24)
        
        result['missing_marker_ids'] = missing_marker_ids
        
        if missing_marker_ids:
            self.logger.info(f"Missing {len(missing_marker_ids)} markers: {missing_marker_ids}")

        # STEP 7: Generate initial compartment boundaries
        compartment_boundaries = []
        
        for marker in marker_data:
            if marker['id'] not in expected_marker_ids:
                continue
                
            # Use marker width as compartment width
            marker_width = marker['width']
            compartment_width = int(marker_width * 1)
            half_width = compartment_width // 2
            
            left_boundary = marker['center_x'] - half_width
            right_boundary = marker['center_x'] + half_width
            
            # Add to boundaries list
            boundary = (
                max(0, left_boundary),
                top_y,
                min(image.shape[1], right_boundary),
                bottom_y
            )
            compartment_boundaries.append(boundary)

        # STEP 7.5: Apply boundary adjustments if provided in metadata
        if metadata and 'boundary_adjustments' in metadata:
            adjustments = metadata['boundary_adjustments']
            print(f"DEBUG: Applying boundary adjustments: {adjustments}")
            
            # Update top and bottom boundaries
            if 'top_boundary' in adjustments:
                top_y = adjustments['top_boundary']
            if 'bottom_boundary' in adjustments:
                bottom_y = adjustments['bottom_boundary']
                
            # Get side height offsets
            left_offset = adjustments.get('left_height_offset', 0)
            right_offset = adjustments.get('right_height_offset', 0)
            
            # Recalculate boundaries with adjustments
            adjusted_boundaries = []
            img_width = image.shape[1]
            
            for x1, _, x2, _ in compartment_boundaries:
                # Calculate y-coordinates based on x-position using boundary lines
                left_top_y = top_y + left_offset
                right_top_y = top_y + right_offset
                left_bottom_y = bottom_y + left_offset
                right_bottom_y = bottom_y + right_offset
                
                # Calculate slopes for top and bottom boundaries
                if img_width > 0:
                    top_slope = (right_top_y - left_top_y) / img_width
                    bottom_slope = (right_bottom_y - left_bottom_y) / img_width
                    
                    # Calculate y values at the x-position of this compartment
                    mid_x = (x1 + x2) / 2
                    new_y1 = int(left_top_y + (top_slope * mid_x))
                    new_y2 = int(left_bottom_y + (bottom_slope * mid_x))
                else:
                    new_y1 = top_y
                    new_y2 = bottom_y
                
                # Create adjusted boundary
                adjusted_boundaries.append((x1, new_y1, x2, new_y2))
            
            # Update compartment boundaries
            compartment_boundaries = adjusted_boundaries
            
            # Update vertical constraints
            result['vertical_constraints'] = (top_y, bottom_y)
        
        # Calculate average compartment width
        if compartment_boundaries:
            widths = [x2 - x1 for x1, _, x2, _ in compartment_boundaries]
            result['avg_compartment_width'] = int(sum(widths) / len(widths))
        
        # STEP 8: Apply wall detection refinement if enabled
        if smart_cropping and compartment_boundaries:
            print("DEBUG: Starting wall detection refinement")
            refined_boundaries = self._refine_balanced_boundaries(
                image, compartment_boundaries, 20)
            compartment_boundaries = refined_boundaries
        
        result['boundaries'] = compartment_boundaries
        
        if hasattr(self, 'app') and hasattr(self.app, 'viz_manager'):
            # Create visualization using viz_manager
            initial_viz = self.app.viz_manager.create_visualization(
                base_version="working",
                viz_name="initial_boundaries",
                draw_function=self.app._draw_boundaries_on_image,
                boundaries=compartment_boundaries,
                markers=markers,
                save_as_version=False  # Just cache it
            )
            result['visualization'] = initial_viz
        else:
            # Fallback - just return the image with boundaries drawn manually
            result['visualization'] = self._create_boundary_visualization(
                image, compartment_boundaries, marker_to_compartment, markers
            )

        # Create boundary to marker mapping
        boundary_to_marker = {}
        for i, boundary in enumerate(compartment_boundaries):
            center_x = (boundary[0] + boundary[2]) // 2
            
            # Find the closest marker
            closest_marker_id = None
            closest_distance = float('inf')
            
            for marker in marker_data:
                distance = abs(marker['center_x'] - center_x)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_marker_id = marker['id']
            
            if closest_marker_id is not None:
                boundary_to_marker[i] = closest_marker_id
        
        result['boundary_to_marker'] = boundary_to_marker
        
        return result

    def _detect_vertical_edges_in_roi(self, image: np.ndarray, roi: Tuple[int, int, int, int],
                                    min_line_length: int = 50, max_line_gap: int = 5) -> List[int]:
        """
        Detect long vertical edges specifically within a region of interest.
        
        This method focuses on finding strong vertical lines that likely represent
        compartment walls, filtering out noise and small curved edges.
        
        Args:
            image: Input image
            roi: Region of interest as (x1, y1, x2, y2)
            min_line_length: Minimum line length to consider (pixels)
            max_line_gap: Maximum allowed gap in a line (pixels)
            
        Returns:
            List of x-coordinates of detected vertical edges
        """
        x1, y1, x2, y2 = roi
        
        # Validate ROI dimensions
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            self.logger.warning(f"Invalid ROI dimensions: {roi}, image shape: {image.shape}")
            return []
        
        # Extract ROI from the image
        roi_img = image[y1:y2, x1:x2].copy()
        
        # Validate extracted ROI is not empty
        if roi_img.size == 0 or roi_img.shape[0] <= 0 or roi_img.shape[1] <= 0:
            self.logger.warning(f"Empty ROI: {roi}, image shape: {image.shape}")
            return []
        
        
        # Convert to grayscale if needed
        if len(roi_img.shape) == 3:
            roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi_img
        
        # Apply Gaussian blur to reduce noise
        roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        
        # Apply adaptive thresholding to handle different lighting conditions
        roi_thresh = cv2.adaptiveThreshold(
            roi_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Use morphological operations to enhance vertical structures
        # Create a vertical kernel
        kernel_size = max(5, min((y2-y1) // 20, 20))  # Adaptive kernel size based on ROI height
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
        
        # Apply morphological operations: opening to isolate vertical lines
        roi_vertical = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, vert_kernel)
        
        # Use Canny edge detector with appropriate thresholds
        edges = cv2.Canny(roi_vertical, 50, 150)
        
        # Use probabilistic Hough transform to detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=40,
            minLineLength=min_line_length, maxLineGap=max_line_gap
        )
        
        # No lines detected
        if lines is None:
            return []
        
        # Store vertical line candidates (lines with very small x-deviation)
        vertical_edges = []
        
        for line in lines:
            x1_line, y1_line, x2_line, y2_line = line[0]
            
            # Check if the line is mostly vertical (small x difference, large y difference)
            x_diff = abs(x2_line - x1_line)
            y_diff = abs(y2_line - y1_line)
            
            # Define verticality as lines with x_diff much smaller than y_diff
            if x_diff < 10 and y_diff > min_line_length:
                # Calculate the center x-coordinate of the line
                center_x = (x1_line + x2_line) // 2
                vertical_edges.append(center_x)
        
        # Convert local ROI coordinates back to global image coordinates
        return [x1 + x for x in vertical_edges]

    def _refine_balanced_boundaries(self, 
                                image: np.ndarray, 
                                boundaries: List[Tuple[int, int, int, int]],
                                margin: int = 20) -> List[Tuple[int, int, int, int]]:
        """
        Refine compartment boundaries by finding walls on both sides and adjusting position.
        Maintains original compartment widths while ensuring walls aren't included.
        
        Args:
            image: Input image
            boundaries: Initial compartment boundaries from ArUco markers
            margin: Margin around expected walls to search for actual walls
            
        Returns:
            Refined compartment boundaries with preserved widths
        """
        if not boundaries or image is None:
            return boundaries
        
        # Store visualization elements instead of creating an image
        viz_elements = {
            'search_regions': [],  # Store (x1, y1, x2, y2, color, thickness)
            'detected_edges': [],  # Store (x1, y1, x2, y2, color, thickness)
            'final_boundaries': []  # Store (x1, y1, x2, y2, color, thickness)
        }
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # First pass: collect all wall positions
        compartment_data = []
        for i, (x1, y1, x2, y2) in enumerate(boundaries):
            # Fixed compartment width from markers
            compartment_width = x2 - x1
            
            # Define ROIs to search for edges
            left_roi = (max(0, x1 - margin), y1, min(image.shape[1], x1 + margin), y2)
            right_roi = (max(0, x2 - margin), y1, min(image.shape[1], x2 + margin), y2)
            
            # Store search regions for visualization
            viz_elements['search_regions'].append((left_roi[0], left_roi[1], left_roi[2], left_roi[3], (0, 255, 255), 1))
            viz_elements['search_regions'].append((right_roi[0], right_roi[1], right_roi[2], right_roi[3], (0, 255, 255), 1))
            
            # Detect vertical edges on both sides
            left_edges = self._detect_vertical_edges_in_roi(gray, left_roi)
            right_edges = self._detect_vertical_edges_in_roi(gray, right_roi)
            
            # Find best wall positions
            best_left_edge = None
            best_right_edge = None
            
            # Find the closest left edge to the expected position
            if left_edges:
                # Sort edges by distance to expected position
                left_edges.sort(key=lambda x: abs(x - x1))
                best_left_edge = left_edges[0]
            
            # Find the closest right edge to the expected position
            if right_edges:
                # Sort edges by distance to expected position
                right_edges.sort(key=lambda x: abs(x - x2))
                best_right_edge = right_edges[0]
                
            # Store compartment data with detected edges
            compartment_data.append({
                'original': (x1, y1, x2, y2),
                'width': compartment_width,
                'left_edge': best_left_edge,
                'right_edge': best_right_edge,
                'compartment_index': i  # Store the compartment index
            })
        
            # Store detected wall positions for visualization (with colors based on even/odd)
            is_even = (i % 2 == 0)
            left_edge_color = (255, 0, 0) if is_even else (0, 0, 255)  # Blue vs Red in BGR
            right_edge_color = (0, 255, 0) if is_even else (255, 0, 255)  # Green vs Magenta in BGR
            
            if best_left_edge is not None:
                viz_elements['detected_edges'].append((best_left_edge, y1, best_left_edge, y2, left_edge_color, 2))
            if best_right_edge is not None:
                viz_elements['detected_edges'].append((best_right_edge, y1, best_right_edge, y2, right_edge_color, 2))
        
        # Start wall detection debug summary
        self.logger.debug("\n=== Wall Detection Summary ===")
        
        # Initialize refined bounds with original boundaries as starting point
        refined_bounds = [comp['original'] for comp in compartment_data]
        buffer = 2  # Buffer pixels to avoid including the wall
        
        # Process compartments in both directions
        for direction in ["left_to_right", "right_to_left"]:
            self.logger.debug(f"\n--- {direction.replace('_', ' ').title()} Pass ---")
            
            # Determine processing order
            indices = range(len(compartment_data)) if direction == "left_to_right" else range(len(compartment_data)-1, -1, -1)
            
            for idx in indices:
                comp_data = compartment_data[idx]
                original_x1, y1, original_x2, y2 = comp_data['original']
                width = comp_data['width']
                left_edge = comp_data['left_edge']
                right_edge = comp_data['right_edge']
                compartment_index = comp_data['compartment_index']
                
                # Get current boundary (potentially already adjusted in previous pass)
                current_x1, current_y1, current_x2, current_y2 = refined_bounds[idx]
                
                # Calculate max allowed adjustment (10% of width)
                max_adjustment = width * 0.1
                
                # Start building summary for this compartment
                summary = f"Compartment {idx+1} ({direction}): "
                
                # Proposed new positions (default to current)
                new_x1, new_x2 = current_x1, current_x2
                
                # Case 1: Both edges detected - center the compartment while maintaining width
                if left_edge is not None and right_edge is not None:
                    # Calculate midpoint between walls with buffer
                    midpoint = (left_edge + buffer + right_edge - buffer) / 2
                    # Calculate proposed positions centered at midpoint
                    proposed_x1 = midpoint - (width / 2)
                    proposed_x2 = midpoint + (width / 2)
                    
                    # Check if adjustment is within allowed range
                    left_adjustment = abs(proposed_x1 - original_x1)
                    right_adjustment = abs(proposed_x2 - original_x2)
                    
                    if left_adjustment <= max_adjustment and right_adjustment <= max_adjustment:
                        new_x1, new_x2 = proposed_x1, proposed_x2
                        summary += f"Both walls detected. Centered within limits. "
                    else:
                        # Limit the adjustment
                        if left_adjustment > max_adjustment:
                            # Limit left adjustment
                            if proposed_x1 < original_x1:
                                new_x1 = original_x1 - max_adjustment
                            else:
                                new_x1 = original_x1 + max_adjustment
                            new_x2 = new_x1 + width
                            summary += f"Both walls detected. Left adjustment limited to {max_adjustment:.1f}px. "
                        else:
                            # Limit right adjustment
                            if proposed_x2 < original_x2:
                                new_x2 = original_x2 - max_adjustment
                            else:
                                new_x2 = original_x2 + max_adjustment
                            new_x1 = new_x2 - width
                            summary += f"Both walls detected. Right adjustment limited to {max_adjustment:.1f}px. "
                
                # Case 2: Only left edge detected
                elif left_edge is not None:
                    min_safe_x1 = left_edge + buffer  # Minimum safe position
                    
                    if current_x1 < min_safe_x1:
                        # Need to shift right to avoid the wall
                        shift_amount = min(max_adjustment, min_safe_x1 - current_x1)
                        new_x1 = current_x1 + shift_amount
                        new_x2 = new_x1 + width
                        summary += f"Left wall detected. Shifted right by {shift_amount:.1f}px (limited). "
                    else:
                        summary += "Left wall detected but already clear. "
                
                # Case 3: Only right edge detected
                elif right_edge is not None:
                    max_safe_x2 = right_edge - buffer  # Maximum safe position
                    
                    if current_x2 > max_safe_x2:
                        # Need to shift left to avoid the wall
                        shift_amount = min(max_adjustment, current_x2 - max_safe_x2)
                        new_x2 = current_x2 - shift_amount
                        new_x1 = new_x2 - width
                        summary += f"Right wall detected. Shifted left by {shift_amount:.1f}px (limited). "
                    else:
                        summary += "Right wall detected but already clear. "
                
                # Case 4: No walls detected
                else:
                    summary += "No walls detected. "
                
                # Check if adjusted position would overlap with adjacent compartments
                # and adjust if necessary
                overlap_fixed = False
                
                # Check left overlap (with previous compartment)
                if idx > 0 and direction == "left_to_right":
                    prev_x1, _, prev_x2, _ = refined_bounds[idx-1]
                    if new_x1 < prev_x2:  # Overlap detected
                        shift_needed = prev_x2 - new_x1
                        # Only shift if it doesn't push into a right wall
                        if right_edge is None or new_x2 + shift_needed < right_edge - buffer:
                            new_x1 = prev_x2
                            new_x2 = new_x1 + width
                            summary += f"Fixed left overlap ({shift_needed:.1f}px). "
                            overlap_fixed = True
                
                # Check right overlap (with next compartment)
                if idx < len(refined_bounds)-1 and direction == "right_to_left":
                    next_x1, _, next_x2, _ = refined_bounds[idx+1]
                    if new_x2 > next_x1:  # Overlap detected
                        shift_needed = new_x2 - next_x1
                        # Only shift if it doesn't push into a left wall
                        if left_edge is None or new_x1 - shift_needed > left_edge + buffer:
                            new_x2 = next_x1
                            new_x1 = new_x2 - width
                            summary += f"Fixed right overlap ({shift_needed:.1f}px). "
                            overlap_fixed = True
                
                # Update refined boundary if changes were made
                if new_x1 != current_x1 or new_x2 != current_x2:
                    refined_bounds[idx] = (int(new_x1), y1, int(new_x2), y2)
                    final_width = new_x2 - new_x1
                    shift_desc = f"Moved from ({current_x1:.1f}, {current_x2:.1f}) to ({new_x1:.1f}, {new_x2:.1f}). "
                    summary += shift_desc + f"Final width: {final_width:.1f}px."
                    self.logger.debug(summary)
                else:
                    summary += "No change needed."
                    self.logger.debug(summary)
        
        # Final validation pass - check for any remaining overlaps
        self.logger.debug("\n--- Final Validation ---")
        has_overlaps = False
        
        for i in range(len(refined_bounds)-1):
            _, _, current_right, _ = refined_bounds[i]
            next_left, _, _, _ = refined_bounds[i+1]
            
            if current_right > next_left:
                has_overlaps = True
                self.logger.warning(
                    f"Remaining overlap between compartments {i+1} and {i+2}: "
                    f"{current_right-next_left:.1f}px overlap"
                )
        
        if not has_overlaps:
            self.logger.debug("Final validation completed - no overlaps detected.")
        
        # Store final boundaries for visualization with colors based on compartment index
        for i, (x1, y1, x2, y2) in enumerate(refined_bounds):
            is_even = (i % 2 == 0)
            boundary_color = (0, 255, 255) if is_even else (255, 255, 0)
            viz_elements['final_boundaries'].append((x1, y1, x2, y2, boundary_color, 1))
        
        self.logger.debug("=== End Wall Detection Summary ===\n")
        
        # Store visualization elements in viz_steps
        if hasattr(self, 'viz_steps'):
            self.viz_steps['wall_detection_elements'] = viz_elements
        
        return refined_bounds
    
