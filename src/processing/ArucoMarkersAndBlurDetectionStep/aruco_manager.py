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
                self.logger.debug(f"DEBUG: Stored {len(final_markers)} markers in visualization_cache")
            
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
    

    def correct_marker_geometry(self, markers: Dict[int, np.ndarray],
                            tolerance_pixels: float = 3,
                            preserve_orientation: bool = False) -> Dict[int, np.ndarray]:
        """
        Correct distorted markers to perfect squares.
        This is called AFTER skew correction, so markers are already properly aligned.
        
        Args:
            markers: Dictionary of detected markers {id: corners}
            tolerance_pixels: Maximum edge length variance allowed
            preserve_orientation: Whether to preserve each marker's rotation
            
        Returns:
            Dictionary of corrected markers with same IDs
        """
        corrected_markers = {}
        
        for marker_id, corners in markers.items():
            # Analyze marker geometry
            edge_data = self._analyze_marker_geometry(corners, tolerance_pixels)
            
            if not edge_data['needs_correction']:
                # Marker is already good
                corrected_markers[marker_id] = corners
                continue
            
            if not edge_data['is_correctable']:
                # Too distorted to correct
                self.logger.warning(f"Marker {marker_id} too distorted to correct")
                corrected_markers[marker_id] = corners
                continue
            
            if preserve_orientation:
                # Preserve the marker's current orientation
                corrected_corners = self._create_oriented_square(
                    edge_data['center'],
                    edge_data['median_edge_length'],
                    edge_data['orientation_angle']
                )
            else:
                # Create axis-aligned square
                corrected_corners = self._create_axis_aligned_square(
                    edge_data['center'],
                    edge_data['median_edge_length']
                )
            
            corrected_markers[marker_id] = corrected_corners
            
            self.logger.debug(
                f"Corrected marker {marker_id}: CV {edge_data['edge_cv']:.2%} -> 0%, "
                f"orientation: {edge_data['orientation_angle']:.1f}°"
            )
        
        return corrected_markers

    def estimate_scale_from_markers(self, markers: Dict[int, np.ndarray],
                                marker_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate image scale from detected markers with known physical sizes.
        
        Args:
            markers: Dictionary of detected markers {id: corners}
            marker_config: Configuration dictionary containing:
                - 'corner_marker_size_cm': Physical size of corner markers
                - 'compartment_marker_size_cm': Physical size of compartment markers
                - 'corner_ids': List of corner marker IDs
                - 'compartment_ids': List of compartment marker IDs
                - 'metadata_ids': List of metadata marker IDs
                - 'use_corner_markers': Whether to include corner markers
                
        Returns:
            Dictionary containing scale estimation results
        """
        if not markers:
            self.logger.warning("No markers provided for scale estimation")
            return {
                'scale_px_per_cm': None,
                'image_width_cm': None,
                'marker_measurements': [],
                'confidence': 0.0
            }
        # Extract configuration values
        corner_size_cm = marker_config.get('corner_marker_size_cm', 1.0)
        compartment_size_cm = marker_config.get('compartment_marker_size_cm', 2.0)
        corner_ids = marker_config.get('corner_ids', [0, 1, 2, 3])
        compartment_ids = marker_config.get('compartment_ids', list(range(4, 24)))
        metadata_ids = marker_config.get('metadata_ids', [24])
        use_corner_markers = marker_config.get('use_corner_markers', True)
        
        # Collect scale measurements
        scale_measurements = []
        marker_measurements = []
        
        for marker_id, corners in markers.items():
            # Skip metadata markers
            if marker_id in metadata_ids:
                continue
            
            # Determine marker type and size
            if marker_id in corner_ids:
                if not use_corner_markers:
                    continue
                physical_size_cm = corner_size_cm
                marker_type = "corner"
            elif marker_id in compartment_ids:
                physical_size_cm = compartment_size_cm
                marker_type = "compartment"
            else:
                continue
            
            # Measure this marker
            measurements = self._measure_marker_for_scale(
                corners, physical_size_cm, marker_id, marker_type
            )
            
            if measurements['valid']:
                scale_measurements.extend(measurements['scales'])
                marker_measurements.append(measurements)
        
        # Calculate robust scale estimate
        if not scale_measurements:
            self.logger.warning("No valid scale measurements obtained")
            return {
                'scale_px_per_cm': None,
                'image_width_cm': None,
                'marker_measurements': marker_measurements,
                'confidence': 0.0
            }
        
        # Remove outliers using IQR method
        scale_array = np.array(scale_measurements)
        q1, q3 = np.percentile(scale_array, [25, 75])
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered_scales = scale_array[(scale_array >= lower_bound) & (scale_array <= upper_bound)]
        
        if len(filtered_scales) == 0:
            filtered_scales = scale_array
            self.logger.warning("All scale measurements were outliers, using unfiltered data")
        
        # Calculate final scale
        final_scale_px_per_cm = float(np.median(filtered_scales))
        
        # Calculate confidence
        if len(filtered_scales) > 1:
            std_dev = np.std(filtered_scales)
            mean_val = np.mean(filtered_scales)
            cv = std_dev / mean_val if mean_val > 0 else 1.0
            confidence = max(0.0, min(1.0, 1.0 - (cv - 0.02) / 0.08))
        else:
            confidence = 0.0
        
        return {
            'scale_px_per_cm': final_scale_px_per_cm,
            'image_width_cm': None,  # Let the caller calculate this if needed
            'marker_measurements': marker_measurements,
            'confidence': confidence,
            'outliers_removed': len(scale_measurements) - len(filtered_scales),
            'total_measurements': len(scale_measurements)
        }

    def _measure_marker_for_scale(self, corners: np.ndarray, physical_size_cm: float, 
                                marker_id: int, marker_type: str) -> Dict[str, Any]:
        """
        Measure a single marker to estimate scale.
        
        Args:
            corners: 4x2 array of corner coordinates
            physical_size_cm: Known physical size of the marker in cm
            marker_id: ID of the marker
            marker_type: Type of marker ('corner' or 'compartment')
            
        Returns:
            Dictionary with measurement results
        """
        # Calculate edge lengths
        edge_lengths = []
        for i in range(4):
            edge_length = np.linalg.norm(corners[(i + 1) % 4] - corners[i])
            edge_lengths.append(edge_length)
        
        # Calculate diagonal lengths
        diagonal_lengths = [
            np.linalg.norm(corners[2] - corners[0]),  # Top-left to bottom-right
            np.linalg.norm(corners[3] - corners[1])   # Top-right to bottom-left
        ]
        
        # Calculate scales from each edge
        scales = []
        for edge_length in edge_lengths:
            if edge_length > 0:
                scale = edge_length / physical_size_cm
                scales.append(scale)
        
        # Check if measurements are valid
        if not scales:
            return {
                'valid': False,
                'marker_id': marker_id,
                'marker_type': marker_type,
                'scales': [],
                'edge_lengths': edge_lengths,
                'diagonal_lengths': diagonal_lengths
            }
        
        # Calculate coefficient of variation to check consistency
        mean_edge = np.mean(edge_lengths)
        std_edge = np.std(edge_lengths)
        edge_cv = std_edge / mean_edge if mean_edge > 0 else float('inf')
        
        # Marker is valid if edges are reasonably consistent (CV < 10%)
        is_valid = edge_cv < 0.1
        
        return {
            'valid': is_valid,
            'marker_id': marker_id,
            'marker_type': marker_type,
            'scales': scales,
            'edge_lengths': edge_lengths,
            'diagonal_lengths': diagonal_lengths,
            'avg_edge_length': mean_edge,
            'edge_cv': edge_cv,
            'physical_size_cm': physical_size_cm
        }

    def auto_place_missing_compartments(self, existing_markers, missing_ids, 
                                      scale_data=None, config=None,
                                      image_shape=None):
        """
        Automatically interpolate positions for missing compartment markers.
        
        This method tries to place missing compartments by:
        1. Interpolating between adjacent existing markers
        2. Extrapolating at edges based on average spacing
        3. Using scale data for proper marker sizing
        4. Respecting configured spacing between compartments
        
        Args:
            existing_markers: Dict of {marker_id: corners_array} for detected markers
            missing_ids: List of missing marker IDs
            scale_data: Dict with scale information (scale_px_per_cm)
            config: Configuration dict
            image_shape: Shape of the image (height, width) for bounds checking
            
        Returns:
            Dict of {marker_id: corners_array} for successfully placed markers
        """
        if not missing_ids or not existing_markers:
            return {}
            
        self.logger.info(f"Attempting to auto-place {len(missing_ids)} missing compartments: {sorted(missing_ids)}")
        
        # Get compartment marker IDs and filter to only compartment markers
        compartment_ids = config.get('compartment_marker_ids', list(range(4, 24))) if config else list(range(4, 24))
        corner_ids = config.get('corner_marker_ids', [0, 1, 2, 3]) if config else [0, 1, 2, 3]
        
        # Extract only compartment markers
        comp_markers = {mid: corners for mid, corners in existing_markers.items() 
                       if mid in compartment_ids}
        
        if not comp_markers:
            self.logger.warning("No existing compartment markers to interpolate from")
            return {}
        
        # Get spacing configuration
        default_spacing_mm = config.get('compartment_spacing_mm', 3.0) if config else 3.0
        spacing_overrides = config.get('compartment_spacing_overrides', {}) if config else {}
        compartment_width_cm = config.get('compartment_width_cm', 2.0) if config else 2.0
        
        # Calculate marker positions and sizes
        marker_info = {}
        for mid, corners in comp_markers.items():
            center_x = np.mean(corners[:, 0])
            center_y = np.mean(corners[:, 1])
            width = np.max(corners[:, 0]) - np.min(corners[:, 0])
            height = np.max(corners[:, 1]) - np.min(corners[:, 1])
            marker_info[mid] = {
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'corners': corners,
                'compartment_num': mid - 3  # Convert marker ID to compartment number
            }
        
        # Sort by marker ID to maintain order
        sorted_ids = sorted(marker_info.keys())
        
        # Get scale information
        scale_px_per_cm = None
        if scale_data:
            scale_px_per_cm = scale_data.get('scale_px_per_cm')
        
        # Calculate expected spacing between consecutive markers
        # This now accounts for variable spacing
        def get_spacing_between_compartments(comp1, comp2, scale_px_per_cm):
            """Get expected spacing between two compartment numbers in pixels."""
            # Check for override
            key = f"{comp1}-{comp2}"
            spacing_mm = spacing_overrides.get(key, default_spacing_mm)
            
            if scale_px_per_cm:
                # Convert mm to pixels
                spacing_cm = spacing_mm / 10.0
                spacing_px = spacing_cm * scale_px_per_cm
                
                # Add compartment width
                total_spacing_px = spacing_px + (compartment_width_cm * scale_px_per_cm)
                return total_spacing_px
            else:
                # Fallback: use detected average
                return None
        
        # Calculate actual spacings from detected markers
        detected_spacings = []
        for i in range(len(sorted_ids) - 1):
            id1, id2 = sorted_ids[i], sorted_ids[i + 1]
            comp1 = marker_info[id1]['compartment_num']
            comp2 = marker_info[id2]['compartment_num']
            
            actual_spacing = marker_info[id2]['center_x'] - marker_info[id1]['center_x']
            
            # Only include if consecutive
            if comp2 - comp1 == 1 and actual_spacing > 0:
                detected_spacings.append(actual_spacing)
        
        avg_detected_spacing = np.mean(detected_spacings) if detected_spacings else None
        
        # Get average marker size
        avg_width = np.mean([info['width'] for info in marker_info.values()])
        avg_height = np.mean([info['height'] for info in marker_info.values()])
        
        # Use scale data if available for marker size
        if scale_data and config:
            marker_size_cm = config.get('compartment_marker_size_cm', 2.0)
            if scale_px_per_cm:
                expected_marker_size = marker_size_cm * scale_px_per_cm
                # Use scale-based size if significantly different from detected
                if abs(expected_marker_size - avg_width) > avg_width * 0.2:
                    self.logger.debug(f"Using scale-based marker size: {expected_marker_size:.1f}px instead of {avg_width:.1f}px")
                    avg_width = expected_marker_size
                    avg_height = expected_marker_size  # Assume square markers
        
        # Try to place each missing marker
        placed_markers = {}
        
        for missing_id in sorted(missing_ids):
            # Skip non-compartment markers
            if missing_id not in compartment_ids:
                continue
                
            placed_corners = None
            placement_method = None
            missing_comp_num = missing_id - 3  # Convert to compartment number
            
            # Method 1: Interpolation between adjacent markers
            prev_id = missing_id - 1
            next_id = missing_id + 1
            
            if prev_id in marker_info and next_id in marker_info:
                # For interpolation, check if special spacing applies
                prev_comp = marker_info[prev_id]['compartment_num']
                next_comp = marker_info[next_id]['compartment_num']
                
                # Check if there's special spacing that affects this interpolation
                if scale_px_per_cm and (f"{prev_comp}-{missing_comp_num}" in spacing_overrides or 
                                       f"{missing_comp_num}-{next_comp}" in spacing_overrides):
                    # Use precise spacing calculation
                    prev_to_missing_spacing = get_spacing_between_compartments(prev_comp, missing_comp_num, scale_px_per_cm)
                    
                    if prev_to_missing_spacing:
                        center_x = marker_info[prev_id]['center_x'] + prev_to_missing_spacing
                        center_y = marker_info[prev_id]['center_y']
                        placement_method = "spacing-aware interpolation"
                    else:
                        # Fallback to simple interpolation
                        center_x = (marker_info[prev_id]['center_x'] + marker_info[next_id]['center_x']) / 2
                        center_y = (marker_info[prev_id]['center_y'] + marker_info[next_id]['center_y']) / 2
                        placement_method = "simple interpolation"
                else:
                    # Simple linear interpolation
                    center_x = (marker_info[prev_id]['center_x'] + marker_info[next_id]['center_x']) / 2
                    center_y = (marker_info[prev_id]['center_y'] + marker_info[next_id]['center_y']) / 2
                    placement_method = "interpolation"
                
                placed_corners = self._create_marker_corners(center_x, center_y, avg_width, avg_height)
                
            # Method 2: Extrapolation from nearby markers
            elif prev_id in marker_info:
                # Extrapolate forward
                prev_comp = marker_info[prev_id]['compartment_num']
                
                # Use configured spacing if available
                if scale_px_per_cm:
                    spacing = get_spacing_between_compartments(prev_comp, missing_comp_num, scale_px_per_cm)
                else:
                    spacing = avg_detected_spacing
                
                if spacing:
                    center_x = marker_info[prev_id]['center_x'] + spacing
                    center_y = marker_info[prev_id]['center_y']
                    placed_corners = self._create_marker_corners(center_x, center_y, avg_width, avg_height)
                    placement_method = "forward extrapolation"
                    
            elif next_id in marker_info:
                # Extrapolate backward
                next_comp = marker_info[next_id]['compartment_num']
                
                # Use configured spacing if available
                if scale_px_per_cm:
                    spacing = get_spacing_between_compartments(missing_comp_num, next_comp, scale_px_per_cm)
                else:
                    spacing = avg_detected_spacing
                
                if spacing:
                    center_x = marker_info[next_id]['center_x'] - spacing
                    center_y = marker_info[next_id]['center_y']
                    placed_corners = self._create_marker_corners(center_x, center_y, avg_width, avg_height)
                    placement_method = "backward extrapolation"
            
            # Validate placement
            if placed_corners is not None:
                # Check if within image bounds
                if image_shape:
                    h, w = image_shape[:2]
                    x_coords = placed_corners[:, 0]
                    y_coords = placed_corners[:, 1]
                    
                    if (np.all(x_coords >= 0) and np.all(x_coords < w) and
                        np.all(y_coords >= 0) and np.all(y_coords < h)):
                        placed_markers[missing_id] = placed_corners
                        self.logger.info(f"Auto-placed marker {missing_id} (compartment {missing_comp_num}) using {placement_method} at x={np.mean(x_coords):.1f}")
                    else:
                        self.logger.debug(f"Marker {missing_id} placement out of bounds")
                else:
                    # No bounds checking
                    placed_markers[missing_id] = placed_corners
                    self.logger.info(f"Auto-placed marker {missing_id} using {placement_method}")
        
        self.logger.info(f"Successfully auto-placed {len(placed_markers)} out of {len(missing_ids)} missing compartments")
        return placed_markers


    def _analyze_marker_geometry(self, corners: np.ndarray, 
                            tolerance_pixels: float) -> Dict[str, Any]:
        """
        Analyze marker geometry to determine if correction is needed.
        
        Returns dictionary with analysis results.
        """
        # Calculate edge lengths
        edges = []
        edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        
        for i, j in edge_indices:
            edge_length = np.linalg.norm(corners[i] - corners[j])
            edges.append(edge_length)
        
        edges = np.array(edges)
        median_edge = np.median(edges)
        edge_std = np.std(edges)
        edge_mean = np.mean(edges)
        edge_cv = edge_std / edge_mean if edge_mean > 0 else 1.0
        
        # Calculate center and orientation
        center = np.mean(corners, axis=0)
        
        # Get orientation from first edge
        edge_vector = corners[1] - corners[0]
        orientation_angle = np.degrees(np.arctan2(edge_vector[1], edge_vector[0]))
        
        # Determine if correction needed
        needs_correction = edge_cv > 0.02  # More than 2% variation
        
        # Check if correctable (at least 2 valid edges)
        valid_edges = sum(1 for e in edges if abs(e - median_edge) <= tolerance_pixels)
        is_correctable = valid_edges >= 2
        
        return {
            'edges': edges,
            'median_edge_length': median_edge,
            'edge_cv': edge_cv,
            'center': center,
            'orientation_angle': orientation_angle,
            'needs_correction': needs_correction,
            'is_correctable': is_correctable,
            'valid_edge_count': valid_edges
        }

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
            metadata: Optional metadata dictionary containing hole_id, depth_from, depth_to,
            
        Returns:
            Dictionary containing:
                - boundaries: List of detected compartment boundaries
                - missing_marker_ids: List of missing marker IDs
                - vertical_constraints: Tuple of (top_y, bottom_y) for compartment placement
                - marker_to_compartment: Mapping of marker IDs to compartment numbers
                - corner_markers: Dictionary of corner marker positions
                - viz_steps: Dictionary of visualization steps which will be used during the dialog to display data dynamically
        """
        self.logger.debug(f"DEBUG: Starting analyze_compartment_boundaries with {len(markers) if markers else 0} markers")
        
        # ===================================================
        # INSERT: Early validation check
        if image is None or not markers:
            self.logger.debug("DEBUG: Cannot analyze boundaries - no image or markers")
            self.logger.warning("Cannot analyze boundaries - no image or markers")
            # Return minimal result structure
            return {
                'boundaries': [],
                'missing_marker_ids': [],
                'vertical_constraints': (0, 0),
                'marker_to_compartment': {},
                'corner_markers': {},
                'boundary_to_marker': {},
                'marker_data': [],
                'detected_compartment_markers': {},
                'expected_marker_ids': [],
                'scale_px_per_cm': None,
                'compartment_width_cm': None,
                'compartment_height_cm': None,
                'compartment_width_px': None,
                'compartment_height_px': None,
                'wall_detection_performed': smart_cropping,
                'wall_detection_results': {},
                'compartment_interval': 1
            }
        # ===================================================

        # Initialize variables that will be populated later
        compartment_boundaries = []
        missing_marker_ids = []
        top_y = 0
        bottom_y = 0
        marker_to_compartment = {}
        corner_markers = {}
        boundary_to_marker = {}
        marker_data = []
        comp_markers = {}
        expected_marker_ids = []
        compartment_interval = 1
        
        # ===================================================
        # GET SCALE DATA EARLY
        # ===================================================
        scale_px_per_cm = None
        compartment_width_cm = self.config.get('compartment_width_cm', 2.0)
        compartment_height_cm = self.config.get('compartment_height_cm', 4.2)
        compartment_width_px = None
        compartment_height_px = None
        
        # Try to get scale from metadata first
        if metadata and 'scale_px_per_cm' in metadata:
            scale_px_per_cm = metadata['scale_px_per_cm']
            self.logger.debug(f"Got scale from metadata: {scale_px_per_cm:.2f} px/cm")
        # Then try from viz_manager
        elif hasattr(self.app, 'viz_manager') and self.app.viz_manager.processing_metadata.get('scale_px_per_cm'):
            scale_px_per_cm = self.app.viz_manager.processing_metadata['scale_px_per_cm']
            self.logger.debug(f"Got scale from viz_manager: {scale_px_per_cm:.2f} px/cm")
        
        # Calculate pixel dimensions if we have scale
        if scale_px_per_cm and scale_px_per_cm > 0:
            compartment_width_px = int(compartment_width_cm * scale_px_per_cm)
            compartment_height_px = int(compartment_height_cm * scale_px_per_cm)
            self.logger.info(f"Compartment dimensions: {compartment_width_cm}cm x {compartment_height_cm}cm = {compartment_width_px}px x {compartment_height_px}px")
        else:
            self.logger.warning("No scale data available - compartment dimensions will be based on marker sizes")
        
        # STEP 1: Get marker ID ranges from config
        corner_ids = self.config.get('corner_marker_ids', [0, 1, 2, 3])
        compartment_ids = self.config.get('compartment_marker_ids', list(range(4, 24)))
        metadata_ids = self.config.get('metadata_marker_ids', [24])
        
        # Get compartment interval from config or metadata
        compartment_interval = metadata.get('compartment_interval', 
                                        self.config.get('compartment_interval', 1.0))
        compartment_interval = int(compartment_interval)
        
        # Get the configured compartment count or use default
        expected_compartment_count = compartment_count or self.config.get('compartment_count', 20)

        # STEP 2: Define ROI using corner markers for vertical boundaries
        corner_markers = {mid: corners for mid, corners in markers.items() if mid in corner_ids}
        
        if not corner_markers:
            self.logger.warning("No corner markers detected")
            # If we have scale, use it to set a reasonable height
            if compartment_height_px:
                # Center the compartment area in the image
                center_y = image.shape[0] // 2
                top_y = max(0, center_y - compartment_height_px // 2)
                bottom_y = min(image.shape[0], center_y + compartment_height_px // 2)
                self.logger.info(f"No corner markers - using centered {compartment_height_px}px height")
            else:
                # Fallback to image boundaries
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
            
            # Set initial boundaries based on detected corners
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
            
            # ===================================================
            # ENFORCE COMPARTMENT HEIGHT SETTINGS FROM CONFIG IF SCALE IS AVAILABLE
            # ===================================================
            if compartment_height_px and compartment_height_px > 0:
                current_height = bottom_y - top_y
                
                if abs(current_height - compartment_height_px) > 5:  # Allow 5px tolerance
                    # Calculate how much to adjust
                    height_diff = compartment_height_px - current_height
                    
                    # Center the markers within the new height
                    center_y = (top_y + bottom_y) // 2
                    new_top_y = int(center_y - compartment_height_px // 2)
                    new_bottom_y = int(center_y + compartment_height_px // 2)
                    
                    # Ensure we stay within image bounds
                    if new_top_y < 0:
                        new_top_y = 0
                        new_bottom_y = min(compartment_height_px, image.shape[0])
                    elif new_bottom_y > image.shape[0]:
                        new_bottom_y = image.shape[0]
                        new_top_y = max(0, image.shape[0] - compartment_height_px)
                    
                    self.logger.info(f"Adjusted vertical constraints from {current_height}px to {compartment_height_px}px (4cm)")
                    self.logger.debug(f"Old bounds: {top_y}-{bottom_y}, New bounds: {new_top_y}-{new_bottom_y}")
                    
                    top_y = new_top_y
                    bottom_y = new_bottom_y
            
        # STEP 3: Extract and sort compartment markers
        comp_markers = {}
        for mid in markers:
            if mid in compartment_ids and mid not in corner_ids and mid not in metadata_ids:
                comp_markers[mid] = markers[mid]
        
        self.logger.debug(f"DEBUG: Found {len(comp_markers)} compartment markers: {sorted(comp_markers.keys())}")
        
        if not comp_markers:
            self.logger.debug("DEBUG: No compartment markers detected")
            self.logger.warning("No compartment markers detected")
            return {
                'boundaries': [],
                'missing_marker_ids': missing_marker_ids,
                'vertical_constraints': (top_y, bottom_y),
                'marker_to_compartment': marker_to_compartment,
                'corner_markers': corner_markers,
                'boundary_to_marker': {},
                'marker_data': [],
                'detected_compartment_markers': {},
                'expected_marker_ids': expected_marker_ids,
                'scale_px_per_cm': scale_px_per_cm,
                'compartment_width_cm': compartment_width_cm,
                'compartment_height_cm': compartment_height_cm,
                'compartment_width_px': compartment_width_px,
                'compartment_height_px': compartment_height_px,
                'wall_detection_performed': smart_cropping,
                'wall_detection_results': {},
                'compartment_interval': compartment_interval
            }

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
        
        # Create mapping between marker IDs and compartment numbers
        marker_to_compartment = {
            4+i: int((i+1) * compartment_interval) for i in range(expected_compartment_count)
        }

        # STEP 6: Identify missing markers
        expected_marker_ids = list(range(4, 4 + expected_compartment_count))
        detected_marker_ids = [marker['id'] for marker in marker_data]
        missing_marker_ids = [marker_id for marker_id in expected_marker_ids if marker_id not in detected_marker_ids]
        
        if missing_marker_ids:
            self.logger.info(f"Missing {len(missing_marker_ids)} markers: {missing_marker_ids}")

        # STEP 7: Generate initial compartment boundaries
        compartment_boundaries = []
        
        for marker in marker_data:
            if marker['id'] not in expected_marker_ids:
                continue
            
            # ===================================================
            # USE SCALE-BASED WIDTH IF AVAILABLE
            # ===================================================
            if compartment_width_px and compartment_width_px > 0:
                # Use the calculated pixel width based on 2.1cm
                compartment_width = compartment_width_px
                self.logger.debug(f"Using scale-based width: {compartment_width}px (2.1cm)")
            else:
                # Fallback to marker width
                compartment_width = int(marker['width'] * 1.0)
                self.logger.debug(f"Using marker-based width: {compartment_width}px")
            
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
            self.logger.debug(f"DEBUG: Applying boundary adjustments: {adjustments}")
            
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
            
        # Apply wall detection refinement if enabled
        if smart_cropping and compartment_boundaries:
            self.logger.debug("DEBUG: Starting wall detection refinement")
            # Pass scale data to refinement if available
            refined_boundaries, viz_elements = self._refine_balanced_boundaries(
                image, compartment_boundaries, 5, scale_px_per_cm)
            compartment_boundaries = refined_boundaries
        
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
        
        # Build complete result dictionary
        result = {
            'boundaries': compartment_boundaries,
            'missing_marker_ids': missing_marker_ids,
            'vertical_constraints': (top_y, bottom_y),
            'marker_to_compartment': marker_to_compartment,
            'corner_markers': corner_markers,
            'boundary_to_marker': boundary_to_marker,
            
            # Add marker analysis data
            'marker_data': marker_data,
            'detected_compartment_markers': comp_markers,
            'expected_marker_ids': expected_marker_ids,
            
            # Add scale-based calculations
            'scale_px_per_cm': scale_px_per_cm,
            'compartment_width_cm': compartment_width_cm,
            'compartment_height_cm': compartment_height_cm,
            'compartment_width_px': compartment_width_px,
            'compartment_height_px': compartment_height_px,
            
            # Wall detection results if performed
            'wall_detection_performed': smart_cropping,
            'wall_detection_results': {},
            
            # Processing metadata
            'compartment_interval': compartment_interval
        }
        
        return result

    def _refine_balanced_boundaries(self, 
                                image: np.ndarray, 
                                boundaries: List[Tuple[int, int, int, int]],
                                margin: int = 5,
                                scale_px_per_cm: float = None) -> List[Tuple[int, int, int, int]]:
        """
        Refine compartment boundaries by finding walls on both sides and adjusting position.
        Maintains compartment widths while ensuring walls aren't included.
        
        Args:
            image: Input image
            boundaries: Initial compartment boundaries from ArUco markers
            margin: Margin around expected walls to search for actual walls (in pixels)
            scale_px_per_cm: Scale factor for converting cm to pixels
            
        Returns:
            Refined compartment boundaries with preserved widths
            Viz elements for dialog display
        """
        if not boundaries or image is None:
            return boundaries
        
        # Calculate scale-based gaps
        gap_px = int(0.2 * scale_px_per_cm) if scale_px_per_cm else 2  # 2mm gap
        special_gap_px = int(0.5 * scale_px_per_cm) if scale_px_per_cm else 5  # 5mm gap between compartments 10-11
        
        # Use scale-based margin if available (search within 5mm of expected position)
        if scale_px_per_cm:
            margin = int(0.8 * scale_px_per_cm)  # 8mm search margin
            self.logger.debug(f"Using scale-based margin: {margin}px (5mm)")
        
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
            # Get compartment width
            compartment_width = x2 - x1
            
            # Define ROIs to search for edges - narrower search area
            left_roi = (max(0, x1 - margin), y1, min(image.shape[1], x1 + margin), y2)
            right_roi = (max(0, x2 - margin), y1, min(image.shape[1], x2 + margin), y2)
            
            # Store search regions for visualization
            viz_elements['search_regions'].append((left_roi[0], left_roi[1], left_roi[2], left_roi[3], (0, 255, 255), 1))
            viz_elements['search_regions'].append((right_roi[0], right_roi[1], right_roi[2], right_roi[3], (0, 255, 255), 1))
            
            # Detect vertical edges on both sides
            left_edges = self._detect_vertical_edges_in_roi(gray, left_roi)
            right_edges = self._detect_vertical_edges_in_roi(gray, right_roi)
            
            # Filter edges to remove those in the middle of compartments
            # Prioritize edges that are close to expected boundary positions
            best_left_edge = None
            best_right_edge = None
            
            if left_edges:
                # For left edge, prefer edges that are to the left of current position
                valid_left_edges = [e for e in left_edges if e <= x1 + gap_px]
                if valid_left_edges:
                    # Get the rightmost valid edge (closest to compartment)
                    best_left_edge = max(valid_left_edges)
                else:
                    # If no valid edges, get the closest one
                    left_edges.sort(key=lambda x: abs(x - x1))
                    best_left_edge = left_edges[0]
            
            if right_edges:
                # For right edge, prefer edges that are to the right of current position
                valid_right_edges = [e for e in right_edges if e >= x2 - gap_px]
                if valid_right_edges:
                    # Get the leftmost valid edge (closest to compartment)
                    best_right_edge = min(valid_right_edges)
                else:
                    # If no valid edges, get the closest one
                    right_edges.sort(key=lambda x: abs(x - x2))
                    best_right_edge = right_edges[0]
                
            # Store compartment data with detected edges
            compartment_data.append({
                'original': (x1, y1, x2, y2),
                'width': compartment_width,
                'left_edge': best_left_edge,
                'right_edge': best_right_edge,
                'compartment_index': i,
                'compartment_number': i + 1  # 1-based compartment number
            })
        
            # Store detected wall positions for visualization
            is_even = (i % 2 == 0)
            left_edge_color = (255, 0, 0) if is_even else (0, 0, 255)  # Blue vs Red
            right_edge_color = (0, 255, 0) if is_even else (255, 0, 255)  # Green vs Magenta
            
            if best_left_edge is not None:
                viz_elements['detected_edges'].append((best_left_edge, y1, best_left_edge, y2, left_edge_color, 2))
            if best_right_edge is not None:
                viz_elements['detected_edges'].append((best_right_edge, y1, best_right_edge, y2, right_edge_color, 2))
        
        # Start wall detection debug summary
        self.logger.debug("\n=== Wall Detection Summary ===")
        self.logger.debug(f"Gap: {gap_px}px (3mm), Special gap (compartments 10-11 - aruco marker IDs 13-14): {special_gap_px}px (8mm)")
        
        # Initialize refined bounds with original boundaries
        refined_bounds = [comp['original'] for comp in compartment_data]
        
        # Calculate maximum allowed adjustment based on scale
        if scale_px_per_cm:
            # Allow adjustment up to 1mm
            max_adjustment_px = int(0.1 * scale_px_per_cm)
        else:
            # Fallback: 2% of compartment width
            max_adjustment_px = None  # Will calculate per compartment
        
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
                comp_num = comp_data['compartment_number']
                
                # Get current boundary
                current_x1, current_y1, current_x2, current_y2 = refined_bounds[idx]
                
                # Calculate max adjustment for this compartment
                if max_adjustment_px is None:
                    max_adjustment = width * 0.02  # 2% of width
                else:
                    max_adjustment = max_adjustment_px
                
                # Determine appropriate gap for this compartment
                if comp_num == 10 and idx < len(compartment_data) - 1:
                    # Special gap after compartment 10
                    required_gap = special_gap_px
                else:
                    required_gap = gap_px
                
                summary = f"Compartment {comp_num} ({direction}): "
                
                # Proposed new positions
                new_x1, new_x2 = current_x1, current_x2
                
                # Case 1: Both edges detected - center between walls with gap
                if left_edge is not None and right_edge is not None:
                    available_space = right_edge - left_edge - (2 * required_gap)
                    
                    if available_space >= width:
                        # Enough space - center the compartment
                        midpoint = (left_edge + required_gap + right_edge - required_gap) / 2
                        proposed_x1 = midpoint - (width / 2)
                        proposed_x2 = midpoint + (width / 2)
                        
                        # Apply adjustment limit
                        left_adjustment = abs(proposed_x1 - original_x1)
                        right_adjustment = abs(proposed_x2 - original_x2)
                        
                        if left_adjustment <= max_adjustment and right_adjustment <= max_adjustment:
                            new_x1, new_x2 = proposed_x1, proposed_x2
                            summary += f"Centered between walls with {required_gap}px gap. "
                        else:
                            summary += f"Adjustment exceeds limit ({max_adjustment:.1f}px). "
                    else:
                        summary += f"Insufficient space between walls ({available_space:.1f}px < {width}px). "
                
                # Case 2: Only left edge detected
                elif left_edge is not None:
                    min_safe_x1 = left_edge + required_gap
                    
                    if current_x1 < min_safe_x1:
                        shift_amount = min(max_adjustment, min_safe_x1 - current_x1)
                        new_x1 = current_x1 + shift_amount
                        new_x2 = new_x1 + width
                        summary += f"Left wall: shifted right {shift_amount:.1f}px. "
                    else:
                        summary += "Left wall: already clear. "
                
                # Case 3: Only right edge detected
                elif right_edge is not None:
                    max_safe_x2 = right_edge - required_gap
                    
                    if current_x2 > max_safe_x2:
                        shift_amount = min(max_adjustment, current_x2 - max_safe_x2)
                        new_x2 = current_x2 - shift_amount
                        new_x1 = new_x2 - width
                        summary += f"Right wall: shifted left {shift_amount:.1f}px. "
                    else:
                        summary += "Right wall: already clear. "
                
                # Case 4: No walls detected
                else:
                    # Check if compartment is undersized and can be expanded
                    if scale_px_per_cm and width < int(2.1 * scale_px_per_cm * 0.95):  # Less than 95% of 2cm
                        target_width = int(2.1 * scale_px_per_cm)
                        expansion = target_width - width
                        
                        # Try to expand equally on both sides
                        expand_left = expansion // 2
                        expand_right = expansion - expand_left
                        
                        # Check boundaries don't overlap with neighbors
                        can_expand = True
                        if idx > 0:
                            prev_x2 = refined_bounds[idx-1][2]
                            if current_x1 - expand_left < prev_x2 + required_gap:
                                can_expand = False
                        if idx < len(refined_bounds) - 1:
                            next_x1 = refined_bounds[idx+1][0]
                            if current_x2 + expand_right > next_x1 - required_gap:
                                can_expand = False
                        
                        if can_expand:
                            new_x1 = current_x1 - expand_left
                            new_x2 = current_x2 + expand_right
                            width = new_x2 - new_x1  # Update width
                            summary += f"Expanded to 2.1cm ({target_width}px). "
                        else:
                            summary += "No walls, undersized but can't expand. "
                    else:
                        summary += "No walls detected. "
                
                # Handle special gap between compartments 10 and 11
                if comp_num == 10 and idx < len(refined_bounds) - 1 and direction == "left_to_right":
                    next_x1 = refined_bounds[idx + 1][0]
                    current_gap = next_x1 - new_x2
                    
                    if current_gap < special_gap_px:
                        # Need to create larger gap
                        adjustment_needed = special_gap_px - current_gap
                        # Try to shift compartment 10 left
                        if left_edge is None or new_x1 - adjustment_needed > left_edge + gap_px:
                            new_x1 -= adjustment_needed
                            new_x2 -= adjustment_needed
                            summary += f"Adjusted for 5mm gap to comp 11. "
                        else:
                            summary += f"Need 5mm gap but constrained by wall. "
                
                # Check for overlaps with adjacent compartments
                if idx > 0 and direction == "left_to_right":
                    prev_x2 = refined_bounds[idx-1][2]
                    min_x1 = prev_x2 + (special_gap_px if idx == 10 else gap_px)
                    
                    if new_x1 < min_x1:
                        new_x1 = min_x1
                        new_x2 = new_x1 + width
                        summary += f"Fixed left overlap. "
                
                if idx < len(refined_bounds)-1 and direction == "right_to_left":
                    next_x1 = refined_bounds[idx+1][0]
                    max_x2 = next_x1 - (special_gap_px if comp_num == 10 else gap_px)
                    
                    if new_x2 > max_x2:
                        new_x2 = max_x2
                        new_x1 = new_x2 - width
                        summary += f"Fixed right overlap. "
                
                # Update refined boundary if changes were made
                if new_x1 != current_x1 or new_x2 != current_x2:
                    refined_bounds[idx] = (int(new_x1), y1, int(new_x2), y2)
                    summary += f"Moved: ({current_x1:.0f},{current_x2:.0f}) → ({new_x1:.0f},{new_x2:.0f})"
                    self.logger.debug(summary)
                else:
                    summary += "No change needed."
                    self.logger.debug(summary)
        
        # Final validation pass
        self.logger.debug("\n--- Final Validation ---")
        
        for i in range(len(refined_bounds)-1):
            x1, y1, x2, y2 = refined_bounds[i]
            next_x1, _, _, _ = refined_bounds[i+1]
            
            actual_gap = next_x1 - x2
            expected_gap = special_gap_px if i == 9 else gap_px  # i=9 is compartment 10
            
            if actual_gap < 0:
                self.logger.warning(f"Overlap between compartments {i+1} and {i+2}: {-actual_gap:.1f}px")
            elif actual_gap < expected_gap * 0.01:  # Allow 01% tolerance
                self.logger.warning(f"Gap between compartments {i+1} and {i+2} too small: {actual_gap:.1f}px (expected {expected_gap}px)")
        
        # Store final boundaries for visualization
        for i, (x1, y1, x2, y2) in enumerate(refined_bounds):
            is_even = (i % 2 == 0)
            boundary_color = (0, 255, 255) if is_even else (255, 255, 0)
            viz_elements['final_boundaries'].append((x1, y1, x2, y2, boundary_color, 1))
        
        self.logger.debug("=== End Wall Detection Summary ===\n")
        
        # Store visualization elements
        if hasattr(self, 'viz_steps'):
            self.viz_steps['wall_detection_elements'] = viz_elements
        
        return refined_bounds, viz_elements

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
        
        # Extract ROI from the image - the ROI is too broad - it should be narrower and try to focus on detecting vertical lines BETWEEN compartment boundaries as priority - currently compartments can be cut in half from false positives within the actual compartment itself.
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
        
        # Convert local ROI coordinates back to global image coordinates TODO - save these to the viz_steps?
        return [x1 + x for x in vertical_edges]
