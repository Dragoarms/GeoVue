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
        # Cache for visualisation images
        self.viz_steps = {}
        

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
            # First try with original image
            self.logger.info("Detecting markers in original image")
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
    
    # Correct Image Orientation and Skew with ArUco markers

    def correct_image_skew(self, image: np.ndarray, 
                        markers: Dict[int, np.ndarray],
                        return_transform: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Correct image orientation by first fixing any major orientation issues,
        then fitting a line through compartment marker centers for fine skew correction.
        
        Args:
            image: Input image
            markers: Dictionary of detected markers
            return_transform: Whether to return transformation matrix
            
        Returns:
            Corrected image, or (corrected image, transformation matrix, total rotation angle) if return_transform is True
        """
        if image is None or not markers:
            self.logger.warning("Cannot correct orientation - image is None or no markers detected")
            if return_transform:
                return image, None, 0.0
            return image
        
        # Get relevant marker IDs from config
        compartment_ids = self.config.get('compartment_marker_ids', list(range(4, 24)))
        corner_ids = self.config.get('corner_marker_ids', [0, 1, 2, 3])
        
        # STEP 1: First correct any major orientation issues (portrait->landscape, 180° flips)
        self.logger.info(f"Checking major orientation issues with {len(markers)} detected markers")
        corrected_orientation, orientation_matrix, orientation_angle = self._determine_and_fix_orientation_with_transform(
            image, markers, corner_ids, compartment_ids
        )
        
        # Track if orientation was corrected
        orientation_corrected = corrected_orientation is not image
        
        # If orientation was corrected, we need to re-detect markers
        if orientation_corrected:
            self.logger.info(f"Image orientation was corrected by {orientation_angle:.2f} degrees, re-detecting markers")
            markers = self.detect_markers(corrected_orientation)
            if not markers:
                self.logger.warning("No markers detected after orientation correction")
                if return_transform:
                    return corrected_orientation, orientation_matrix, orientation_angle
                return corrected_orientation
        else:
            corrected_orientation = image
        
        # STEP 2: Fine-tune skew by fitting a line through compartment markers
        compartment_centers = []
        for marker_id in compartment_ids:
            if marker_id in markers and marker_id != 24:  # Exclude metadata marker 24
                center = np.mean(markers[marker_id], axis=0)
                compartment_centers.append((marker_id, center))
        
        # Need enough markers to estimate the skew angle
        if len(compartment_centers) < 4:
            self.logger.warning(f"Only {len(compartment_centers)} compartment markers found. "
                        "At least 4 are required to estimate skew.")
            if return_transform:
                return corrected_orientation, orientation_matrix, orientation_angle
            return corrected_orientation
        
        # Sort centers by their ID to get left-to-right order
        compartment_centers.sort(key=lambda x: x[0])
        center_points = np.array([c[1] for c in compartment_centers])
        
        # Fit a line through the marker centers for skew correction
        y_vals = center_points[:, 1]
        x_vals = center_points[:, 0]
        
        try:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            slope = coeffs[0]
            
            # Calculate angle in radians and convert to degrees
            angle_rad = np.arctan(slope)
            skew_angle_deg = np.rad2deg(angle_rad)
        except Exception as e:
            self.logger.error(f"Error fitting line to compartment centers: {str(e)}")
            if return_transform:
                return corrected_orientation, orientation_matrix, orientation_angle
            return corrected_orientation
        
        # Ensure angle is within -90 to 90 degrees for most stable rotation
        if skew_angle_deg > 90:
            skew_angle_deg -= 180
        elif skew_angle_deg < -90:
            skew_angle_deg += 180
        
        # Don't bother with tiny rotations
        if abs(skew_angle_deg) < 0.01:
            self.logger.info("Skew angle too small, no correction needed")
            if return_transform:
                return corrected_orientation, orientation_matrix, orientation_angle
            return corrected_orientation
        
        # Apply rotation for skew correction
        self.logger.info(f"Correcting image skew: {skew_angle_deg:.2f} degrees")
        
        # Get image center for rotation
        h, w = corrected_orientation.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix for skew correction
        skew_rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle_deg, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(corrected_orientation, skew_rotation_matrix, (w, h), 
                            flags=cv2.INTER_LANCZOS4, 
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))
        
        if return_transform:
            # Combine orientation and skew transformations
            total_angle = orientation_angle + skew_angle_deg
            total_matrix = self._combine_transformation_matrices(
                orientation_matrix, skew_rotation_matrix, image.shape, corrected_orientation.shape
            )
            return rotated, total_matrix, total_angle
        return rotated

    def _determine_and_fix_orientation_with_transform(self, image: np.ndarray, 
                            markers: Dict[int, np.ndarray],
                            corner_ids: List[int],
                            compartment_ids: List[int]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Determine if the image has the correct orientation and fix it if needed.
        Returns the corrected image, transformation matrix, and rotation angle.
        
        Args:
            image: Original image
            markers: Dictionary of detected markers
            corner_ids: List of corner marker IDs
            compartment_ids: List of compartment marker IDs
            
        Returns:
            Tuple of (corrected image, transformation matrix, rotation angle in degrees)
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get all corner markers that are detected
        detected_corner_markers = [(id, markers[id]) for id in corner_ids if id in markers]
        
        # Get all compartment markers that are detected, excluding metadata marker (24)
        detected_compartment_markers = [(id, markers[id]) for id in compartment_ids 
                                    if id in markers and id != 24]
        
        # If we don't have enough markers for orientation analysis, return original
        if len(detected_corner_markers) < 1 or len(detected_compartment_markers) < 2:
            self.logger.warning(f"Not enough markers to determine orientation: {len(detected_corner_markers)} corner and {len(detected_compartment_markers)} compartment markers")
            return image, np.array([[1, 0, 0], [0, 1, 0]]), 0.0
        
        # Check if the image is in landscape or portrait orientation
        is_landscape = w > h
        
        # Calculate the centers of all detected markers
        corner_centers = []
        for marker_id, corners in detected_corner_markers:
            center_pt = np.mean(corners, axis=0)
            corner_centers.append((marker_id, center_pt[0], center_pt[1]))
        
        compartment_centers = []
        for marker_id, corners in detected_compartment_markers:
            center_pt = np.mean(corners, axis=0)
            compartment_centers.append((marker_id, center_pt[0], center_pt[1]))
        
        # Compute average positions
        corner_x_avg = sum(x for _, x, _ in corner_centers) / len(corner_centers)
        corner_y_avg = sum(y for _, _, y in corner_centers) / len(corner_centers)
        
        compartment_x_avg = sum(x for _, x, _ in compartment_centers) / len(compartment_centers)
        compartment_y_avg = sum(y for _, _, y in compartment_centers) / len(compartment_centers)
        
        # Sort compartment centers by marker ID (should be left-to-right in correct orientation)
        sorted_comp_centers = sorted(compartment_centers, key=lambda c: c[0])
        
        # Check if the sequence of compartment IDs is aligned with X-coordinate order
        # In correct orientation, lower ID markers should be on the left (lower X)
        markers_sorted_by_x = sorted(compartment_centers, key=lambda c: c[1])
        
        # Compare the order of IDs when sorted by ID vs when sorted by X
        id_order_correct = True
        if len(sorted_comp_centers) >= 2 and len(markers_sorted_by_x) >= 2:
            ids_by_id = [c[0] for c in sorted_comp_centers]
            ids_by_x = [c[0] for c in markers_sorted_by_x]
            
            # If these don't match, the image might be flipped or rotated
            # Calculate correlation between sequence position and ID
            id_x_correlation = np.corrcoef(ids_by_id, ids_by_x)[0, 1]
            id_order_correct = id_x_correlation > 0  # Positive correlation means correct order
        
        # Determine if compartment markers with lower IDs are at higher Y positions
        lower_ids_higher_y = False
        if len(sorted_comp_centers) >= 4:
            # Compare average Y of low IDs vs high IDs
            low_id_y_avg = sum(y for _, _, y in sorted_comp_centers[:len(sorted_comp_centers)//2]) / (len(sorted_comp_centers)//2)
            high_id_y_avg = sum(y for _, _, y in sorted_comp_centers[len(sorted_comp_centers)//2:]) / (len(sorted_comp_centers) - len(sorted_comp_centers)//2)
            lower_ids_higher_y = low_id_y_avg < high_id_y_avg
        
        rotation_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        angle_deg = 0.0
        rotated_image = image
        
        # For Portrait images
        if not is_landscape:
            self.logger.info("Portrait image detected, determining correct rotation...")
            
            # Portrait - check if we need to rotate clockwise or counter-clockwise
            if (lower_ids_higher_y or corner_x_avg > compartment_x_avg):
                # Rotate 90 degrees counter-clockwise
                self.logger.info("Portrait image - rotating 90 degrees counter-clockwise")
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # Create a rotation matrix for 90 degrees counter-clockwise
                angle_deg = 90.0
                rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            else:
                # Rotate clockwise
                self.logger.info("Portrait image - rotating 90 degrees clockwise")
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                # Create a rotation matrix for 90 degrees clockwise
                angle_deg = -90.0
                rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        
        # For Landscape images - check if upside down (180° rotation needed)
        else:
            if (compartment_y_avg < corner_y_avg) or (not id_order_correct):
                # Rotate 180 degrees
                self.logger.info("Landscape image appears to be upside down, rotating 180 degrees")
                rotated_image = cv2.rotate(image, cv2.ROTATE_180)
                # Create a rotation matrix for 180 degrees rotation
                angle_deg = 180.0
                rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            else:
                # No orientation issues detected
                self.logger.info("Image is correct way up - no rotation needed")
        
        return rotated_image, rotation_matrix, angle_deg
   
    def _combine_transformation_matrices(self, orientation_matrix: Optional[np.ndarray], 
                                        skew_matrix: Optional[np.ndarray],
                                        original_shape: Tuple[int, int],
                                        intermediate_shape: Tuple[int, int]) -> np.ndarray:
        """
        Combine orientation and skew transformation matrices into a single transformation.
        
        Args:
            orientation_matrix: Matrix for major orientation correction
            skew_matrix: Matrix for fine skew correction
            original_shape: Shape of the original image (h, w)
            intermediate_shape: Shape after orientation correction (h, w)
            
        Returns:
            Combined transformation matrix
        """
        if orientation_matrix is None and skew_matrix is None:
            # Identity matrix (no transformation)
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        
        if orientation_matrix is None:
            return skew_matrix
        
        if skew_matrix is None:
            return orientation_matrix
        
        # For correct combination, we need to use homogeneous coordinates
        # Convert 2x3 matrices to 3x3 homogeneous matrices
        orientation_homog = np.vstack([orientation_matrix, [0, 0, 1]])
        skew_homog = np.vstack([skew_matrix, [0, 0, 1]])
        
        # The combined transformation is applied in order: first orientation, then skew
        # We need the inverse order in matrix multiplication
        combined_homog = np.matmul(skew_homog, orientation_homog)
        
        # Extract the 2x3 transformation matrix
        combined_matrix = combined_homog[:2, :]
        
        return combined_matrix
   
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
        
        # # Create initial visualization
        # initial_viz = self._create_boundary_visualization(
        #     image, 
        #     compartment_boundaries,
        #     marker_to_compartment=marker_to_compartment,
        #     markers=markers
        # )
        # self.viz_steps['initial_boundaries'] = initial_viz
        # result['visualization'] = initial_viz
        # result['viz_steps'] = self.viz_steps
        
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


    def get_compartment_corners(self, hole_id: str, depth_from: int, depth_to: int) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Retrieve the four corner coordinates for a specific compartment.
        
        Returns:
            Dictionary with 'top_left', 'top_right', 'bottom_right', 'bottom_left' corners
            or None if not found
        """
        with self._get_file_lock(self.compartment_lock):
            try:
                data = self._read_json_file(self.compartment_json_path, self.compartment_lock)
                
                for record in data:
                    if (record['HoleID'] == hole_id and 
                        record['From'] == depth_from and 
                        record['To'] == depth_to):
                        
                        # Check if corner data exists
                        if 'Top_Left_X' in record:
                            return {
                                'top_left': (record['Top_Left_X'], record['Top_Left_Y']),
                                'top_right': (record['Top_Right_X'], record['Top_Right_Y']),
                                'bottom_right': (record['Bottom_Right_X'], record['Bottom_Right_Y']),
                                'bottom_left': (record['Bottom_Left_X'], record['Bottom_Left_Y'])
                            }
                return None
                
            except Exception as e:
                self.logger.error(f"Error retrieving compartment corners: {e}")
                return None

    def boundaries_to_corners(boundaries: List[Tuple[int, int, int, int]], 
                            boundary_adjustments: Optional[Dict] = None,
                            img_width: int = None) -> List[Dict[str, Tuple[int, int]]]:
        """
        Convert rectangular boundaries to four corners, applying any height adjustments.
        
        Args:
            boundaries: List of (x1, y1, x2, y2) tuples
            boundary_adjustments: Dict with 'top_boundary', 'bottom_boundary', 
                                'left_height_offset', 'right_height_offset'
            img_width: Image width for calculating slopes
        
        Returns:
            List of corner dictionaries for each compartment
        """
        corners_list = []
        
        for x1, y1, x2, y2 in boundaries:
            if boundary_adjustments and img_width and img_width > 0:
                # Get adjustment parameters
                left_offset = boundary_adjustments.get('left_height_offset', 0)
                right_offset = boundary_adjustments.get('right_height_offset', 0)
                top_base = boundary_adjustments.get('top_boundary', y1)
                bottom_base = boundary_adjustments.get('bottom_boundary', y2)
                
                # Calculate adjusted y coordinates at left and right edges
                left_top_y = top_base + left_offset
                right_top_y = top_base + right_offset
                left_bottom_y = bottom_base + left_offset
                right_bottom_y = bottom_base + right_offset
                
                # Calculate slopes
                top_slope = (right_top_y - left_top_y) / img_width
                bottom_slope = (right_bottom_y - left_bottom_y) / img_width
                
                # Calculate y values at compartment x positions
                y1_left = int(left_top_y + (top_slope * x1))
                y1_right = int(left_top_y + (top_slope * x2))
                y2_left = int(left_bottom_y + (bottom_slope * x1))
                y2_right = int(left_bottom_y + (bottom_slope * x2))
            else:
                # No adjustments, use rectangular boundaries
                y1_left = y1_right = y1
                y2_left = y2_right = y2
                
            corners = {
                'top_left': (x1, y1_left),
                'top_right': (x2, y1_right),
                'bottom_right': (x2, y2_right),
                'bottom_left': (x1, y2_left)
            }
            corners_list.append(corners)
        
        return corners_list

    def extract_compartment_boundaries(self, image, markers, compartment_count=20, smart_cropping=True, parent_window=None, metadata=None):
        """
        Extract compartment boundaries from ArUco markers.
        
        This is a legacy wrapper that maintains backward compatibility.
        For new code, use analyze_compartment_boundaries() instead.
        
        Args:
            image: Input image
            markers: Detected ArUco markers dictionary
            compartment_count: Number of expected compartments
            smart_cropping: Whether to use smart cropping for gaps
            parent_window: Parent window for manual annotation dialog (deprecated)
            metadata: Optional metadata dictionary containing hole_id, depth_from, depth_to
        
        Returns:
            Tuple of (boundaries, visualization image)
        """
        # Analyze boundaries without UI
        analysis_result = self.analyze_compartment_boundaries(
            image, markers, compartment_count, smart_cropping, metadata
        )
        
        # Return just the boundaries and visualization for backward compatibility
        return analysis_result['boundaries'], analysis_result['visualization']

    def _calculate_average_compartment_width(self, detected_boundaries=None):
        """
        Calculate the average width of detected compartments.
        
        Args:
            detected_boundaries: Optional list of boundaries to calculate width from
                            
        Returns:
            Average width as integer
        """
        if detected_boundaries is None:
            detected_boundaries = []
        
        # If no boundaries provided, return reasonable default
        if not detected_boundaries:
            if hasattr(self, 'image') and self.image is not None:
                w = self.image.shape[1]
                return int(w * 0.04)  # 4% of image width as default
            else:
                return 80  # Reasonable default value
        
        # Calculate width of each detected compartment
        widths = [x2 - x1 for x1, y1, x2, y2 in detected_boundaries]
        
        if widths:
            return int(sum(widths) / len(widths))
        else:
            return 80  # Default width if calculation fails

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
            boundary_color = (0, 255, 255) if is_even else (255, 255, 0)  # Yellow vs Cyan in BGR
            viz_elements['final_boundaries'].append((x1, y1, x2, y2, boundary_color, 1))
        
        self.logger.debug("=== End Wall Detection Summary ===\n")
        
        # Store visualization elements in viz_steps
        if hasattr(self, 'viz_steps'):
            self.viz_steps['wall_detection_elements'] = viz_elements
        
        return refined_bounds
    

    def estimate_image_scale(self, markers: Dict[int, np.ndarray], 
                            use_corner_markers: bool = True) -> Dict[str, float]:
        """
        Estimate the image scale (pixels per centimeter) using detected ArUco markers.
        
        This method calculates the scale by measuring the edge lengths and diagonals
        of detected markers with known physical sizes, then computes a robust median
        estimate after filtering outliers.
        
        Args:
            markers: Dictionary of detected markers {id: corners}
            use_corner_markers: Whether to include corner markers in calculation
                            (default False due to frequent occlusion)
        
        Returns:
            Dictionary containing:
                - scale_px_per_cm: Median scale value in pixels/cm
                - image_width_cm: Image width in centimeters
                - marker_measurements: List of individual measurements for debugging
                - confidence: Confidence score based on measurement consistency
        """
        if not markers:
            self.logger.warning("No markers provided for scale estimation")
            return {
                'scale_px_per_cm': None,
                'image_width_cm': None,
                'marker_measurements': [],
                'confidence': 0.0
            }
        
        # Get marker physical sizes from config (with defaults)
        compartment_marker_size_cm = self.config.get('compartment_marker_size_cm', 2.0)
        corner_marker_size_cm = self.config.get('corner_marker_size_cm', 1.0)
        
        # Get marker ID ranges
        corner_ids = self.config.get('corner_marker_ids', [0, 1, 2, 3])
        compartment_ids = self.config.get('compartment_marker_ids', list(range(4, 24)))
        metadata_ids = self.config.get('metadata_marker_ids', [24])
        
        # Collect all scale measurements
        scale_measurements = []
        marker_measurements = []
        
        for marker_id, corners in markers.items():
            # Skip metadata markers (they might have different sizes)
            if marker_id in metadata_ids:
                continue
                
            # Determine marker type and size
            if marker_id in corner_ids:
                if not use_corner_markers:
                    continue
                physical_size_cm = corner_marker_size_cm
                marker_type = "corner"
            elif marker_id in compartment_ids:
                physical_size_cm = compartment_marker_size_cm
                marker_type = "compartment"
            else:
                # Unknown marker type, skip
                continue
            
            # Calculate measurements for this marker
            measurements = self._measure_marker_edges(corners, physical_size_cm, marker_id, marker_type)
            
            if measurements['valid']:
                scale_measurements.extend(measurements['scales'])
                marker_measurements.append(measurements)
        
        # Calculate final scale using robust statistics
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
        
        # Define outlier bounds (1.5 * IQR is standard)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter out outliers
        filtered_scales = scale_array[(scale_array >= lower_bound) & (scale_array <= upper_bound)]
        
        if len(filtered_scales) == 0:
            # If all values were outliers, use original array
            filtered_scales = scale_array
            self.logger.warning("All scale measurements were outliers, using unfiltered data")
        
        # Calculate final scale as median of filtered values
        final_scale_px_per_cm = float(np.median(filtered_scales))
        
        # Calculate confidence based on consistency
        if len(filtered_scales) > 1:
            # Use coefficient of variation as consistency measure
            std_dev = np.std(filtered_scales)
            mean_val = np.mean(filtered_scales)
            cv = std_dev / mean_val if mean_val > 0 else 1.0
            
            # Convert CV to confidence score (lower CV = higher confidence)
            # CV of 0.02 (2%) = 95% confidence, CV of 0.10 (10%) = 50% confidence
            confidence = max(0.0, min(1.0, 1.0 - (cv - 0.02) / 0.08))
        else:
            confidence = 0.5  # Single measurement
        
        # Calculate image width in cm if we have image shape
        image_width_cm = None
        if hasattr(self, 'app') and hasattr(self.app, 'small_image'):
            image_width_px = self.app.small_image.shape[1]
            image_width_cm = image_width_px / final_scale_px_per_cm
        
        # Log results
        self.logger.info(f"Scale estimation complete: {final_scale_px_per_cm:.2f} px/cm")
        self.logger.info(f"Used {len(filtered_scales)}/{len(scale_measurements)} measurements after filtering")
        self.logger.info(f"Confidence: {confidence:.2%}")
        if image_width_cm:
            self.logger.info(f"Image width: {image_width_cm:.1f} cm")
        
        return {
            'scale_px_per_cm': final_scale_px_per_cm,
            'image_width_cm': image_width_cm,
            'marker_measurements': marker_measurements,
            'confidence': confidence,
            'outliers_removed': len(scale_measurements) - len(filtered_scales),
            'total_measurements': len(scale_measurements)
        }

    def correct_skewed_markers(self, markers: Dict[int, np.ndarray], 
                            tolerance_pixels: float = 5.0,
                            max_correction_angle: float = 30.0) -> Dict[int, np.ndarray]:
        """
        Correct skewed/rotated markers to perfect squares if they have at least one valid edge.
        
        Args:
            markers: Dictionary of detected markers {id: corners}
            tolerance_pixels: Maximum pixel difference allowed for edge lengths (default 5 pixels)
            max_correction_angle: Maximum rotation angle to correct (degrees)
            
        Returns:
            Dictionary of corrected markers
        """
        corrected_markers = {}
        
        # Get marker physical sizes from config
        compartment_marker_size_cm = self.config.get('compartment_marker_size_cm', 2.0)
        corner_marker_size_cm = self.config.get('corner_marker_size_cm', 1.0)
        
        # Get marker ID ranges
        corner_ids = self.config.get('corner_marker_ids', [0, 1, 2, 3])
        compartment_ids = self.config.get('compartment_marker_ids', list(range(4, 24)))
        metadata_ids = self.config.get('metadata_marker_ids', [24])
        
        for marker_id, corners in markers.items():
            # Determine expected size for this marker
            if marker_id in corner_ids:
                expected_size_cm = corner_marker_size_cm
            elif marker_id in compartment_ids:
                expected_size_cm = compartment_marker_size_cm
            elif marker_id in metadata_ids:
                expected_size_cm = self.config.get('metadata_marker_size_cm', 2.0)
            else:
                # Unknown marker, skip correction
                corrected_markers[marker_id] = corners
                continue
            
            # Calculate edge lengths
            edge_lengths = []
            edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
            
            for i, j in edge_indices:
                edge_length = np.linalg.norm(corners[i] - corners[j])
                edge_lengths.append(edge_length)
            
            # Find the median edge length (most likely to be correct)
            median_edge = np.median(edge_lengths)
            
            # Check if at least one edge is within tolerance of the median
            valid_edges = []
            for idx, length in enumerate(edge_lengths):
                if abs(length - median_edge) <= tolerance_pixels:
                    valid_edges.append(idx)
            
            if len(valid_edges) >= 2:
                # We have at least 2 valid edges, we can reconstruct a square
                
                # Find the center of the marker
                center = np.mean(corners, axis=0)
                
                # Calculate the angle of the marker
                # Use the most reliable edge (closest to median)
                best_edge_idx = min(valid_edges, key=lambda i: abs(edge_lengths[i] - median_edge))
                edge_start_idx, edge_end_idx = edge_indices[best_edge_idx]
                
                # Calculate angle from this edge
                edge_vector = corners[edge_end_idx] - corners[edge_start_idx]
                angle = np.arctan2(edge_vector[1], edge_vector[0])
                
                # Check if we need to correct by 90 degrees
                if best_edge_idx in [1, 3]:  # Vertical edges
                    angle -= np.pi / 2
                
                # Create a perfect square centered at the same location
                half_size = median_edge / 2
                
                # Create square corners before rotation
                square_corners = np.array([
                    [-half_size, -half_size],
                    [half_size, -half_size],
                    [half_size, half_size],
                    [-half_size, half_size]
                ])
                
                # Rotate the square
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                
                # Apply rotation and translation
                corrected_corners = np.dot(square_corners, rotation_matrix.T) + center
                
                # Check if the correction angle is within limits
                original_angle = np.arctan2(corners[1][1] - corners[0][1], 
                                        corners[1][0] - corners[0][0])
                correction_angle = abs(angle - original_angle) * 180 / np.pi
                
                if correction_angle > max_correction_angle:
                    self.logger.warning(f"Marker {marker_id} requires {correction_angle:.1f}° correction, exceeding limit of {max_correction_angle}°")
                    corrected_markers[marker_id] = corners
                else:
                    self.logger.info(f"Corrected marker {marker_id} from skewed to square (correction angle: {correction_angle:.1f}°)")
                    corrected_markers[marker_id] = corrected_corners.astype(np.float32)
            else:
                # Not enough valid edges to correct
                self.logger.warning(f"Marker {marker_id} has insufficient valid edges for correction")
                corrected_markers[marker_id] = corners
        
        return corrected_markers


    def _measure_marker_edges(self, corners: np.ndarray, physical_size_cm: float,
                            marker_id: int, marker_type: str) -> Dict[str, Any]:
        """
        Measure all edges and diagonals of a single marker to estimate scale.
        
        Args:
            corners: 4x2 array of corner coordinates
            physical_size_cm: Known physical size of marker edge in cm
            marker_id: ID of the marker for logging
            marker_type: Type of marker ("corner" or "compartment")
        
        Returns:
            Dictionary with measurement results
        """
        measurements = {
            'marker_id': marker_id,
            'marker_type': marker_type,
            'physical_size_cm': physical_size_cm,
            'scales': [],
            'edge_lengths': [],
            'diagonal_lengths': [],
            'valid': False,
            'rejection_reason': None
        }
        
        # Ensure we have 4 corners
        if corners.shape[0] != 4:
            measurements['rejection_reason'] = f"Invalid corner count: {corners.shape[0]}"
            return measurements
        
        # Calculate edge lengths (4 sides)
        edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        edge_lengths = []
        
        for i, j in edge_indices:
            edge_length = np.linalg.norm(corners[i] - corners[j])
            edge_lengths.append(edge_length)
            
            # Calculate scale for this edge
            scale = edge_length / physical_size_cm
            measurements['scales'].append(scale)
        
        measurements['edge_lengths'] = edge_lengths
        
        # Calculate diagonal lengths (2 diagonals)
        diagonal_indices = [(0, 2), (1, 3)]
        diagonal_lengths = []
        diagonal_physical_size = physical_size_cm * np.sqrt(2)  # Diagonal of square
        
        for i, j in diagonal_indices:
            diagonal_length = np.linalg.norm(corners[i] - corners[j])
            diagonal_lengths.append(diagonal_length)
            
            # Calculate scale for this diagonal
            scale = diagonal_length / diagonal_physical_size
            measurements['scales'].append(scale)
        
        measurements['diagonal_lengths'] = diagonal_lengths
        
        # Check marker quality by internal consistency
        edge_array = np.array(edge_lengths)
        edge_std = np.std(edge_array)
        edge_mean = np.mean(edge_array)
        
        # Calculate coefficient of variation for edges
        edge_cv = edge_std / edge_mean if edge_mean > 0 else 1.0
        
        # Reject markers with high internal variance (distorted squares)
        max_acceptable_cv = 0.05  # 15% variation
        if edge_cv > max_acceptable_cv:
            measurements['rejection_reason'] = f"High edge variance (CV={edge_cv:.2%})"
            measurements['valid'] = False
        else:
            measurements['valid'] = True
            measurements['edge_cv'] = edge_cv
        
        # Additional validation: check if it's roughly square
        if measurements['valid']:
            # Check diagonal ratio (should be close to 1.0 for square)
            if len(diagonal_lengths) == 2 and min(diagonal_lengths) > 0:
                diagonal_ratio = max(diagonal_lengths) / min(diagonal_lengths)
                if diagonal_ratio > 1.05:  # More than 20% difference
                    measurements['rejection_reason'] = f"Non-square marker (diagonal ratio={diagonal_ratio:.2f})"
                    measurements['valid'] = False
        
        return measurements
