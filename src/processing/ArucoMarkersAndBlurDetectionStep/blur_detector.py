import cv2
import os
import re
import numpy as np
import logging
from typing import List, Tuple, Dict, Any



class BlurDetector:
    """
    A class to detect blurry images using the Laplacian variance method.
    
    The Laplacian operator is used to measure the second derivative of an image.
    The variance of the Laplacian is a simple measure of the amount of edges 
    present in an image - blurry images tend to have fewer edges and thus lower variance.
    """

    def __init__(self, threshold: float = 100.0, roi_ratio: float = 0.8):
        """
        Initialize the blur detector with configurable parameters.
        
        Args:
            threshold: Laplacian variance threshold below which an image is considered blurry
            roi_ratio: Ratio of central image area to use for blur detection (0.0-1.0)
        """
        self.threshold = threshold
        self.roi_ratio = max(0.1, min(1.0, roi_ratio))  # Clamp between 0.1 and 1.0
        self.logger = logging.getLogger(__name__)
    
    def get_laplacian_variance(self, image: np.ndarray) -> float:
        """
        Calculate the variance of the Laplacian for an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Variance of the Laplacian as a measure of blurriness (lower = more blurry)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Extract region of interest if ratio < 1.0
            if self.roi_ratio < 1.0:
                h, w = gray.shape
                center_h, center_w = h // 2, w // 2
                roi_h, roi_w = int(h * self.roi_ratio), int(w * self.roi_ratio)
                start_h, start_w = center_h - (roi_h // 2), center_w - (roi_w // 2)
                
                # Ensure ROI is within image bounds
                start_h = max(0, start_h)
                start_w = max(0, start_w)
                end_h = min(h, start_h + roi_h)
                end_w = min(w, start_w + roi_w)
                
                # Extract ROI
                gray = gray[start_h:end_h, start_w:end_w]
            
            # Calculate Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Calculate variance
            variance = laplacian.var()
            return variance
            
        except Exception as e:
            self.logger.error(f"Error calculating Laplacian variance: {str(e)}")
            return 0.0
    
    def is_blurry(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Determine if an image is blurry.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (is_blurry, variance_score)
        """
        variance = self.get_laplacian_variance(image)
        return variance < self.threshold, variance
    
    def analyze_image_with_visualization(self, image: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """
        Analyze an image and create a visualization of the blur detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (is_blurry, variance_score, visualization_image)
        """
        # Make a copy for visualization
        viz_image = image.copy()
        
        # Calculate blur metrics
        is_blurry, variance = self.is_blurry(image)
        
        # Add text with blur metrics
        status = "BLURRY" if is_blurry else "SHARP"
        color = (0, 0, 255) if is_blurry else (0, 255, 0)  # Red for blurry, green for sharp
        
        # Add a background box for better text visibility
        h, w = viz_image.shape[:2]
        cv2.rectangle(viz_image, (10, h - 60), (w - 10, h - 10), (0, 0, 0), -1)
        
        # Add text
        cv2.putText(
            viz_image,
            f"Status: {status}", 
            (20, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            color, 
            2
        )
        
        cv2.putText(
            viz_image,
            f"Laplacian Variance: {variance:.2f} (threshold: {self.threshold:.2f})", 
            (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        return is_blurry, variance, viz_image
    
    def batch_analyze_images(self, 
                           images: List[np.ndarray],
                           generate_visualizations: bool = False) -> List[Dict[str, Any]]:
        """
        Analyze a batch of images for blurriness.
        
        Args:
            images: List of input images
            generate_visualizations: Whether to create visualization images
            
        Returns:
            List of dictionaries with analysis results
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                # Perform analysis
                if generate_visualizations:
                    is_blurry, variance, viz_image = self.analyze_image_with_visualization(image)
                    result = {
                        'index': i,
                        'is_blurry': is_blurry,
                        'variance': variance,
                        'visualization': viz_image
                    }
                else:
                    is_blurry, variance = self.is_blurry(image)
                    result = {
                        'index': i,
                        'is_blurry': is_blurry,
                        'variance': variance
                    }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error analyzing image {i}: {str(e)}")
                # Add a placeholder result for the failed image
                results.append({
                    'index': i,
                    'is_blurry': False,
                    'variance': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def calibrate_threshold(self, 
                           sharp_images: List[np.ndarray], 
                           blurry_images: List[np.ndarray],
                           safety_factor: float = 1.5) -> float:
        """
        Calibrate the blur threshold based on example images.
        
        Args:
            sharp_images: List of sharp (good quality) images
            blurry_images: List of blurry (poor quality) images
            safety_factor: Factor to apply to the calculated threshold
            
        Returns:
            Calibrated threshold value
        """
        if not sharp_images or not blurry_images:
            self.logger.warning("Not enough sample images for calibration, using default threshold")
            return self.threshold
        
        try:
            # Calculate variances for sharp images
            sharp_variances = [self.get_laplacian_variance(img) for img in sharp_images]
            
            # Calculate variances for blurry images
            blurry_variances = [self.get_laplacian_variance(img) for img in blurry_images]
            
            # Find the minimum variance of sharp images
            min_sharp = min(sharp_variances)
            
            # Find the maximum variance of blurry images
            max_blurry = max(blurry_variances)
            
            # Calculate the threshold
            if min_sharp > max_blurry:
                # Clear separation - use the midpoint
                threshold = (min_sharp + max_blurry) / 2
            else:
                # Overlap - use a weighted average
                threshold = (min_sharp * 0.7 + max_blurry * 0.3)
            
            # Apply safety factor (lower the threshold to err on the side of detecting blurry images)
            threshold = threshold / safety_factor
            
            self.logger.info(f"Calibrated blur threshold: {threshold:.2f}")
            
            # Update the instance threshold
            self.threshold = threshold
            
            return threshold
            
        except Exception as e:
            self.logger.error(f"Error during threshold calibration: {str(e)}")
            return self.threshold
