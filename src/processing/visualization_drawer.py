# processing/visualization_drawer.py

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any

class VisualizationDrawer:
    """Collection of drawing functions for use with VisualizationManager."""
    
    @staticmethod
    def draw_boundaries(image: np.ndarray, **kwargs) -> np.ndarray:
        """Draw compartment boundaries on image."""
        # Implementation as above
        pass
    
    @staticmethod
    def draw_markers(image: np.ndarray, **kwargs) -> np.ndarray:
        """Draw only markers on image."""
        viz = image.copy()
        markers = kwargs.get('markers', {})
        color = kwargs.get('color', (255, 0, 0))
        
        for marker_id, corners in markers.items():
            if isinstance(corners, np.ndarray):
                pts = corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(viz, [pts], True, color, 2)
                
                center = np.mean(corners, axis=0).astype(int)
                cv2.putText(viz, str(marker_id), tuple(center),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return viz
    
    @staticmethod
    def draw_debug_info(image: np.ndarray, **kwargs) -> np.ndarray:
        """Draw debug information overlay."""
        # Add debug info like scale, image dimensions, etc.
        pass