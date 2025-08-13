# visualization_drawer.py

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class VisualizationDrawer:
    """Collection of drawing functions for use with VisualizationManager."""
    
    @staticmethod
    def draw_boundaries(image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Draw compartment boundaries on image.
        
        Args:
            image: Input image to draw on
            **kwargs:
                - boundaries: List of (x1, y1, x2, y2) tuples
                - color: BGR color tuple (default: green)
                - thickness: Line thickness (default: 2)
                - labels: Optional dict mapping boundary index to label text
                - label_color: Color for labels (default: white on black background)
                - rotation_angle: Optional rotation angle for boundaries
                - top_slope: Optional slope for top boundary line
                - bottom_slope: Optional slope for bottom boundary line
                
        Returns:
            Image with boundaries drawn
        """
        viz = image.copy()
        boundaries = kwargs.get('boundaries', [])
        color = kwargs.get('color', (0, 255, 0))  # Green default
        thickness = kwargs.get('thickness', 2)
        labels = kwargs.get('labels', {})
        label_color = kwargs.get('label_color', (255, 165, 0))  # Orange default
        rotation_angle = kwargs.get('rotation_angle', 0)
        
        # Get slopes for angled boundaries
        top_slope = kwargs.get('top_slope', 0)
        bottom_slope = kwargs.get('bottom_slope', 0)
        img_height, img_width = viz.shape[:2]
        
        for i, boundary in enumerate(boundaries):
            if len(boundary) == 4:
                x1, y1, x2, y2 = boundary
                
                # If rotation angle is provided, create rotated rectangle
                if rotation_angle != 0 or top_slope != 0 or bottom_slope != 0:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Create rectangle corners
                    rect_corners = np.array([
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ], dtype=np.float32)
                    
                    # Apply rotation if needed
                    if rotation_angle != 0:
                        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
                        ones = np.ones(shape=(len(rect_corners), 1))
                        rect_corners_homog = np.hstack([rect_corners, ones])
                        rotated_corners = np.dot(rotation_matrix, rect_corners_homog.T).T
                        rotated_corners = rotated_corners.astype(np.int32)
                        cv2.polylines(viz, [rotated_corners], True, color, thickness)
                    else:
                        # Just draw rectangle
                        cv2.rectangle(viz, (x1, y1), (x2, y2), color, thickness)
                else:
                    # Simple rectangle
                    cv2.rectangle(viz, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label if provided
                if i in labels:
                    label_text = str(labels[i])
                    mid_x = int((x1 + x2) / 2)
                    mid_y = int((y1 + y2) / 2)
                    
                    # Get text size
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    font_thickness = 2
                    text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
                    
                    # Draw background rectangle for text
                    cv2.rectangle(viz,
                                (mid_x - text_size[0]//2 - 5, mid_y - text_size[1]//2 - 5),
                                (mid_x + text_size[0]//2 + 5, mid_y + text_size[1]//2 + 5),
                                (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(viz, label_text,
                              (mid_x - text_size[0]//2, mid_y + text_size[1]//2),
                              font, font_scale, label_color, font_thickness)
        
        return viz
    
    @staticmethod
    def draw_markers(image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Draw ArUco markers on image.
        
        Args:
            image: Input image
            **kwargs:
                - markers: Dict of {marker_id: corners_array}
                - color: Marker outline color (default: cyan)
                - thickness: Line thickness (default: 2)
                - draw_ids: Whether to draw marker IDs (default: True)
                - id_color: Color for marker IDs (default: same as marker color)
                - draw_centers: Whether to draw center points (default: True)
                
        Returns:
            Image with markers drawn
        """
        viz = image.copy()
        markers = kwargs.get('markers', {})
        color = kwargs.get('color', (255, 255, 0))  # Cyan default
        thickness = kwargs.get('thickness', 2)
        draw_ids = kwargs.get('draw_ids', True)
        id_color = kwargs.get('id_color', color)
        draw_centers = kwargs.get('draw_centers', True)
        
        for marker_id, corners in markers.items():
            if isinstance(corners, np.ndarray):
                # Draw marker outline
                pts = corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(viz, [pts], True, color, thickness)
                
                # Draw center point if requested
                if draw_centers:
                    center = np.mean(corners, axis=0).astype(int)
                    cv2.circle(viz, tuple(center), 5, color, -1)
                
                # Draw marker ID if requested
                if draw_ids:
                    center = np.mean(corners, axis=0).astype(int)
                    cv2.putText(viz, str(marker_id), 
                              (center[0] - 10, center[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, id_color, 2)
        
        return viz
    
    @staticmethod
    def draw_lines(image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Draw lines on image.
        
        Args:
            image: Input image
            **kwargs:
                - lines: List of ((x1, y1), (x2, y2)) tuples
                - color: Line color (default: green)
                - thickness: Line thickness (default: 2)
                - line_type: cv2 line type (default: cv2.LINE_8)
                
        Returns:
            Image with lines drawn
        """
        viz = image.copy()
        lines = kwargs.get('lines', [])
        color = kwargs.get('color', (0, 255, 0))
        thickness = kwargs.get('thickness', 2)
        line_type = kwargs.get('line_type', cv2.LINE_8)
        
        for line in lines:
            if len(line) == 2:
                pt1, pt2 = line
                cv2.line(viz, tuple(pt1), tuple(pt2), color, thickness, line_type)
        
        return viz
    
    @staticmethod
    def draw_boundary_lines(image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Draw top and bottom boundary lines with optional slope.
        
        Args:
            image: Input image
            **kwargs:
                - top_y: Base top Y coordinate
                - bottom_y: Base bottom Y coordinate
                - left_offset: Left side height offset
                - right_offset: Right side height offset
                - color: Line color (default: green)
                - thickness: Line thickness (default: 2)
                
        Returns:
            Image with boundary lines drawn
        """
        viz = image.copy()
        img_height, img_width = viz.shape[:2]
        
        top_y = kwargs.get('top_y', 0)
        bottom_y = kwargs.get('bottom_y', img_height)
        left_offset = kwargs.get('left_offset', 0)
        right_offset = kwargs.get('right_offset', 0)
        color = kwargs.get('color', (0, 255, 0))
        thickness = kwargs.get('thickness', 2)
        
        # Calculate line endpoints
        left_top_y = top_y + left_offset
        right_top_y = top_y + right_offset
        left_bottom_y = bottom_y + left_offset
        right_bottom_y = bottom_y + right_offset
        
        # Draw top boundary line
        cv2.line(viz, (0, left_top_y), (img_width, right_top_y), color, thickness)
        
        # Draw bottom boundary line
        cv2.line(viz, (0, left_bottom_y), (img_width, right_bottom_y), color, thickness)
        
        return viz
    
    @staticmethod
    def draw_scale_bar(image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Draw a scale bar on the image.
        
        Args:
            image: Input image
            **kwargs:
                - scale_px_per_cm: Pixels per centimeter
                - position: 'bottom-left', 'bottom-right', 'top-left', 'top-right'
                - length_cm: Length of scale bar in cm (default: 10)
                - margin: Margin from edge (default: 20)
                - height: Scale bar height (default: 20)
                - confidence: Optional confidence percentage to display
                
        Returns:
            Image with scale bar drawn
        """
        viz = image.copy()
        scale_px_per_cm = kwargs.get('scale_px_per_cm')
        if not scale_px_per_cm:
            return viz
            
        img_height, img_width = viz.shape[:2]
        position = kwargs.get('position', 'bottom-left')
        length_cm = kwargs.get('length_cm', 10)
        margin = kwargs.get('margin', 20)
        bar_height = kwargs.get('height', 20)
        confidence = kwargs.get('confidence')
        
        # Calculate scale bar dimensions
        bar_length_px = int(length_cm * scale_px_per_cm)
        
        # Determine position
        if 'left' in position:
            bar_x = margin
        else:
            bar_x = img_width - margin - bar_length_px
            
        if 'top' in position:
            bar_y = margin
        else:
            bar_y = img_height - margin - bar_height - 30  # Extra space for labels
        
        # Draw white background
        bg_padding = 10
        cv2.rectangle(viz,
                    (bar_x - bg_padding, bar_y - 20 - bg_padding),
                    (bar_x + bar_length_px + bg_padding, bar_y + bar_height + 30 + bg_padding),
                    (255, 255, 255), -1)
        
        # Draw checkered scale bar (1cm segments)
        for i in range(length_cm):
            segment_start = bar_x + int(i * scale_px_per_cm)
            segment_end = bar_x + int((i + 1) * scale_px_per_cm)
            
            # Alternate black and white
            color = (0, 0, 0) if i % 2 == 0 else (200, 200, 200)
            cv2.rectangle(viz,
                        (segment_start, bar_y),
                        (segment_end, bar_y + bar_height),
                        color, -1)
        
        # Draw border
        cv2.rectangle(viz,
                    (bar_x, bar_y),
                    (bar_x + bar_length_px, bar_y + bar_height),
                    (0, 0, 0), 2)
        
        # Add labels
        cv2.putText(viz, "0cm",
                  (bar_x, bar_y + bar_height + 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        text_size = cv2.getTextSize(f"{length_cm}cm", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(viz, f"{length_cm}cm",
                  (bar_x + bar_length_px - text_size[0], bar_y + bar_height + 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Scale info above the bar
        if confidence is not None:
            scale_text = f"{scale_px_per_cm:.1f} px/cm ({confidence:.0%} confidence)"
        else:
            scale_text = f"{scale_px_per_cm:.1f} px/cm"
        
        text_size = cv2.getTextSize(scale_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = bar_x + (bar_length_px - text_size[0]) // 2
        cv2.putText(viz, scale_text,
                  (text_x, bar_y - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return viz
    
    @staticmethod
    def draw_regions(image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Draw rectangular regions (e.g., search regions, ROIs).
        
        Args:
            image: Input image
            **kwargs:
                - regions: List of (x1, y1, x2, y2) tuples
                - color: Region color (default: yellow)
                - thickness: Line thickness (default: 1)
                - fill: Whether to fill regions (default: False)
                - alpha: Transparency for filled regions (0-1, default: 0.3)
                
        Returns:
            Image with regions drawn
        """
        viz = image.copy()
        regions = kwargs.get('regions', [])
        color = kwargs.get('color', (0, 255, 255))  # Yellow default
        thickness = kwargs.get('thickness', 1)
        fill = kwargs.get('fill', False)
        alpha = kwargs.get('alpha', 0.3)
        
        for region in regions:
            if len(region) == 4:
                x1, y1, x2, y2 = region
                
                if fill:
                    # Create overlay for transparency
                    overlay = viz.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, alpha, viz, 1 - alpha, 0, viz)
                else:
                    cv2.rectangle(viz, (x1, y1), (x2, y2), color, thickness)
        
        return viz
    
    @staticmethod
    def draw_text(image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Draw text with optional background.
        
        Args:
            image: Input image
            **kwargs:
                - text: Text string to draw
                - position: (x, y) tuple for text position
                - font: OpenCV font (default: cv2.FONT_HERSHEY_SIMPLEX)
                - font_scale: Font scale (default: 1.0)
                - color: Text color (default: white)
                - thickness: Text thickness (default: 2)
                - bg_color: Optional background color
                - padding: Padding around text for background (default: 5)
                
        Returns:
            Image with text drawn
        """
        viz = image.copy()
        text = kwargs.get('text', '')
        position = kwargs.get('position', (10, 30))
        font = kwargs.get('font', cv2.FONT_HERSHEY_SIMPLEX)
        font_scale = kwargs.get('font_scale', 1.0)
        color = kwargs.get('color', (255, 255, 255))
        thickness = kwargs.get('thickness', 2)
        bg_color = kwargs.get('bg_color')
        padding = kwargs.get('padding', 5)
        
        if text:
            # Get text size
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Draw background if specified
            if bg_color is not None:
                x, y = position
                cv2.rectangle(viz,
                            (x - padding, y - text_size[1] - padding),
                            (x + text_size[0] + padding, y + padding),
                            bg_color, -1)
            
            # Draw text
            cv2.putText(viz, text, position, font, font_scale, color, thickness)
        
        return viz
    
    @staticmethod
    def draw_composite(image: np.ndarray, operations: List[Dict[str, Any]]) -> np.ndarray:
        """
        Apply multiple drawing operations in sequence.
        
        Args:
            image: Input image
            operations: List of dicts, each with 'type' and operation-specific kwargs
                Example: [
                    {'type': 'markers', 'markers': {...}, 'color': (255, 0, 0)},
                    {'type': 'boundaries', 'boundaries': [...], 'color': (0, 255, 0)},
                    {'type': 'scale_bar', 'scale_px_per_cm': 100}
                ]
                
        Returns:
            Image with all operations applied
        """
        viz = image.copy()
        
        drawer = VisualizationDrawer()
        
        for op in operations:
            op_type = op.get('type')
            op_kwargs = {k: v for k, v in op.items() if k != 'type'}
            
            if op_type == 'markers':
                viz = drawer.draw_markers(viz, **op_kwargs)
            elif op_type == 'boundaries':
                viz = drawer.draw_boundaries(viz, **op_kwargs)
            elif op_type == 'lines':
                viz = drawer.draw_lines(viz, **op_kwargs)
            elif op_type == 'boundary_lines':
                viz = drawer.draw_boundary_lines(viz, **op_kwargs)
            elif op_type == 'scale_bar':
                viz = drawer.draw_scale_bar(viz, **op_kwargs)
            elif op_type == 'regions':
                viz = drawer.draw_regions(viz, **op_kwargs)
            elif op_type == 'text':
                viz = drawer.draw_text(viz, **op_kwargs)
        
        return viz