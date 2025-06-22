# core/visualization_manager.py


class VisualizationManager:
    """Manages creation and storage of debug visualizations."""
    
    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.visualization_cache = {}
    
    def create_step_visualization(self, original_image, processing_steps):
        """Create a visualization showing processing steps."""
        # Visualization creation code
    
    def create_marker_visualization(self, image, markers):
        """Create a visualization of detected markers."""
        # Marker visualization code
    
    def save_visualization(self, image, metadata, type_name):
        """Save visualization through FileManager."""
        if metadata and 'hole_id' in metadata:
            self.file_manager.save_debug_image(
                image,
                metadata['hole_id'],
                metadata.get('depth_from', 0),
                metadata.get('depth_to', 0),
                type_name
            )
        else:
            self.file_manager.save_temp_debug_image(image, "unknown", type_name)