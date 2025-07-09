"""
Handles creation of data visualizations for drillhole traces.
Supports both stitchable (for full traces) and detailed (for individual compartments) modes.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Suppress matplotlib font manager debug messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


class VisualizationMode(Enum):
    """Visualization modes for drillhole traces."""
    STITCHABLE = "stitchable"  # For creating full traces
    DETAILED = "detailed"       # For individual compartment plots


class PlotType(Enum):
    """Available plot types."""
    SOLID_COLUMN = "solid_column"
    STACKED_BAR = "stacked_bar"
    LINE_GRAPH = "line_graph"
    TERNARY = "ternary"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"


class PlotConfig:
    """Configuration for a single plot column."""
    
    def __init__(self, plot_type: PlotType, width: int = 100):
        self.plot_type = plot_type
        self.width = width
        self.data_source = None
        self.columns = []  # Column names to plot
        self.color_map = {}
        self.scale_min = None
        self.scale_max = None
        self.title = ""
        self.show_legend = True
        self.custom_params = {}  # Plot-specific parameters


class DrillholeDataVisualizer:
    """
    Creates data visualizations for drillhole traces.
    
    Supports two modes:
    - Stitchable: Creates row-based visualizations that can be vertically concatenated
    - Detailed: Creates full-context plots with current interval highlighted
    """
    
    def __init__(self, mode: VisualizationMode = VisualizationMode.STITCHABLE):
        """
        Initialize the visualizer.
        
        Args:
            mode: Visualization mode (stitchable or detailed)
        """
        self.mode = mode
        self.plot_configs: List[PlotConfig] = []
        self.logger = logging.getLogger(__name__)
        
        # Default color schemes
        self.default_colors = {
            'categorical': [
                (255, 99, 71),    # Tomato
                (60, 179, 113),   # Medium sea green
                (106, 90, 205),   # Slate blue
                (255, 165, 0),    # Orange
                (147, 112, 219),  # Medium purple
                (46, 139, 87),    # Sea green
                (255, 20, 147),   # Deep pink
                (72, 61, 139),    # Dark slate blue
            ],
            'gradient': {
                'low': (0, 0, 255),      # Blue
                'mid': (0, 255, 0),      # Green  
                'high': (255, 0, 0)      # Red
            }
        }
        
    def add_plot_column(self, config: PlotConfig) -> bool:
            """
            Add a plot column configuration.
            
            Args:
                config: Plot configuration
                
            Returns:
                True if added successfully
            """
            self.plot_configs.append(config)
            return True
    
    def generate_compartment_visualization(self,
                                            compartment_data: Dict[str, Any],
                                            full_hole_data: Optional[pd.DataFrame] = None,
                                            depth_from: float = None,
                                            depth_to: float = None,
                                            height: int = 300) -> np.ndarray:
            """
            Generate visualization for a single compartment.
            
            Args:
                compartment_data: Data for this specific compartment
                full_hole_data: Complete hole data (for detailed mode)
                depth_from: Start depth of compartment
                depth_to: End depth of compartment
                height: Height of the output image
                
            Returns:
                Composite image with all configured plots
            """
            if not self.plot_configs:
                return np.ones((height, 100, 3), dtype=np.uint8) * 255
                
            plot_images = []
            
            for config in self.plot_configs:
                if config.plot_type in [PlotType.SOLID_COLUMN, PlotType.STACKED_BAR]:
                    plot_img = self._generate_stitchable_plot(
                        config, compartment_data, height
                    )
                else:
                    plot_img = self._generate_detailed_plot(
                        config, compartment_data, full_hole_data, 
                        depth_from, depth_to, height
                    )
                    
                if plot_img is not None:
                    # ===================================================
                    # Ensure all images have the exact same height
                    # ===================================================
                    if plot_img.shape[0] != height:
                        plot_img = cv2.resize(plot_img, (plot_img.shape[1], height), 
                                            interpolation=cv2.INTER_AREA)
                    plot_images.append(plot_img)
                    
            # Combine plots horizontally
            if plot_images:
                return np.hstack(plot_images)
            else:
                return np.ones((height, 100, 3), dtype=np.uint8) * 255

    def _generate_stitchable_plot(self, 
                                config: PlotConfig,
                                data: Dict[str, Any],
                                height: int) -> Optional[np.ndarray]:
        """Generate a stitchable plot (solid column or stacked bar)."""
        if config.plot_type == PlotType.SOLID_COLUMN:
            return self._create_solid_column(config, data, height)
        elif config.plot_type == PlotType.STACKED_BAR:
            return self._create_stacked_bar(config, data, height)
        else:
            return None
            
    def _generate_detailed_plot(self,
                              config: PlotConfig,
                              compartment_data: Dict[str, Any],
                              full_data: pd.DataFrame,
                              depth_from: float,
                              depth_to: float,
                              height: int) -> Optional[np.ndarray]:
        """Generate a detailed plot with full context."""
        if config.plot_type == PlotType.LINE_GRAPH:
            return self._create_line_graph(
                config, full_data, depth_from, depth_to, height
            )
        elif config.plot_type == PlotType.TERNARY:
            return self._create_ternary_plot(
                config, full_data, compartment_data, height
            )
        elif config.plot_type == PlotType.SCATTER:
            return self._create_scatter_plot(
                config, full_data, depth_from, depth_to, height
            )
        else:
            return None
            

    def _create_solid_column(self, 
                           config: PlotConfig,
                           data: Dict[str, Any],
                           height: int) -> np.ndarray:
        """
        Create a solid colored column based on categorical value.
        
        Args:
            config: Plot configuration
            data: Compartment data
            height: Column height
            
        Returns:
            BGR image array
        """
        # Create column
        column = np.ones((height, config.width, 3), dtype=np.uint8) * 255
        
        # Get value for the specified column
        if config.columns and config.columns[0] in data:
            value = data[config.columns[0]]
            
            # Check if we have a ColorMap object
            if 'color_map_obj' in config.custom_params:
                color_map_obj = config.custom_params['color_map_obj']
                color = color_map_obj.get_color(value)
            elif value in config.color_map:
                color = config.color_map[value]
            else:
                # Assign a default color
                cat_index = hash(str(value)) % len(self.default_colors['categorical'])
                color = self.default_colors['categorical'][cat_index]
                
            # Fill column with color
            column[:] = color
            
            # Add text label if space permits
            if config.width > 40:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(value)[:10]  # Truncate long text
                text_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                
                if text_size[0] < config.width - 10:
                    # Determine text color based on background brightness
                    brightness = sum(color) / 3
                    text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
                    
                    # Center text
                    text_x = (config.width - text_size[0]) // 2
                    text_y = height // 2
                    
                    cv2.putText(column, text, (text_x, text_y),
                              font, 0.4, text_color, 1)
                    
        return column


    def _create_stacked_bar(self,
                          config: PlotConfig,
                          data: Dict[str, Any],
                          height: int) -> np.ndarray:
        """
        Create a 100% stacked horizontal bar.
        
        Args:
            config: Plot configuration
            data: Compartment data with multiple columns
            height: Bar height
            
        Returns:
            BGR image array
        """
        # Create column
        column = np.ones((height, config.width, 3), dtype=np.uint8) * 255
        
        # Collect values for stacking
        values = {}
        total = 0
        
        for col in config.columns:
            if col in data and data[col] is not None:
                try:
                    val = float(data[col])
                    if val > 0:
                        values[col] = val
                        total += val
                except (ValueError, TypeError):
                    pass
                    
        if total == 0 or not values:
            return column
            
        # Draw stacked bars
        current_x = 0
        
        for col, value in values.items():
            proportion = value / total
            bar_width = int(config.width * proportion)
            
            # Ensure we fill the entire width
            if col == list(values.keys())[-1]:
                bar_width = config.width - current_x
                
            # Get color
            if col in config.color_map:
                color = config.color_map[col]
            else:
                cat_index = hash(col) % len(self.default_colors['categorical'])
                color = self.default_colors['categorical'][cat_index]
                
            # Draw bar segment
            column[:, current_x:current_x + bar_width] = color
            
            # Add percentage text if segment is wide enough
            if bar_width > 20:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"{int(proportion * 100)}%"
                text_size = cv2.getTextSize(text, font, 0.3, 1)[0]
                
                if text_size[0] < bar_width - 4:
                    brightness = sum(color) / 3
                    text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
                    
                    text_x = current_x + (bar_width - text_size[0]) // 2
                    text_y = height // 2
                    
                    cv2.putText(column, text, (text_x, text_y),
                              font, 0.3, text_color, 1)
                    
            current_x += bar_width
            
        return column
        
    def _create_line_graph(self,
                         config: PlotConfig,
                         full_data: pd.DataFrame,
                         depth_from: float,
                         depth_to: float,
                         height: int) -> np.ndarray:
        """
        Create a line graph with current interval highlighted.
        
        Args:
            config: Plot configuration
            full_data: Complete hole data
            depth_from: Start depth to highlight
            depth_to: End depth to highlight
            height: Plot height
            
        Returns:
            BGR image array
        """
        # Set up matplotlib figure
        dpi = 100
        fig_width = config.width / dpi
        fig_height = height / dpi
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        fig.patch.set_facecolor('white')
        
        # Plot each configured column
        for i, col in enumerate(config.columns):
            if col in full_data.columns:
                # Get data
                plot_data = full_data[[col, 'from', 'to']].copy()
                plot_data['depth'] = (plot_data['from'] + plot_data['to']) / 2
                plot_data = plot_data.dropna(subset=[col])
                
                if len(plot_data) > 0:
                    # Plot line
                    ax.plot(plot_data[col], plot_data['depth'], 
                           label=col, linewidth=1.5)
                    
        # Highlight current interval
        ax.axhspan(depth_from, depth_to, alpha=0.2, color='red', zorder=0)
        
        # Add horizontal lines at interval boundaries
        ax.axhline(y=depth_from, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=depth_to, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        # Configure axes
        ax.invert_yaxis()  # Depth increases downward
        ax.set_ylabel('Depth (m)', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        
        # Set x-axis limits if configured
        if config.scale_min is not None and config.scale_max is not None:
            ax.set_xlim(config.scale_min, config.scale_max)
            
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend if requested and there's space
        if config.show_legend and config.width > 150:
            ax.legend(fontsize=6, loc='best')
            
        # Tight layout
        plt.tight_layout(pad=0.5)
        
        # Draw the canvas first
        fig.canvas.draw()
        
        # Convert to numpy array
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        ncols, nrows = fig.canvas.get_width_height()
        img_array = buf.reshape(nrows, ncols, 4)
        
        # Convert RGBA to BGR
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        
        return img_bgr
        
    def _create_ternary_plot(self,
                            config: PlotConfig,
                            full_data: pd.DataFrame,
                            current_data: Dict[str, Any],
                            height: int) -> np.ndarray:
            """
            Create a ternary diagram with current point highlighted.
            
            Args:
                config: Plot configuration (should have 3 columns)
                full_data: Complete hole data
                current_data: Current compartment data
                height: Plot height
                
            Returns:
                BGR image array
            """
            try:
                import ternary
            except ImportError:
                self.logger.error("python-ternary package not installed")
                return np.ones((height, config.width, 3), dtype=np.uint8) * 255
                
            if len(config.columns) != 3:
                self.logger.error("Ternary plot requires exactly 3 columns")
                return np.ones((height, config.width, 3), dtype=np.uint8) * 255
                
            # ===================================================
            # Force square aspect ratio for ternary plots
            # ===================================================
            width = height  # Make it square
            
            # Set up figure
            scale = 100
            fig, tax = ternary.figure(scale=scale)
            fig.set_size_inches(width / 100, height / 100)
            fig.patch.set_facecolor('white')
            
            # ===================================================
            # Add gridlines, ticks, and boundaries
            # ===================================================
            # Draw boundary and gridlines
            tax.boundary(linewidth=2.0)
            tax.gridlines(color="gray", multiple=10, linewidth=0.5, alpha=0.7)
            
            # Add ticks
            tax.ticks(axis='lbr', linewidth=1, multiple=10, offset=0.015, 
                    tick_formats="%.0f", fontsize=8)
            
            # Get data for the three components
            plot_data = full_data[config.columns].dropna()
            
            if len(plot_data) > 0:
                # Normalize to sum to scale
                points = []
                for _, row in plot_data.iterrows():
                    total = row.sum()
                    if total > 0:
                        normalized = (row / total * scale).values
                        points.append(normalized)
                        
                # Plot all points in gray
                if points:
                    tax.scatter(points, color='gray', alpha=0.5, s=20)
                    
            # Plot current point if available
            current_values = []
            for col in config.columns:
                if col in current_data and current_data[col] is not None:
                    try:
                        current_values.append(float(current_data[col]))
                    except (ValueError, TypeError):
                        current_values.append(0)
                else:
                    current_values.append(0)
                    
            if sum(current_values) > 0:
                # Normalize
                total = sum(current_values)
                current_normalized = [v / total * scale for v in current_values]
                
                # Plot with highlight
                tax.scatter([current_normalized], color='red', s=100,
                        edgecolor='black', linewidth=2, zorder=10)
                
            # ===================================================
            # Add corner labels with better positioning
            # ===================================================
            fontsize = 10
            offset = 0.14
            
            # Bottom axis label (component 1)
            tax.bottom_axis_label(config.columns[0], fontsize=fontsize, offset=offset)
            
            # Right axis label (component 2) 
            tax.right_axis_label(config.columns[1], fontsize=fontsize, offset=offset)
            
            # Left axis label (component 3)
            tax.left_axis_label(config.columns[2], fontsize=fontsize, offset=offset)
            
            # ===================================================
            # Add corner labels (actual corner text)
            # ===================================================
            tax.top_corner_label(config.columns[2], fontsize=fontsize - 2)
            tax.right_corner_label(config.columns[0], fontsize=fontsize - 2)
            tax.left_corner_label(config.columns[1], fontsize=fontsize - 2)
            
            # Remove default matplotlib axes
            tax.get_axes().axis('off')
            
            # Tight layout
            plt.tight_layout(pad=0.5)
            
            # Convert to image using the newer approach
            # Draw the canvas first
            fig.canvas.draw()
            
            # Get the RGBA buffer
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            ncols, nrows = fig.canvas.get_width_height()
            img_array = buf.reshape(nrows, ncols, 4)
            
            # Convert RGBA to BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            
            plt.close(fig)
            
            # ===================================================
            # Ensure exact dimensions (square)
            # ===================================================
            if img_bgr.shape[0] != height or img_bgr.shape[1] != width:
                img_bgr = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
            
            return img_bgr

    def _create_scatter_plot(self,
                           config: PlotConfig,
                           full_data: pd.DataFrame,
                           depth_from: float,
                           depth_to: float,
                           height: int) -> np.ndarray:
        """
        Create a scatter plot with current interval highlighted.
        
        Args:
            config: Plot configuration (should have 2 columns)
            full_data: Complete hole data
            depth_from: Start depth to highlight
            depth_to: End depth to highlight
            height: Plot height
            
        Returns:
            BGR image array
        """
        if len(config.columns) < 2:
            self.logger.error("Scatter plot requires at least 2 columns")
            return np.ones((height, config.width, 3), dtype=np.uint8) * 255
            
        # Set up matplotlib figure
        dpi = 100
        fig_width = config.width / dpi
        fig_height = height / dpi
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        fig.patch.set_facecolor('white')
        
        # Get data
        x_col = config.columns[0]
        y_col = config.columns[1]
        
        if x_col in full_data.columns and y_col in full_data.columns:
            # Calculate depth for each point
            plot_data = full_data[[x_col, y_col, 'from', 'to']].copy()
            plot_data['depth'] = (plot_data['from'] + plot_data['to']) / 2
            plot_data = plot_data.dropna(subset=[x_col, y_col])
            
            if len(plot_data) > 0:
                # Separate current interval data
                current_mask = ((plot_data['from'] >= depth_from) & 
                              (plot_data['to'] <= depth_to))
                
                # Plot all points
                ax.scatter(plot_data[~current_mask][x_col],
                         plot_data[~current_mask][y_col],
                         c='gray', alpha=0.5, s=20, label='Other intervals')
                         
                # Highlight current interval
                if current_mask.any():
                    ax.scatter(plot_data[current_mask][x_col],
                             plot_data[current_mask][y_col],
                             c='red', s=50, edgecolor='black',
                             linewidth=1, label=f'{depth_from}-{depth_to}m')
                             
        # Configure axes
        ax.set_xlabel(x_col, fontsize=8)
        ax.set_ylabel(y_col, fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend if space permits
        if config.show_legend and config.width > 150:
            ax.legend(fontsize=6, loc='best')
            
        # Tight layout
        plt.tight_layout(pad=0.5)
        
        # Draw the canvas first
        fig.canvas.draw()
        
        # Convert to numpy array
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        ncols, nrows = fig.canvas.get_width_height()
        img_array = buf.reshape(nrows, ncols, 4)
        
        # Convert RGBA to BGR
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        
        return img_bgr
        
    def create_color_legend(self, width: int = 200, max_height: int = 400) -> np.ndarray:
        """
        Create a color legend for all configured plots.
        
        Args:
            width: Legend width
            max_height: Maximum legend height
            
        Returns:
            BGR image array of the legend
        """
        # Set up matplotlib figure
        fig, ax = plt.subplots(figsize=(width/100, max_height/100), dpi=100)
        fig.patch.set_facecolor('white')
        ax.axis('off')
        
        y_pos = 0.95
        
        for config in self.plot_configs:
            if config.color_map and config.show_legend:
                # Plot title
                ax.text(0.1, y_pos, config.title or config.plot_type.value,
                       fontsize=10, fontweight='bold')
                y_pos -= 0.05
                
                # Plot color entries
                for label, color in config.color_map.items():
                    if y_pos < 0.05:
                        break
                        
                    # Convert BGR to RGB for matplotlib
                    rgb_color = (color[2]/255, color[1]/255, color[0]/255)
                    
                    # Color box
                    ax.add_patch(plt.Rectangle((0.1, y_pos - 0.03), 0.1, 0.025,
                                             facecolor=rgb_color, edgecolor='black'))
                    
                    # Label
                    ax.text(0.25, y_pos - 0.02, str(label), fontsize=8,
                           verticalalignment='center')
                    
                    y_pos -= 0.04
                    
                y_pos -= 0.05  # Space between plot types
                
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Draw the canvas first
        fig.canvas.draw()
        
        # Convert to numpy array
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        ncols, nrows = fig.canvas.get_width_height()
        img_array = buf.reshape(nrows, ncols, 4)
        
        # Convert RGBA to BGR
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        
        # Crop to actual content if needed, but maintain the requested width
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(255 - gray)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Crop height only, keep full width
            img_bgr = img_bgr[y:y+h+10, :]
            
        # Ensure the legend has exactly the requested width
        if img_bgr.shape[1] != width:
            # Resize to match the requested width
            height_ratio = width / img_bgr.shape[1]
            new_height = int(img_bgr.shape[0] * height_ratio)
            img_bgr = cv2.resize(img_bgr, (width, new_height), interpolation=cv2.INTER_AREA)
            
        return img_bgr