 
import os
import cv2
import numpy as np
import pandas as pd
import re
import logging
import traceback
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional, Tuple, Any

# Assuming DialogHelper is from a local helper module
from gui.dialog_helper import DialogHelper
from gui.drillhole_trace_designer import DrillholeTraceDesigner
from processing.LoggingReviewStep.drillhole_data_manager import DrillholeDataManager
from processing.LoggingReviewStep.drillhole_data_visualizer import DrillholeDataVisualizer, VisualizationMode



class DrillholeTraceGenerator:
    """
    A class to generate drillhole trace images by stitching together chip tray compartment images.
    Integrates with ChipTrayExtractor to create complete drillhole visualization.
    """
    
    def __init__(self, 
                config: Dict[str, Any] = None, 
                progress_queue: Optional[Any] = None,
                root: Optional[tk.Tk] = None,
                file_manager: Optional[Any] = None):
        """
        Initialize the Drillhole Trace Generator.
        
        Args:
            config: Configuration dictionary
            progress_queue: Optional queue for reporting progress
            root: Optional Tkinter root for dialog windows
        """
        self.progress_queue = progress_queue
        self.root = root
        self.file_manager = file_manager
        
        # Default configuration
        self.config = {
            'output_folder': 'drillhole_traces',
            'metadata_box_color': (200, 200, 200, 150),  # BGRA (light gray with transparency)
            'metadata_text_color': (0, 0, 0),  # BGR (black)
            'metadata_font_scale': 0.7,
            'metadata_font_thickness': 1,
            'metadata_font_face': cv2.FONT_HERSHEY_SIMPLEX,
            'metadata_box_padding': 10,
            'metadata_pattern': r'(.+)_CC_(\d+\.?\d*)-(\d+\.?\d*)m',  # Pattern to extract metadata from filename
            'max_width': 2000,  # Maximum width for output image
            'min_width': 800,   # Minimum width for output image
            'box_alpha': 0.7,    # Transparency of metadata box (0-1)
            'additional_columns': [],  # Additional columns from CSV to include
            'save_individual_compartments': True  # Save un-stitched images
        }
        
        # Update with provided config if any
        if config:
            self.config.update(config)
            
        # Logger
        self.logger = logging.getLogger(__name__)

    def process_selected_holes(self, 
                        compartment_dir: str,
                        csv_path: Optional[str] = None,
                        selected_columns: Optional[List[str]] = None,
                        hole_ids: Optional[List[str]] = None) -> List[str]:
        """
        Process specific holes to create trace images.
        
        Args:
            compartment_dir: Directory containing compartment images
            csv_path: Optional path to CSV file with additional data
            selected_columns: Optional list of columns to include from CSV
            hole_ids: Optional list of specific hole IDs to process
            
        Returns:
            List of paths to generated trace images
        """
        # Load CSV data
        csv_data = None
        if csv_path and os.path.exists(csv_path):
            try:
                # Load the CSV
                csv_data = pd.read_csv(csv_path)
                
                # Convert column names to lowercase for case-insensitive matching
                csv_data.columns = [col.lower() for col in csv_data.columns]
                
                # Ensure numeric columns are properly typed
                for col in csv_data.columns:
                    if col not in ['holeid', 'cutoffs1', 'cutoffs2']:  # Don't convert text columns
                        try:
                            csv_data[col] = pd.to_numeric(csv_data[col], errors='coerce')
                        except:
                            self.logger.warning(f"Could not convert column {col} to numeric")
                
                self.logger.info(f"Loaded CSV with {len(csv_data)} rows and {len(csv_data.columns)} columns")
                if selected_columns:
                    self.logger.info(f"Selected columns: {selected_columns}")
                    
            except Exception as e:
                self.logger.error(f"Error loading CSV data: {str(e)}")
                if self.progress_queue:
                    self.progress_queue.put((f"Error loading CSV data: {str(e)}", None))
                    
        # Get the output directory from FileManager
        output_dir = self.file_manager.dir_structure["drill_traces"] if self.file_manager else None
        
        if not output_dir:
            self.logger.error("No output directory available")
            return []
        
        # Process each hole separately based on subdirectories
        generated_traces = []
        
        # Process only specified holes if provided
        for i, hole_id in enumerate(hole_ids or []):
            try:
                # Update progress
                if self.progress_queue:
                    progress = ((i + 1) / len(hole_ids)) * 100
                    self.progress_queue.put((f"Processing hole {i+1}/{len(hole_ids)}: {hole_id}", progress))
                
                # Path to hole directory inside compartment_dir
                hole_dir = os.path.join(compartment_dir, hole_id)
                
                if not os.path.isdir(hole_dir):
                    self.logger.warning(f"Directory not found for hole {hole_id}")
                    continue
                    
                # Collect compartment images for this hole
                compartments = []
                
                # Look for compartment images
                image_files = [f for f in os.listdir(hole_dir) 
                            if os.path.isfile(os.path.join(hole_dir, f)) and 
                            f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
                
                # Process each compartment image
                for filename in image_files:
                    # Parse metadata from filename
                    hole_id_from_file, depth_from, depth_to = self.parse_filename_metadata(filename)
                    
                    if hole_id_from_file and depth_from is not None and depth_to is not None:
                        # Check if this belongs to the current hole
                        if hole_id_from_file.upper() == hole_id.upper():
                            file_path = os.path.join(hole_dir, filename)
                            compartments.append((hole_id, depth_from, depth_to, file_path))
                
                # Sort compartments by depth
                compartments = sorted(compartments, key=lambda x: x[1])
                
                if not compartments:
                    self.logger.warning(f"No valid compartment images found for hole {hole_id}")
                    continue
                    
                # Generate trace
                trace_path = self.generate_drillhole_trace_cv2(
                    hole_id, compartments, csv_data, output_dir
                )
                
                if trace_path:
                    generated_traces.append(trace_path)
                    self.logger.info(f"Generated trace for hole {hole_id}")
                else:
                    self.logger.warning(f"Failed to generate trace for hole {hole_id}")
                    
            except Exception as e:
                self.logger.error(f"Error processing hole {hole_id}: {str(e)}")
                self.logger.error(traceback.format_exc())
        
        # Final status update
        status_msg = f"Completed trace generation: {len(generated_traces)}/{len(hole_ids) if hole_ids else 0} successful"
        self.logger.info(status_msg)
        if self.progress_queue:
            self.progress_queue.put((status_msg, 100))
        
        return generated_traces

    def get_csv_columns(self, csv_path: str) -> List[str]:
        """
        Get column names from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            List of column names
        """
        try:
            # Read just the header row for efficiency
            df = pd.read_csv(csv_path, nrows=0)
            columns = df.columns.tolist()
            
            # Check for required columns
            required_columns = ['holeid', 'from', 'to']
            missing_columns = [col for col in required_columns if col.lower() not in [c.lower() for c in columns]]
            
            if missing_columns:
                msg = f"CSV missing required columns: {', '.join(missing_columns)}"
                self.logger.error(msg)
                if self.progress_queue:
                    self.progress_queue.put((msg, None))
                return []
            
            return columns
        except Exception as e:
            self.logger.error(f"Error reading CSV columns: {str(e)}")
            if self.progress_queue:
                self.progress_queue.put((f"Error reading CSV: {str(e)}", None))
            return []

    def select_csv_columns(self, columns: List[str]) -> List[str]:
        """
        Open a dialog to let the user select columns from a CSV.
        
        Args:
            columns: List of column names from the CSV
            
        Returns:
            List of selected column names
        """
        if not self.root:
            self.logger.warning("No Tkinter root available for column selection dialog")
            return []
            
        # Create a dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(DialogHelper.t("Select CSV Columns"))
        dialog.geometry("500x400")
        dialog.grab_set()  # Make dialog modal
        
        # Explanatory text
        header_label = ttk.Label(
            dialog, 
            text=DialogHelper.t("Select additional columns to display in the metadata box (max 5):"),
            wraplength=480,
            justify=tk.LEFT,
            padding=(10, 10)
        )
        header_label.pack(fill=tk.X)
        
        # Required columns notice
        required_label = ttk.Label(
            dialog, 
            text=DialogHelper.t("Note: 'holeid', 'from', and 'to' are always included."),
            font=("Arial", 9, "italic"),
            foreground="gray",
            padding=(10, 0, 10, 10)
        )
        required_label.pack(fill=tk.X)
        
        # Create a frame for the column checkboxes
        column_frame = ttk.Frame(dialog, padding=10)
        column_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas with scrollbar for many columns
        canvas = tk.Canvas(column_frame)
        scrollbar = ttk.Scrollbar(column_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Dictionary to track which columns are selected
        selected_columns = {}
        
        # Exclude required columns from the selection list
        required_columns = ['holeid', 'from', 'to']
        selectable_columns = [col for col in columns if col.lower() not in [c.lower() for c in required_columns]]
        
        # Add checkboxes for each column
        for column in selectable_columns:
            var = tk.BooleanVar(value=False)
            selected_columns[column] = var
            
            checkbox = ttk.Checkbutton(
                scrollable_frame,
                text=DialogHelper.t(column),
                variable=var,
                command=lambda: self._update_selection_count(selected_columns, selection_label)
            )
            checkbox.pack(anchor="w", padx=10, pady=5)
        
        # Label to show how many columns are selected
        selection_label = ttk.Label(
            dialog,
            text=DialogHelper.t("0 columns selected (max 5)"),
            padding=(10, 10)
        )
        selection_label.pack()
        
        # Buttons frame
        button_frame = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        button_frame.pack(fill=tk.X)
        
        # Result container
        result = []
        
        # OK button handler
        def on_ok():
            nonlocal result
            result = [col for col, var in selected_columns.items() if var.get()]
            if len(result) > 5:
                # TODO - Update this message box with the DialogHelper method
                messagebox.showwarning("Too Many Columns", "Please select at most 5 columns.")
                return
            dialog.destroy()
        
        # Cancel button handler
        def on_cancel():
            nonlocal result
            result = []
            dialog.destroy()
        
        # Add buttons
        ok_button = ttk.Button(button_frame, text=DialogHelper.t("OK"), command=on_ok)
        ok_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        cancel_button = ttk.Button(button_frame, text=DialogHelper.t("Cancel"), command=on_cancel)
        cancel_button.pack(side=tk.RIGHT)
        
        # Wait for dialog to close
        dialog.wait_window()
        return result
        
    def _update_selection_count(self, selected_columns, label_widget):
        """Helper method to update the selection count label."""
        count = sum(var.get() for var in selected_columns.values())
        color = "red" if count > 5 else "black"
        label_widget.config(text=self.t(f"{count} columns selected (max 5)"), foreground=color)


    def show_trace_designer(self) -> Optional[Dict[str, Any]]:
            """
            Show the trace designer GUI and get configuration.
            
            Returns:
                Configuration dictionary or None if cancelled
            """
            # Get GUI manager from parent if available
            gui_manager = None
            if hasattr(self, 'app') and hasattr(self.app, 'gui_manager'):
                gui_manager = self.app.gui_manager
            elif hasattr(self, 'parent') and hasattr(self.parent, 'gui_manager'):
                gui_manager = self.parent.gui_manager
            else:
                self.logger.warning("No GUI manager available for trace designer")
                return None
                
            # Create and show designer
            designer = DrillholeTraceDesigner(
                parent=self.root,
                file_manager=self.file_manager,
                gui_manager=gui_manager,
                config=self.config
            )
            
            result = designer.show()
            
            if result:
                # Store the configuration
                self.data_manager = result['data_manager']
                self.visualizer = result['visualizer']
                # ===================================================
                # Store orientation instead of mode
                # ===================================================
                self.compartment_orientation = result.get('orientation', 'horizontal')
                
            return result
        
    def generate_configured_traces(self, 
                                    compartment_dir: str,
                                    hole_ids: Optional[List[str]] = None) -> List[str]:
            """
            Generate traces using the configured data manager and visualizer.
            
            Args:
                compartment_dir: Directory containing compartment images
                hole_ids: Optional list of specific hole IDs to process
                
            Returns:
                List of paths to generated trace images
            """
            if not hasattr(self, 'data_manager') or not hasattr(self, 'visualizer'):
                self.logger.error("No configuration available. Run show_trace_designer() first.")
                return []
                
            generated_traces = []
            
            # Process each hole
            holes_to_process = hole_ids or self.data_manager.hole_ids
            
            for i, hole_id in enumerate(holes_to_process):
                try:
                    # Update progress
                    if self.progress_queue:
                        progress = ((i + 1) / len(holes_to_process)) * 100
                        self.progress_queue.put((f"Processing hole {i+1}/{len(holes_to_process)}: {hole_id}", progress))
                        
                    # Get compartment images
                    hole_dir = os.path.join(compartment_dir, hole_id)
                    if not os.path.isdir(hole_dir):
                        self.logger.warning(f"Directory not found for hole {hole_id}")
                        continue
                        
                    # Collect compartment images
                    compartments = []
                    image_files = [f for f in os.listdir(hole_dir) 
                                if os.path.isfile(os.path.join(hole_dir, f)) and 
                                f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
                    
                    for filename in image_files:
                        hole_id_from_file, depth_from, depth_to = self.parse_filename_metadata(filename)
                        if hole_id_from_file and hole_id_from_file.upper() == hole_id.upper():
                            file_path = os.path.join(hole_dir, filename)
                            compartments.append((hole_id, depth_from, depth_to, file_path))
                            
                    # Sort by depth
                    compartments = sorted(compartments, key=lambda x: x[1])
                    
                    if not compartments:
                        self.logger.warning(f"No compartment images found for hole {hole_id}")
                        continue
                        
                    # Get data for this hole
                    hole_data = self.data_manager.get_data_for_hole(hole_id)
                    
                    trace_path = self._generate_stitchable_trace(
                        hole_id, compartments, hole_data
                    )
                        
                    if trace_path:
                        generated_traces.append(trace_path)
                        
                except Exception as e:
                    self.logger.error(f"Error processing hole {hole_id}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    
            return generated_traces
        
    def _generate_stitchable_trace(self,
                                  hole_id: str,
                                  compartments: List[Tuple[str, float, float, str]],
                                  hole_data: pd.DataFrame) -> Optional[str]:
        """
        Generate a stitchable trace with configured visualizations.
        
        Args:
            hole_id: Hole ID
            compartments: List of compartment tuples
            hole_data: DataFrame with hole data
            
        Returns:
            Path to generated trace or None
        """
        if not compartments:
            return None
            
        try:
            output_dir = self.file_manager.dir_structure["drill_traces"]
            
            # Process each compartment
            processed_images = []
            
            for hole_id, depth_from, depth_to, file_path in compartments:
                # Load compartment image
                img = cv2.imread(file_path)
                if img is None:
                    self.logger.warning(f"Couldn't read image: {file_path}")
                    continue
                    
                # ===================================================
                # Handle orientation based on configuration
                # ===================================================
                if hasattr(self, 'compartment_orientation') and self.compartment_orientation == 'vertical':
                    # Keep vertical orientation
                    oriented_img = img
                else:
                    # Default to horizontal (rotate 90 degrees clockwise)
                    oriented_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                
                # Get data for this interval
                interval_data = hole_data[
                    (hole_data['from'] <= depth_from) & 
                    (hole_data['to'] >= depth_to)
                ]
                
                if not interval_data.empty:
                    compartment_data = interval_data.iloc[0].to_dict()
                else:
                    compartment_data = {}
                    
                # Generate visualization
                data_viz = self.visualizer.generate_compartment_visualization(
                    compartment_data=compartment_data,
                    full_hole_data=hole_data,
                    depth_from=depth_from,
                    depth_to=depth_to,
                    height=oriented_img.shape[0]
                )
                
                # Combine compartment image with data
                if data_viz is not None and data_viz.shape[1] > 0:
                    combined = np.hstack([oriented_img, data_viz])
                else:
                    combined = oriented_img
                    
                # Add depth label
                cv2.putText(combined, f"{hole_id} - {int(depth_from)}-{int(depth_to)}m",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                           
                processed_images.append(combined)
                
            if not processed_images:
                self.logger.warning(f"No images processed for hole {hole_id}")
                return None
                
            # Determine the standard width from the first processed image
            standard_width = processed_images[0].shape[1]
            
            # Create header
            header_height = 80
            header = np.ones((header_height, standard_width, 3), dtype=np.uint8) * 255
            
            cv2.putText(header, f"Drillhole: {hole_id}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
                       
            # Create legend with matching width
            legend = self.visualizer.create_color_legend(width=standard_width)
            
            # Ensure all images have the same width and number of channels
            normalized_images = []
            
            # Add header
            normalized_images.append(header)
            
            # Process each compartment image
            for img in processed_images:
                # Ensure it's BGR (3 channels)
                if len(img.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:  # BGRA
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    
                # Ensure width matches
                if img.shape[1] != standard_width:
                    self.logger.warning(f"Resizing image from width {img.shape[1]} to {standard_width}")
                    img = cv2.resize(img, (standard_width, img.shape[0]), interpolation=cv2.INTER_AREA)
                    
                normalized_images.append(img)
                
            # Add legend if it exists
            if legend is not None:
                # Ensure legend is BGR
                if len(legend.shape) == 2:
                    legend = cv2.cvtColor(legend, cv2.COLOR_GRAY2BGR)
                elif legend.shape[2] == 4:
                    legend = cv2.cvtColor(legend, cv2.COLOR_BGRA2BGR)
                    
                normalized_images.append(legend)
                
            # Now all images should have the same width and channel count
            try:
                final_image = cv2.vconcat(normalized_images)
            except Exception as e:
                self.logger.error(f"Error concatenating images: {str(e)}")
                # Debug info
                for i, img in enumerate(normalized_images):
                    self.logger.error(f"Image {i}: shape={img.shape}, dtype={img.dtype}")
                raise
                
            # Save the result
            output_filename = f"{hole_id}_trace.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, final_image)
            
            self.logger.info(f"Successfully created drillhole trace for {hole_id} at {output_path}")
            return output_path
                
        except Exception as e:
            self.logger.error(f"Error generating stitchable trace: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _generate_detailed_traces(self,
                                hole_id: str,
                                compartments: List[Tuple[str, float, float, str]],
                                hole_data: pd.DataFrame) -> List[str]:
        """
        Generate detailed traces with full-context plots stitched together.
        
        Args:
            hole_id: Hole ID
            compartments: List of compartment tuples
            hole_data: DataFrame with hole data
            
        Returns:
            List containing path to the generated stitched image
        """
        if not compartments:
            return []
            
        try:
            output_dir = self.file_manager.dir_structure["drill_traces"]
            
            # Process each compartment
            processed_images = []
            
            for hole_id_comp, depth_from, depth_to, file_path in compartments:
                try:
                    # Load compartment image
                    img = cv2.imread(file_path)
                    if img is None:
                        self.logger.warning(f"Couldn't read image: {file_path}")
                        continue
                        
                    # Rotate 90 degrees clockwise for horizontal display
                    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    
                    # Get data for this interval
                    interval_data = hole_data[
                        (hole_data['from'] <= depth_from) & 
                        (hole_data['to'] >= depth_to)
                    ]
                    
                    if not interval_data.empty:
                        compartment_data = interval_data.iloc[0].to_dict()
                    else:
                        compartment_data = {}
                        
                    # Generate visualization with full context
                    data_viz = self.visualizer.generate_compartment_visualization(
                        compartment_data=compartment_data,
                        full_hole_data=hole_data,
                        depth_from=depth_from,
                        depth_to=depth_to,
                        height=rotated_img.shape[0]  # Match compartment height
                    )
                    
                    # Combine compartment image with data
                    if data_viz is not None and data_viz.shape[1] > 0:
                        combined = np.hstack([rotated_img, data_viz])
                    else:
                        combined = rotated_img
                        
                    # Add depth label
                    cv2.putText(combined, f"{hole_id} - {int(depth_from)}-{int(depth_to)}m",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                               cv2.LINE_AA)
                               
                    processed_images.append(combined)
                    
                except Exception as e:
                    self.logger.error(f"Error processing compartment: {str(e)}")
                    
            # Create header
            if processed_images:
                header_height = 100
                header_width = processed_images[0].shape[1]
                header = np.ones((header_height, header_width, 3), dtype=np.uint8) * 255
                
                # Add title
                cv2.putText(header, f"Drillhole: {hole_id} - Detailed Analysis", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
                           
                # Add mode info
                cv2.putText(header, "Full downhole context shown for each interval", (20, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (64, 64, 64), 1)
                           
                # Create legend
                legend = self.visualizer.create_color_legend(width=header_width)
                
                # Combine all vertically
                all_images = [header] + processed_images
                if legend is not None:
                    all_images.append(legend)
                    
                final_image = cv2.vconcat(all_images)
                
                # Save the stitched image
                output_filename = f"{hole_id}_detailed_trace.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, final_image)
                
                self.logger.info(f"Generated detailed trace for {hole_id}")
                return [output_path]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error generating detailed trace: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []
        

    def parse_filename_metadata(self, filename: str) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """
        Parse metadata from a filename based on various patterns including leading zeros.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Tuple of (hole_id, depth_from, depth_to)
        """
        try:
            # Get compartment interval from config
            compartment_interval = self.config.get('compartment_interval', 1)
            
            # Try the pattern with any number of digits: HoleID_CC_EndDepth
            match = re.search(r'([A-Za-z]{2}\d{4})_CC_(\d{1,3})(?:\..*)?$', filename)
            if match:
                hole_id = match.group(1)
                depth_to = float(match.group(2))
                # Use the configured interval
                depth_from = depth_to - compartment_interval
                self.logger.info(f"Parsed from CC format: {hole_id}, {depth_from}-{depth_to}m (interval: {compartment_interval}m)")
                return hole_id, depth_from, depth_to
            
            # Try with 3-digit format explicitly
            match = re.search(r'([A-Za-z]{2}\d{4})_CC_(\d{3})(?:\..*)?$', filename)
            if match:
                hole_id = match.group(1)
                depth_to = float(match.group(2))
                # Use the configured interval
                depth_from = depth_to - compartment_interval
                self.logger.info(f"Parsed from 3-digit format: {hole_id}, {depth_from}-{depth_to}m (interval: {compartment_interval}m)")
                return hole_id, depth_from, depth_to
            
            # Try the old pattern as fallback
            match = re.match(self.config['metadata_pattern'], filename)
            if match:
                hole_id = match.group(1)
                depth_from = float(match.group(2))
                depth_to = float(match.group(3))
                return hole_id, depth_from, depth_to
            
            # Try another generic pattern if needed
            match = re.search(r'([A-Za-z]{2}\d{4}).*?(\d+\.?\d*)-(\d+\.?\d*)m', filename)
            if match:
                hole_id = match.group(1)
                depth_from = float(match.group(2))
                depth_to = float(match.group(3))
                return hole_id, depth_from, depth_to
                
            # No match found
            self.logger.warning(f"Could not parse metadata from filename: {filename}")
            return None, None, None
            
        except Exception as e:
            self.logger.error(f"Error parsing filename metadata: {str(e)}")
            return None, None, None


    def collect_compartment_images(self, compartment_dir: str) -> Dict[str, List[Tuple[str, float, float, str]]]:
        """
        Collect and organize compartment images by hole ID.
        
        Args:
            compartment_dir: Directory containing compartment images
            
        Returns:
            Dictionary mapping hole IDs to lists of (hole_id, depth_from, depth_to, file_path) tuples
        """
        # Dictionary to store compartment info by hole ID
        hole_compartments: Dict[str, List[Tuple[str, float, float, str]]] = {}
        
        # Valid image extensions
        valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        
        try:
            # Get all image files
            image_files = [f for f in os.listdir(compartment_dir) 
                        if os.path.isfile(os.path.join(compartment_dir, f)) and 
                        f.lower().endswith(valid_extensions)]
            
            if not image_files:
                self.logger.warning(f"No image files found in {compartment_dir}")
                return hole_compartments
                
            self.logger.info(f"Found {len(image_files)} image files to process")
            
            # Process each file
            for filename in image_files:
                # Parse metadata from filename
                hole_id, depth_from, depth_to = self.parse_filename_metadata(filename)
                
                if hole_id and depth_from is not None and depth_to is not None:
                    # Add to appropriate hole ID list
                    if hole_id not in hole_compartments:
                        hole_compartments[hole_id] = []
                        
                    file_path = os.path.join(compartment_dir, filename)
                    hole_compartments[hole_id].append((hole_id, depth_from, depth_to, file_path))
            
            # Sort each hole's compartments by depth
            for hole_id, compartments in hole_compartments.items():
                hole_compartments[hole_id] = sorted(compartments, key=lambda x: x[1])  # Sort by depth_from
                
            self.logger.info(f"Organized compartments for {len(hole_compartments)} holes")
            
            # Log some statistics
            for hole_id, compartments in hole_compartments.items():
                self.logger.info(f"Hole {hole_id}: {len(compartments)} compartments")
                
            return hole_compartments
            
        except Exception as e:
            self.logger.error(f"Error collecting compartment images: {str(e)}")
            if self.progress_queue:
                self.progress_queue.put((f"Error collecting images: {str(e)}", None))
            return {}
 
    def generate_compartment_with_data(self,
                                    image: np.ndarray,
                                    hole_id: str,
                                    depth_from: float,
                                    depth_to: float,
                                    fe_value: Optional[float] = None,
                                    sio2_value: Optional[float] = None,
                                    al2o3_value: Optional[float] = None,
                                    p_value: Optional[float] = None,
                                    cutoffs1_value: Optional[str] = None,
                                    cutoffs2_value: Optional[str] = None,
                                    is_missing: bool = False,
                                    data_column_width: int = 100) -> np.ndarray:
        # Get the image dimensions
        h, w = image.shape[:2]
        
        # Use the provided fixed width for data columns
        column_width = data_column_width
        
        # Total width of all data columns
        data_width = column_width * 6
        
        # Total width for the new image: original width + data columns
        total_width = w + data_width
        
        # Create new image with exact original dimensions plus data columns
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # BGRA
                composite = np.ones((h, total_width, 4), dtype=np.uint8) * 255
                # Set alpha channel to fully opaque
                composite[:, :, 3] = 255
            else:  # BGR
                composite = np.ones((h, total_width, 3), dtype=np.uint8) * 255
        else:  # Grayscale
            composite = np.ones((h, total_width), dtype=np.uint8) * 255
        
        # Copy the image to the left portion with exact dimensions
        if is_missing:
            # For missing intervals, create a black box with diagonal stripes
            composite[:, :w] = 0  # Black box
            
            # Add diagonal stripes
            for i in range(0, h + w, 20):
                cv2.line(
                    composite[:, :w],
                    (0, i),
                    (i, 0),
                    (50, 50, 150),  # Dark red stripes
                    2
                )
            
            # Add MISSING text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize("MISSING", font, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            cv2.putText(
                composite,
                "MISSING",
                (text_x, text_y),
                font,
                1.0,
                (255, 255, 255),  # White text
                2
            )
        else:
            # Normal image - copy the full original image 
            composite[:, :w] = image
        
        # Calculate x positions for data columns - each starts after the original image
        x_positions = [
            w,                      # Fe
            w + column_width,       # SiO2
            w + column_width * 2,   # Al2O3
            w + column_width * 3,   # P
            w + column_width * 4,   # Cutoffs1
            w + column_width * 5    # Cutoffs2
        ]

        # Add column headers
        header_bg = (220, 220, 220)  # Light gray
        header_height = 30
        
        # Draw header backgrounds and add header text
        headers = ["Fe %", "SiO2 %", "Al2O3 %", "P %", "Cutoffs1", "Cutoffs2"]
        font = cv2.FONT_HERSHEY_SIMPLEX  # Define font for later text
        
        for i, header in enumerate(headers):
            x = x_positions[i]
            width = column_width
            
            # Draw header background
            cv2.rectangle(composite, (x, 0), (x + width, header_height), header_bg, -1)
            
            # Add header text
            cv2.putText(composite, header, (x + 10, 20), font, 0.6, (0, 0, 0), 1)

        # Function to draw column
        def draw_column(x_pos, width, value, color_func, format_str="{:.2f}"):
            # If value is None or NaN, keep column white
            if value is None or pd.isna(value):
                return
                
            # Get color for the value
            color = color_func(value)
            
            # Draw the colored column
            cv2.rectangle(
                composite,
                (x_pos, header_height),
                (x_pos + width, h),
                color,
                -1  # Fill
            )
            
            # Add value text if not None
            if value is not None and not pd.isna(value):
                # Black or white text depending on background
                text_color = (255, 255, 255) if sum(color) < 380 else (0, 0, 0)
                
                # Format the value
                if isinstance(value, (int, float)):
                    text = format_str.format(value)
                else:
                    text = str(value)
                    
                # Get text size
                text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
                text_x = x_pos + (width - text_size[0]) // 2
                text_y = h // 2
                
                cv2.putText(
                    composite,
                    text,
                    (text_x, text_y),
                    font,
                    0.8,
                    text_color,
                    2
                )
        
        # Define color functions for numeric data
        def get_fe_color(value):
            # Fe color scale (modified from matplotlib version)
            if value is None or np.isnan(value) or value < 30:
                return (224, 224, 224)  # Light gray (BGRs)
            elif value < 40:
                return (48, 48, 48)      # Dark gray
            elif value < 45:
                return (255, 0, 0)       # Blue (BGR)
            elif value < 50:
                return (255, 255, 0)     # Cyan
            elif value < 54:
                return (0, 255, 0)       # Green  
            elif value < 56:
                return (0, 255, 255)     # Yellow
            elif value < 58:
                return (0, 128, 255)     # Orange
            elif value < 60:
                return (0, 0, 255)       # Red
            else:
                return (255, 0, 255)     # Magenta
        
        def get_sio2_color(value):
            # SiO2 color scale (modified from matplotlib version)
            if value is None or np.isnan(value) or value < 5:
                return (255, 255, 255)  # White (BGR)
            elif value < 15:
                return (0, 255, 0)      # Green
            elif value < 35:
                return (255, 0, 0)      # Blue
            else:
                return (0, 0, 255)      # Red

        def get_al2o3_color(value):
            # Al2O3 scale based on your Image 1
            if value is None or np.isnan(value):
                return (200, 200, 200)  # Light gray for no value
            elif value < 2:
                return (255, 0, 255)    # Magenta (0)
            elif value < 5:
                return (0, 0, 255)      # Red (2)
            elif value < 10:
                return (0, 165, 255)    # Orange (5) 
            elif value < 15:
                return (0, 255, 255)    # Yellow (10)
            else:
                return (0, 0, 255)      # Blue (15+)

        def get_p_color(value):
            # P scale based on your Image 2 (3 gradient scheme)
            if value is None or np.isnan(value):
                return (200, 200, 200)  # Light gray for no value
            elif value < 0.02:
                return (255, 0, 255)    # Magenta
            elif value < 0.08:
                return (0, 255, 0)      # Green
            else:
                return (0, 0, 255)      # Red
        
        def get_cutoffs_color(value):
            # Color mapping for Cutoffs text
            cutoffs_colors = {
                'Other': (31, 161, 217),    # #D9A11F
                'BID/Fs?': (0, 255, 255),   # #FFFF00
                'BIFf': (255, 77, 77),      # #4DFFFF
                'BIFf?': (0, 165, 255),     # #FFA500
                'BIFhm': (0, 0, 255),       # #FF0101
                'Mineralised': (200, 0, 254),# #FE00C4
                'High Confidence': (0, 255, 0),  # Green
                'Potential BID/Fs': (255, 255, 0)  # Cyan
            }
            
            # If value is None or empty, return default color
            if value is None or pd.isna(value) or str(value).strip() == "":
                return (200, 200, 200)  # Light gray for no value
                
            # Convert value to string just in case it's a number
            value_str = str(value).strip()
            
            # Check if value matches any of our known labels
            return cutoffs_colors.get(value_str, (200, 200, 200))  # Default to light gray

        # Draw each data column
        draw_column(x_positions[0], column_width, fe_value, get_fe_color)
        draw_column(x_positions[1], column_width, sio2_value, get_sio2_color)
        draw_column(x_positions[2], column_width, al2o3_value, get_al2o3_color)
        draw_column(x_positions[3], column_width, p_value, get_p_color, format_str="{:.3f}")
        
        # Draw cutoffs columns with text
        if cutoffs1_value and not pd.isna(cutoffs1_value) and str(cutoffs1_value).lower() != 'nan':
            x_pos = x_positions[4]
            width = column_width
            color = get_cutoffs_color(cutoffs1_value)
            
            # Draw colored background
            cv2.rectangle(composite, (x_pos, header_height), (x_pos + width, h), color, -1)
            
            # Add text
            text_color = (255, 255, 255) if sum(color) < 380 else (0, 0, 0)
            
            # Word wrap for longer labels
            text = str(cutoffs1_value)
            if len(text) > 10:
                # Split into multiple lines
                words = text.split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if len(test_line) <= 10:
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                    
                # Draw each line
                for i, line in enumerate(lines):
                    y_pos = h // 2 - (len(lines) - 1) * 15 + i * 30
                    cv2.putText(
                        composite,
                        line,
                        (x_pos + width//2 - 25, y_pos),
                        font,
                        0.7,
                        text_color,
                        2
                    )
            else:
                # Single line text
                cv2.putText(
                    composite,
                    text,
                    (x_pos + width//2 - 25, h//2),
                    font,
                    0.8,
                    text_color,
                    2
                )
        
        # Same for cutoffs2
        if cutoffs2_value and not pd.isna(cutoffs2_value) and str(cutoffs2_value).lower() != 'nan':
            x_pos = x_positions[5]
            width = column_width
            color = get_cutoffs_color(cutoffs2_value)
            
            # Draw colored background
            cv2.rectangle(composite, (x_pos, header_height), (x_pos + width, h), color, -1)
            
            # Add text
            text_color = (255, 255, 255) if sum(color) < 380 else (0, 0, 0)
            
            # Word wrap for longer labels
            text = str(cutoffs2_value)
            if len(text) > 10:
                # Split into multiple lines
                words = text.split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if len(test_line) <= 10:
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                    
                # Draw each line
                for i, line in enumerate(lines):
                    y_pos = h // 2 - (len(lines) - 1) * 15 + i * 30
                    cv2.putText(
                        composite,
                        line,
                        (x_pos + width//2 - 25, y_pos),
                        font,
                        0.7,
                        text_color,
                        2
                    )
            else:
                # Single line text
                cv2.putText(
                    composite,
                    text,
                    (x_pos + width//2 - 25, h//2),
                    font,
                    0.8,
                    text_color,
                    2
                )
        
        # Add depth information at the top of the compartment in HoleID - From-To format
        depth_text = f"{hole_id} - {int(depth_from)}-{int(depth_to)}m"
        text_size = cv2.getTextSize(depth_text, font, 0.7, 2)[0]  # Bold text (thickness=2)

        # Add a dark background for better visibility
        bg_margin = 5
        cv2.rectangle(
            composite,
            (10 - bg_margin, 30 - text_size[1] - bg_margin),
            (10 + text_size[0] + bg_margin, 30 + bg_margin),
            (0, 0, 0),  # Black background
            -1  # Fill
        )

        # Add white bold text
        cv2.putText(
            composite,
            depth_text,
            (10, 30),
            font,
            0.7,
            (255, 255, 255),  # White text
            2  # Bold
        )
        
        # Add separator lines between data columns
        for x in x_positions[1:]:
            cv2.line(composite, (x, 0), (x, h), (0, 0, 0), 1)
        
        return composite

        
    def create_value_legend(self, width: int = 800, height: int = 300) -> np.ndarray:
        """
        Create a comprehensive legend image for all data columns.
        
        Args:
            width: Width of the legend image
            height: Height of the legend image
            
        Returns:
            Legend image as numpy array
        """
        # Create white image for legend
        legend = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Define color scales and boundaries for each column
        legend_configs = [
            {
                'title': 'Fe %',
                'colors': [
                    (224, 224, 224),  # Light gray (BGR)
                    (48, 48, 48),     # Dark gray
                    (255, 0, 0),      # Blue
                    (255, 255, 0),    # Cyan
                    (0, 255, 0),      # Green
                    (0, 255, 255),    # Yellow
                    (0, 128, 255),    # Orange
                    (0, 0, 255),      # Red
                    (255, 0, 255)     # Magenta
                ],
                'bounds': ["<30", "30-40", "40-45", "45-50", "50-54", "54-56", "56-58", "58-60", ">60"]
            },
            {
                'title': 'SiO2 %',
                'colors': [
                    (255, 255, 255),  # White
                    (0, 255, 0),      # Green
                    (255, 0, 0),      # Blue
                    (0, 0, 255)       # Red
                ],
                'bounds': ["<5", "5-15", "15-35", ">35"]
            },
            {
                'title': 'Al2O3 %',
                'colors': [
                    (255, 255, 255),  # White
                    (200, 200, 255),  # Light blue
                    (120, 120, 255),  # Medium blue
                    (50, 50, 255),    # Dark blue
                    (0, 0, 255)       # Darkest blue
                ],
                'bounds': ["<2", "2-4", "4-6", "6-8", ">8"]
            },
            {
                'title': 'P %',
                'colors': [
                    (255, 255, 255),  # White
                    (200, 255, 200),  # Light green
                    (100, 255, 100),  # Medium green
                    (0, 255, 0),      # Green
                    (0, 200, 0)       # Dark green
                ],
                'bounds': ["<0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", ">0.7"]
            },
            {
                'title': 'Cutoffs',
                'colors': [
                    (31, 161, 217),   # Other - #D9A11F
                    (0, 255, 255),    # BID/Fs? - #FFFF00
                    (255, 77, 77),    # BIFf - #4DFFFF
                    (0, 165, 255),    # BIFf? - #FFA500
                    (0, 0, 255),      # BIFhm - #FF0101
                    (200, 0, 254)     # Mineralised - #FE00C4
                ],
                'bounds': [
                    "Other", 
                    "BID/Fs?", 
                    "BIFf", 
                    "BIFf?", 
                    "BIFhm", 
                    "Mineralised"
                ]
            }
        ]
        
        # Draw legend
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        cv2.putText(legend, "Geochemical Data Legend", (20, 30), font, 1.0, (0, 0, 0), 2)
        
        # Positioning
        box_width = 50
        box_height = 25
        x_start = 20
        y_pos = 50
        
        # Render each column's legend
        for config in legend_configs:
            # Column title
            cv2.putText(legend, config['title'], (x_start, y_pos), font, 0.7, (0, 0, 0), 2)
            y_pos += 30
            
            # Render color boxes and labels
            for i in range(len(config['colors'])):
                # Draw color box
                cv2.rectangle(
                    legend,
                    (x_start, y_pos),
                    (x_start + box_width, y_pos + box_height),
                    config['colors'][i],
                    -1
                )
                
                # Add border
                cv2.rectangle(
                    legend,
                    (x_start, y_pos),
                    (x_start + box_width, y_pos + box_height),
                    (0, 0, 0),
                    1
                )
                
                # Add label
                cv2.putText(
                    legend,
                    config['bounds'][i],
                    (x_start + box_width + 10, y_pos + 20),
                    font,
                    0.5,
                    (0, 0, 0),
                    1
                )
                
                # Move to next position
                y_pos += box_height + 5
            
            # Add spacing between columns
            y_pos += 20
            
            # Move to next column if running out of vertical space
            if y_pos > height - 100:
                x_start += box_width + 150
                y_pos = 50
        
        return legend

        
    def generate_drillhole_trace_cv2(self, 
                                hole_id: str, 
                                compartments: List[Tuple[str, float, float, str]],
                                csv_data: Optional[pd.DataFrame] = None,
                                output_dir: Optional[str] = None) -> Optional[str]:
        """
        Generate a drillhole trace using OpenCV instead of matplotlib.
        This approach preserves image quality and creates a more accurate representation.
        
        Args:
            hole_id: Hole ID
            compartments: List of (hole_id, depth_from, depth_to, file_path) tuples
            csv_data: Optional DataFrame with CSV data
            output_dir: Optional directory (uses FileManager's directory if None)
            
        Returns:
            Path to the generated image file, or None if failed
        """
        if not compartments:
            self.logger.warning(f"No compartments provided for hole {hole_id}")
            return None
            
        try:
            # Use output_dir parameter if provided, otherwise use FileManager
            if output_dir is None and hasattr(self, 'file_manager') and self.file_manager is not None:
                output_dir = self.file_manager.dir_structure["drill_traces"]
                
            # Make sure we have a valid output directory
            if not output_dir:
                self.logger.error("No output directory available for saving trace images")
                return None
                
            # Sort compartments by depth
            sorted_compartments = sorted(compartments, key=lambda x: x[1])
            
            # Extract depth range
            min_depth = sorted_compartments[0][1]
            max_depth = sorted_compartments[-1][2]
            
            # Prepare chemical data if available
            hole_csv_data = {}
            if csv_data is not None and not csv_data.empty:
                # Filter for this hole
                hole_data = csv_data[csv_data['holeid'].str.upper() == hole_id.upper()]
                
                # Convert to dictionary format for easier lookup
                for _, row in hole_data.iterrows():
                    depth_from = row.get('from', 0)
                    depth_to = row.get('to', 0)
                    if depth_from is not None and depth_to is not None:
                        # Store interval with all available data
                        interval_data = {
                            'from': depth_from,
                            'to': depth_to
                        }
                        
                        # Add all other columns to the interval data
                        for col in hole_data.columns:
                            if col not in ['holeid', 'from', 'to']:
                                interval_data[col] = row.get(col)
                        
                        # Use midpoint as key for interval
                        interval_key = (depth_from + depth_to) / 2
                        hole_csv_data[interval_key] = interval_data
            
            # Process each compartment image - store results for concatenation
            processed_images = []
            
            # Create a list to track missing intervals
            missing_intervals = []
            processed_depths = []
            
            # Track all depth ranges we have processed
            for _, depth_from, depth_to, _ in sorted_compartments:
                processed_depths.append((depth_from, depth_to))
            
            # Load first image to determine standard width
            standard_width = None
            for _, _, _, file_path in sorted_compartments:
                try:
                    test_img = cv2.imread(file_path)
                    if test_img is not None:
                        # Rotate to get proper dimensions
                        test_rotated = cv2.rotate(test_img, cv2.ROTATE_90_CLOCKWISE)
                        h, w = test_rotated.shape[:2]
                        
                        # Calculate data column widths - fixed size
                        data_column_width = 100  # Fixed width for each data column
                        total_data_width = data_column_width * 6  # 6 data columns
                        
                        # Get maximum width we'll need to accommodate both the image and data columns
                        standard_width = w + total_data_width
                        
                        # No need to keep checking - we've got our reference width
                        self.logger.info(f"Using standard width of {standard_width} pixels based on first valid image")
                        break
                except Exception:
                    continue

            # Set default if no images loaded
            if standard_width is None:
                standard_width = 800
                self.logger.warning("No valid images found to determine width, using default width of 800 pixels")
            
            # Find missing intervals by checking gaps between processed depths
            processed_depths.sort()  # Ensure they're in order
            prev_depth = min_depth
            missing_intervals = []  # Initialize an empty list for missing intervals
            
            for depth_from, depth_to in processed_depths:
                if depth_from > prev_depth:
                    # Found a gap
                    missing_intervals.append((prev_depth, depth_from))
                prev_depth = max(prev_depth, depth_to)
            
            # Also check if there's a gap at the end
            if prev_depth < max_depth:
                missing_intervals.append((prev_depth, max_depth))
                
            # Process each compartment
            for i, (hole_id, depth_from, depth_to, file_path) in enumerate(sorted_compartments):
                try:
                    # Load and process image
                    img = cv2.imread(file_path)
                    if img is None:
                        self.logger.warning(f"Couldn't read image: {file_path}")
                        continue
                    
                    if hasattr(self, 'compartment_orientation') and self.compartment_orientation == 'vertical':
                        # Keep vertical orientation
                        rotated_img = img
                    else:
                        # Default to horizontal (rotate 90 degrees clockwise)
                        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        
                    # Get the original image width before any processing
                    original_width = rotated_img.shape[1]
                    
                    # Look up chemical data for this depth interval
                    fe_value = None
                    sio2_value = None
                    al2o3_value = None
                    p_value = None
                    cutoffs1_value = None
                    cutoffs2_value = None

                    
                    # Find best matching interval in CSV data
                    if hole_csv_data:
                        # Calculate midpoint of current interval
                        midpoint = (depth_from + depth_to) / 2
                        
                        # Find closest interval in CSV data
                        closest_interval = None
                        min_distance = float('inf')
                        
                        for interval_mid, interval_data in hole_csv_data.items():
                            interval_from = interval_data['from']
                            interval_to = interval_data['to']
                            
                            # Check if midpoint is within interval
                            if interval_from <= midpoint <= interval_to:
                                closest_interval = interval_data
                                break
                            
                            # Otherwise check if it's the closest interval
                            distance = min(abs(midpoint - interval_from), abs(midpoint - interval_to))
                            if distance < min_distance:
                                min_distance = distance
                                closest_interval = interval_data
                        
                        # Extract chemistry and cutoff1 and cutoffs 2 values if available
                        if closest_interval:
                            fe_value = closest_interval.get('fe_pct')
                            sio2_value = closest_interval.get('sio2_pct')
                            al2o3_value = closest_interval.get('al2o3_pct')
                            p_value = closest_interval.get('p_pct')
                            
                            # Properly handle text columns for cutoffs - don't display 'nan'
                            cutoffs1_value = closest_interval.get('cutoffs1')
                            if pd.isna(cutoffs1_value) or cutoffs1_value == 'nan':
                                cutoffs1_value = None
                                
                            cutoffs2_value = closest_interval.get('cutoffs2')
                            if pd.isna(cutoffs2_value) or cutoffs2_value == 'nan':
                                cutoffs2_value = None
                    
                    # Create composite image with data
                    data_column_width = 100  # Fixed width for data columns
                    composite = self.generate_compartment_with_data(
                        rotated_img, 
                        hole_id, 
                        depth_from, 
                        depth_to, 
                        fe_value=fe_value, 
                        sio2_value=sio2_value, 
                        al2o3_value=al2o3_value, 
                        p_value=p_value,
                        cutoffs1_value=cutoffs1_value, 
                        cutoffs2_value=cutoffs2_value,
                        data_column_width=data_column_width
                    )
                    
                    # Add to processed images
                    processed_images.append(composite)
                    
                except Exception as e:
                    self.logger.error(f"Error processing compartment {file_path}: {str(e)}")
                    self.logger.error(traceback.format_exc())

            # Now process missing intervals - only if there are any
            if missing_intervals:
                self.logger.info(f"Processing {len(missing_intervals)} missing intervals")
                for gap_from, gap_to in missing_intervals:
                    # Log the gap being processed
                    self.logger.info(f"Processing gap from {gap_from}m to {gap_to}m")

                    # For each 1-meter interval in the gap
                    for meter_start in range(int(gap_from), int(gap_to)):
                        # The compartment at meter_start is identified by its end depth: meter_start + 1
                        meter_end = meter_start + 1
                        compartment_depth = meter_end
                        self.logger.info(f"Creating missing compartment for {meter_start}-{meter_end}m (depth {compartment_depth})")
                        
                        # Determine a reasonable height for the missing interval
                        if processed_images:
                            avg_height = sum(img.shape[0] for img in processed_images) // len(processed_images)
                        else:
                            avg_height = 300  # Default height if no other images available
                            
                        # Create black image with appropriate dimensions and type
                        if processed_images and len(processed_images[0].shape) == 3:
                            if processed_images[0].shape[2] == 4:  # With alpha
                                blank_image = np.zeros((avg_height, standard_width, 4), dtype=np.uint8)
                            else:  # Regular BGR
                                blank_image = np.zeros((avg_height, standard_width, 3), dtype=np.uint8)
                        else:
                            blank_image = np.zeros((avg_height, standard_width, 3), dtype=np.uint8)
                        
                        # Initialize all values properly
                        fe_value = None
                        sio2_value = None
                        al2o3_value = None
                        p_value = None
                        cutoffs1_value = None
                        cutoffs2_value = None
                            
                        # Here we need to look up data in the CSV for this specific interval
                        if hole_csv_data:
                            # Calculate midpoint of current interval
                            midpoint = (meter_start + meter_end) / 2
                            
                            # Only use exact interval matches
                            matching_interval = None
                            
                            for interval_mid, interval_data in hole_csv_data.items():
                                interval_from = interval_data['from']
                                interval_to = interval_data['to']
                                
                                # Check if midpoint is EXACTLY within interval
                                if interval_from <= midpoint <= interval_to:
                                    matching_interval = interval_data
                                    break
                            
                            # Extract values if available (only if exact match found)
                            if matching_interval:
                                fe_value = matching_interval.get('fe_pct')
                                sio2_value = matching_interval.get('sio2_pct')
                                al2o3_value = matching_interval.get('al2o3_pct')
                                p_value = matching_interval.get('p_pct')
                                
                                # Properly handle text columns for cutoffs - don't display 'nan'
                                cutoffs1_value = matching_interval.get('cutoffs1')
                                if pd.isna(cutoffs1_value) or str(cutoffs1_value).lower() == 'nan':
                                    cutoffs1_value = None
                                    
                                cutoffs2_value = matching_interval.get('cutoffs2')
                                if pd.isna(cutoffs2_value) or str(cutoffs2_value).lower() == 'nan':
                                    cutoffs2_value = None
                        
                        # Create missing box with the data
                        black_box = self.generate_compartment_with_data(
                            blank_image, hole_id, meter_start, meter_end,
                            fe_value=fe_value,
                            sio2_value=sio2_value,
                            al2o3_value=al2o3_value,
                            p_value=p_value,
                            cutoffs1_value=cutoffs1_value,
                            cutoffs2_value=cutoffs2_value,
                            is_missing=True,
                            data_column_width=data_column_width
                        )

                        # Find where to insert based on depth
                        insert_index = 0
                        for i, (comp_hole_id, comp_from, comp_to, _) in enumerate(sorted_compartments):
                            if meter_start < comp_from:
                                insert_index = i
                                break
                            insert_index = i + 1

                        # Insert the black box in the processed_images list
                        if insert_index < len(processed_images):
                            processed_images.insert(insert_index, black_box)
                        else:
                            processed_images.append(black_box)

                        # Also insert into sorted_compartments so the insert_index keeps working
                        sorted_compartments.insert(insert_index, (hole_id, meter_start, meter_end, None))
            else:
                self.logger.info("No missing intervals detected - all depths are continuous")

            # Check if we should save individual compartments
            if self.config.get('save_individual_compartments', True):
                # Use FileManager if available
                if hasattr(self, 'file_manager') and self.file_manager is not None:
                    # Save each individual compartment with data columns through FileManager
                    for i, (h_id, d_from, d_to, _) in enumerate(sorted_compartments):
                        if i < len(processed_images):
                            # Calculate compartment depth
                            compartment_depth = int(d_to)
                            
                            # Use the FileManager method to save the compartment with data
                            self.file_manager.save_compartment_with_data(
                                processed_images[i],
                                hole_id,
                                compartment_depth,
                                output_format="png"
                            )
                else:
                    # Fallback for cases without FileManager (should be rare)
                    self.logger.warning("No FileManager available for saving individual compartments")

            # Create a legend image
            legend = self.create_value_legend(width=standard_width)
            
            # Create hole info header
            header_height = 80
            header = np.ones((header_height, standard_width, 3), dtype=np.uint8) * 255
            
            # Add hole ID and depth range
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                header,
                f"Drillhole: {hole_id}",
                (20, 40),
                font,
                1.2,
                (0, 0, 0),
                2
            )
            
            cv2.putText(
                header,
                f"Depth range: {int(min_depth)}-{int(max_depth)}m",
                (20, 70),
                font,
                0.8,
                (0, 0, 0),
                1
            )
            
            # Verify all images have the same width and type for vconcat
            all_images = [header] + processed_images + [legend]

            # Debug widths before concat
            for i, img in enumerate(all_images):
                img_type = "header" if i == 0 else "legend" if i == len(all_images)-1 else f"compartment {i-1}"
                self.logger.info(f"Image {i} ({img_type}) shape: {img.shape}, width diff from standard: {img.shape[1] - standard_width}")
                
                # Force resize to standard width if needed
                if img.shape[1] != standard_width:
                    self.logger.warning(f"Resizing {img_type} from width {img.shape[1]} to {standard_width}")
                    all_images[i] = cv2.resize(img, (standard_width, img.shape[0]), interpolation=cv2.INTER_AREA)
            
            # Combine all images vertically - verify all have same width
            if processed_images:
                try:
                    # Try concatenating in chunks to avoid potential memory issues
                    chunk_size = 10  # Adjust as needed
                    final_chunks = []
                    
                    for i in range(0, len(all_images), chunk_size):
                        chunk = all_images[i:i+chunk_size]
                        try:
                            chunk_image = cv2.vconcat(chunk)
                            final_chunks.append(chunk_image)
                        except Exception as e:
                            self.logger.error(f"Error concatenating chunk {i//chunk_size}: {str(e)}")
                            # Try individual images if chunk fails
                            for j, img in enumerate(chunk):
                                try:
                                    final_chunks.append(img)
                                except Exception:
                                    self.logger.error(f"Error with image {i+j}")
                    
                    # Final concatenation of chunks
                    final_image = cv2.vconcat(final_chunks)
                    
                    # Save the result
                    output_filename = f"{hole_id}_drillhole_trace.png"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save using cv2.imwrite without creating directories
                    cv2.imwrite(output_path, final_image)
                    
                    self.logger.info(f"Successfully created drillhole trace for {hole_id} at {output_path}")
                    return output_path
                    
                except Exception as e:
                    self.logger.error(f"Error in final concatenation: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    
                    # Use FileManager for fallback if available
                    if hasattr(self, 'file_manager') and self.file_manager is not None:
                        # Get a path from FileManager's drill_traces directory
                        fallback_dir = os.path.join(self.file_manager.dir_structure["drill_traces"], f"{hole_id}_sections")
                    else:
                        # Fallback to output_dir if provided
                        fallback_dir = os.path.join(output_dir or ".", f"{hole_id}_sections")
                    
                    self.logger.info(f"Falling back to saving individual sections in {fallback_dir}")
                    
                    # Save individual sections
                    for i, img in enumerate(all_images):
                        section_path = os.path.join(fallback_dir, f"{hole_id}_section_{i:03d}.png")
                        
                        # Save without creating directories
                        cv2.imwrite(section_path, img)
                    
                    return fallback_dir
            else:
                self.logger.error(f"No valid images processed for hole {hole_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating drillhole trace for {hole_id}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def process_all_drillholes(self, 
                            compartment_dir: str,
                            csv_path: Optional[str] = None,
                            selected_columns: Optional[List[str]] = None) -> List[str]:
        """
        Process all drillholes in a directory to create trace images.
        
        Args:
            compartment_dir: Directory containing compartment images
            csv_path: Optional path to CSV file with additional data
            selected_columns: Optional list of columns to include from CSV
            
        Returns:
            List of paths to generated trace images
        """
        # Load CSV data
        csv_data = None
        if csv_path and os.path.exists(csv_path):
            try:
                # Load the CSV
                csv_data = pd.read_csv(csv_path)
                
                # Convert column names to lowercase for case-insensitive matching
                csv_data.columns = [col.lower() for col in csv_data.columns]
                
                # Ensure numeric columns are properly typed, but keep text columns as strings
                for col in csv_data.columns:
                    if col not in ['holeid', 'cutoffs1', 'cutoffs2']:  # Don't convert text columns
                        try:
                            csv_data[col] = pd.to_numeric(csv_data[col], errors='coerce')
                        except:
                            self.logger.warning(f"Could not convert column {col} to numeric")
                
                self.logger.info(f"Loaded CSV with {len(csv_data)} rows and {len(csv_data.columns)} columns")
                
            except Exception as e:
                self.logger.error(f"Error loading CSV data: {str(e)}")
        
        # Collect compartment images
        hole_compartments = self.collect_compartment_images(compartment_dir)
        
        if not hole_compartments:
            self.logger.warning("No valid compartment images found")
            if self.progress_queue:
                self.progress_queue.put(("No valid compartment images found", None))
            return []
        
        # Get the output directory from FileManager if available
        output_dir = None
        if hasattr(self, 'file_manager') and self.file_manager is not None:
            output_dir = self.file_manager.dir_structure["drill_traces"]
            self.logger.info(f"Using FileManager directory for drill traces: {output_dir}")
        else:
            # Fall back to a centralized location if no FileManager available
            # This should almost never happen in normal operation
            output_dir = os.path.join("C:/Excel Automation Local Outputs/Chip Tray Photo Processor", 
                                    "Processed", "Drill Traces")
            self.logger.warning(f"FileManager not available, using fallback directory: {output_dir}")
        
        # Get compartment interval for logging
        compartment_interval = self.config.get('compartment_interval', 1)
        
        # Process each hole
        generated_traces = []
        
        for i, (hole_id, compartments) in enumerate(hole_compartments.items()):
            # Update progress
            if self.progress_queue:
                progress = ((i + 1) / len(hole_compartments)) * 100
                self.progress_queue.put((f"Processing hole {i+1}/{len(hole_compartments)}: {hole_id} (interval: {compartment_interval}m)", progress))
            
            # Generate trace using the OpenCV-based method
            trace_path = self.generate_drillhole_trace_cv2(
                hole_id, compartments, csv_data, output_dir
            )
            
            if trace_path:
                generated_traces.append(trace_path)
        
        # Final status update
        status_msg = f"Completed drillhole trace generation: {len(generated_traces)}/{len(hole_compartments)} successful"
        self.logger.info(status_msg)
        if self.progress_queue:
            self.progress_queue.put((status_msg, 100))
        
        return generated_traces
