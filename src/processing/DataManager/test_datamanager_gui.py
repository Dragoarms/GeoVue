#!/usr/bin/env python3
"""
test_datamanager_gui.py - Simple GUI to test DataManager components.

This provides a visual test harness for:
- DataCoordinator initialization and timing
- ScatterSelectionWidget functionality
- DynamicFilterRow functionality

Place this file in: src/processing/DataManager/
Run with: python test_datamanager_gui.py

Author: George Symonds / Claude
"""

import os
import sys
import tempfile
import shutil
import time
import logging
import random
from pathlib import Path
from unittest.mock import Mock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# PATH SETUP
# =============================================================================

def setup_paths():
    """Set up import paths."""
    script_dir = Path(__file__).parent.resolve()
    src_dir = script_dir.parent.parent  # Go up to src/
    
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    logger.info(f"Script directory: {script_dir}")

setup_paths()

# =============================================================================
# IMPORTS
# =============================================================================

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

# Import DataManager components
try:
    from data_coordinator import DataCoordinator, CompartmentData
    from keys import ImageKey
    from schema import DataType
    logger.info("DataManager imports successful")
except ImportError as e:
    logger.error(f"Failed to import DataManager: {e}")
    sys.exit(1)

# =============================================================================
# TEST DATA GENERATION
# =============================================================================

def create_test_csv(path: str, rows: int = 200, holes: int = 5):
    """Create a test CSV file with geological data."""
    with open(path, 'w') as f:
        f.write("HOLEID,SAMPFROM,SAMPTO,Fe_pct_BEST,SiO2_pct_BEST,Al2O3_pct_BEST,STRATSUM,LITHCODE\n")
        
        hole_ids = [f"BA{str(i).zfill(4)}" for i in range(1, holes + 1)]
        strat_values = ["BIF", "SHALE", "GRANITE", "LATERITE", "BIFf", "BIFhm"]
        lith_codes = ["FeBIF", "Qtz", "Hema", "Goeth", "Clay"]
        
        depth = 0
        for i in range(rows):
            hole_id = hole_ids[i % holes]
            if i % holes == 0:
                depth = 0
            depth_from = depth
            depth_to = depth + 1
            depth = depth_to
            
            fe = round(random.uniform(20, 70), 2)
            sio2 = round(random.uniform(5, 40), 2)
            al2o3 = round(random.uniform(1, 15), 2)
            strat = random.choice(strat_values)
            lith = random.choice(lith_codes)
            
            f.write(f"{hole_id},{depth_from},{depth_to},{fe},{sio2},{al2o3},{strat},{lith}\n")
    
    logger.info(f"Created test CSV with {rows} rows at {path}")


def create_test_compartment_images(folder: str, holes: int = 5, depths_per_hole: int = 20):
    """Create dummy compartment image files."""
    os.makedirs(folder, exist_ok=True)
    
    count = 0
    for h in range(1, holes + 1):
        hole_id = f"BA{str(h).zfill(4)}"
        hole_folder = os.path.join(folder, "PROJECT", hole_id)
        os.makedirs(hole_folder, exist_ok=True)
        
        for d in range(1, depths_per_hole + 1):
            for moisture in ["Wet", "Dry"]:
                filename = f"{hole_id}_CC_{str(d).zfill(3)}_{moisture}.png"
                filepath = os.path.join(hole_folder, filename)
                with open(filepath, 'w') as f:
                    f.write("")
                count += 1
    
    logger.info(f"Created {count} test compartment images")
    return count


def create_mock_json_manager():
    """Create a mock JSONRegisterManager."""
    manager = Mock()
    
    mock_df = pd.DataFrame({
        'hole_id': ['BA0001'] * 5 + ['BA0002'] * 5,
        'depth_from': list(range(0, 5)) + list(range(0, 5)),
        'depth_to': list(range(1, 6)) + list(range(1, 6)),
        'photo_status': ['reviewed'] * 10
    })
    manager.get_all_compartments_all_users.return_value = mock_df
    
    def mock_get_user_review(hole_id, depth_from, depth_to):
        classifications = ["BIFf", "BIFhm", "Compact", "Porous", None]
        return {"classification": random.choice(classifications), "tags": [], "comments": ""}
    
    manager.get_user_review.side_effect = mock_get_user_review
    manager.get_all_reviews_for_compartment.return_value = []
    
    def mock_get_hex_colors(hole_id, depth_from, depth_to):
        r = random.randint(100, 200)
        g = random.randint(60, 120)
        b = random.randint(40, 80)
        return {
            'wet_hex': f"#{r:02X}{g:02X}{b:02X}",
            'dry_hex': f"#{r+20:02X}{g+20:02X}{b+20:02X}",
            'combined_hex': f"#{r+10:02X}{g+10:02X}{b+10:02X}"
        }
    
    manager.get_hex_colors_for_interval.side_effect = mock_get_hex_colors
    
    return manager


# =============================================================================
# MOCK GUI MANAGER (for widgets that require it)
# =============================================================================

class MockGUIManager:
    """Minimal mock GUIManager for widget testing."""
    
    def __init__(self, root):
        self.root = root
        
        # Theme colors (dark theme)
        self.theme_colors = {
            "background": "#1e1e1e",
            "secondary_bg": "#252526",
            "field_bg": "#3c3c3c",
            "text": "#e0e0e0",
            "field_border": "#555555",
            "accent_blue": "#3a7ca5",
            "accent_green": "#4caf50",
            "accent_red": "#f44336",
            "accent_orange": "#ff9800",
        }
        
        # Fonts
        self.fonts = {
            "normal": ("Segoe UI", 10),
            "small": ("Segoe UI", 9),
            "heading": ("Segoe UI", 12, "bold"),
        }
    
    def apply_theme(self, widget):
        """Apply theme to a widget."""
        try:
            widget.configure(bg=self.theme_colors["background"])
        except tk.TclError:
            pass
    
    def create_modern_button(self, parent, text, color, command=None, **kwargs):
        """Create a simple button."""
        btn = tk.Button(
            parent,
            text=text,
            bg=color,
            fg="white",
            command=command,
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        return btn
    
    def create_searchable_optionmenu(self, parent, items, variable, width=15, 
                                      placeholder="", on_change=None):
        """Create a simple optionmenu (non-searchable fallback)."""
        if items:
            menu = tk.OptionMenu(parent, variable, *items)
        else:
            menu = tk.OptionMenu(parent, variable, "")
        menu.config(width=width, bg=self.theme_colors["field_bg"], 
                   fg=self.theme_colors["text"])
        return menu
    
    def style_dropdown(self, dropdown, width=15):
        """Style a dropdown."""
        dropdown.config(
            width=width,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            activebackground=self.theme_colors["accent_blue"],
            activeforeground="white",
            highlightthickness=0
        )


# =============================================================================
# MAIN TEST GUI
# =============================================================================

class DataManagerTestGUI:
    """Main test GUI for DataManager components."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("DataManager Test GUI")
        self.root.geometry("1200x800")
        
        # Theme
        self.bg_color = "#1e1e1e"
        self.fg_color = "#e0e0e0"
        self.accent_color = "#3a7ca5"
        
        self.root.configure(bg=self.bg_color)
        
        # Test data paths (will be set during initialization)
        self.temp_dir = None
        self.coordinator = None
        self.dataframe = None
        
        # Create GUI Manager mock
        self.gui_manager = MockGUIManager(root)
        
        # Build UI
        self._build_ui()
        
        # Initialize test data
        self.root.after(100, self._initialize_test_data)
    
    def _build_ui(self):
        """Build the main UI."""
        # Top frame - Status and controls
        top_frame = tk.Frame(self.root, bg=self.bg_color)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            top_frame,
            text="DataManager Component Test GUI",
            font=("Segoe UI", 16, "bold"),
            bg=self.bg_color,
            fg=self.fg_color
        )
        title_label.pack(side=tk.LEFT)
        
        # Reinitialize button
        reinit_btn = tk.Button(
            top_frame,
            text="Reinitialize",
            bg=self.accent_color,
            fg="white",
            command=self._reinitialize
        )
        reinit_btn.pack(side=tk.RIGHT, padx=5)
        
        # Status frame
        status_frame = tk.LabelFrame(
            self.root,
            text="Initialization Status",
            bg=self.bg_color,
            fg=self.fg_color,
            font=("Segoe UI", 10, "bold")
        )
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_text = tk.Text(
            status_frame,
            height=6,
            bg="#252526",
            fg=self.fg_color,
            font=("Consolas", 9),
            wrap=tk.WORD
        )
        self.status_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Main content - notebook with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Style the notebook
        style = ttk.Style()
        style.configure("TNotebook", background=self.bg_color)
        style.configure("TNotebook.Tab", background="#3c3c3c", foreground="black", padding=[10, 5])
        
        # Tab 1: DataFrame Preview
        self.df_frame = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.df_frame, text="DataFrame Preview")
        self._build_dataframe_tab()
        
        # Tab 2: Scatter Plot (placeholder)
        self.scatter_frame = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.scatter_frame, text="ScatterSelectionWidget")
        self._build_scatter_tab()
        
        # Tab 3: Filter Row (placeholder)
        self.filter_frame = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.filter_frame, text="DynamicFilterRow")
        self._build_filter_tab()
    
    def _build_dataframe_tab(self):
        """Build the DataFrame preview tab."""
        # Info label
        info_label = tk.Label(
            self.df_frame,
            text="DataFrame built from DataCoordinator.build_dataframe():",
            bg=self.bg_color,
            fg=self.fg_color,
            font=("Segoe UI", 10)
        )
        info_label.pack(anchor=tk.W, padx=10, pady=5)
        
        # Treeview for DataFrame
        tree_frame = tk.Frame(self.df_frame, bg=self.bg_color)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.tree_scroll_y = tk.Scrollbar(tree_frame)
        self.tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree_scroll_x = tk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        self.tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tree = ttk.Treeview(
            tree_frame,
            yscrollcommand=self.tree_scroll_y.set,
            xscrollcommand=self.tree_scroll_x.set
        )
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        self.tree_scroll_y.config(command=self.tree.yview)
        self.tree_scroll_x.config(command=self.tree.xview)
    
    def _build_scatter_tab(self):
        """Build the scatter plot test tab."""
        # Info
        info_frame = tk.Frame(self.scatter_frame, bg=self.bg_color)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        info_label = tk.Label(
            info_frame,
            text="ScatterSelectionWidget Test (requires matplotlib):",
            bg=self.bg_color,
            fg=self.fg_color,
            font=("Segoe UI", 10)
        )
        info_label.pack(anchor=tk.W)
        
        # Column selection
        ctrl_frame = tk.Frame(self.scatter_frame, bg=self.bg_color)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(ctrl_frame, text="X Column:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT)
        self.x_col_var = tk.StringVar(value="fe_pct_best")
        self.x_col_combo = ttk.Combobox(ctrl_frame, textvariable=self.x_col_var, width=20)
        self.x_col_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Label(ctrl_frame, text="Y Column:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=(20, 0))
        self.y_col_var = tk.StringVar(value="sio2_pct_best")
        self.y_col_combo = ttk.Combobox(ctrl_frame, textvariable=self.y_col_var, width=20)
        self.y_col_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Label(ctrl_frame, text="Color By:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=(20, 0))
        self.color_col_var = tk.StringVar(value="stratsum")
        self.color_col_combo = ttk.Combobox(ctrl_frame, textvariable=self.color_col_var, width=20)
        self.color_col_combo.pack(side=tk.LEFT, padx=5)
        
        create_scatter_btn = tk.Button(
            ctrl_frame,
            text="Create Scatter Plot",
            bg=self.accent_color,
            fg="white",
            command=self._create_scatter_widget
        )
        create_scatter_btn.pack(side=tk.LEFT, padx=20)
        
        # Scatter widget container
        self.scatter_container = tk.Frame(self.scatter_frame, bg="#252526")
        self.scatter_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Selection info
        self.selection_info_var = tk.StringVar(value="No selection")
        self.selection_label = tk.Label(
            self.scatter_frame,
            textvariable=self.selection_info_var,
            bg=self.bg_color,
            fg=self.fg_color,
            font=("Consolas", 9)
        )
        self.selection_label.pack(anchor=tk.W, padx=10, pady=5)
    
    def _build_filter_tab(self):
        """Build the filter row test tab."""
        # Info
        info_label = tk.Label(
            self.filter_frame,
            text="DynamicFilterRow Test:",
            bg=self.bg_color,
            fg=self.fg_color,
            font=("Segoe UI", 10)
        )
        info_label.pack(anchor=tk.W, padx=10, pady=5)
        
        # Add filter button
        btn_frame = tk.Frame(self.filter_frame, bg=self.bg_color)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        add_filter_btn = tk.Button(
            btn_frame,
            text="Add Filter Row",
            bg=self.accent_color,
            fg="white",
            command=self._add_filter_row
        )
        add_filter_btn.pack(side=tk.LEFT)
        
        apply_filters_btn = tk.Button(
            btn_frame,
            text="Apply Filters",
            bg="#4caf50",
            fg="white",
            command=self._apply_filters
        )
        apply_filters_btn.pack(side=tk.LEFT, padx=10)
        
        # Filter rows container
        self.filter_container = tk.Frame(self.filter_frame, bg="#252526")
        self.filter_container.pack(fill=tk.X, padx=10, pady=5)
        
        # Results
        self.filter_results_var = tk.StringVar(value="No filters applied")
        results_label = tk.Label(
            self.filter_frame,
            textvariable=self.filter_results_var,
            bg=self.bg_color,
            fg=self.fg_color,
            font=("Consolas", 9)
        )
        results_label.pack(anchor=tk.W, padx=10, pady=5)
        
        # Filtered data preview
        self.filter_tree_frame = tk.Frame(self.filter_frame, bg=self.bg_color)
        self.filter_tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Store filter rows
        self.filter_rows = []
    
    def _log_status(self, message):
        """Log a message to the status text."""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def _initialize_test_data(self):
        """Initialize test data and DataCoordinator."""
        self._log_status("Creating test data...")
        
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp(prefix="datamanager_test_")
        self._log_status(f"Temp directory: {self.temp_dir}")
        
        # Create test files
        compartment_folder = os.path.join(self.temp_dir, "compartments")
        csv_path = os.path.join(self.temp_dir, "drillhole_data.csv")
        
        start_time = time.time()
        img_count = create_test_compartment_images(compartment_folder, holes=5, depths_per_hole=20)
        create_test_csv(csv_path, rows=200, holes=5)
        data_creation_time = time.time() - start_time
        
        self._log_status(f"Test data created in {data_creation_time:.3f}s ({img_count} images)")
        
        # Create mock manager
        mock_manager = create_mock_json_manager()
        
        # Initialize DataCoordinator
        self._log_status("Initializing DataCoordinator...")
        start_time = time.time()
        
        self.coordinator = DataCoordinator()
        self.coordinator.initialize(
            compartment_folders=[compartment_folder],
            csv_files=[csv_path],
            json_manager=mock_manager
        )
        
        init_time = time.time() - start_time
        
        self._log_status(f"DataCoordinator initialized in {init_time:.3f}s")
        self._log_status(f"  Images indexed: {self.coordinator.image_index.image_count}")
        self._log_status(f"  Unique holes: {self.coordinator.image_index.hole_count}")
        self._log_status(f"  CSV rows: {self.coordinator.geological_store.total_rows}")
        
        # Build DataFrame
        self._log_status("Building DataFrame...")
        start_time = time.time()
        
        all_keys = list(self.coordinator.image_index.keys())
        self.dataframe = self.coordinator.build_dataframe(all_keys)
        
        df_time = time.time() - start_time
        self._log_status(f"DataFrame built in {df_time:.3f}s ({len(self.dataframe)} rows, {len(self.dataframe.columns)} columns)")
        
        # Update UI
        self._populate_dataframe_view()
        self._update_column_combos()
    
    def _reinitialize(self):
        """Reinitialize with fresh test data."""
        # Clean up old temp dir
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clear status
        self.status_text.delete(1.0, tk.END)
        
        # Clear filter rows
        for fr in self.filter_rows:
            fr.destroy()
        self.filter_rows.clear()
        
        # Reinitialize
        self._initialize_test_data()
    
    def _populate_dataframe_view(self):
        """Populate the treeview with DataFrame data."""
        if self.dataframe is None:
            return
        
        # Clear existing
        self.tree.delete(*self.tree.get_children())
        
        # Set columns
        columns = list(self.dataframe.columns)[:15]  # Limit for display
        self.tree["columns"] = columns
        self.tree["show"] = "headings"
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        
        # Add rows (limit to first 100)
        for idx, row in self.dataframe.head(100).iterrows():
            values = [str(row[col])[:20] for col in columns]
            self.tree.insert("", tk.END, values=values)
    
    def _update_column_combos(self):
        """Update column combo boxes with available columns."""
        if self.dataframe is None:
            return
        
        columns = list(self.dataframe.columns)
        
        # Numeric columns for scatter
        numeric_cols = [col for col in columns if self.dataframe[col].dtype in ['float64', 'int64']]
        
        self.x_col_combo['values'] = numeric_cols
        self.y_col_combo['values'] = numeric_cols
        self.color_col_combo['values'] = columns
        
        # Set defaults if available
        if 'fe_pct_best' in numeric_cols:
            self.x_col_var.set('fe_pct_best')
        elif numeric_cols:
            self.x_col_var.set(numeric_cols[0])
        
        if 'sio2_pct_best' in numeric_cols:
            self.y_col_var.set('sio2_pct_best')
        elif len(numeric_cols) > 1:
            self.y_col_var.set(numeric_cols[1])
        
        if 'stratsum' in columns:
            self.color_col_var.set('stratsum')
    
    def _create_scatter_widget(self):
        """Create the scatter selection widget."""
        if self.dataframe is None:
            messagebox.showwarning("No Data", "Please wait for data initialization")
            return
        
        # Clear existing
        for child in self.scatter_container.winfo_children():
            child.destroy()
        
        x_col = self.x_col_var.get()
        y_col = self.y_col_var.get()
        color_col = self.color_col_var.get()
        
        if not x_col or not y_col:
            messagebox.showwarning("Missing Columns", "Please select X and Y columns")
            return
        
        try:
            # Import the widget
            from select_on_scatterplot_widget import ScatterSelectionWidget
            
            # Create simple color map from unique values
            if color_col and color_col in self.dataframe.columns:
                unique_vals = self.dataframe[color_col].dropna().unique()
                color_map = {}
                colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
                for i, val in enumerate(unique_vals[:len(colors)]):
                    color_map[str(val)] = colors[i % len(colors)]
            else:
                color_map = None
            
            # Create widget
            scatter = ScatterSelectionWidget(
                parent=self.scatter_container,
                gui_manager=self.gui_manager,
                data=self.dataframe,
                x_col=x_col,
                y_col=y_col,
                color_by=color_col,
                color_map=color_map,
                on_selection=self._on_scatter_selection,
                point_size=30.0,
            )
            scatter.pack(fill=tk.BOTH, expand=True)
            
            self.selection_info_var.set("Scatter plot created - use lasso to select points")
            
        except ImportError as e:
            self._log_status(f"Could not import ScatterSelectionWidget: {e}")
            messagebox.showerror("Import Error", f"Could not import ScatterSelectionWidget:\n{e}")
        except Exception as e:
            self._log_status(f"Error creating scatter widget: {e}")
            messagebox.showerror("Error", f"Error creating scatter widget:\n{e}")
    
    def _on_scatter_selection(self, selected_df):
        """Handle scatter plot selection."""
        count = len(selected_df)
        if count > 0:
            self.selection_info_var.set(
                f"Selected {count} points | "
                f"Holes: {selected_df['hole_id'].nunique()} | "
                f"Fe avg: {selected_df.get('fe_pct_best', pd.Series([0])).mean():.1f}%"
            )
        else:
            self.selection_info_var.set("No selection")
    
    def _add_filter_row(self):
        """Add a new filter row."""
        if self.dataframe is None:
            messagebox.showwarning("No Data", "Please wait for data initialization")
            return
        
        try:
            from dynamic_filter_row import DynamicFilterRow
            
            # Get columns info
            columns_info = {col: DataType.TEXT for col in self.dataframe.columns}
            for col in self.dataframe.columns:
                if self.dataframe[col].dtype in ['float64', 'int64']:
                    columns_info[col] = DataType.NUMERIC
            
            index = len(self.filter_rows)
            
            filter_row = DynamicFilterRow(
                parent=self.filter_container,
                gui_manager=self.gui_manager,
                columns_info=columns_info,
                register_data=self.dataframe,
                on_remove_callback=self._remove_filter_row,
                index=index
            )
            
            self.filter_rows.append(filter_row)
            self.filter_results_var.set(f"{len(self.filter_rows)} filter(s) added")
            
        except ImportError as e:
            self._log_status(f"Could not import DynamicFilterRow: {e}")
            messagebox.showerror("Import Error", f"Could not import DynamicFilterRow:\n{e}")
        except Exception as e:
            self._log_status(f"Error creating filter row: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error creating filter row:\n{e}")
    
    def _remove_filter_row(self, index):
        """Remove a filter row."""
        if 0 <= index < len(self.filter_rows):
            self.filter_rows[index].destroy()
            self.filter_rows.pop(index)
            # Re-index remaining rows
            for i, fr in enumerate(self.filter_rows):
                fr.index = i
            self.filter_results_var.set(f"{len(self.filter_rows)} filter(s) remaining")
    
    def _apply_filters(self):
        """Apply all filters to the DataFrame."""
        if self.dataframe is None or not self.filter_rows:
            return
        
        try:
            filtered_df = self.dataframe.copy()
            
            for filter_row in self.filter_rows:
                config = filter_row.get_filter_config()
                
                # Apply filter
                mask = filtered_df.apply(
                    lambda row: filter_row.apply_filter(row.to_dict()),
                    axis=1
                )
                filtered_df = filtered_df[mask]
            
            self.filter_results_var.set(
                f"Filter applied: {len(filtered_df)} / {len(self.dataframe)} rows match"
            )
            
        except Exception as e:
            self._log_status(f"Error applying filters: {e}")
            self.filter_results_var.set(f"Filter error: {e}")
    
    def cleanup(self):
        """Clean up temp files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    root = tk.Tk()
    
    # Create app
    app = DataManagerTestGUI(root)
    
    # Handle cleanup on close
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Run
    root.mainloop()


if __name__ == "__main__":
    main()
