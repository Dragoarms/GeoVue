# src/gui/csv_export_dialog.py
"""
CSV Export Dialog - Comprehensive export with column selection from multiple data sources.

Provides a modal dialog for exporting classifications and geological data with:
- Column selection from multiple sources (Register, CSV data, Image properties)
- Filter by classification status
- Include/exclude comments, tags, hex colors
- Preview of export data
- Merge data from DataCoordinator stores

Author: George Symonds
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
from typing import List, Dict, Optional, Set, Any, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime
from collections import OrderedDict

from gui.dialog_helper import DialogHelper
from gui.widgets.modern_button import ModernButton
from gui.widgets.collapsible_frame import CollapsibleFrame
from gui.widgets.custom_checkbox import create_custom_checkbox

logger = logging.getLogger(__name__)


class CSVExportDialog:
    """
    Dialog for exporting classification data to CSV with column selection.
    
    Allows selecting columns from:
    - Image metadata (hole_id, depth_from, depth_to, filename)
    - Classification data (classification, tags, comments)
    - Review data (consensus, agreement, review_count)
    - Image properties (combined_hex)
    - Geological CSV data (Fe%, SiO2%, etc.)
    
    Usage:
        >>> dialog = CSVExportDialog(
        ...     parent=self.dialog,
        ...     all_images=self.all_images,
        ...     gui_manager=self.gui_manager,
        ...     data_coordinator=self.data_coordinator,
        ...     item_manager=self.item_manager,
        ... )
        >>> dialog.show()
    """
    
    # Column groups for organization
    COLUMN_GROUPS = OrderedDict([
        ("Image Metadata", [
            ("hole_id", "Hole ID", True),
            ("depth_from", "Depth From", True),
            ("depth_to", "Depth To", True),
            ("moisture_status", "Moisture Status", False),
            ("filename", "Filename", False),
            ("image_path", "Image Path", False),
        ]),
        ("Consensus (All Reviewers)", [
            ("consensus_classification", "Consensus Classification", True),
            ("review_count", "Review Count", True),
            ("agreement", "Agreement Level", True),
        ]),
        ("Image Properties", [
            ("combined_hex", "Hex Color", False),
            ("wet_hex", "Wet Hex Color", False),
            ("dry_hex", "Dry Hex Color", False),
        ]),
    ])
    
    def __init__(
        self,
        parent,
        all_images: List,
        gui_manager,
        data_coordinator=None,
        drillhole_data_manager=None,
        item_manager=None,
        config_manager=None,
        json_manager=None,
        hex_color_cache: Dict[Tuple, str] = None,
        other_user_reviews: Dict[Tuple, List[Dict]] = None,
    ):
        """
        Initialize the CSV export dialog.
        
        Args:
            parent: Parent window
            all_images: List of CompartmentImage objects
            gui_manager: GUIManager for theming
            data_coordinator: DataCoordinator for accessing stores
            drillhole_data_manager: Legacy DrillholeDataManager (fallback)
            item_manager: ImageClassificationAndTagManager for tag definitions
            config_manager: ConfigManager for saving preferences
            json_manager: JSONRegisterManager for direct review data access
        """
        self.parent = parent
        self.all_images = all_images
        self.gui_manager = gui_manager
        self.data_coordinator = data_coordinator
        self.drillhole_data_manager = drillhole_data_manager
        self.item_manager = item_manager
        self.config_manager = config_manager
        self.json_manager = json_manager
        self.hex_color_cache = hex_color_cache or {}
        self.other_user_reviews = other_user_reviews or {}
        
        self.theme = gui_manager.theme_colors
        self.dialog = None
        self.result = None
        
        # Column selection state
        self.column_vars: Dict[str, tk.BooleanVar] = {}
        self.csv_column_vars: Dict[str, tk.BooleanVar] = {}
        self.tag_column_vars: Dict[str, tk.BooleanVar] = {}

        # Dynamic reviewer columns: reviewer -> [list of cols]
        self.reviewer_columns: Dict[str, List[str]] = {}
        
        # Store checkbox frames for potential updates
        self.checkbox_frames: List = []
        
        # Filter options
        self.filter_var = tk.StringVar(value="classified")
        self.include_header_var = tk.BooleanVar(value=True)
        
        # Available CSV columns (populated from data sources)
        self.available_csv_columns: Dict[str, List[Tuple[str, str]]] = {}
        
        logger.info(f"CSVExportDialog initialized with {len(all_images)} images")
    
    def show(self) -> Optional[str]:
        """
        Show the dialog and return the exported filepath or None if cancelled.
        
        Returns:
            Path to exported file, or None if cancelled
        """
        logger.info("=" * 60)
        logger.info("CSV EXPORT DIALOG SHOW()")
        logger.info("=" * 60)
        logger.info(f"Total images received: {len(self.all_images)}")
        
        # Detect reviewers before building UI
        self._detect_reviewers()
        self._create_dialog()
        self._populate_csv_columns()
        
        # Log summary of all variable dictionaries
        logger.info("=" * 40)
        logger.info("VARIABLE SUMMARY AFTER INITIALIZATION")
        logger.info("=" * 40)
        logger.info(f"column_vars: {len(self.column_vars)} entries")
        logger.info(f"  Keys: {list(self.column_vars.keys())}")
        logger.info(f"tag_column_vars: {len(self.tag_column_vars)} entries")
        logger.info(f"  Keys: {list(self.tag_column_vars.keys())}")
        logger.info(f"csv_column_vars: {len(self.csv_column_vars)} entries")
        logger.info(f"  Keys: {list(self.csv_column_vars.keys())[:10]}...")  # First 10 only
        logger.info(f"reviewer_columns: {len(self.reviewer_columns)} reviewers")
        logger.info("=" * 40)
        
        # Center on parent
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Wait for dialog to close
        self.parent.wait_window(self.dialog)
        
        return self.result
    
    def _detect_reviewers(self):
        """Scan all images and discover all reviewer names."""
        reviewers = set()
        
        logger.info("=" * 60)
        logger.info("DETECTING REVIEWERS FOR CSV EXPORT")
        logger.info("=" * 60)
        logger.info(f"Total images to scan: {len(self.all_images)}")
        
        images_with_other_reviews = 0
        
        for img in self.all_images:
            # Current user (classification)
            if getattr(img, "classified_by", None):
                reviewers.add(img.classified_by)
                logger.debug(f"Found current user reviewer: {img.classified_by}")

            # Other reviewers stored in img.other_reviews (list of dicts)
            # Field names can be: "Reviewed_By", "_file_user", or "reviewer"
            if hasattr(img, "other_reviews") and img.other_reviews:
                images_with_other_reviews += 1
                for r in img.other_reviews:
                    if isinstance(r, dict):
                        # Try multiple field names for reviewer
                        reviewer_name = (
                            r.get("Reviewed_By") or 
                            r.get("_file_user") or 
                            r.get("reviewer") or
                            r.get("classified_by")
                        )
                        if reviewer_name:
                            reviewers.add(reviewer_name)
                            logger.debug(f"Found other reviewer: {reviewer_name}")

        logger.info(f"Images with other_reviews populated: {images_with_other_reviews}")
        logger.info(f"Unique reviewers found: {reviewers}")

        # Build columns per reviewer (classification, comments, tags, date)
        self.reviewer_columns = {}
        for reviewer in sorted(reviewers):  # Sort for consistent ordering
            safe = reviewer.lower().replace(" ", "_").replace(".", "_")
            self.reviewer_columns[reviewer] = [
                f"rev_{safe}_classification",
                f"rev_{safe}_comments",
                f"rev_{safe}_tags",
                f"rev_{safe}_date",
            ]
            logger.info(f"  Reviewer '{reviewer}' -> columns: {self.reviewer_columns[reviewer]}")
        
        logger.info(f"Total reviewer columns to create: {len(self.reviewer_columns) * 4}")
        logger.info("=" * 60)


    def _create_dialog(self):
        """Create the dialog window and widgets."""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Export to CSV")
        self.dialog.geometry("900x700")
        self.dialog.minsize(800, 600)
        self.dialog.configure(bg=self.theme["background"])
        
        # Apply theme
        self.gui_manager.apply_theme(self.dialog)
        
        # Main container
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Export Classifications to CSV",
            font=("Arial", 14, "bold"),
            style="Heading.TLabel"
        )
        title_label.pack(pady=(0, 10))
        
        # Paned window for columns selection and options
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left panel: Column selection
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=2)
        self._create_column_selection(left_frame)
        
        # Right panel: Options and preview
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        self._create_options_panel(right_frame)
        
        # Bottom: Action buttons
        self._create_action_buttons(main_frame)
        
        # Bind escape to cancel
        self.dialog.bind("<Escape>", lambda e: self._on_cancel())
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
    
    def _create_column_selection(self, parent):
        """Create the column selection panel with checkboxes."""
        # Container with scrollbar
        container = ttk.LabelFrame(parent, text="Select Columns to Export", padding=5)
        container.pack(fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Canvas for scrolling
        canvas = tk.Canvas(
            container,
            bg=self.theme["background"],
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mousewheel
        def on_mousewheel(event):
            if canvas.winfo_exists():
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)
        
        # Create column groups
        for group_name, columns in self.COLUMN_GROUPS.items():
            self._create_column_group(scroll_frame, group_name, columns)

        # NEW: individual reviewer columns (per-reviewer classification/date/by)
        if self.reviewer_columns:
            reviewer_group = ttk.LabelFrame(scroll_frame, text="Individual Reviewer Classifications", padding=5)
            reviewer_group.pack(fill=tk.X, pady=2, padx=2)

            for reviewer, cols in self.reviewer_columns.items():
                # Header label per reviewer
                sub_label = ttk.Label(
                    reviewer_group,
                    text=f"Reviewer: {reviewer}",
                    style="Content.TLabel"
                )
                sub_label.pack(anchor=tk.W, pady=(2, 0))

                # One checkbox per generated column id
                for col_id in cols:
                    var = tk.BooleanVar(value=True)  # on by default
                    self.column_vars[col_id] = var

                    cb = ttk.Checkbutton(
                        reviewer_group,
                        text=col_id,
                        variable=var,
                        style="Content.TCheckbutton"
                    )
                    cb.pack(anchor=tk.W, padx=20, pady=1)
        
        # Add tag columns if available
        if self.item_manager:
            self._create_tag_columns_group(scroll_frame)
        
        # Add CSV data columns
        self._create_csv_columns_group(scroll_frame)
        
        # Select/Deselect all buttons
        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ModernButton(
            btn_frame,
            text="Select All",
            command=self._select_all_columns,
            color=self.theme["accent_blue"],
            theme_colors=self.theme,
            width=10,
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ModernButton(
            btn_frame,
            text="Select None",
            command=self._deselect_all_columns,
            color=self.theme.get("accent_orange", "#ff9800"),
            theme_colors=self.theme,
            width=10,
        ).pack(side=tk.LEFT)
        
        ModernButton(
            btn_frame,
            text="Select Defaults",
            command=self._select_default_columns,
            color=self.theme["secondary_bg"],
            theme_colors=self.theme,
            width=12,
        ).pack(side=tk.LEFT, padx=(5, 0))
    
    def _create_column_group(self, parent, group_name: str, columns: List[Tuple[str, str, bool]]):
        """Create a collapsible group of column checkboxes."""
        group_frame = ttk.LabelFrame(parent, text=group_name, padding=5)
        group_frame.pack(fill=tk.X, pady=2, padx=2)
        
        for col_id, display_name, default_selected in columns:
            var = tk.BooleanVar(value=default_selected)
            self.column_vars[col_id] = var
            
            cb = ttk.Checkbutton(
                group_frame,
                text=display_name,
                variable=var,
                style="Content.TCheckbutton"
            )
            cb.pack(anchor=tk.W, pady=1)
    
    def _create_tag_columns_group(self, parent):
        """Create checkboxes for tag columns."""
        tags = self.item_manager.get_all_tags()
        if not tags:
            return
        
        group_frame = ttk.LabelFrame(parent, text="Tags", padding=5)
        group_frame.pack(fill=tk.X, pady=2, padx=2)
        
        for tag_def in tags:
            col_id = f"tag_{tag_def.id}"
            var = tk.BooleanVar(value=True)  # Tags selected by default
            self.tag_column_vars[col_id] = var
            
            cb = ttk.Checkbutton(
                group_frame,
                text=tag_def.label or tag_def.id,
                variable=var,
                style="Content.TCheckbutton"
            )
            cb.pack(anchor=tk.W, pady=1)
    
    def _create_csv_columns_group(self, parent):
        """Create checkboxes for CSV data columns (from geological stores)."""
        # This will be populated after dialog creation
        self.csv_columns_frame = ttk.LabelFrame(parent, text="Geological Data (CSV)", padding=5)
        self.csv_columns_frame.pack(fill=tk.X, pady=2, padx=2)
        
        # Placeholder until columns are populated
        self.csv_loading_label = ttk.Label(
            self.csv_columns_frame,
            text="Loading available columns...",
            style="Content.TLabel"
        )
        self.csv_loading_label.pack(anchor=tk.W)
    
    def _populate_csv_columns(self):
        """Populate CSV columns from data sources."""
        logger.info("=" * 40)
        logger.info("POPULATING CSV COLUMNS")
        logger.info("=" * 40)
        
        # Clear loading label
        self.csv_loading_label.destroy()
        
        # Get columns from DataCoordinator or DrillholeDataManager
        available = None
        if self.data_coordinator and self.data_coordinator.geological_store.is_loaded:
            available = self.data_coordinator.geological_store.get_available_columns()
            logger.info("Using DataCoordinator geological_store for columns")
        elif self.drillhole_data_manager and hasattr(self.drillhole_data_manager, 'get_available_columns'):
            available = self.drillhole_data_manager.get_available_columns()
            logger.info("Using DrillholeDataManager for columns")
        
        if not available:
            logger.warning("No CSV data source available")
            ttk.Label(
                self.csv_columns_frame,
                text="No CSV data loaded",
                style="Content.TLabel"
            ).pack(anchor=tk.W)
            return
        
        logger.info(f"Available column sources: {list(available.keys())}")
        for source, cols in available.items():
            logger.info(f"  {source}: {len(cols)} columns")
        
        # Key columns to skip (already in Image Metadata group)
        skip_columns = {
            'holeid', 'hole_id', 'bhid', 'drillhole_id',
            'from', 'depth_from', 'sampfrom', 'geolfrom',
            'to', 'depth_to', 'sampto', 'geolto',
        }
        
        # Create collapsible frames per source
        source_count = len(available)
        columns_added = 0
        
        for source_name, cols in available.items():
            # Create sub-frame for each source
            source_frame = ttk.Frame(self.csv_columns_frame)
            source_frame.pack(fill=tk.X, pady=2)
            
            if source_count > 1:
                source_label = ttk.Label(
                    source_frame,
                    text=f"[{source_name}]:",
                    font=("Arial", 9, "bold"),
                    style="Content.TLabel"
                )
                source_label.pack(anchor=tk.W)
            
            # Add columns (in a grid for compact display)
            col_container = ttk.Frame(source_frame)
            col_container.pack(fill=tk.X, padx=(10, 0))
            
            row = 0
            col = 0
            max_cols = 3
            
            for col_name, col_type in cols:
                # Skip key columns
                col_lower = col_name.lower().replace('_', '')
                if col_lower in skip_columns or col_name.lower() in skip_columns:
                    continue
                
                # Create unique ID
                if source_count > 1:
                    col_id = f"{col_name} ({source_name})"
                else:
                    col_id = col_name
                
                var = tk.BooleanVar(value=False)  # CSV columns off by default
                self.csv_column_vars[col_id] = var
                
                # Truncate long names
                display_name = col_name[:20] + "..." if len(col_name) > 20 else col_name
                
                cb = ttk.Checkbutton(
                    col_container,
                    text=display_name,
                    variable=var,
                    style="Content.TCheckbutton"
                )
                cb.grid(row=row, column=col, sticky=tk.W, padx=2, pady=1)
                
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
                
                columns_added += 1
        
        if columns_added == 0:
            ttk.Label(
                self.csv_columns_frame,
                text="No additional CSV columns available",
                style="Content.TLabel"
            ).pack(anchor=tk.W)
        else:
            # Add select all/none for CSV columns
            csv_btn_frame = ttk.Frame(self.csv_columns_frame)
            csv_btn_frame.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Button(
                csv_btn_frame,
                text="Select All CSV",
                command=lambda: self._set_all_csv_columns(True),
                style="Accent.TButton"
            ).pack(side=tk.LEFT, padx=(0, 5))
            
            ttk.Button(
                csv_btn_frame,
                text="Deselect All CSV",
                command=lambda: self._set_all_csv_columns(False),
            ).pack(side=tk.LEFT)
        
        logger.info(f"Populated {columns_added} CSV columns from {source_count} sources")
        logger.info(f"Total csv_column_vars: {len(self.csv_column_vars)}")
        logger.info("=" * 40)
    
    def _create_options_panel(self, parent):
        """Create the options and preview panel."""
        # Filter options
        filter_frame = ttk.LabelFrame(parent, text="Export Filter", padding=10)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        filters = [
            ("all", "All Images"),
            ("classified", "Classified Only"),
            ("unclassified", "Unclassified Only"),
            ("displayed", "Currently Displayed"),
        ]
        
        for value, text in filters:
            rb = ttk.Radiobutton(
                filter_frame,
                text=text,
                variable=self.filter_var,
                value=value,
                style="Content.TRadiobutton",
                command=self._update_preview,
            )
            rb.pack(anchor=tk.W, pady=2)
        
        # Export options
        options_frame = ttk.LabelFrame(parent, text="Options", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(
            options_frame,
            text="Include header row",
            variable=self.include_header_var,
            style="Content.TCheckbutton"
        ).pack(anchor=tk.W, pady=2)
        
        # Preview section
        preview_frame = ttk.LabelFrame(parent, text="Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Row count
        self.row_count_label = ttk.Label(
            preview_frame,
            text="Rows to export: calculating...",
            style="Content.TLabel"
        )
        self.row_count_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Column count
        self.col_count_label = ttk.Label(
            preview_frame,
            text="Columns selected: 0",
            style="Content.TLabel"
        )
        self.col_count_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Preview text
        self.preview_text = tk.Text(
            preview_frame,
            height=10,
            width=40,
            wrap=tk.NONE,
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            font=("Courier", 8),
        )
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        
        # H scrollbar for preview
        h_scroll = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.preview_text.xview)
        h_scroll.pack(fill=tk.X)
        self.preview_text.configure(xscrollcommand=h_scroll.set)
        
        # Initial update
        self.dialog.after(100, self._update_preview)
    
    def _create_action_buttons(self, parent):
        """Create the action buttons at the bottom."""
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Export button
        ModernButton(
            btn_frame,
            text="Export CSV",
            command=self._on_export,
            color=self.theme["accent_green"],
            theme_colors=self.theme,
            width=15,
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Cancel button
        ModernButton(
            btn_frame,
            text="Cancel",
            command=self._on_cancel,
            color=self.theme["secondary_bg"],
            theme_colors=self.theme,
            width=10,
        ).pack(side=tk.RIGHT)
    
    def _get_selected_columns(self) -> List[str]:
        """Get list of selected column IDs."""
        selected = []
        
        logger.debug("=" * 40)
        logger.debug("GETTING SELECTED COLUMNS")
        logger.debug("=" * 40)
        
        # Standard columns (includes reviewer columns)
        standard_selected = 0
        for col_id, var in self.column_vars.items():
            if var.get():
                selected.append(col_id)
                standard_selected += 1
                logger.debug(f"  Standard column selected: {col_id}")
        logger.debug(f"Standard columns: {standard_selected} selected from {len(self.column_vars)} total")
        
        # Tag columns
        tag_selected = 0
        for col_id, var in self.tag_column_vars.items():
            if var.get():
                selected.append(col_id)
                tag_selected += 1
                logger.debug(f"  Tag column selected: {col_id}")
        logger.debug(f"Tag columns: {tag_selected} selected from {len(self.tag_column_vars)} total")
        
        # CSV columns
        csv_selected = 0
        for col_id, var in self.csv_column_vars.items():
            if var.get():
                selected.append(col_id)
                csv_selected += 1
                logger.debug(f"  CSV column selected: {col_id}")
        logger.debug(f"CSV columns: {csv_selected} selected from {len(self.csv_column_vars)} total")
        
        logger.debug(f"TOTAL SELECTED: {len(selected)} columns")
        logger.debug("=" * 40)
        
        return selected
    
    def _get_filtered_images(self) -> List:
        """Get images based on current filter selection."""
        filter_mode = self.filter_var.get()
        logger.debug(f"_get_filtered_images: filter_mode={filter_mode}, total_images={len(self.all_images)}")
        
        if filter_mode == "all":
            result = self.all_images
        elif filter_mode == "classified":
            result = [img for img in self.all_images 
                    if img.classification and str(img.classification) not in ('', 'UNASSIGNED', 'ClassificationCategory.UNASSIGNED')]
        elif filter_mode == "unclassified":
            result = [img for img in self.all_images 
                    if not img.classification or str(img.classification) in ('', 'UNASSIGNED', 'ClassificationCategory.UNASSIGNED')]
        elif filter_mode == "displayed":
            # Would need access to displayed_images - use all for now
            result = self.all_images
        else:
            result = self.all_images
        
        logger.debug(f"_get_filtered_images: returning {len(result)} images")
        return result
    
    def _update_preview(self, *args):
        """Update the preview panel."""
        images = self._get_filtered_images()
        columns = self._get_selected_columns()
        
        self.row_count_label.config(text=f"Rows to export: {len(images):,}")
        self.col_count_label.config(text=f"Columns selected: {len(columns)}")
        
        # Generate preview (first 5 rows)
        if columns and images:
            preview_data = self._build_export_dataframe(images[:5], columns)
            preview_str = preview_data.to_string(index=False, max_cols=10)
        else:
            preview_str = "(Select columns to see preview)"
        
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert("1.0", preview_str)
    
    def _build_export_dataframe(self, images: List, columns: List[str]) -> pd.DataFrame:
        """
        Build the export DataFrame using cached data and pandas merges.
        
        Args:
            images: List of CompartmentImage objects
            columns: List of column IDs to include
            
        Returns:
            pandas DataFrame ready for export
        """
        logger.info(f"Building export DataFrame: {len(images)} images, {len(columns)} columns")
        
        # =================================================================
        # STEP 1: Build base DataFrame from images with all cached data
        # =================================================================
        logger.info("Step 1: Building base DataFrame from images...")
        
        base_data = []
        for img in images:
            # Get the lookup key for caches
            cache_key = (img.hole_id, img.depth_from, img.depth_to)
            
            row = {
                'hole_id': img.hole_id,
                'depth_from': img.depth_from,
                'depth_to': img.depth_to,
                'moisture_status': img.moisture_status or "",
                'filename': img.filename,
                'image_path': img.image_path,
                # Merge keys for CSV data
                '_hole_upper': img.hole_id.upper() if img.hole_id else '',
                '_depth_int': int(img.depth_to) if img.depth_to else 0,
            }
            
            # Tags (from image object)
            for col_id in columns:
                if col_id.startswith('tag_'):
                    tag_id = col_id[4:]
                    row[col_id] = "Yes" if (img.tags and tag_id in img.tags) else "No"
            
            # Hex colors from cache (cache stores dict with combined_hex, wet_hex, dry_hex)
            hex_data = self.hex_color_cache.get(cache_key, {})
            if isinstance(hex_data, str):
                # Old format - just combined_hex string
                hex_data = {"combined_hex": hex_data, "wet_hex": "", "dry_hex": ""}
            
            if 'combined_hex' in columns:
                row['combined_hex'] = hex_data.get("combined_hex", "")
            if 'wet_hex' in columns:
                row['wet_hex'] = hex_data.get("wet_hex", "")
            if 'dry_hex' in columns:
                row['dry_hex'] = hex_data.get("dry_hex", "")
            
            # Collect all reviews for this image (current user + others)
            all_reviews = self._collect_all_reviews(img, cache_key)
            
            # Consensus columns
            if 'consensus_classification' in columns or 'review_count' in columns or 'agreement' in columns:
                classifications = [self._normalize_classification(r.get("classification")) 
                                   for r in all_reviews if r.get("classification")]
                classifications = [c for c in classifications if c]  # Remove empty strings
                
                if 'review_count' in columns:
                    row['review_count'] = len(all_reviews)
                
                if 'consensus_classification' in columns:
                    if classifications:
                        from collections import Counter
                        row['consensus_classification'] = Counter(classifications).most_common(1)[0][0]
                    else:
                        row['consensus_classification'] = ""
                
                if 'agreement' in columns:
                    if len(all_reviews) == 0:
                        row['agreement'] = "None"
                    elif len(all_reviews) == 1:
                        row['agreement'] = "Single"
                    elif classifications:
                        unique = set(classifications)
                        if len(unique) == 1:
                            row['agreement'] = "Unanimous"
                        else:
                            from collections import Counter
                            most_common_count = Counter(classifications).most_common(1)[0][1]
                            row['agreement'] = "Majority" if most_common_count > len(classifications) / 2 else "Split"
                    else:
                        row['agreement'] = "None"
            
            # Per-reviewer columns
            for col_id in columns:
                if col_id.startswith('rev_'):
                    row[col_id] = self._get_reviewer_column_value(col_id, all_reviews)
            
            base_data.append(row)
        
        df = pd.DataFrame(base_data)
        logger.info(f"  Base DataFrame: {len(df)} rows")
        
        # =================================================================
        # STEP 2: Merge CSV geological data
        # =================================================================
        csv_cols = [c for c in columns if ' (' in c]  # Columns with source suffix
        
        if csv_cols and self.data_coordinator and self.data_coordinator.geological_store.is_loaded:
            logger.info(f"Step 2: Merging {len(csv_cols)} CSV columns...")
            
            # Build mapping of requested column -> base name
            col_mapping = {}
            for col_id in csv_cols:
                base_col = col_id.split(" (")[0] if " (" in col_id else col_id
                col_mapping[col_id] = base_col.lower()
            
            needed_base_cols = set(col_mapping.values())
            
            geo_store = self.data_coordinator.geological_store
            
            for source_name, source in geo_store.get_data_sources().items():
                if not source.is_loaded or source.df is None:
                    continue
                
                src_df = source.df
                
                # Find which needed columns are in this source
                available = [c for c in needed_base_cols if c in src_df.columns]
                if not available:
                    continue
                
                # Determine merge strategy based on dataset type
                has_depth = '_depth_int' in src_df.columns
                
                if has_depth:
                    # Interval data - merge on hole + depth
                    logger.info(f"  Merging {len(available)} columns from {source_name} (interval join)...")
                    merge_df = src_df[['_hole_upper', '_depth_int'] + available].drop_duplicates(
                        subset=['_hole_upper', '_depth_int'], keep='first'
                    )
                    df = df.merge(merge_df, on=['_hole_upper', '_depth_int'], how='left', suffixes=('', f'__{source_name}'))
                else:
                    # Collar/header data - merge on hole only (one-to-many)
                    logger.info(f"  Merging {len(available)} columns from {source_name} (hole join)...")
                    merge_df = src_df[['_hole_upper'] + available].drop_duplicates(
                        subset=['_hole_upper'], keep='first'
                    )
                    df = df.merge(merge_df, on=['_hole_upper'], how='left', suffixes=('', f'__{source_name}'))
                
                needed_base_cols -= set(available)
            
            # Build rename dict and apply all at once to avoid fragmentation
            rename_dict = {}
            for col_id in csv_cols:
                base_col = col_mapping[col_id]
                if base_col in df.columns and col_id != base_col:
                    rename_dict[base_col] = col_id
            
            # Apply renames
            if rename_dict:
                df = df.rename(columns=rename_dict)
            
            # Fill NaN and add missing columns
            for col_id in csv_cols:
                if col_id in df.columns:
                    df[col_id] = df[col_id].fillna("")
                elif col_mapping[col_id] in df.columns:
                    # Column exists under base name, copy it
                    df[col_id] = df[col_mapping[col_id]].fillna("")
                else:
                    df[col_id] = ""
            
            # Defragment the DataFrame
            df = df.copy()
        
        # =================================================================
        # STEP 3: Select final columns in order
        # =================================================================
        logger.info("Step 3: Selecting final columns...")
        
        final_cols = [c for c in columns if c in df.columns]
        df = df[final_cols]
        
        logger.info(f"DataFrame built: {len(df)} rows x {len(df.columns)} columns")
        
        return df
    
    def _collect_all_reviews(self, img, cache_key: Tuple) -> List[Dict]:
        """
        Collect all reviews for an image - current user + other users.
        
        Args:
            img: CompartmentImage object
            cache_key: Tuple of (hole_id, depth_from, depth_to) for cache lookup
            
        Returns:
            List of normalized review dictionaries
        """
        all_reviews = []
        
        # 1. Current user's review from the image object
        if img.classification and str(img.classification) not in ('', 'UNASSIGNED', 'ClassificationCategory.UNASSIGNED'):
            all_reviews.append({
                "classification": self._normalize_classification(img.classification),
                "Reviewed_By": img.classified_by or "",
                "classified_date": img.classified_date or "",
                "comments": img.comments or "",
                "tags": list(img.tags) if img.tags else []
            })
        
        # 2. Other users' reviews from the cache
        other_reviews = self.other_user_reviews.get(cache_key, [])
        for review in other_reviews:
            if isinstance(review, dict):
                all_reviews.append({
                    "classification": self._normalize_classification(
                        review.get("classification") or 
                        review.get("Classification") or 
                        review.get("Lithology") or ""
                    ),
                    "Reviewed_By": (
                        review.get("Reviewed_By") or 
                        review.get("_file_user") or 
                        review.get("reviewer") or ""
                    ),
                    "classified_date": (
                        review.get("classified_date") or 
                        review.get("Classified_Date") or ""
                    ),
                    "comments": review.get("comments") or review.get("Comments") or "",
                    "tags": review.get("tags") or review.get("Tags") or []
                })
        
        return all_reviews
    
    def _normalize_classification(self, value) -> str:
        """Normalize classification value to consistent format."""
        if value is None:
            return ""
        
        val_str = str(value).strip()
        
        # Remove enum prefix if present
        val_str = val_str.replace("ClassificationCategory.", "")
        
        # Skip empty/unassigned
        if not val_str or val_str.upper() == "UNASSIGNED":
            return ""
        
        # Normalize to lowercase for consistent comparison
        return val_str.lower()
    
    def _get_reviewer_column_value(self, col_id: str, all_reviews: List[Dict]) -> str:
        """Get value for a reviewer-specific column."""
        # Format: rev_<safe_name>_<field>
        parts = col_id.split("_")
        if len(parts) < 3:
            return ""
        
        field = parts[-1]  # classification, comments, tags, or date
        safe_name = "_".join(parts[1:-1])  # Everything between rev_ and _field
        
        # Find matching reviewer
        for review in all_reviews:
            reviewer = review.get("Reviewed_By", "")
            reviewer_safe = reviewer.lower().replace(" ", "_").replace(".", "_")
            
            if reviewer_safe == safe_name:
                if field == "classification":
                    return review.get("classification", "")
                elif field == "date":
                    return review.get("classified_date", "")
                elif field == "comments":
                    return review.get("comments", "")
                elif field == "tags":
                    tags = review.get("tags", [])
                    return ",".join(tags) if isinstance(tags, list) else str(tags)
        
        return ""

    def _get_all_reviews_for_image(self, img) -> List[Dict]:
        """
        Fetch all reviews for an image from the JSON register manager.
        
        Args:
            img: CompartmentImage object
            
        Returns:
            List of review dictionaries from all users
        """
        all_reviews = []
        
        # Try to get from json_manager first (most complete source)
        if self.json_manager:
            try:
                reviews = self.json_manager.get_all_reviews_for_compartment(
                    img.hole_id,
                    int(img.depth_from),
                    int(img.depth_to)
                )
                if reviews:
                    all_reviews = reviews
                    logger.debug(f"Fetched {len(reviews)} reviews from json_manager for {img.hole_id}/{img.depth_to}")
            except Exception as e:
                logger.debug(f"Could not get reviews from json_manager: {e}")
        
        # Fallback: use other_reviews if populated on image
        if not all_reviews and hasattr(img, "other_reviews") and img.other_reviews:
            all_reviews = list(img.other_reviews)
            logger.debug(f"Using {len(all_reviews)} reviews from img.other_reviews")
        
        # Last fallback: create review from image's own classification
        if not all_reviews and img.classification:
            all_reviews = [{
                "classification": str(img.classification).replace("ClassificationCategory.", ""),
                "Reviewed_By": img.classified_by or "",
                "classified_date": img.classified_date or "",
                "comments": img.comments or "",
                "tags": list(img.tags) if img.tags else []
            }]
            logger.debug(f"Created fallback review from img.classification")
        
        return all_reviews
    
    def _get_column_value(self, img, col_id: str, all_reviews: List[Dict] = None) -> Any:
        """
        Get a column value for an image.
        
        Args:
            img: CompartmentImage object
            col_id: Column identifier
            all_reviews: Pre-fetched list of all reviews for this compartment
            
        Returns:
            Value for the column
        """
        # Image metadata
        if col_id == "hole_id":
            return img.hole_id
        elif col_id == "depth_from":
            return img.depth_from
        elif col_id == "depth_to":
            return img.depth_to
        elif col_id == "moisture_status":
            return img.moisture_status or ""
        elif col_id == "filename":
            return img.filename
        elif col_id == "image_path":
            return img.image_path
        
        # Dynamic reviewer columns (rev_<username>_<field>)
        if col_id.startswith("rev_"):
            # Format: rev_<safe_name>_<field> where field is classification/comments/tags/date
            parts = col_id.split("_")
            if len(parts) < 3:
                logger.warning(f"Invalid reviewer column format: {col_id}")
                return ""
            
            field = parts[-1]  # Last part is the field
            safe = "_".join(parts[1:-1])  # Everything between rev_ and _field

            # Find the reviewer matching this safe name
            target_reviewer = None
            for reviewer in self.reviewer_columns.keys():
                safe_name = reviewer.lower().replace(" ", "_").replace(".", "_")
                if safe_name == safe:
                    target_reviewer = reviewer
                    break
            
            if not target_reviewer:
                return ""
            
            # Search through all_reviews for this reviewer
            if all_reviews:
                for review in all_reviews:
                    if not isinstance(review, dict):
                        continue
                    
                    # Get reviewer name from multiple possible fields
                    review_by = (
                        review.get("Reviewed_By") or 
                        review.get("_file_user") or 
                        review.get("reviewer") or
                        review.get("classified_by", "")
                    )
                    
                    if review_by and review_by.lower() == target_reviewer.lower():
                        if field == "classification":
                            return (
                                review.get("classification") or 
                                review.get("Classification") or 
                                review.get("Lithology") or 
                                ""
                            )
                        elif field == "date":
                            return (
                                review.get("classified_date") or 
                                review.get("Classified_Date") or 
                                review.get("Review_Date") or 
                                ""
                            )
                        elif field == "comments":
                            return review.get("comments") or review.get("Comments") or ""
                        elif field == "tags":
                            tags = review.get("tags") or review.get("Tags") or []
                            return ",".join(tags) if isinstance(tags, list) else str(tags)
            
            return ""

        elif col_id == "classification":
            # Legacy column - return current user's classification
            classification = str(img.classification) if img.classification else ""
            return classification.replace("ClassificationCategory.", "")
        elif col_id == "classified_by":
            return img.classified_by or ""
        elif col_id == "classified_date":
            return img.classified_date or ""
        elif col_id == "comments":
            return img.comments or ""
        
        # Consensus columns - computed from all_reviews
        elif col_id == "consensus_classification":
            if all_reviews:
                classifications = []
                for r in all_reviews:
                    if isinstance(r, dict):
                        cls = r.get("classification") or r.get("Classification") or r.get("Lithology")
                        if cls:
                            cls_clean = str(cls).replace("ClassificationCategory.", "")
                            classifications.append(cls_clean)
                if classifications:
                    from collections import Counter
                    return Counter(classifications).most_common(1)[0][0]
            # Fallback to image's classification
            if img.classification:
                return str(img.classification).replace("ClassificationCategory.", "")
            return ""
        
        elif col_id == "review_count":
            if all_reviews:
                return len(all_reviews)
            return 1 if img.classification else 0
        
        elif col_id == "agreement":
            if all_reviews and len(all_reviews) > 1:
                classifications = []
                for r in all_reviews:
                    if isinstance(r, dict):
                        cls = r.get("classification") or r.get("Classification") or r.get("Lithology")
                        if cls:
                            classifications.append(str(cls))
                if classifications:
                    unique = set(classifications)
                    if len(unique) == 1:
                        return "Unanimous"
                    else:
                        from collections import Counter
                        most_common_count = Counter(classifications).most_common(1)[0][1]
                        if most_common_count > len(classifications) / 2:
                            return "Majority"
                        return "Split"
            return "Single"
        
        # Image properties
        elif col_id == "combined_hex":
            if img.csv_data and 'combined_hex' in img.csv_data:
                return img.csv_data.get('combined_hex', '')
            return ""
        elif col_id in ("wet_hex", "dry_hex"):
            # Look up from register store
            if self.data_coordinator:
                try:
                    from processing.DataManager.keys import ImageKey
                    key = ImageKey(img.hole_id, img.depth_to, img.moisture_status)
                    props = self.data_coordinator.register_store.get_image_properties(key)
                    if col_id == "wet_hex":
                        return props.wet_hex or ""
                    else:
                        return props.dry_hex or ""
                except Exception as e:
                    logger.debug(f"Register store lookup failed for {col_id}: {e}")
            return ""
        
        # Tag columns
        elif col_id.startswith("tag_"):
            tag_id = col_id[4:]  # Remove "tag_" prefix
            has_tag = tag_id in img.tags if img.tags else False
            return "Yes" if has_tag else "No"
        
        # CSV data columns - fetch from data sources
        else:
            # First try DataCoordinator's geological_store (most reliable)
            if self.data_coordinator and self.data_coordinator.geological_store.is_loaded:
                try:
                    from processing.DataManager.keys import ImageKey
                    key = ImageKey(img.hole_id, img.depth_to, img.moisture_status)
                    
                    # Strip source suffix if present to get base column name
                    base_col = col_id.split(" (")[0] if " (" in col_id else col_id
                    
                    # Debug: log lookup attempt for first few rows
                    if hasattr(self, '_debug_count'):
                        self._debug_count += 1
                    else:
                        self._debug_count = 1
                    
                    if self._debug_count <= 5:
                        logger.info(f"CSV lookup: col_id='{col_id}', base_col='{base_col}', key={key}")
                    
                    # geological_store normalizes columns to lowercase, so always try lowercase
                    value = self.data_coordinator.geological_store.get_value(key, base_col.lower())
                    
                    if self._debug_count <= 5:
                        logger.info(f"  -> value={value}")
                    if value is not None:
                        return value
                    
                    # Also try original case just in case
                    value = self.data_coordinator.geological_store.get_value(key, base_col)
                    if value is not None:
                        return value
                        
                except Exception as e:
                    logger.debug(f"DataCoordinator geological_store lookup failed for {col_id}: {e}")
            
            # Fallback: Try img.csv_data
            if img.csv_data:
                if col_id in img.csv_data:
                    return img.csv_data.get(col_id, "")
                if " (" in col_id:
                    base_col = col_id.split(" (")[0]
                    if base_col in img.csv_data:
                        return img.csv_data.get(base_col, "")
                    for k, v in img.csv_data.items():
                        if k.lower() == base_col.lower():
                            return v
            
            # Fallback: DrillholeDataManager
            if self.drillhole_data_manager:
                try:
                    base_col = col_id.split(" (")[0] if " (" in col_id else col_id
                    value = self.drillhole_data_manager.get_value(img.hole_id, img.depth_to, base_col)
                    if value is not None:
                        return value
                except Exception:
                    pass
            
            return ""
    
    def _select_all_columns(self):
        """Select all columns."""
        logger.info("=" * 40)
        logger.info("SELECT ALL COLUMNS CLICKED")
        logger.info("=" * 40)
        
        logger.info(f"column_vars count: {len(self.column_vars)}")
        logger.info(f"tag_column_vars count: {len(self.tag_column_vars)}")
        logger.info(f"csv_column_vars count: {len(self.csv_column_vars)}")
        
        for col_id, var in self.column_vars.items():
            var.set(True)
            logger.debug(f"  Set column_vars[{col_id}] = True")
        
        for col_id, var in self.tag_column_vars.items():
            var.set(True)
            logger.debug(f"  Set tag_column_vars[{col_id}] = True")
        
        for col_id, var in self.csv_column_vars.items():
            var.set(True)
            logger.debug(f"  Set csv_column_vars[{col_id}] = True")
        
        self._update_preview()
        logger.info("Select all complete")
    
    def _deselect_all_columns(self):
        """Deselect all columns."""
        for var in self.column_vars.values():
            var.set(False)
        for var in self.tag_column_vars.values():
            var.set(False)
        for var in self.csv_column_vars.values():
            var.set(False)
        self._update_preview()
    
    def _select_default_columns(self):
        """Select default columns only."""
        # Reset to defaults
        for group_name, columns in self.COLUMN_GROUPS.items():
            for col_id, display_name, default_selected in columns:
                if col_id in self.column_vars:
                    self.column_vars[col_id].set(default_selected)
        
        # Tags on by default
        for var in self.tag_column_vars.values():
            var.set(True)
        
        # CSV columns off by default
        for var in self.csv_column_vars.values():
            var.set(False)
        
        self._update_preview()
    
    def _set_all_csv_columns(self, value: bool):
        """Set all CSV column checkboxes to value."""
        logger.info(f"Setting all {len(self.csv_column_vars)} CSV columns to {value}")
        for col_id, var in self.csv_column_vars.items():
            var.set(value)
            logger.debug(f"  Set csv_column_vars[{col_id}] = {value}")
        self._update_preview()
    
    def _on_export(self):
        """Handle export button click."""
        logger.info("=" * 60)
        logger.info("CSV EXPORT INITIATED")
        logger.info("=" * 60)
        
        columns = self._get_selected_columns()
        
        logger.info(f"Selected columns count: {len(columns)}")
        logger.info(f"Column vars total: {len(self.column_vars)}")
        logger.info(f"Tag column vars total: {len(self.tag_column_vars)}")
        logger.info(f"CSV column vars total: {len(self.csv_column_vars)}")
        
        if not columns:
            logger.warning("No columns selected for export")
            messagebox.showwarning(
                "No Columns",
                "Please select at least one column to export.",
                parent=self.dialog
            )
            return
        
        images = self._get_filtered_images()
        logger.info(f"Filtered images count: {len(images)}")
        
        if not images:
            logger.warning("No images match filter")
            messagebox.showwarning(
                "No Data",
                "No images match the selected filter.",
                parent=self.dialog
            )
            return
        
        # Ask for file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"classifications_export_{timestamp}.csv"
        
        filepath = filedialog.asksaveasfilename(
            parent=self.dialog,
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile=default_name,
        )
        
        if not filepath:
            return
        
        try:
            # Build and export
            logger.info(f"Exporting {len(images)} rows with {len(columns)} columns to {filepath}")
            
            df = self._build_export_dataframe(images, columns)
            df.to_csv(
                filepath,
                index=False,
                header=self.include_header_var.get(),
            )
            
            self.result = filepath
            
            messagebox.showinfo(
                "Export Complete",
                f"Successfully exported {len(images):,} rows to:\n{filepath}",
                parent=self.dialog
            )
            
            self.dialog.destroy()
            
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            messagebox.showerror(
                "Export Failed",
                f"Failed to export CSV:\n{str(e)}",
                parent=self.dialog
            )
    
    def _on_cancel(self):
        """Handle cancel/close."""
        self.result = None
        self.dialog.destroy()