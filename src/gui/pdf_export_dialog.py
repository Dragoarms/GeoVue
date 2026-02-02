# # src/gui/pdf_export_dialog.py
# """
# PDF Export Dialog for Logging Review
# Provides advanced filtering and export customization
# """

# import os
# import logging
# import tkinter as tk
# from tkinter import ttk, filedialog, messagebox
# from typing import List, Set, Optional, Any
# from datetime import datetime

# from gui.widgets.modern_button import ModernButton
# from gui.widgets.collapsible_frame import CollapsibleFrame
# from gui.dialog_helper import DialogHelper
# from gui.progress_dialog import ProgressDialog
# from processing.pdf_report_generator import PDFReportGenerator


# class PDFExportDialog:
#     """Dialog for exporting image grid to PDF with filtering and customization"""

#     def __init__(
#         self,
#         parent,
#         all_images: List[Any],
#         displayed_images: List[Any],
#         gui_manager,
#         drillhole_data_manager=None,
#         drillhole_visualizer=None,
#         item_manager=None,
#     ):
#         """
#         Initialize PDF export dialog

#         Args:
#             parent: Parent window
#             all_images: Complete list of available images
#             displayed_images: Currently displayed/filtered images
#             gui_manager: GUI manager for theming
#             drillhole_data_manager: Data manager for CSV data
#             drillhole_visualizer: Visualizer for data plots
#             item_manager: Classification and tag manager
#         """
#         self.parent = parent
#         self.all_images = all_images
#         self.displayed_images = displayed_images
#         self.gui_manager = gui_manager
#         self.drillhole_data_manager = drillhole_data_manager
#         self.drillhole_visualizer = drillhole_visualizer
#         self.item_manager = item_manager
#         self.logger = logging.getLogger(__name__)

#         self.dialog = None
#         self.generator = PDFReportGenerator()

#         # Export settings
#         self.export_source = tk.StringVar(value="displayed")  # "displayed" or "filtered"
#         self.page_size = tk.StringVar(value="A4 Portrait")
#         self.images_per_row = tk.IntVar(value=3)
#         self.group_by_hole = tk.BooleanVar(value=True)
#         self.hole_spacing = tk.IntVar(value=20)
#         self.image_spacing = tk.IntVar(value=5)

#         # Label settings
#         self.label_font_size = tk.IntVar(value=9)
#         self.label_bg_color = tk.StringVar(value="#333333")
#         self.show_depth_labels = tk.BooleanVar(value=True)
#         self.show_classification_labels = tk.BooleanVar(value=True)
#         self.show_hole_headers = tk.BooleanVar(value=True)

#         # Visualization settings
#         self.include_visualizations = tk.BooleanVar(value=True)
#         self.viz_width = tk.IntVar(value=40)

#         # Document settings
#         self.report_title = tk.StringVar(value="Geological Image Report")
#         self.show_page_numbers = tk.BooleanVar(value=True)
#         self.show_date = tk.BooleanVar(value=True)

#         # Filtering settings
#         self.filter_by_intervals = tk.BooleanVar(value=False)
#         self.intervals_text = tk.StringVar(value="")
#         self.filter_by_tags = tk.BooleanVar(value=False)
#         self.selected_tags = set()
#         self.filter_by_classification = tk.BooleanVar(value=False)
#         self.selected_classifications = set()
#         self.sort_by = tk.StringVar(value="hole_depth")  # "hole_depth", "classification", "tags"

#         # Preview data
#         self.filtered_images = None
#         self.preview_stats = {}

#     def show(self):
#         """Show the export dialog"""
#         self.logger.info("Opening PDF export dialog")
#         self._create_dialog()
#         self._update_preview()

#     def _create_dialog(self):
#         """Create the dialog window"""
#         self.dialog = tk.Toplevel(self.parent)
#         self.dialog.title("Export to PDF")
#         self.dialog.geometry("800x700")
#         self.dialog.configure(bg=self.gui_manager.theme_colors["background"])

#         # Make modal
#         self.dialog.transient(self.parent)
#         self.dialog.grab_set()

#         # Main container with scrollbar
#         main_canvas = tk.Canvas(
#             self.dialog,
#             bg=self.gui_manager.theme_colors["background"],
#             highlightthickness=0,
#         )
#         scrollbar = ttk.Scrollbar(self.dialog, orient="vertical", command=main_canvas.yview)
#         scrollable_frame = ttk.Frame(main_canvas)

#         scrollable_frame.bind(
#             "<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
#         )

#         main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
#         main_canvas.configure(yscrollcommand=scrollbar.set)

#         main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
#         scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

#         # Enable mousewheel scrolling
#         def _on_mousewheel(event):
#             main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

#         self.dialog.bind("<MouseWheel>", _on_mousewheel)
#         main_canvas.bind("<MouseWheel>", _on_mousewheel)

#         # Title
#         title_label = ttk.Label(
#             scrollable_frame,
#             text="Export Image Grid to PDF",
#             font=("Arial", 14, "bold"),
#         )
#         title_label.pack(pady=(0, 15))

#         # === Source Selection ===
#         source_frame = CollapsibleFrame(
#             scrollable_frame, "1. Image Source", expanded=True
#         )
#         source_frame.pack(fill=tk.X, pady=5)

#         source_content = source_frame.content_frame
#         ttk.Radiobutton(
#             source_content,
#             text=f"Current Display ({len(self.displayed_images)} images)",
#             variable=self.export_source,
#             value="displayed",
#             command=self._update_preview,
#         ).pack(anchor=tk.W, pady=2)

#         ttk.Radiobutton(
#             source_content,
#             text="Apply Custom Filters (see below)",
#             variable=self.export_source,
#             value="filtered",
#             command=self._update_preview,
#         ).pack(anchor=tk.W, pady=2)

#         # === Filtering Section ===
#         filter_frame = CollapsibleFrame(
#             scrollable_frame, "2. Filters (Optional)", expanded=True
#         )
#         filter_frame.pack(fill=tk.X, pady=5)

#         filter_content = filter_frame.content_frame

#         # Multi-interval filter
#         interval_check = ttk.Checkbutton(
#             filter_content,
#             text="Filter by Drillhole Intervals",
#             variable=self.filter_by_intervals,
#             command=self._update_preview,
#         )
#         interval_check.pack(anchor=tk.W, pady=(5, 2))

#         ttk.Label(
#             filter_content,
#             text="Enter intervals (one per line): HoleID From-To",
#             font=("Arial", 8, "italic"),
#         ).pack(anchor=tk.W, padx=20)

#         ttk.Label(
#             filter_content,
#             text="Example: GAB24-001 50-100 or GAB24-001 50-100, GAB24-002 20-50",
#             font=("Arial", 8, "italic"),
#         ).pack(anchor=tk.W, padx=20)

#         intervals_text = tk.Text(filter_content, height=4, width=60)
#         intervals_text.pack(padx=20, pady=2, fill=tk.X)
#         intervals_text.bind("<<Modified>>", lambda e: self._on_intervals_changed(intervals_text))

#         # Classification filter
#         class_check = ttk.Checkbutton(
#             filter_content,
#             text="Filter by Classification",
#             variable=self.filter_by_classification,
#             command=self._update_preview,
#         )
#         class_check.pack(anchor=tk.W, pady=(10, 2))

#         class_container = ttk.Frame(filter_content)
#         class_container.pack(fill=tk.X, padx=20, pady=2)

#         if self.item_manager:
#             classifications = self.item_manager.get_active_classifications()
#             for i, class_def in enumerate(classifications):
#                 var = tk.BooleanVar(value=False)
#                 cb = ttk.Checkbutton(
#                     class_container,
#                     text=class_def.label,
#                     variable=var,
#                     command=lambda cid=class_def.id, v=var: self._on_classification_toggled(
#                         cid, v
#                     ),
#                 )
#                 cb.grid(row=i // 3, column=i % 3, sticky=tk.W, padx=5)

#         # Tag filter
#         tag_check = ttk.Checkbutton(
#             filter_content,
#             text="Filter by Tags (include images with ANY selected tag)",
#             variable=self.filter_by_tags,
#             command=self._update_preview,
#         )
#         tag_check.pack(anchor=tk.W, pady=(10, 2))

#         tag_container = ttk.Frame(filter_content)
#         tag_container.pack(fill=tk.X, padx=20, pady=2)

#         if self.item_manager:
#             tags = self.item_manager.get_active_tags()
#             for i, tag_def in enumerate(tags):
#                 var = tk.BooleanVar(value=False)
#                 cb = ttk.Checkbutton(
#                     tag_container,
#                     text=f"{tag_def.icon} {tag_def.label}" if tag_def.icon else tag_def.label,
#                     variable=var,
#                     command=lambda tid=tag_def.id, v=var: self._on_tag_toggled(tid, v),
#                 )
#                 cb.grid(row=i // 3, column=i % 3, sticky=tk.W, padx=5)

#         # Sort options
#         ttk.Label(filter_content, text="Sort By:", font=("Arial", 9, "bold")).pack(
#             anchor=tk.W, pady=(10, 2)
#         )

#         sort_frame = ttk.Frame(filter_content)
#         sort_frame.pack(fill=tk.X, padx=20)

#         ttk.Radiobutton(
#             sort_frame,
#             text="Hole ID & Depth",
#             variable=self.sort_by,
#             value="hole_depth",
#             command=self._update_preview,
#         ).pack(side=tk.LEFT, padx=5)

#         ttk.Radiobutton(
#             sort_frame,
#             text="Classification",
#             variable=self.sort_by,
#             value="classification",
#             command=self._update_preview,
#         ).pack(side=tk.LEFT, padx=5)

#         ttk.Radiobutton(
#             sort_frame,
#             text="Tag Count",
#             variable=self.sort_by,
#             value="tags",
#             command=self._update_preview,
#         ).pack(side=tk.LEFT, padx=5)

#         # === Layout Settings ===
#         layout_frame = CollapsibleFrame(
#             scrollable_frame, "3. Layout Settings", expanded=True
#         )
#         layout_frame.pack(fill=tk.X, pady=5)

#         layout_content = layout_frame.content_frame

#         # Page size
#         ttk.Label(layout_content, text="Page Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
#         page_combo = ttk.Combobox(
#             layout_content,
#             textvariable=self.page_size,
#             values=list(PDFReportGenerator.PAGE_SIZES.keys()),
#             state="readonly",
#             width=20,
#         )
#         page_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

#         # Images per row
#         ttk.Label(layout_content, text="Images per Row:").grid(
#             row=1, column=0, sticky=tk.W, pady=2
#         )
#         ttk.Spinbox(
#             layout_content,
#             from_=1,
#             to=6,
#             textvariable=self.images_per_row,
#             width=10,
#         ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

#         # Group by hole
#         ttk.Checkbutton(
#             layout_content,
#             text="Group by Drillhole",
#             variable=self.group_by_hole,
#         ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)

#         # Spacing
#         ttk.Label(layout_content, text="Hole Spacing (mm):").grid(
#             row=3, column=0, sticky=tk.W, pady=2
#         )
#         ttk.Spinbox(
#             layout_content,
#             from_=0,
#             to=50,
#             textvariable=self.hole_spacing,
#             width=10,
#         ).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

#         ttk.Label(layout_content, text="Image Spacing (mm):").grid(
#             row=4, column=0, sticky=tk.W, pady=2
#         )
#         ttk.Spinbox(
#             layout_content,
#             from_=0,
#             to=20,
#             textvariable=self.image_spacing,
#             width=10,
#         ).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)

#         # === Label Settings ===
#         label_settings_frame = CollapsibleFrame(
#             scrollable_frame, "4. Label Settings", expanded=False
#         )
#         label_settings_frame.pack(fill=tk.X, pady=5)

#         label_content = label_settings_frame.content_frame

#         ttk.Label(label_content, text="Font Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
#         ttk.Spinbox(
#             label_content,
#             from_=6,
#             to=16,
#             textvariable=self.label_font_size,
#             width=10,
#         ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

#         ttk.Label(label_content, text="Background Color:").grid(
#             row=1, column=0, sticky=tk.W, pady=2
#         )
#         color_frame = ttk.Frame(label_content)
#         color_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

#         color_entry = ttk.Entry(color_frame, textvariable=self.label_bg_color, width=10)
#         color_entry.pack(side=tk.LEFT, padx=(0, 5))

#         ModernButton(
#             color_frame,
#             text="Pick",
#             color="#6c757d",
#             command=self._pick_color,
#             theme_colors=self.gui_manager.theme_colors,
#         ).pack(side=tk.LEFT)

#         ttk.Checkbutton(
#             label_content, text="Show Depth Labels", variable=self.show_depth_labels
#         ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)

#         ttk.Checkbutton(
#             label_content,
#             text="Show Classification Labels",
#             variable=self.show_classification_labels,
#         ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

#         ttk.Checkbutton(
#             label_content, text="Show Hole Headers", variable=self.show_hole_headers
#         ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)

#         # === Visualization Settings ===
#         viz_frame = CollapsibleFrame(
#             scrollable_frame, "5. Data Visualizations", expanded=False
#         )
#         viz_frame.pack(fill=tk.X, pady=5)

#         viz_content = viz_frame.content_frame

#         ttk.Checkbutton(
#             viz_content,
#             text="Include Data Visualizations",
#             variable=self.include_visualizations,
#         ).pack(anchor=tk.W, pady=2)

#         viz_width_frame = ttk.Frame(viz_content)
#         viz_width_frame.pack(fill=tk.X, pady=2)

#         ttk.Label(viz_width_frame, text="Visualization Width (mm):").pack(side=tk.LEFT)
#         ttk.Spinbox(
#             viz_width_frame,
#             from_=20,
#             to=100,
#             textvariable=self.viz_width,
#             width=10,
#         ).pack(side=tk.LEFT, padx=5)

#         # === Document Settings ===
#         doc_frame = CollapsibleFrame(
#             scrollable_frame, "6. Document Settings", expanded=False
#         )
#         doc_frame.pack(fill=tk.X, pady=5)

#         doc_content = doc_frame.content_frame

#         ttk.Label(doc_content, text="Report Title:").pack(anchor=tk.W, pady=2)
#         ttk.Entry(doc_content, textvariable=self.report_title, width=50).pack(
#             fill=tk.X, pady=2
#         )

#         ttk.Checkbutton(
#             doc_content, text="Show Page Numbers", variable=self.show_page_numbers
#         ).pack(anchor=tk.W, pady=2)

#         ttk.Checkbutton(doc_content, text="Show Date", variable=self.show_date).pack(
#             anchor=tk.W, pady=2
#         )

#         # === Preview ===
#         preview_frame = ttk.LabelFrame(scrollable_frame, text="Preview", padding=10)
#         preview_frame.pack(fill=tk.X, pady=5)

#         self.preview_label = ttk.Label(
#             preview_frame,
#             text="Calculating preview...",
#             font=("Arial", 9),
#             justify=tk.LEFT,
#         )
#         self.preview_label.pack(anchor=tk.W)

#         # === Action Buttons ===
#         button_frame = ttk.Frame(scrollable_frame)
#         button_frame.pack(fill=tk.X, pady=15)

#         ModernButton(
#             button_frame,
#             text="Export to PDF",
#             color="#28a745",
#             command=self._export_pdf,
#             theme_colors=self.gui_manager.theme_colors,
#         ).pack(side=tk.LEFT, padx=5)

#         ModernButton(
#             button_frame,
#             text="Cancel",
#             color="#6c757d",
#             command=self.dialog.destroy,
#             theme_colors=self.gui_manager.theme_colors,
#         ).pack(side=tk.LEFT, padx=5)

#     def _on_intervals_changed(self, text_widget):
#         """Handle interval text changes"""
#         if text_widget.edit_modified():
#             self.intervals_text.set(text_widget.get("1.0", tk.END))
#             text_widget.edit_modified(False)
#             self._update_preview()

#     def _on_tag_toggled(self, tag_id: str, var: tk.BooleanVar):
#         """Handle tag checkbox toggle"""
#         if var.get():
#             self.selected_tags.add(tag_id)
#         else:
#             self.selected_tags.discard(tag_id)
#         self._update_preview()

#     def _on_classification_toggled(self, class_id: str, var: tk.BooleanVar):
#         """Handle classification checkbox toggle"""
#         if var.get():
#             self.selected_classifications.add(class_id)
#         else:
#             self.selected_classifications.discard(class_id)
#         self._update_preview()

#     def _pick_color(self):
#         """Pick label background color"""
#         from tkinter import colorchooser

#         color = colorchooser.askcolor(
#             color=self.label_bg_color.get(),
#             title="Choose Label Background Color",
#             parent=self.dialog,
#         )
#         if color[1]:
#             self.label_bg_color.set(color[1])

#     def _update_preview(self):
#         """Update preview statistics"""
#         try:
#             # Apply filters to get final image list
#             if self.export_source.get() == "displayed":
#                 self.filtered_images = self.displayed_images.copy()
#             else:
#                 self.filtered_images = self._apply_filters(self.all_images)

#             # Apply sorting
#             self.filtered_images = self._apply_sorting(self.filtered_images)

#             # Calculate statistics
#             total_images = len(self.filtered_images)
#             hole_count = len(set(img.hole_id for img in self.filtered_images))

#             # Classification breakdown
#             class_counts = {}
#             for img in self.filtered_images:
#                 class_name = (
#                     img.classification.value
#                     if hasattr(img.classification, "value")
#                     else str(img.classification)
#                 )
#                 class_counts[class_name] = class_counts.get(class_name, 0) + 1

#             # Tag breakdown
#             tag_counts = {}
#             for img in self.filtered_images:
#                 if hasattr(img, "tags"):
#                     for tag in img.tags:
#                         tag_counts[tag] = tag_counts.get(tag, 0) + 1

#             # Update preview label
#             preview_text = f"📊 Export Preview:\n"
#             preview_text += f"  • Total Images: {total_images}\n"
#             preview_text += f"  • Drillholes: {hole_count}\n"

#             if class_counts:
#                 preview_text += f"  • Classifications:\n"
#                 for class_name, count in sorted(class_counts.items()):
#                     preview_text += f"      - {class_name}: {count}\n"

#             if tag_counts:
#                 preview_text += f"  • Tags:\n"
#                 for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[
#                     :5
#                 ]:
#                     preview_text += f"      - {tag}: {count}\n"

#             self.preview_label.config(text=preview_text)

#         except Exception as e:
#             self.logger.error(f"Error updating preview: {e}", exc_info=True)
#             self.preview_label.config(text=f"Preview error: {str(e)}")

#     def _apply_filters(self, images: List[Any]) -> List[Any]:
#         """Apply all selected filters to image list"""
#         filtered = images.copy()

#         # Filter by intervals
#         if self.filter_by_intervals.get() and self.intervals_text.get().strip():
#             filtered = self._filter_by_intervals(filtered)

#         # Filter by classification
#         if self.filter_by_classification.get() and self.selected_classifications:
#             filtered = [
#                 img
#                 for img in filtered
#                 if (
#                     hasattr(img.classification, "value")
#                     and img.classification.value in self.selected_classifications
#                 )
#                 or str(img.classification) in self.selected_classifications
#             ]

#         # Filter by tags
#         if self.filter_by_tags.get() and self.selected_tags:
#             filtered = [
#                 img
#                 for img in filtered
#                 if hasattr(img, "tags") and any(tag in img.tags for tag in self.selected_tags)
#             ]

#         return filtered

#     def _filter_by_intervals(self, images: List[Any]) -> List[Any]:
#         """Filter images by specified drillhole intervals"""
#         intervals_text = self.intervals_text.get().strip()
#         if not intervals_text:
#             return images

#         # Parse intervals
#         intervals = []
#         for line in intervals_text.split("\n"):
#             line = line.strip()
#             if not line:
#                 continue

#             # Try to parse: HoleID From-To or HoleID From To
#             # Support multiple intervals per line separated by commas
#             for segment in line.split(","):
#                 segment = segment.strip()
#                 parts = segment.split()

#                 if len(parts) >= 2:
#                     hole_id = parts[0]

#                     # Parse depth range (could be "50-100" or "50 100")
#                     depth_str = parts[1] if len(parts) == 2 else " ".join(parts[1:])

#                     if "-" in depth_str:
#                         try:
#                             from_depth, to_depth = map(float, depth_str.split("-"))
#                             intervals.append((hole_id, from_depth, to_depth))
#                         except ValueError:
#                             self.logger.warning(f"Invalid interval format: {segment}")
#                     elif len(parts) == 3:
#                         try:
#                             from_depth = float(parts[1])
#                             to_depth = float(parts[2])
#                             intervals.append((hole_id, from_depth, to_depth))
#                         except ValueError:
#                             self.logger.warning(f"Invalid depth values: {segment}")

#         if not intervals:
#             return images

#         # Filter images
#         filtered = []
#         for img in images:
#             for hole_id, from_depth, to_depth in intervals:
#                 if img.hole_id == hole_id:
#                     # Check if image overlaps with interval
#                     if (
#                         img.depth_from <= to_depth
#                         and img.depth_to >= from_depth
#                     ):
#                         filtered.append(img)
#                         break

#         return filtered

#     def _apply_sorting(self, images: List[Any]) -> List[Any]:
#         """Sort images based on selected criteria"""
#         if self.sort_by.get() == "hole_depth":
#             return sorted(images, key=lambda img: (img.hole_id, img.depth_from))
#         elif self.sort_by.get() == "classification":
#             return sorted(
#                 images,
#                 key=lambda img: (
#                     img.classification.value
#                     if hasattr(img.classification, "value")
#                     else str(img.classification),
#                     img.hole_id,
#                     img.depth_from,
#                 ),
#             )
#         elif self.sort_by.get() == "tags":
#             return sorted(
#                 images,
#                 key=lambda img: (
#                     -len(img.tags) if hasattr(img, "tags") else 0,
#                     img.hole_id,
#                     img.depth_from,
#                 ),
#             )
#         else:
#             return images

#     def _export_pdf(self):
#         """Export to PDF file"""
#         try:
#             # Check if there are images to export
#             if not self.filtered_images:
#                 DialogHelper.show_message(
#                     self.dialog,
#                     "No Images",
#                     "No images to export. Please adjust your filters.",
#                 )
#                 return

#             # Ask for save location
#             default_filename = f"geological_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
#             output_path = filedialog.asksaveasfilename(
#                 parent=self.dialog,
#                 title="Save PDF Report",
#                 defaultextension=".pdf",
#                 filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
#                 initialfile=default_filename,
#             )

#             if not output_path:
#                 return

#             # Show progress dialog
#             progress = ProgressDialog(
#                 self.dialog,
#                 "Exporting PDF",
#                 "Generating PDF report...",
#             )

#             try:
#                 # Configure generator
#                 self.generator.configure(
#                     page_size_name=self.page_size.get(),
#                     images_per_row=self.images_per_row.get(),
#                     group_by_hole=self.group_by_hole.get(),
#                     hole_spacing=self.hole_spacing.get(),
#                     image_spacing=self.image_spacing.get(),
#                     label_font_size=self.label_font_size.get(),
#                     label_bg_color=self.label_bg_color.get(),
#                     show_depth_labels=self.show_depth_labels.get(),
#                     show_classification_labels=self.show_classification_labels.get(),
#                     show_hole_headers=self.show_hole_headers.get(),
#                     include_visualizations=self.include_visualizations.get(),
#                     viz_width=self.viz_width.get(),
#                     title=self.report_title.get(),
#                     show_page_numbers=self.show_page_numbers.get(),
#                     show_date=self.show_date.get(),
#                 )

#                 # Generate PDF
#                 success = self.generator.generate_report(
#                     output_path,
#                     self.filtered_images,
#                     self.drillhole_data_manager,
#                     self.drillhole_visualizer,
#                 )

#                 progress.close()

#                 if success:
#                     DialogHelper.show_message(
#                         self.dialog,
#                         "Export Complete",
#                         f"PDF report exported successfully to:\n{output_path}",
#                     )
#                     self.dialog.destroy()
#                 else:
#                     DialogHelper.show_message(
#                         self.dialog,
#                         "Export Failed",
#                         "Failed to generate PDF report. Check logs for details.",
#                     )

#             except Exception as e:
#                 progress.close()
#                 raise

#         except Exception as e:
#             self.logger.error(f"Error exporting PDF: {e}", exc_info=True)
#             DialogHelper.show_message(
#                 self.dialog,
#                 "Export Error",
#                 f"An error occurred during export:\n{str(e)}",
#             )


# src/gui/pdf_export_dialog.py
"""
PDF Export Dialog for Logging Review
Provides advanced filtering and export customization.

Enhanced Features:
- Export mode selection (Grid vs Strip Log)
- Dynamic page sizing
- Strip log with context intervals
- Customizable label fields
- Collar location minimap
"""

import os
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Set, Optional, Any, Dict, Tuple
from datetime import datetime

from gui.widgets.modern_button import ModernButton
from gui.widgets.collapsible_frame import CollapsibleFrame
from gui.dialog_helper import DialogHelper
from gui.progress_dialog import ProgressDialog
from processing.pdf_report_generator import PDFReportGenerator


class PDFExportDialog:
    """Dialog for exporting image grid to PDF with filtering and customization"""

    def __init__(
        self,
        parent,
        all_images: List[Any],
        displayed_images: List[Any],
        gui_manager,
        drillhole_data_manager=None,
        drillhole_visualizer=None,
        item_manager=None,
        data_coordinator=None,
    ):
        """
        Initialize PDF export dialog

        Args:
            parent: Parent window
            all_images: Complete list of available images
            displayed_images: Currently displayed/filtered images
            gui_manager: GUI manager for theming
            drillhole_data_manager: Data manager for CSV data
            drillhole_visualizer: Visualizer for data plots
            item_manager: Classification and tag manager
            data_coordinator: DataCoordinator for collar data access
        """
        self.parent = parent
        self.all_images = all_images
        self.displayed_images = displayed_images
        self.gui_manager = gui_manager
        self.drillhole_data_manager = drillhole_data_manager
        self.drillhole_visualizer = drillhole_visualizer
        self.item_manager = item_manager
        self.data_coordinator = data_coordinator
        self.logger = logging.getLogger(__name__)

        self.dialog = None
        self.generator = PDFReportGenerator()

        # Export mode
        self.export_mode = tk.StringVar(value="strip_log")  # "grid" or "strip_log"

        # Export settings
        self.export_source = tk.StringVar(value="displayed")  # "displayed" or "filtered"
        self.page_size = tk.StringVar(value="Dynamic (Fit Content)")
        self.images_per_row = tk.IntVar(value=3)
        self.group_by_hole = tk.BooleanVar(value=True)
        self.one_hole_per_page = tk.BooleanVar(value=True)  # Each drillhole on its own page
        self.spatial_grouping = tk.BooleanVar(value=True)  # Group nearby holes together
        self.hole_spacing = tk.IntVar(value=20)
        self.image_spacing = tk.IntVar(value=5)

        # Strip log settings
        self.strip_log_image_width = tk.IntVar(value=60)  # Compact image width
        self.strip_log_context_meters = tk.DoubleVar(value=10.0)
        self.strip_log_cell_height = tk.IntVar(value=20)  # Tighter packing

        # Minimap settings
        self.show_minimap = tk.BooleanVar(value=True)
        self.minimap_width = tk.IntVar(value=80)  # Compact minimap
        self.minimap_height = tk.IntVar(value=60)

        # Label settings
        self.label_font_size = tk.IntVar(value=9)
        self.label_bg_color = tk.StringVar(value="#333333")
        self.show_depth_labels = tk.BooleanVar(value=True)
        self.show_classification_labels = tk.BooleanVar(value=True)
        self.show_hole_headers = tk.BooleanVar(value=True)
        
        # Extended label fields
        self.label_field_depth = tk.BooleanVar(value=True)
        self.label_field_classification = tk.BooleanVar(value=True)
        self.label_field_tags = tk.BooleanVar(value=True)
        self.label_field_comments = tk.BooleanVar(value=True)  # Show comments by default
        self.label_field_consensus = tk.BooleanVar(value=False)
        self.label_field_review_count = tk.BooleanVar(value=False)
        self.label_csv_columns = tk.StringVar(value="")  # Comma-separated

        # Visualization settings
        self.include_visualizations = tk.BooleanVar(value=True)
        self.viz_width = tk.IntVar(value=15)  # Narrow columns like GUI

        # Document settings
        self.report_title = tk.StringVar(value="Geological Image Report")
        self.show_page_numbers = tk.BooleanVar(value=True)
        self.show_date = tk.BooleanVar(value=True)

        # Filtering settings
        self.filter_by_intervals = tk.BooleanVar(value=False)
        self.intervals_text = tk.StringVar(value="")
        self.filter_by_tags = tk.BooleanVar(value=False)
        self.selected_tags = set()
        self.filter_by_classification = tk.BooleanVar(value=False)
        self.selected_classifications = set()
        self.sort_by = tk.StringVar(value="hole_depth")

        # Preview data
        self.filtered_images = None
        self.preview_stats = {}
        self.selected_intervals_dict = {}  # {hole_id: (from, to)}

    def show(self):
        """Show the export dialog"""
        self.logger.info("Opening PDF export dialog")
        self._create_dialog()
        self._update_preview()

    def _create_dialog(self):
        """Create the dialog window"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Export to PDF")
        self.dialog.geometry("900x800")
        self.dialog.configure(bg=self.gui_manager.theme_colors["background"])

        # Make modal
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Main container with scrollbar
        main_canvas = tk.Canvas(
            self.dialog,
            bg=self.gui_manager.theme_colors["background"],
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(self.dialog, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )

        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.dialog.bind("<MouseWheel>", _on_mousewheel)
        main_canvas.bind("<MouseWheel>", _on_mousewheel)

        # Title
        title_label = ttk.Label(
            scrollable_frame,
            text="Export Image Grid to PDF",
            font=("Arial", 14, "bold"),
        )
        title_label.pack(pady=(0, 15))

        # === Export Mode Selection ===
        mode_frame = CollapsibleFrame(
            scrollable_frame, "1. Export Mode", expanded=True
        )
        mode_frame.pack(fill=tk.X, pady=5)

        mode_content = mode_frame.content_frame
        
        ttk.Radiobutton(
            mode_content,
            text="Strip Log (Continuous depth view with context)",
            variable=self.export_mode,
            value="strip_log",
            command=self._on_mode_changed,
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            mode_content,
            text="Grid (Traditional image grid layout)",
            variable=self.export_mode,
            value="grid",
            command=self._on_mode_changed,
        ).pack(anchor=tk.W, pady=2)
        
        # Mode description
        self.mode_description = ttk.Label(
            mode_content,
            text="Strip Log: Shows continuous depth with images and data columns. "
                 "Each drillhole on its own dynamically-sized page.",
            font=("Arial", 8, "italic"),
            foreground="gray",
            wraplength=600,
        )
        self.mode_description.pack(anchor=tk.W, pady=(5, 0), padx=20)

        # === Source Selection ===
        source_frame = CollapsibleFrame(
            scrollable_frame, "2. Image Source", expanded=True
        )
        source_frame.pack(fill=tk.X, pady=5)

        source_content = source_frame.content_frame
        ttk.Radiobutton(
            source_content,
            text=f"Current Display ({len(self.displayed_images)} images)",
            variable=self.export_source,
            value="displayed",
            command=self._update_preview,
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            source_content,
            text="Apply Custom Filters (see below)",
            variable=self.export_source,
            value="filtered",
            command=self._update_preview,
        ).pack(anchor=tk.W, pady=2)

        # === Filtering Section ===
        filter_frame = CollapsibleFrame(
            scrollable_frame, "3. Filters (Optional)", expanded=False
        )
        filter_frame.pack(fill=tk.X, pady=5)

        filter_content = filter_frame.content_frame

        # Multi-interval filter
        interval_check = ttk.Checkbutton(
            filter_content,
            text="Filter by Drillhole Intervals",
            variable=self.filter_by_intervals,
            command=self._update_preview,
        )
        interval_check.pack(anchor=tk.W, pady=(5, 2))

        ttk.Label(
            filter_content,
            text="Enter intervals (one per line): HoleID From-To",
            font=("Arial", 8, "italic"),
        ).pack(anchor=tk.W, padx=20)

        ttk.Label(
            filter_content,
            text="Example: GAB24-001 50-100 or GAB24-001 50-100, GAB24-002 20-50",
            font=("Arial", 8, "italic"),
        ).pack(anchor=tk.W, padx=20)

        self.intervals_text_widget = tk.Text(filter_content, height=4, width=60)
        self.intervals_text_widget.pack(padx=20, pady=2, fill=tk.X)
        self.intervals_text_widget.bind("<<Modified>>", lambda e: self._on_intervals_changed(self.intervals_text_widget))

        # Classification filter
        class_check = ttk.Checkbutton(
            filter_content,
            text="Filter by Classification",
            variable=self.filter_by_classification,
            command=self._update_preview,
        )
        class_check.pack(anchor=tk.W, pady=(10, 2))

        class_container = ttk.Frame(filter_content)
        class_container.pack(fill=tk.X, padx=20, pady=2)

        if self.item_manager:
            classifications = self.item_manager.get_active_classifications()
            for i, class_def in enumerate(classifications):
                var = tk.BooleanVar(value=False)
                cb = ttk.Checkbutton(
                    class_container,
                    text=class_def.label,
                    variable=var,
                    command=lambda cid=class_def.id, v=var: self._on_classification_toggled(
                        cid, v
                    ),
                )
                cb.grid(row=i // 3, column=i % 3, sticky=tk.W, padx=5)

        # Tag filter
        tag_check = ttk.Checkbutton(
            filter_content,
            text="Filter by Tags (include images with ANY selected tag)",
            variable=self.filter_by_tags,
            command=self._update_preview,
        )
        tag_check.pack(anchor=tk.W, pady=(10, 2))

        tag_container = ttk.Frame(filter_content)
        tag_container.pack(fill=tk.X, padx=20, pady=2)

        if self.item_manager:
            tags = self.item_manager.get_active_tags()
            for i, tag_def in enumerate(tags):
                var = tk.BooleanVar(value=False)
                cb = ttk.Checkbutton(
                    tag_container,
                    text=f"{tag_def.icon} {tag_def.label}" if tag_def.icon else tag_def.label,
                    variable=var,
                    command=lambda tid=tag_def.id, v=var: self._on_tag_toggled(tid, v),
                )
                cb.grid(row=i // 3, column=i % 3, sticky=tk.W, padx=5)

        # Sort options
        ttk.Label(filter_content, text="Sort By:", font=("Arial", 9, "bold")).pack(
            anchor=tk.W, pady=(10, 2)
        )

        sort_frame = ttk.Frame(filter_content)
        sort_frame.pack(fill=tk.X, padx=20)

        ttk.Radiobutton(
            sort_frame,
            text="Hole ID & Depth",
            variable=self.sort_by,
            value="hole_depth",
            command=self._update_preview,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            sort_frame,
            text="Classification",
            variable=self.sort_by,
            value="classification",
            command=self._update_preview,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            sort_frame,
            text="Tag Count",
            variable=self.sort_by,
            value="tags",
            command=self._update_preview,
        ).pack(side=tk.LEFT, padx=5)

        # === Strip Log Settings ===
        self.strip_log_frame = CollapsibleFrame(
            scrollable_frame, "4. Strip Log Settings", expanded=True
        )
        self.strip_log_frame.pack(fill=tk.X, pady=5)

        strip_content = self.strip_log_frame.content_frame

        # Context meters
        context_frame = ttk.Frame(strip_content)
        context_frame.pack(fill=tk.X, pady=2)
        ttk.Label(context_frame, text="Context Above/Below (m):").pack(side=tk.LEFT)
        ttk.Spinbox(
            context_frame,
            from_=0,
            to=50,
            textvariable=self.strip_log_context_meters,
            width=10,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(
            context_frame,
            text="Meters of images to include above/below selected intervals",
            font=("Arial", 8, "italic"),
        ).pack(side=tk.LEFT, padx=10)

        # Image width
        width_frame = ttk.Frame(strip_content)
        width_frame.pack(fill=tk.X, pady=2)
        ttk.Label(width_frame, text="Image Width (mm):").pack(side=tk.LEFT)
        ttk.Spinbox(
            width_frame,
            from_=40,
            to=150,
            textvariable=self.strip_log_image_width,
            width=10,
        ).pack(side=tk.LEFT, padx=5)

        # Minimap settings
        ttk.Separator(strip_content, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Checkbutton(
            strip_content,
            text="Show Collar Location Minimap",
            variable=self.show_minimap,
        ).pack(anchor=tk.W, pady=2)

        minimap_size_frame = ttk.Frame(strip_content)
        minimap_size_frame.pack(fill=tk.X, pady=2, padx=20)
        
        ttk.Label(minimap_size_frame, text="Minimap Width (mm):").pack(side=tk.LEFT)
        ttk.Spinbox(
            minimap_size_frame,
            from_=80,
            to=250,
            textvariable=self.minimap_width,
            width=8,
        ).pack(side=tk.LEFT, padx=5)
        
        # Spatial grouping option
        ttk.Separator(strip_content, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Checkbutton(
            strip_content,
            text="Group Drillholes Spatially (nearby holes together)",
            variable=self.spatial_grouping,
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Label(
            strip_content,
            text="Uses collar coordinates to sort holes by proximity rather than alphabetically",
            font=("Arial", 8, "italic"),
            foreground="gray",
        ).pack(anchor=tk.W, padx=20)

        # === Grid Layout Settings (shown only for grid mode) ===
        self.grid_layout_frame = CollapsibleFrame(
            scrollable_frame, "4. Grid Layout Settings", expanded=True
        )
        self.grid_layout_frame.pack(fill=tk.X, pady=5)
        self.grid_layout_frame.pack_forget()  # Hidden by default

        grid_content = self.grid_layout_frame.content_frame

        # Page size
        ttk.Label(grid_content, text="Page Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        page_combo = ttk.Combobox(
            grid_content,
            textvariable=self.page_size,
            values=list(PDFReportGenerator.PAGE_SIZES.keys()),
            state="readonly",
            width=20,
        )
        page_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Images per row
        ttk.Label(grid_content, text="Images per Row:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        ttk.Spinbox(
            grid_content,
            from_=1,
            to=6,
            textvariable=self.images_per_row,
            width=10,
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Group by hole
        ttk.Checkbutton(
            grid_content,
            text="Group by Drillhole",
            variable=self.group_by_hole,
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)

        # One hole per page
        ttk.Checkbutton(
            grid_content,
            text="One Drillhole per Page",
            variable=self.one_hole_per_page,
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Spacing
        ttk.Label(grid_content, text="Hole Spacing (mm):").grid(
            row=4, column=0, sticky=tk.W, pady=2
        )
        ttk.Spinbox(
            grid_content,
            from_=0,
            to=50,
            textvariable=self.hole_spacing,
            width=10,
        ).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(grid_content, text="Image Spacing (mm):").grid(
            row=5, column=0, sticky=tk.W, pady=2
        )
        ttk.Spinbox(
            grid_content,
            from_=0,
            to=20,
            textvariable=self.image_spacing,
            width=10,
        ).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)

        # === Label Settings ===
        label_settings_frame = CollapsibleFrame(
            scrollable_frame, "5. Label Settings", expanded=False
        )
        label_settings_frame.pack(fill=tk.X, pady=5)

        label_content = label_settings_frame.content_frame

        ttk.Label(label_content, text="Font Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(
            label_content,
            from_=6,
            to=16,
            textvariable=self.label_font_size,
            width=10,
        ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(label_content, text="Background Color:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        color_frame = ttk.Frame(label_content)
        color_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        color_entry = ttk.Entry(color_frame, textvariable=self.label_bg_color, width=10)
        color_entry.pack(side=tk.LEFT, padx=(0, 5))

        ModernButton(
            color_frame,
            text="Pick",
            color="#6c757d",
            command=self._pick_color,
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.LEFT)

        # Label field selection
        ttk.Separator(label_content, orient='horizontal').grid(
            row=2, column=0, columnspan=2, sticky='ew', pady=10
        )
        
        ttk.Label(
            label_content, 
            text="Label Fields to Include:", 
            font=("Arial", 9, "bold")
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        fields_frame = ttk.Frame(label_content)
        fields_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=10)

        ttk.Checkbutton(
            fields_frame, text="Depth Range", variable=self.label_field_depth
        ).grid(row=0, column=0, sticky=tk.W, padx=5)
        
        ttk.Checkbutton(
            fields_frame, text="Classification", variable=self.label_field_classification
        ).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Checkbutton(
            fields_frame, text="Tags", variable=self.label_field_tags
        ).grid(row=0, column=2, sticky=tk.W, padx=5)
        
        ttk.Checkbutton(
            fields_frame, text="Comments", variable=self.label_field_comments
        ).grid(row=1, column=0, sticky=tk.W, padx=5)
        
        ttk.Checkbutton(
            fields_frame, text="Consensus", variable=self.label_field_consensus
        ).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Checkbutton(
            fields_frame, text="Review Count", variable=self.label_field_review_count
        ).grid(row=1, column=2, sticky=tk.W, padx=5)

        # CSV columns
        csv_frame = ttk.Frame(label_content)
        csv_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)
        
        ttk.Label(csv_frame, text="CSV Columns:").pack(side=tk.LEFT)
        ttk.Entry(
            csv_frame, 
            textvariable=self.label_csv_columns, 
            width=40
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(
            csv_frame, 
            text="(comma-separated)", 
            font=("Arial", 8, "italic")
        ).pack(side=tk.LEFT)

        # === Visualization Settings ===
        viz_frame = CollapsibleFrame(
            scrollable_frame, "6. Data Visualizations", expanded=False
        )
        viz_frame.pack(fill=tk.X, pady=5)

        viz_content = viz_frame.content_frame

        ttk.Checkbutton(
            viz_content,
            text="Include Data Visualizations",
            variable=self.include_visualizations,
        ).pack(anchor=tk.W, pady=2)

        viz_width_frame = ttk.Frame(viz_content)
        viz_width_frame.pack(fill=tk.X, pady=2)

        ttk.Label(viz_width_frame, text="Visualization Width (mm):").pack(side=tk.LEFT)
        ttk.Spinbox(
            viz_width_frame,
            from_=20,
            to=100,
            textvariable=self.viz_width,
            width=10,
        ).pack(side=tk.LEFT, padx=5)

        # === Document Settings ===
        doc_frame = CollapsibleFrame(
            scrollable_frame, "7. Document Settings", expanded=False
        )
        doc_frame.pack(fill=tk.X, pady=5)

        doc_content = doc_frame.content_frame

        ttk.Label(doc_content, text="Report Title:").pack(anchor=tk.W, pady=2)
        ttk.Entry(doc_content, textvariable=self.report_title, width=50).pack(
            fill=tk.X, pady=2
        )

        ttk.Checkbutton(
            doc_content, text="Show Page Numbers", variable=self.show_page_numbers
        ).pack(anchor=tk.W, pady=2)

        ttk.Checkbutton(doc_content, text="Show Date", variable=self.show_date).pack(
            anchor=tk.W, pady=2
        )

        # === Preview ===
        preview_frame = ttk.LabelFrame(scrollable_frame, text="Preview", padding=10)
        preview_frame.pack(fill=tk.X, pady=5)

        self.preview_label = ttk.Label(
            preview_frame,
            text="Calculating preview...",
            font=("Arial", 9),
            justify=tk.LEFT,
        )
        self.preview_label.pack(anchor=tk.W)

        # === Action Buttons ===
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=15)

        ModernButton(
            button_frame,
            text="Export to PDF",
            color="#28a745",
            command=self._export_pdf,
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.LEFT, padx=5)

        ModernButton(
            button_frame,
            text="Cancel",
            color="#6c757d",
            command=self.dialog.destroy,
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.LEFT, padx=5)

        # Initial mode setup
        self._on_mode_changed()

    def _on_mode_changed(self):
        """Handle export mode change"""
        mode = self.export_mode.get()
        
        if mode == "strip_log":
            self.strip_log_frame.pack(fill=tk.X, pady=5, after=self.strip_log_frame.master.winfo_children()[3])
            self.grid_layout_frame.pack_forget()
            self.mode_description.config(
                text="Strip Log: Shows continuous depth with images and data columns. "
                     "Each drillhole on its own dynamically-sized page."
            )
        else:
            self.grid_layout_frame.pack(fill=tk.X, pady=5, after=self.strip_log_frame.master.winfo_children()[3])
            self.strip_log_frame.pack_forget()
            self.mode_description.config(
                text="Grid: Traditional grid layout with multiple images per row. "
                     "Uses standard page sizes (A4, Letter, etc.)."
            )
        
        self._update_preview()

    def _on_intervals_changed(self, text_widget):
        """Handle interval text changes"""
        if text_widget.edit_modified():
            self.intervals_text.set(text_widget.get("1.0", tk.END))
            text_widget.edit_modified(False)
            self._parse_intervals()
            self._update_preview()

    def _parse_intervals(self):
        """Parse interval text into selected_intervals_dict.
        
        For each hole, stores the MIN and MAX depth across all intervals,
        so that context can be calculated from the full selected range.
        """
        self.selected_intervals_dict = {}
        intervals_text = self.intervals_text.get().strip()
        
        if not intervals_text:
            return
        
        for line in intervals_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            for segment in line.split(","):
                segment = segment.strip()
                parts = segment.split()
                
                if len(parts) >= 2:
                    hole_id = parts[0]
                    depth_str = parts[1] if len(parts) == 2 else " ".join(parts[1:])
                    
                    from_depth = None
                    to_depth = None
                    
                    if "-" in depth_str:
                        try:
                            from_depth, to_depth = map(float, depth_str.split("-"))
                        except ValueError:
                            pass
                    elif len(parts) == 3:
                        try:
                            from_depth = float(parts[1])
                            to_depth = float(parts[2])
                        except ValueError:
                            pass
                    
                    # Merge with existing interval to get full range
                    if from_depth is not None and to_depth is not None:
                        if hole_id in self.selected_intervals_dict:
                            existing_from, existing_to = self.selected_intervals_dict[hole_id]
                            self.selected_intervals_dict[hole_id] = (
                                min(existing_from, from_depth),
                                max(existing_to, to_depth)
                            )
                        else:
                            self.selected_intervals_dict[hole_id] = (from_depth, to_depth)

    def _on_tag_toggled(self, tag_id: str, var: tk.BooleanVar):
        """Handle tag checkbox toggle"""
        if var.get():
            self.selected_tags.add(tag_id)
        else:
            self.selected_tags.discard(tag_id)
        self._update_preview()

    def _on_classification_toggled(self, class_id: str, var: tk.BooleanVar):
        """Handle classification checkbox toggle"""
        if var.get():
            self.selected_classifications.add(class_id)
        else:
            self.selected_classifications.discard(class_id)
        self._update_preview()

    def _pick_color(self):
        """Pick label background color"""
        from tkinter import colorchooser

        color = colorchooser.askcolor(
            color=self.label_bg_color.get(),
            title="Choose Label Background Color",
            parent=self.dialog,
        )
        if color[1]:
            self.label_bg_color.set(color[1])

    def _update_preview(self):
        """Update preview statistics"""
        try:
            # Apply filters to get final image list
            if self.export_source.get() == "displayed":
                self.filtered_images = self.displayed_images.copy()
            else:
                self.filtered_images = self._apply_filters(self.all_images)

            # Apply sorting
            self.filtered_images = self._apply_sorting(self.filtered_images)

            # Calculate statistics
            total_images = len(self.filtered_images)
            hole_count = len(set(img.hole_id for img in self.filtered_images))

            # Classification breakdown
            class_counts = {}
            for img in self.filtered_images:
                class_name = (
                    img.classification.value
                    if hasattr(img.classification, "value")
                    else str(img.classification)
                )
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Tag breakdown
            tag_counts = {}
            for img in self.filtered_images:
                if hasattr(img, "tags"):
                    for tag in img.tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Update preview label
            mode = self.export_mode.get()
            preview_text = f"📊 Export Preview ({mode.replace('_', ' ').title()}):\n"
            preview_text += f"  • Total Images: {total_images}\n"
            preview_text += f"  • Drillholes: {hole_count}\n"
            
            if mode == "strip_log":
                if self.selected_intervals_dict:
                    preview_text += f"  • Selected Depth Ranges: {len(self.selected_intervals_dict)} holes\n"
                    for hole_id, (from_d, to_d) in list(self.selected_intervals_dict.items())[:3]:
                        range_m = to_d - from_d
                        preview_text += f"      - {hole_id}: {from_d:.1f}m - {to_d:.1f}m ({range_m:.0f}m)\n"
                    if len(self.selected_intervals_dict) > 3:
                        preview_text += f"      - ... and {len(self.selected_intervals_dict) - 3} more\n"
                context = self.strip_log_context_meters.get()
                preview_text += f"  • Context: ±{context:.1f}m above/below\n"
                preview_text += f"  • Pages: ~{hole_count} (one per drillhole)\n"

            if class_counts:
                preview_text += f"  • Classifications:\n"
                for class_name, count in sorted(class_counts.items()):
                    preview_text += f"      - {class_name}: {count}\n"

            if tag_counts:
                preview_text += f"  • Tags:\n"
                for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[
                    :5
                ]:
                    preview_text += f"      - {tag}: {count}\n"

            self.preview_label.config(text=preview_text)

        except Exception as e:
            self.logger.error(f"Error updating preview: {e}", exc_info=True)
            self.preview_label.config(text=f"Preview error: {str(e)}")

    def _apply_filters(self, images: List[Any]) -> List[Any]:
        """Apply all selected filters to image list"""
        filtered = images.copy()

        # Filter by intervals
        if self.filter_by_intervals.get() and self.intervals_text.get().strip():
            filtered = self._filter_by_intervals(filtered)

        # Filter by classification
        if self.filter_by_classification.get() and self.selected_classifications:
            filtered = [
                img
                for img in filtered
                if (
                    hasattr(img.classification, "value")
                    and img.classification.value in self.selected_classifications
                )
                or str(img.classification) in self.selected_classifications
            ]

        # Filter by tags
        if self.filter_by_tags.get() and self.selected_tags:
            filtered = [
                img
                for img in filtered
                if hasattr(img, "tags") and any(tag in img.tags for tag in self.selected_tags)
            ]

        return filtered

    def _filter_by_intervals(self, images: List[Any]) -> List[Any]:
        """Filter images by specified drillhole intervals"""
        intervals_text = self.intervals_text.get().strip()
        if not intervals_text:
            return images

        # Parse intervals
        intervals = []
        for line in intervals_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            for segment in line.split(","):
                segment = segment.strip()
                parts = segment.split()

                if len(parts) >= 2:
                    hole_id = parts[0]
                    depth_str = parts[1] if len(parts) == 2 else " ".join(parts[1:])

                    if "-" in depth_str:
                        try:
                            from_depth, to_depth = map(float, depth_str.split("-"))
                            intervals.append((hole_id, from_depth, to_depth))
                        except ValueError:
                            self.logger.warning(f"Invalid interval format: {segment}")
                    elif len(parts) == 3:
                        try:
                            from_depth = float(parts[1])
                            to_depth = float(parts[2])
                            intervals.append((hole_id, from_depth, to_depth))
                        except ValueError:
                            self.logger.warning(f"Invalid depth values: {segment}")

        if not intervals:
            return images

        # Filter images
        filtered = []
        for img in images:
            for hole_id, from_depth, to_depth in intervals:
                if img.hole_id == hole_id:
                    if img.depth_from <= to_depth and img.depth_to >= from_depth:
                        filtered.append(img)
                        break

        return filtered

    def _apply_sorting(self, images: List[Any]) -> List[Any]:
        """Sort images based on selected criteria"""
        if self.sort_by.get() == "hole_depth":
            return sorted(images, key=lambda img: (img.hole_id, img.depth_from))
        elif self.sort_by.get() == "classification":
            return sorted(
                images,
                key=lambda img: (
                    img.classification.value
                    if hasattr(img.classification, "value")
                    else str(img.classification),
                    img.hole_id,
                    img.depth_from,
                ),
            )
        elif self.sort_by.get() == "tags":
            return sorted(
                images,
                key=lambda img: (
                    -len(img.tags) if hasattr(img, "tags") else 0,
                    img.hole_id,
                    img.depth_from,
                ),
            )
        else:
            return images

    def _get_viz_columns_config(self) -> List[Dict[str, Any]]:
        """Get visualization column configuration from the visualizer."""
        viz_columns = []
        
        if not self.include_visualizations.get():
            return viz_columns
        
        # Get from the visualizer's plot_configs if available
        if self.drillhole_visualizer and hasattr(self.drillhole_visualizer, 'plot_configs'):
            for config in self.drillhole_visualizer.plot_configs:
                if hasattr(config, 'columns') and config.columns:
                    col_name = config.columns[0] if config.columns else ""
                    color_map = None
                    if hasattr(config, 'custom_params') and config.custom_params:
                        color_map = config.custom_params.get('color_map_obj')
                    
                    viz_columns.append({
                        "column": col_name,
                        "color_map": color_map,
                        "width": getattr(config, 'width', 80),
                    })
        
        self.logger.info(f"Configured {len(viz_columns)} viz columns for PDF export")
        return viz_columns

    def _populate_image_csv_data(self):
        """Populate csv_data and register attributes on filtered images."""
        if not self.data_coordinator:
            self.logger.warning("No DataCoordinator available for CSV data")
            return
        
        from processing.DataManager.keys import ImageKey
        
        geological_store = getattr(self.data_coordinator, '_geological_store', None)
        register_store = getattr(self.data_coordinator, '_register_store', None)
        
        populated_csv = 0
        populated_register = 0
        
        for img in self.filtered_images:
            try:
                moisture = getattr(img, 'moisture_status', 'Wet')
                key = ImageKey(img.hole_id, img.depth_to, moisture)
                
                # Populate CSV data
                if geological_store:
                    csv_row = geological_store.get_row(key)
                    if csv_row:
                        img.csv_data = csv_row
                        populated_csv += 1
                    else:
                        img.csv_data = {}
                else:
                    img.csv_data = {}
                
                # Populate register data (classification, comments, tags)
                if register_store:
                    review_meta = register_store.get_review_metadata(key)
                    if review_meta:
                        img.classification = review_meta.classification
                        img.tags = review_meta.tags
                        img.comments = review_meta.comments
                        img.consensus_classification = review_meta.consensus_classification
                        img.review_count = review_meta.review_count
                        if review_meta.classification or review_meta.comments:
                            populated_register += 1
                            
            except Exception as e:
                self.logger.debug(f"Could not get data for {img.hole_id} {img.depth_from}-{img.depth_to}: {e}")
                if not hasattr(img, 'csv_data'):
                    img.csv_data = {}
        
        self.logger.info(f"Populated data: {populated_csv} CSV, {populated_register} register for {len(self.filtered_images)} images")
        
        # Debug: log sample data
        for img in self.filtered_images[:3]:
            if hasattr(img, 'csv_data') and img.csv_data:
                self.logger.debug(f"Sample csv_data keys: {list(img.csv_data.keys())[:10]}")
            if hasattr(img, 'comments') and img.comments:
                self.logger.debug(f"Sample comment: {img.comments[:50]}")
            break

    def _get_full_hole_images(self) -> List[Any]:
        """Get ALL images for holes that appear in filtered_images (for strip log mode)."""
        from processing.DataManager.keys import ImageKey
        
        # Get unique hole IDs from filtered images
        filtered_hole_ids = set(img.hole_id for img in self.filtered_images)
        self.logger.info(f"Strip log mode: Getting full hole data for {len(filtered_hole_ids)} holes")
        
        # Get all images for these holes from the data coordinator
        all_images = []
        if hasattr(self.data_coordinator, 'image_index') and self.data_coordinator.image_index:
            for hole_id in filtered_hole_ids:
                hole_images = self.data_coordinator.image_index.get_images_for_hole(hole_id)
                if hole_images:
                    all_images.extend(hole_images)
                    self.logger.debug(f"  {hole_id}: {len(hole_images)} images")
        
        # If we couldn't get from data_coordinator, fall back to checking all_images if available
        if not all_images and hasattr(self, 'all_images') and self.all_images:
            for img in self.all_images:
                if img.hole_id in filtered_hole_ids:
                    all_images.append(img)
        
        # If still no images, just use filtered
        if not all_images:
            self.logger.warning("Could not get full hole images, using filtered images")
            return self.filtered_images
        
        # Get store references for efficient batch population
        geological_store = getattr(self.data_coordinator, '_geological_store', None)
        register_store = getattr(self.data_coordinator, '_register_store', None)
        
        populated_csv = 0
        populated_register = 0
        
        # Populate data for all images
        for img in all_images:
            moisture = getattr(img, 'moisture_status', 'Wet')
            key = ImageKey(img.hole_id, img.depth_to, moisture)
            
            # Populate CSV data
            if not hasattr(img, 'csv_data') or not img.csv_data:
                if geological_store:
                    try:
                        csv_row = geological_store.get_row(key)
                        if csv_row:
                            img.csv_data = csv_row
                            populated_csv += 1
                        else:
                            img.csv_data = {}
                    except:
                        img.csv_data = {}
                else:
                    img.csv_data = {}
            
            # Populate register data (classification, comments, tags)
            if register_store:
                try:
                    review_meta = register_store.get_review_metadata(key)
                    if review_meta:
                        img.classification = review_meta.classification
                        img.tags = review_meta.tags
                        img.comments = review_meta.comments
                        img.consensus_classification = review_meta.consensus_classification
                        img.review_count = review_meta.review_count
                        if review_meta.classification or review_meta.comments:
                            populated_register += 1
                except Exception as e:
                    self.logger.debug(f"Could not get register data for {key}: {e}")
        
        self.logger.info(f"Strip log mode: {len(all_images)} total images ({populated_csv} CSV, {populated_register} register) for {len(filtered_hole_ids)} holes")
        return all_images

    def _export_pdf(self):
        """Export to PDF file"""
        try:
            # Check if there are images to export
            if not self.filtered_images:
                DialogHelper.show_message(
                    self.dialog,
                    "No Images",
                    "No images to export. Please adjust your filters.",
                )
                return

            # Ask for save location
            default_filename = f"geological_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            output_path = filedialog.asksaveasfilename(
                parent=self.dialog,
                title="Save PDF Report",
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                initialfile=default_filename,
            )

            if not output_path:
                return

            # Show progress dialog
            progress = ProgressDialog(
                self.dialog,
                "Exporting PDF",
                "Generating PDF report...",
            )

            try:
                # Build label fields config
                label_fields = {
                    "depth": self.label_field_depth.get(),
                    "classification": self.label_field_classification.get(),
                    "tags": self.label_field_tags.get(),
                    "comments": self.label_field_comments.get(),
                    "consensus": self.label_field_consensus.get(),
                    "review_count": self.label_field_review_count.get(),
                    "csv_columns": [c.strip() for c in self.label_csv_columns.get().split(",") if c.strip()],
                }

                # Configure generator
                self.generator.configure(
                    export_mode=self.export_mode.get(),
                    spatial_grouping=self.spatial_grouping.get(), 
                    page_size_name=self.page_size.get(),
                    images_per_row=self.images_per_row.get(),
                    group_by_hole=self.group_by_hole.get(),
                    one_hole_per_page=self.one_hole_per_page.get(),
                    hole_spacing=self.hole_spacing.get(),
                    image_spacing=self.image_spacing.get(),
                    strip_log_image_width=self.strip_log_image_width.get(),
                    strip_log_context_meters=self.strip_log_context_meters.get(),
                    strip_log_cell_height=self.strip_log_cell_height.get(),
                    show_minimap=self.show_minimap.get(),
                    minimap_width=self.minimap_width.get(),
                    minimap_height=self.minimap_height.get(),
                    label_font_size=self.label_font_size.get(),
                    label_bg_color=self.label_bg_color.get(),
                    show_depth_labels=self.show_depth_labels.get(),
                    show_classification_labels=self.show_classification_labels.get(),
                    show_hole_headers=self.show_hole_headers.get(),
                    label_fields=label_fields,
                    include_visualizations=self.include_visualizations.get(),
                    viz_width=self.viz_width.get(),
                    viz_columns=self._get_viz_columns_config(),
                    title=self.report_title.get(),
                    show_page_numbers=self.show_page_numbers.get(),
                    show_date=self.show_date.get(),
                )
                
                # Populate csv_data on images for viz columns and labels
                self._populate_image_csv_data()

                # Set collar data for minimap
                if self.data_coordinator and self.show_minimap.get():
                    try:
                        collar_data = self.data_coordinator.get_collar_data()
                        if collar_data is not None and not collar_data.empty:
                            self.generator.set_collar_data(collar_data)
                    except Exception as e:
                        self.logger.warning(f"Could not load collar data for minimap: {e}")

                # Log what we're sending
                self.logger.info(f"Exporting {len(self.filtered_images)} images to PDF")
                
                # Count images per hole
                hole_counts = {}
                for img in self.filtered_images:
                    hole_counts[img.hole_id] = hole_counts.get(img.hole_id, 0) + 1
                self.logger.info(f"Holes: {len(hole_counts)}, Images per hole: {dict(list(hole_counts.items())[:10])}")
                
                # Get images to export
                # For strip log mode, get ALL images for the filtered holes (not just filtered intervals)
                images_to_export = self.filtered_images
                if self.export_mode.get() == "strip_log" and self.data_coordinator:
                    images_to_export = self._get_full_hole_images()
                
                # Generate PDF
                success = self.generator.generate_report(
                    output_path,
                    images_to_export,
                    self.drillhole_data_manager,
                    self.drillhole_visualizer,
                    selected_intervals=self.selected_intervals_dict if self.selected_intervals_dict else None,
                )

                progress.close()

                if success:
                    DialogHelper.show_message(
                        self.dialog,
                        "Export Complete",
                        f"PDF report exported successfully to:\n{output_path}",
                    )
                    self.dialog.destroy()
                else:
                    DialogHelper.show_message(
                        self.dialog,
                        "Export Failed",
                        "Failed to generate PDF report. Check logs for details.",
                    )

            except Exception as e:
                progress.close()
                raise

        except Exception as e:
            self.logger.error(f"Error exporting PDF: {e}", exc_info=True)
            DialogHelper.show_message(
                self.dialog,
                "Export Error",
                f"An error occurred during export:\n{str(e)}",
            )