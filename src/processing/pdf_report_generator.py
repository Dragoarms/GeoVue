# # src/processing/pdf_report_generator.py
# """
# PDF Report Generator for Geological Image Grids
# Generates professional PDF reports from image grids with data visualizations
# """

# import os
# import logging
# from typing import List, Dict, Any, Optional, Tuple
# from datetime import datetime
# from PIL import Image as PILImage
# import cv2
# import numpy as np
# from io import BytesIO

# # ReportLab imports
# from reportlab.lib.pagesizes import A4, A3, LETTER, LEGAL
# from reportlab.lib.units import mm, inch
# from reportlab.lib import colors
# from reportlab.lib.utils import ImageReader
# from reportlab.platypus import (
#     SimpleDocTemplate,
#     Table,
#     TableStyle,
#     Paragraph,
#     Spacer,
#     PageBreak,
#     Image as RLImage,
#     KeepTogether,
# )
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
# from reportlab.pdfgen import canvas as pdf_canvas


# class PDFReportGenerator:
#     """Generates PDF reports from image grids with customizable layout"""

#     # Page size configurations
#     PAGE_SIZES = {
#         "A4 Portrait": (A4[0], A4[1]),
#         "A4 Landscape": (A4[1], A4[0]),
#         "A3 Portrait": (A3[0], A3[1]),
#         "A3 Landscape": (A3[1], A3[0]),
#         "Letter Portrait": (LETTER[0], LETTER[1]),
#         "Letter Landscape": (LETTER[1], LETTER[0]),
#         "Legal Portrait": (LEGAL[0], LEGAL[1]),
#         "Legal Landscape": (LEGAL[1], LEGAL[0]),
#     }

#     def __init__(self):
#         """Initialize the PDF generator"""
#         self.logger = logging.getLogger(__name__)

#         # Default settings
#         self.page_size_name = "A4 Portrait"
#         self.page_size = self.PAGE_SIZES[self.page_size_name]
#         self.margins = (15 * mm, 15 * mm, 15 * mm, 15 * mm)  # left, right, top, bottom

#         # Layout settings
#         self.images_per_row = 3
#         self.group_by_hole = True
#         self.hole_spacing = 20  # mm between holes
#         self.image_spacing = 5  # mm between images

#         # Label settings
#         self.label_font_size = 9
#         self.label_bg_color = colors.HexColor("#333333")
#         self.label_text_color = colors.white
#         self.show_depth_labels = True
#         self.show_classification_labels = True
#         self.show_hole_headers = True

#         # Visualization settings
#         self.include_visualizations = True
#         self.viz_width = 40  # mm

#         # Header/Footer settings
#         self.title = "Geological Image Report"
#         self.show_page_numbers = True
#         self.show_date = True

#     def configure(self, **kwargs):
#         """
#         Configure generator settings

#         Args:
#             page_size_name: Page size key from PAGE_SIZES
#             images_per_row: Number of images per row
#             group_by_hole: Whether to group images by drillhole
#             hole_spacing: Spacing between hole groups (mm)
#             image_spacing: Spacing between images (mm)
#             label_font_size: Font size for labels
#             label_bg_color: Background color for labels (hex string or colors object)
#             label_text_color: Text color for labels
#             show_depth_labels: Show depth interval labels
#             show_classification_labels: Show classification labels
#             show_hole_headers: Show hole ID headers
#             include_visualizations: Include data visualizations
#             viz_width: Width of visualization columns (mm)
#             title: Report title
#             show_page_numbers: Show page numbers
#             show_date: Show generation date
#         """
#         for key, value in kwargs.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)

#         # Update page size if changed
#         if "page_size_name" in kwargs and kwargs["page_size_name"] in self.PAGE_SIZES:
#             self.page_size = self.PAGE_SIZES[kwargs["page_size_name"]]

#         # Convert hex color to ReportLab color if needed
#         if "label_bg_color" in kwargs and isinstance(kwargs["label_bg_color"], str):
#             self.label_bg_color = colors.HexColor(kwargs["label_bg_color"])

#     def generate_report(
#         self,
#         output_path: str,
#         images: List[Any],
#         drillhole_data_manager: Optional[Any] = None,
#         drillhole_visualizer: Optional[Any] = None,
#     ) -> bool:
#         """
#         Generate PDF report from image list

#         Args:
#             output_path: Path to save PDF
#             images: List of CompartmentImage objects
#             drillhole_data_manager: Optional data manager for CSV data
#             drillhole_visualizer: Optional visualizer for data plots

#         Returns:
#             True if successful
#         """
#         try:
#             self.logger.info(f"Generating PDF report: {output_path}")
#             self.logger.info(f"  Images: {len(images)}")
#             self.logger.info(f"  Page size: {self.page_size_name}")
#             self.logger.info(f"  Images per row: {self.images_per_row}")

#             # Create the PDF document
#             doc = SimpleDocTemplate(
#                 output_path,
#                 pagesize=self.page_size,
#                 leftMargin=self.margins[0],
#                 rightMargin=self.margins[1],
#                 topMargin=self.margins[2],
#                 bottomMargin=self.margins[3],
#                 title=self.title,
#             )

#             # Build content
#             story = []

#             # Add title
#             styles = getSampleStyleSheet()
#             title_style = ParagraphStyle(
#                 "CustomTitle",
#                 parent=styles["Title"],
#                 fontSize=16,
#                 textColor=colors.HexColor("#1a1a1a"),
#                 spaceAfter=12,
#                 alignment=TA_CENTER,
#             )
#             story.append(Paragraph(self.title, title_style))

#             # Add metadata
#             if self.show_date:
#                 date_style = ParagraphStyle(
#                     "DateStyle",
#                     parent=styles["Normal"],
#                     fontSize=9,
#                     textColor=colors.grey,
#                     alignment=TA_CENTER,
#                     spaceAfter=20,
#                 )
#                 date_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
#                 story.append(Paragraph(date_text, date_style))

#             # Add summary
#             summary_style = ParagraphStyle(
#                 "SummaryStyle",
#                 parent=styles["Normal"],
#                 fontSize=10,
#                 spaceAfter=15,
#             )
#             summary_text = f"Total Images: {len(images)}"
#             if self.group_by_hole:
#                 hole_count = len(set(img.hole_id for img in images))
#                 summary_text += f" | Drillholes: {hole_count}"
#             story.append(Paragraph(summary_text, summary_style))
#             story.append(Spacer(1, 10))

#             # Group images if needed
#             if self.group_by_hole:
#                 # Group by hole_id
#                 holes_dict = {}
#                 for img in images:
#                     if img.hole_id not in holes_dict:
#                         holes_dict[img.hole_id] = []
#                     holes_dict[img.hole_id].append(img)

#                 # Sort by hole_id
#                 sorted_holes = sorted(holes_dict.items())

#                 for hole_id, hole_images in sorted_holes:
#                     # Add hole header
#                     if self.show_hole_headers:
#                         hole_header_style = ParagraphStyle(
#                             "HoleHeader",
#                             parent=styles["Heading2"],
#                             fontSize=12,
#                             textColor=colors.HexColor("#0066cc"),
#                             spaceAfter=8,
#                             spaceBefore=15,
#                         )
#                         story.append(Paragraph(f"Drillhole: {hole_id}", hole_header_style))

#                     # Add images for this hole
#                     self._add_image_grid(
#                         story,
#                         hole_images,
#                         drillhole_data_manager,
#                         drillhole_visualizer,
#                     )

#                     # Add spacing between holes
#                     story.append(Spacer(1, self.hole_spacing))

#             else:
#                 # Add all images without grouping
#                 self._add_image_grid(
#                     story, images, drillhole_data_manager, drillhole_visualizer
#                 )

#             # Build PDF with page numbers
#             if self.show_page_numbers:
#                 doc.build(story, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)
#             else:
#                 doc.build(story)

#             self.logger.info(f"✅ PDF report generated: {output_path}")
#             return True

#         except Exception as e:
#             self.logger.error(f"❌ Error generating PDF report: {e}", exc_info=True)
#             return False

#     def _add_image_grid(
#         self,
#         story: List,
#         images: List[Any],
#         data_manager: Optional[Any],
#         visualizer: Optional[Any],
#     ):
#         """Add image grid to PDF story"""
#         # Calculate available width
#         page_width = self.page_size[0] - self.margins[0] - self.margins[1]

#         # Calculate cell dimensions
#         total_spacing = (self.images_per_row - 1) * self.image_spacing
#         if self.include_visualizations:
#             viz_total_width = self.viz_width * self.images_per_row
#         else:
#             viz_total_width = 0

#         cell_width = (page_width - total_spacing - viz_total_width) / self.images_per_row

#         # Process images in rows
#         for i in range(0, len(images), self.images_per_row):
#             row_images = images[i : i + self.images_per_row]

#             # Create table data for this row
#             table_data = []
#             image_row = []

#             for img in row_images:
#                 # Create composite image (image + visualization)
#                 composite = self._create_composite_image(
#                     img, cell_width, data_manager, visualizer
#                 )
#                 image_row.append(composite)

#             # Pad row if incomplete
#             while len(image_row) < self.images_per_row:
#                 image_row.append("")

#             table_data.append(image_row)

#             # Create labels row if enabled
#             if self.show_depth_labels or self.show_classification_labels:
#                 label_row = []
#                 for img in row_images:
#                     label_text = self._create_label_text(img)
#                     label_para = Paragraph(
#                         label_text,
#                         ParagraphStyle(
#                             "LabelStyle",
#                             fontSize=self.label_font_size,
#                             textColor=self.label_text_color,
#                             alignment=TA_CENTER,
#                             leading=self.label_font_size + 2,
#                         ),
#                     )
#                     label_row.append(label_para)

#                 # Pad label row
#                 while len(label_row) < self.images_per_row:
#                     label_row.append("")

#                 table_data.append(label_row)

#             # Create table
#             col_widths = [cell_width + (viz_total_width / self.images_per_row if self.include_visualizations else 0)] * self.images_per_row
            
#             table = Table(table_data, colWidths=col_widths)

#             # Apply table style
#             table_style = [
#                 ("VALIGN", (0, 0), (-1, -1), "TOP"),
#                 ("ALIGN", (0, 0), (-1, -1), "CENTER"),
#             ]

#             # Add label background if labels are shown
#             if self.show_depth_labels or self.show_classification_labels:
#                 table_style.append(
#                     ("BACKGROUND", (0, 1), (-1, 1), self.label_bg_color)
#                 )
#                 table_style.append(("TOPPADDING", (0, 1), (-1, 1), 3))
#                 table_style.append(("BOTTOMPADDING", (0, 1), (-1, 1), 3))

#             table.setStyle(TableStyle(table_style))

#             # Add to story with KeepTogether to avoid page breaks within rows
#             story.append(KeepTogether(table))
#             story.append(Spacer(1, self.image_spacing))

#     def _create_composite_image(
#         self,
#         img: Any,
#         target_width: float,
#         data_manager: Optional[Any],
#         visualizer: Optional[Any],
#     ) -> RLImage:
#         """
#         Create composite image (main image + visualization)

#         Args:
#             img: CompartmentImage object
#             target_width: Target width in points
#             data_manager: Data manager for CSV data
#             visualizer: Visualizer for plots

#         Returns:
#             ReportLab Image object
#         """
#         try:
#             # Debug logging
#             self.logger.debug(f"Processing image: {img.filename}")
#             self.logger.debug(f"  img.image type: {type(img.image)}")
#             self.logger.debug(f"  img.image is None: {img.image is None}")
#             self.logger.debug(f"  hasattr image_path: {hasattr(img, 'image_path')}")
#             if hasattr(img, 'image_path'):
#                 self.logger.debug(f"  img.image_path type: {type(img.image_path)}")
#                 self.logger.debug(f"  img.image_path value: {img.image_path}")
            
#             # Load main image - handle multiple input types
#             main_img = None
            
#             # Case 1: img.image is already a numpy array
#             if img.image is not None and isinstance(img.image, np.ndarray):
#                 self.logger.debug("  ✓ Using img.image (numpy array)")
#                 main_img = img.image.copy()
            
#             # Case 2: img.image is a PIL Image
#             elif img.image is not None and isinstance(img.image, PILImage.Image):
#                 self.logger.debug("  ✓ Converting img.image from PIL to numpy")
#                 # Convert PIL to numpy array
#                 main_img = np.array(img.image)
#                 # Convert RGB to BGR for OpenCV compatibility
#                 if len(main_img.shape) == 3 and main_img.shape[2] == 3:
#                     main_img = cv2.cvtColor(main_img, cv2.COLOR_RGB2BGR)
            
#             # Case 3: image_path is a string path
#             elif hasattr(img, 'image_path') and isinstance(img.image_path, (str, bytes, os.PathLike)):
#                 if os.path.exists(str(img.image_path)):
#                     self.logger.debug(f"  ✓ Loading from image_path: {img.image_path}")
#                     main_img = cv2.imread(str(img.image_path))
#                     if main_img is None:
#                         self.logger.error(f"  ✗ cv2.imread failed for: {img.image_path}")
#                 else:
#                     self.logger.warning(f"  ✗ image_path does not exist: {img.image_path}")
            
#             # Case 4: image_path is incorrectly a PIL Image object
#             elif hasattr(img, 'image_path') and isinstance(img.image_path, PILImage.Image):
#                 self.logger.warning("  ! image_path is PIL Image (should be path string), converting")
#                 main_img = np.array(img.image_path)
#                 if len(main_img.shape) == 3 and main_img.shape[2] == 3:
#                     main_img = cv2.cvtColor(main_img, cv2.COLOR_RGB2BGR)
            
#             # Case 5: Try to load image if not already loaded
#             elif hasattr(img, 'load_image'):
#                 self.logger.debug("  ⟳ Attempting to load image via load_image()")
#                 img.load_image()
#                 if img.image is not None and isinstance(img.image, np.ndarray):
#                     main_img = img.image.copy()
#                     self.logger.debug("  ✓ Successfully loaded via load_image()")
#                 elif img.image is not None and isinstance(img.image, PILImage.Image):
#                     self.logger.debug("  ✓ Loaded PIL image via load_image(), converting")
#                     main_img = np.array(img.image)
#                     if len(main_img.shape) == 3 and main_img.shape[2] == 3:
#                         main_img = cv2.cvtColor(main_img, cv2.COLOR_RGB2BGR)
#                 else:
#                     self.logger.warning("  ✗ load_image() did not produce valid image")
            
#             # Fallback: Create placeholder
#             if main_img is None:
#                 self.logger.warning(f"  Creating placeholder for {img.filename}")
#                 main_img = np.ones((300, 200, 3), dtype=np.uint8) * 240
#                 cv2.putText(
#                     main_img,
#                     "No Image",
#                     (50, 150),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (100, 100, 100),
#                     2,
#                 )
#             else:
#                 self.logger.debug(f"  ✓ Final image shape: {main_img.shape}")

#             # Generate visualization if enabled
#             self.logger.info(f"[PDF_VIZ_DEBUG] Checking viz for {img.filename}")
#             self.logger.info(f"[PDF_VIZ_DEBUG]   include_visualizations: {self.include_visualizations}")
#             self.logger.info(f"[PDF_VIZ_DEBUG]   visualizer is not None: {visualizer is not None}")
#             if visualizer:
#                 self.logger.info(f"[PDF_VIZ_DEBUG]   visualizer.plot_configs count: {len(visualizer.plot_configs)}")
#             self.logger.info(f"[PDF_VIZ_DEBUG]   hasattr csv_data: {hasattr(img, 'csv_data')}")
#             self.logger.info(f"[PDF_VIZ_DEBUG]   csv_data truthy: {bool(img.csv_data) if hasattr(img, 'csv_data') else 'N/A'}")
            
#             if self.include_visualizations and visualizer and hasattr(img, "csv_data") and img.csv_data:
#                 self.logger.info(f"[PDF_VIZ_DEBUG]   PASSED all checks, generating viz...")
#                 self.logger.info(f"[PDF_VIZ_DEBUG]   csv_data keys: {list(img.csv_data.keys())[:10]}")
#                 try:
#                     viz_height = main_img.shape[0]
#                     self.logger.info(f"[PDF_VIZ_DEBUG]   viz_height: {viz_height}")
                    
#                     viz_img = visualizer.generate_compartment_visualization(
#                         compartment_data=img.csv_data,
#                         height=viz_height,
#                         depth_from=img.depth_from,
#                         depth_to=img.depth_to,
#                     )
                    
#                     self.logger.info(f"[PDF_VIZ_DEBUG]   viz_img returned: {viz_img is not None}")
#                     if viz_img is not None:
#                         self.logger.info(f"[PDF_VIZ_DEBUG]   viz_img shape: {viz_img.shape}")
#                         self.logger.info(f"[PDF_VIZ_DEBUG]   viz_img size: {viz_img.size}")

#                     # Concatenate horizontally
#                     if viz_img is not None and viz_img.size > 0:
#                         self.logger.info(f"[PDF_VIZ_DEBUG]   Concatenating viz to main image")
#                         main_img = np.hstack([main_img, viz_img])
#                         self.logger.info(f"[PDF_VIZ_DEBUG]   New main_img shape: {main_img.shape}")
#                     else:
#                         self.logger.warning(f"[PDF_VIZ_DEBUG]   viz_img is None or empty!")
#                 except Exception as e:
#                     self.logger.warning(f"Failed to generate visualization for {img.filename}: {e}")
#                     self.logger.exception("[PDF_VIZ_DEBUG] Full exception:")
#             else:
#                 self.logger.info(f"[PDF_VIZ_DEBUG]   SKIPPED viz generation - failed one or more checks")

#             # Convert to PIL Image
#             self.logger.debug("  Converting to RGB for PIL")
#             main_img_rgb = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
#             pil_img = PILImage.fromarray(main_img_rgb)
#             self.logger.debug(f"  PIL image size: {pil_img.size}")

#             # Calculate dimensions maintaining aspect ratio
#             aspect_ratio = pil_img.height / pil_img.width
#             img_width = target_width
#             img_height = img_width * aspect_ratio
#             self.logger.debug(f"  Target dimensions: {img_width:.1f}x{img_height:.1f} points")

#             # Save PIL image to BytesIO for ReportLab
#             self.logger.debug("  Saving PIL image to BytesIO")
#             img_buffer = BytesIO()
#             pil_img.save(img_buffer, format='PNG')
#             img_buffer.seek(0)  # Reset buffer position to beginning
            
#             # Create ReportLab image from BytesIO
#             self.logger.debug("  Creating ReportLab Image object from BytesIO")
#             rl_image = RLImage(img_buffer, width=img_width, height=img_height)
#             self.logger.debug(f"  ✓ Successfully created RLImage for {img.filename}")

#             return rl_image

#         except Exception as e:
#             self.logger.error(f"Error creating composite image for {img.filename}: {e}")
#             self.logger.error(f"  Exception type: {type(e).__name__}")
#             self.logger.error(f"  Exception details:", exc_info=True)
            
#             # Create a placeholder image instead of returning Paragraph
#             try:
#                 self.logger.debug("  Creating error placeholder image")
#                 # Create red placeholder with error text
#                 placeholder = np.ones((300, 200, 3), dtype=np.uint8) * 255
#                 placeholder[:, :, 0] = 200  # Red channel
#                 placeholder[:, :, 1] = 200  # Green channel
#                 placeholder[:, :, 2] = 200  # Blue channel
                
#                 # Add error text
#                 cv2.putText(
#                     placeholder,
#                     "Image Error",
#                     (20, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (255, 0, 0),
#                     2,
#                 )
#                 cv2.putText(
#                     placeholder,
#                     img.filename[:25],
#                     (20, 150),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.4,
#                     (100, 100, 100),
#                     1,
#                 )
                
#                 # Convert to PIL and ReportLab Image
#                 placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
#                 pil_placeholder = PILImage.fromarray(placeholder_rgb)
                
#                 # Save to BytesIO
#                 placeholder_buffer = BytesIO()
#                 pil_placeholder.save(placeholder_buffer, format='PNG')
#                 placeholder_buffer.seek(0)
                
#                 # Calculate dimensions
#                 aspect_ratio = pil_placeholder.height / pil_placeholder.width
#                 img_width = target_width
#                 img_height = img_width * aspect_ratio
                
#                 return RLImage(placeholder_buffer, width=img_width, height=img_height)
#             except Exception as placeholder_error:
#                 self.logger.error(f"Failed to create placeholder: {placeholder_error}")
#                 # Last resort: create minimal placeholder
#                 minimal_placeholder = np.ones((100, 100, 3), dtype=np.uint8) * 200
#                 minimal_rgb = cv2.cvtColor(minimal_placeholder, cv2.COLOR_BGR2RGB)
#                 minimal_pil = PILImage.fromarray(minimal_rgb)
                
#                 # Save to BytesIO
#                 minimal_buffer = BytesIO()
#                 minimal_pil.save(minimal_buffer, format='PNG')
#                 minimal_buffer.seek(0)
                
#                 return RLImage(minimal_buffer, width=100, height=100)

#     def _create_label_text(self, img: Any) -> str:
#         """Create label text for image"""
#         parts = []

#         if self.show_depth_labels:
#             parts.append(f"{img.hole_id} | {img.depth_from:.1f}-{img.depth_to:.1f}m")

#         if self.show_classification_labels:
#             # Get classification text
#             if hasattr(img.classification, "value"):
#                 class_text = img.classification.value
#             else:
#                 class_text = str(img.classification)

#             if class_text and class_text != "":
#                 parts.append(f"<b>{class_text}</b>")

#             # Add tags if present
#             if hasattr(img, "tags") and img.tags:
#                 tag_text = ", ".join(sorted(img.tags))
#                 parts.append(f"[{tag_text}]")

#         return " | ".join(parts) if parts else f"{img.hole_id}"

#     def _add_page_number(self, canvas, doc):
#         """Add page number to bottom of page"""
#         page_num = canvas.getPageNumber()
#         text = f"Page {page_num}"
#         canvas.saveState()
#         canvas.setFont("Helvetica", 9)
#         canvas.setFillColor(colors.grey)
#         canvas.drawCentredString(
#             self.page_size[0] / 2, 10 * mm, text
#         )
#         canvas.restoreState()


# src/processing/pdf_report_generator.py
"""
PDF Report Generator for Geological Image Grids
Generates professional PDF reports from image grids with data visualizations.

Enhanced Features:
- Dynamic page sizing (fit to content)
- One drillhole per page
- Spatial minimap showing collar locations
- Strip log view with context intervals
- Customizable label fields
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from PIL import Image as PILImage
import cv2
import numpy as np
from io import BytesIO

# ReportLab imports
from reportlab.lib.pagesizes import A4, A3, LETTER, LEGAL
from reportlab.lib.units import mm, inch
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
    Image as RLImage,
    KeepTogether,
    Flowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.pdfbase import pdfmetrics




class PDFReportGenerator:
    """Generates PDF reports from image grids with customizable layout"""

    # Standard page size configurations (kept for backward compatibility)
    PAGE_SIZES = {
        "A4 Portrait": (A4[0], A4[1]),
        "A4 Landscape": (A4[1], A4[0]),
        "A3 Portrait": (A3[0], A3[1]),
        "A3 Landscape": (A3[1], A3[0]),
        "Letter Portrait": (LETTER[0], LETTER[1]),
        "Letter Landscape": (LETTER[1], LETTER[0]),
        "Legal Portrait": (LEGAL[0], LEGAL[1]),
        "Legal Landscape": (LEGAL[1], LEGAL[0]),
        "Dynamic (Fit Content)": None,  # Special marker for dynamic sizing
    }

    # Export modes
    EXPORT_MODE_GRID = "grid"
    EXPORT_MODE_STRIP_LOG = "strip_log"

    def __init__(self):
        """Initialize the PDF generator"""
        self.logger = logging.getLogger(__name__)

        # Default settings
        self.page_size_name = "Dynamic (Fit Content)"
        self.page_size = None  # Will be calculated per drillhole
        self.margins = (15 * mm, 15 * mm, 15 * mm, 15 * mm)  # left, right, top, bottom

        # Export mode
        self.export_mode = self.EXPORT_MODE_STRIP_LOG  # Default to strip log

        # Layout settings
        self.images_per_row = 3
        self.group_by_hole = True
        self.hole_spacing = 20  # mm between holes
        self.image_spacing = 5  # mm between images
        self.one_hole_per_page = True  # New: each drillhole on its own page


        # Spatial grouping
        self.spatial_grouping = True  # Sort holes by spatial proximity

        # Strip log settings
        self.strip_log_image_width = 80  # mm width for each image in strip log
        self.strip_log_context_meters = 10.0  # meters above/below selection
        self.strip_log_cell_height = 25  # mm per interval cell
        self.pixels_per_meter = 25  # for continuous depth scaling (match cell height)

        # Minimap settings
        self.show_minimap = True
        self.minimap_width = 150  # mm
        self.minimap_height = 120  # mm
        self.minimap_collar_size = 6
        self.minimap_highlight_size = 10
        self.minimap_padding = 50  # Map units padding around collars

        # Label settings
        self.label_font_size = 9
        self.label_bg_color = colors.HexColor("#333333")
        self.label_text_color = colors.white
        self.show_depth_labels = True
        self.show_classification_labels = True
        self.show_hole_headers = True
        
        # Extended label options
        self.label_fields = {
            "depth": True,
            "classification": True,
            "tags": True,
            "comments": False,
            "hole_id": True,
            "consensus": False,
            "review_count": False,
            "csv_columns": [],  # List of CSV column names to include
        }

        # Visualization settings
        self.include_visualizations = True
        self.viz_width = 15  # mm - narrow columns to match GUI appearance
        self.viz_columns = []  # List of {column, color_map, type}

        # Header/Footer settings
        self.title = "Geological Image Report"
        self.show_page_numbers = True
        self.show_date = True
        
        # Collar data for minimap (set externally)
        self._collar_data = None

    def configure(self, **kwargs):
        """
        Configure generator settings

        Args:
            page_size_name: Page size key from PAGE_SIZES or "Dynamic (Fit Content)"
            export_mode: "grid" or "strip_log"
            images_per_row: Number of images per row (grid mode)
            group_by_hole: Whether to group images by drillhole
            one_hole_per_page: Put each drillhole on its own page
            hole_spacing: Spacing between hole groups (mm)
            image_spacing: Spacing between images (mm)
            
            strip_log_image_width: Width of images in strip log (mm)
            strip_log_context_meters: Meters of context above/below
            strip_log_cell_height: Height per interval (mm)
            
            show_minimap: Show collar location minimap
            minimap_width: Width of minimap (mm)
            minimap_height: Height of minimap (mm)
            
            label_font_size: Font size for labels
            label_bg_color: Background color for labels (hex string or colors object)
            label_text_color: Text color for labels
            show_depth_labels: Show depth interval labels
            show_classification_labels: Show classification labels
            show_hole_headers: Show hole ID headers
            label_fields: Dict of which label fields to show
            
            include_visualizations: Include data visualizations
            viz_width: Width of visualization columns (mm)
            viz_columns: List of visualization column configs
            
            title: Report title
            show_page_numbers: Show page numbers
            show_date: Show generation date
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Update page size if changed
        if "page_size_name" in kwargs:
            if kwargs["page_size_name"] == "Dynamic (Fit Content)":
                self.page_size = None
            elif kwargs["page_size_name"] in self.PAGE_SIZES:
                self.page_size = self.PAGE_SIZES[kwargs["page_size_name"]]

        # Convert hex color to ReportLab color if needed
        if "label_bg_color" in kwargs and isinstance(kwargs["label_bg_color"], str):
            self.label_bg_color = colors.HexColor(kwargs["label_bg_color"])

    def set_collar_data(self, collar_df):
        """
        Set collar data for minimap rendering.
        
        Args:
            collar_df: DataFrame with columns: holeid, x, y (and optionally z)
        """
        self._collar_data = collar_df
        self.logger.info(f"Set collar data with {len(collar_df)} collars")

    def get_spatially_sorted_holes(self, hole_ids: List[str]) -> List[str]:
        """
        Sort hole IDs spatially using nearest-neighbor traversal.
        
        Starts from the hole closest to the centroid, then repeatedly
        picks the nearest unvisited hole. This keeps nearby holes together.
        
        Args:
            hole_ids: List of hole IDs to sort
            
        Returns:
            Spatially sorted list of hole IDs
        """
        if self._collar_data is None or self._collar_data.empty:
            self.logger.warning("No collar data for spatial sorting, using original order")
            return hole_ids
        
        if len(hole_ids) <= 1:
            return hole_ids
        
        try:
            # Build coordinate lookup
            coords = {}
            for _, row in self._collar_data.iterrows():
                hid = row.get('holeid', '')
                if hid in hole_ids:
                    coords[hid] = (float(row['x']), float(row['y']))
            
            # Holes without coordinates go at the end
            holes_with_coords = [h for h in hole_ids if h in coords]
            holes_without_coords = [h for h in hole_ids if h not in coords]
            
            if len(holes_with_coords) <= 1:
                return hole_ids
            
            # Find centroid
            cx = sum(coords[h][0] for h in holes_with_coords) / len(holes_with_coords)
            cy = sum(coords[h][1] for h in holes_with_coords) / len(holes_with_coords)
            
            # Start from hole closest to centroid
            def dist_to_centroid(h):
                return (coords[h][0] - cx) ** 2 + (coords[h][1] - cy) ** 2
            
            sorted_holes = []
            remaining = set(holes_with_coords)
            
            # Start with hole closest to centroid
            current = min(remaining, key=dist_to_centroid)
            sorted_holes.append(current)
            remaining.remove(current)
            
            # Nearest-neighbor traversal
            while remaining:
                curr_x, curr_y = coords[current]
                
                def dist_to_current(h):
                    return (coords[h][0] - curr_x) ** 2 + (coords[h][1] - curr_y) ** 2
                
                nearest = min(remaining, key=dist_to_current)
                sorted_holes.append(nearest)
                remaining.remove(nearest)
                current = nearest
            
            # Add holes without coordinates at the end
            sorted_holes.extend(holes_without_coords)
            
            self.logger.info(f"Spatially sorted {len(sorted_holes)} holes")
            return sorted_holes
            
        except Exception as e:
            self.logger.error(f"Error in spatial sorting: {e}", exc_info=True)
            return hole_ids

    def generate_report(
        self,
        output_path: str,
        images: List[Any],
        drillhole_data_manager: Optional[Any] = None,
        drillhole_visualizer: Optional[Any] = None,
        selected_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> bool:
        """
        Generate PDF report from image list.

        Args:
            output_path: Path to save PDF
            images: List of CompartmentImage objects
            drillhole_data_manager: Optional data manager for CSV data
            drillhole_visualizer: Optional visualizer for data plots
            selected_intervals: Optional dict of {hole_id: (from_depth, to_depth)} 
                               for strip log context range

        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Generating PDF report: {output_path}")
            self.logger.info(f"  Images: {len(images)}")
            self.logger.info(f"  Export mode: {self.export_mode}")
            self.logger.info(f"  Page sizing: {self.page_size_name}")

            if self.export_mode == self.EXPORT_MODE_STRIP_LOG:
                return self._generate_strip_log_report(
                    output_path, images, drillhole_data_manager, 
                    drillhole_visualizer, selected_intervals
                )
            else:
                return self._generate_grid_report(
                    output_path, images, drillhole_data_manager, drillhole_visualizer
                )

        except Exception as e:
            self.logger.error(f"❌ Error generating PDF report: {e}", exc_info=True)
            return False

    def _generate_strip_log_report(
        self,
        output_path: str,
        images: List[Any],
        data_manager: Optional[Any],
        visualizer: Optional[Any],
        selected_intervals: Optional[Dict[str, Tuple[float, float]]],
    ) -> bool:
        """Generate strip log style report with one drillhole per page."""
        from reportlab.pdfgen.canvas import Canvas
        
        # Group images by hole
        holes_dict = {}
        for img in images:
            if img.hole_id not in holes_dict:
                holes_dict[img.hole_id] = []
            holes_dict[img.hole_id].append(img)
        
        # Log image counts per hole
        self.logger.info(f"  Total images received: {len(images)}")
        self.logger.info(f"  Unique holes: {len(holes_dict)}")
        for hole_id, hole_images in sorted(holes_dict.items())[:10]:  # First 10 holes
            self.logger.info(f"    {hole_id}: {len(hole_images)} images ({hole_images[0].depth_from:.0f}m - {hole_images[-1].depth_to:.0f}m)")

        # Sort images within each hole by depth
        for hole_id in holes_dict:
            holes_dict[hole_id].sort(key=lambda x: x.depth_from)

        # Sort holes spatially if enabled
        hole_ids = list(holes_dict.keys())
        if self.spatial_grouping and self._collar_data is not None:
            hole_ids = self.get_spatially_sorted_holes(hole_ids)
            self.logger.info(f"  Using spatial grouping for {len(hole_ids)} drillholes")
        else:
            hole_ids = sorted(hole_ids)
            self.logger.info(f"  Using alphabetical sorting for {len(hole_ids)} drillholes")
        
        sorted_holes = [(hid, holes_dict[hid]) for hid in hole_ids]

        # We'll build each page manually for dynamic sizing
        pages_data = []
        
        for hole_id, hole_images in sorted_holes:
            # Always use FULL range of available images for display
            depth_from = min(img.depth_from for img in hole_images)
            depth_to = max(img.depth_to for img in hole_images)
            
            # Get selected interval for highlighting (if any)
            selected_interval = None
            if selected_intervals and hole_id in selected_intervals:
                selected_interval = selected_intervals[hole_id]
            
            # Include ALL images for this hole
            range_images = hole_images
            
            if not range_images:
                continue
            
            # Calculate page dimensions
            page_width, page_height = self._calculate_strip_log_page_size(
                range_images, depth_from, depth_to
            )
            
            pages_data.append({
                "hole_id": hole_id,
                "images": range_images,
                "depth_from": depth_from,
                "depth_to": depth_to,
                "selected_interval": selected_interval,  # Already calculated above
                "page_width": page_width,
                "page_height": page_height,
            })

        if not pages_data:
            self.logger.error("No pages to generate")
            return False

        # Create PDF with custom page sizes
        c = Canvas(output_path)
        
        for page_idx, page_data in enumerate(pages_data):
            self.logger.info(f"  Rendering page {page_idx + 1}: {page_data['hole_id']}")
            
            # Set page size
            c.setPageSize((page_data["page_width"], page_data["page_height"]))
            
            # Render page content
            self._render_strip_log_page(
                c, page_data, data_manager, visualizer, page_idx + 1, len(pages_data)
            )
            
            c.showPage()

        c.save()
        self.logger.info(f"✅ Strip log PDF report generated: {output_path}")
        return True

    def _calculate_strip_log_page_size(
        self,
        images: List[Any],
        depth_from: float,
        depth_to: float,
    ) -> Tuple[float, float]:
        """
        Calculate dynamic page size for strip log.
        
        Returns:
            (width, height) in points
        """
        depth_range = depth_to - depth_from
        
        # Calculate content width
        content_width = self.margins[0] + self.margins[1]  # Left + right margins
        
        # Minimap
        if self.show_minimap:
            content_width += self.minimap_width * mm + 10 * mm  # minimap + spacing
        
        # Depth ruler
        content_width += 30 * mm  # Depth scale
        
        # Images column
        content_width += self.strip_log_image_width * mm
        
        # Visualization columns
        if self.include_visualizations and self.viz_columns:
            content_width += len(self.viz_columns) * self.viz_width * mm
        
        # Labels column (must match _draw_labels_column)
        content_width += 50 * mm  # Compact labels column
        
        # Calculate content height
        # Header: title + hole info
        header_height = 40 * mm
        
        # Strip log content height based on number of intervals
        # Each interval needs enough height for landscape image + label
        # For landscape images (typically 3:1 to 4:1 aspect), height = width / aspect
        # With 120mm width, ~40-45mm height per image for proper display
        cell_height = max(self.strip_log_cell_height, 45)  # mm per interval
        strip_height = len(images) * cell_height * mm
        
        # Alternative: depth-based if cell height would be too small
        depth_based_height = depth_range * self.pixels_per_meter * mm
        strip_height = max(strip_height, depth_based_height)
        
        # Maximum reasonable height for single page
        # For 180 images at 45mm each = 8100mm needed
        # PDF viewers handle tall pages well
        max_height = 10000 * mm  # ~10 meters for very long holes
        strip_height = min(strip_height, max_height)
        
        content_height = (
            self.margins[2] + self.margins[3] +  # Top + bottom margins
            header_height +
            strip_height +
            20 * mm  # Footer
        )
        
        # Ensure minimum page size
        min_width = 200 * mm
        min_height = 250 * mm
        
        page_width = max(content_width, min_width)
        page_height = max(content_height, min_height)
        
        self.logger.debug(f"  Page size: {page_width/mm:.0f}mm x {page_height/mm:.0f}mm")
        return page_width, page_height

    def _render_strip_log_page(
        self,
        canvas: "Canvas",
        page_data: Dict[str, Any],
        data_manager: Optional[Any],
        visualizer: Optional[Any],
        page_num: int,
        total_pages: int,
    ):
        """Render a single strip log page."""
        hole_id = page_data["hole_id"]
        images = page_data["images"]
        depth_from = page_data["depth_from"]
        depth_to = page_data["depth_to"]
        selected_interval = page_data["selected_interval"]
        page_width = page_data["page_width"]
        page_height = page_data["page_height"]
        
        # Current drawing position
        x_pos = self.margins[0]
        y_pos = page_height - self.margins[2]
        
        # === HEADER ===
        # Title
        canvas.setFont("Helvetica-Bold", 14)
        canvas.setFillColor(colors.HexColor("#1a1a1a"))
        canvas.drawString(x_pos, y_pos - 15, self.title)
        
        # Date
        if self.show_date:
            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(colors.grey)
            date_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            canvas.drawRightString(page_width - self.margins[1], y_pos - 15, date_text)
        
        y_pos -= 25 * mm
        
        # Hole ID header
        canvas.setFont("Helvetica-Bold", 12)
        canvas.setFillColor(colors.HexColor("#0066cc"))
        canvas.drawString(x_pos, y_pos, f"Drillhole: {hole_id}")
        
        # Depth range info
        canvas.setFont("Helvetica", 10)
        canvas.setFillColor(colors.HexColor("#333333"))
        depth_info = f"Depth Range: {depth_from:.1f}m - {depth_to:.1f}m ({depth_to - depth_from:.1f}m)"
        canvas.drawString(x_pos + 150, y_pos, depth_info)
        
        if selected_interval:
            sel_text = f"Selected: {selected_interval[0]:.1f}m - {selected_interval[1]:.1f}m"
            canvas.drawString(x_pos + 400, y_pos, sel_text)
        
        y_pos -= 15 * mm
        
        # === MINIMAP ===
        minimap_x = x_pos
        if self.show_minimap and self._collar_data is not None:
            minimap_img = self._render_collar_minimap(hole_id)
            if minimap_img:
                # Draw minimap image
                canvas.drawImage(
                    ImageReader(minimap_img),
                    minimap_x,
                    y_pos - self.minimap_height * mm,
                    width=self.minimap_width * mm,
                    height=self.minimap_height * mm,
                )
                
                # Minimap border
                canvas.setStrokeColor(colors.grey)
                canvas.setLineWidth(1)
                canvas.rect(
                    minimap_x,
                    y_pos - self.minimap_height * mm,
                    self.minimap_width * mm,
                    self.minimap_height * mm,
                )
                
                # Minimap label
                canvas.setFont("Helvetica", 8)
                canvas.setFillColor(colors.grey)
                canvas.drawString(minimap_x, y_pos - self.minimap_height * mm - 10, "Collar Location")
            
            x_pos += self.minimap_width * mm + 10 * mm
        
        # === STRIP LOG CONTENT ===
        strip_top_y = y_pos
        strip_bottom_y = self.margins[3] + 20 * mm
        strip_height = strip_top_y - strip_bottom_y
        
        # Depth ruler
        self._draw_depth_ruler(canvas, x_pos, strip_top_y, strip_bottom_y, depth_from, depth_to)
        x_pos += 30 * mm
        
        # Images column
        images_x = x_pos
        self._draw_strip_log_images(
            canvas, images, images_x, strip_top_y, strip_bottom_y,
            depth_from, depth_to, selected_interval, data_manager, visualizer
        )
        x_pos += self.strip_log_image_width * mm
        
        # Visualization columns
        if self.include_visualizations and self.viz_columns:
            for viz_config in self.viz_columns:
                self._draw_viz_column(
                    canvas, images, x_pos, strip_top_y, strip_bottom_y,
                    depth_from, depth_to, viz_config, visualizer
                )
                x_pos += self.viz_width * mm
        
        # Labels column
        self._draw_labels_column(
            canvas, images, x_pos, strip_top_y, strip_bottom_y,
            depth_from, depth_to, selected_interval
        )
        
        # === FOOTER ===
        if self.show_page_numbers:
            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(colors.grey)
            canvas.drawCentredString(
                page_width / 2,
                10 * mm,
                f"Page {page_num} of {total_pages}"
            )

    def _render_collar_minimap(self, highlight_hole_id: str) -> Optional[PILImage.Image]:
        """
        Render a static minimap image showing collar locations.
        
        Args:
            highlight_hole_id: Hole ID to highlight
            
        Returns:
            PIL Image or None
        """
        if self._collar_data is None or self._collar_data.empty:
            return None
        
        try:
            # Get collar coordinates
            if 'x' not in self._collar_data.columns or 'y' not in self._collar_data.columns:
                self.logger.warning("Collar data missing x/y columns")
                return None
            
            # Create high-resolution image (300 DPI equivalent)
            dpi_scale = 4  # 4x resolution for crisp rendering
            width_px = int(self.minimap_width * 2.83 * dpi_scale)
            height_px = int(self.minimap_height * 2.83 * dpi_scale)
            
            img = np.ones((height_px, width_px, 3), dtype=np.uint8) * 245  # Light gray background
            
            # Calculate bounds
            min_x = self._collar_data['x'].min()
            max_x = self._collar_data['x'].max()
            min_y = self._collar_data['y'].min()
            max_y = self._collar_data['y'].max()
            
            # Add padding
            range_x = max_x - min_x
            range_y = max_y - min_y
            
            if range_x == 0:
                range_x = 100
            if range_y == 0:
                range_y = 100
            
            padding = max(range_x, range_y) * 0.1
            min_x -= padding
            max_x += padding
            min_y -= padding
            max_y += padding
            range_x = max_x - min_x
            range_y = max_y - min_y
            
            # Scale to fit
            scale_x = (width_px - 20) / range_x
            scale_y = (height_px - 20) / range_y
            scale = min(scale_x, scale_y)
            
            def map_to_pixel(x, y):
                margin = int(10 * dpi_scale)
                px = int((x - min_x) * scale + margin)
                py = int(height_px - ((y - min_y) * scale + margin))  # Flip Y
                return px, py
            
            # Draw grid (scaled line width)
            grid_width = max(1, int(dpi_scale / 2))
            margin = int(10 * dpi_scale)
            cv2.line(img, (margin, height_px // 2), (width_px - margin, height_px // 2), (220, 220, 220), grid_width)
            cv2.line(img, (width_px // 2, margin), (width_px // 2, height_px - margin), (220, 220, 220), grid_width)
            
            # Scale marker sizes for high-res rendering
            collar_radius = max(2, int(3 * dpi_scale))  # Smaller markers
            highlight_radius = max(4, int(5 * dpi_scale))
            outline_width = max(1, int(dpi_scale / 2))
            font_scale = 0.3 * dpi_scale
            
            # Draw all collars
            highlight_pos = None
            for _, row in self._collar_data.iterrows():
                hole_id = row.get('holeid', '')
                x, y = row['x'], row['y']
                px, py = map_to_pixel(x, y)
                
                if hole_id == highlight_hole_id:
                    highlight_pos = (px, py)
                else:
                    cv2.circle(img, (px, py), collar_radius, (52, 152, 219), -1)  # Blue
                    cv2.circle(img, (px, py), collar_radius, (0, 0, 0), outline_width)  # Black outline
            
            # Draw highlight collar on top
            if highlight_pos:
                px, py = highlight_pos
                cv2.circle(img, (px, py), highlight_radius, (231, 76, 60), -1)  # Red
                cv2.circle(img, (px, py), highlight_radius, (0, 0, 0), max(1, int(dpi_scale)))  # Black outline
                
                # Label
                cv2.putText(
                    img, highlight_hole_id,
                    (px + highlight_radius + int(5 * dpi_scale), py + int(4 * dpi_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, int(dpi_scale / 2))
                )
            
            # Draw north arrow (scaled)
            arrow_x = width_px - int(25 * dpi_scale)
            arrow_y = int(30 * dpi_scale)
            arrow_len = int(20 * dpi_scale)
            cv2.arrowedLine(img, (arrow_x, arrow_y + arrow_len), (arrow_x, arrow_y), (0, 0, 0), max(1, int(dpi_scale)), tipLength=0.3)
            cv2.putText(img, "N", (arrow_x - int(4 * dpi_scale), arrow_y - int(5 * dpi_scale)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, int(dpi_scale / 2)))
            
            # Convert to PIL and downscale for final output (keeps quality via anti-aliasing)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(img_rgb)
            
            # Resize down to target size (this applies anti-aliasing)
            final_width = int(self.minimap_width * 2.83)
            final_height = int(self.minimap_height * 2.83)
            pil_img = pil_img.resize((final_width, final_height), PILImage.LANCZOS)
            
            return pil_img
            
        except Exception as e:
            self.logger.error(f"Error rendering minimap: {e}", exc_info=True)
            return None

    def _draw_depth_ruler(
        self,
        canvas: "Canvas",
        x: float,
        y_top: float,
        y_bottom: float,
        depth_from: float,
        depth_to: float,
    ):
        """Draw depth scale ruler."""
        height = y_top - y_bottom
        depth_range = depth_to - depth_from
        
        # Draw ruler line
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(1)
        canvas.line(x + 25 * mm, y_top, x + 25 * mm, y_bottom)
        
        # Calculate tick interval (nice round numbers)
        if depth_range <= 20:
            tick_interval = 2
        elif depth_range <= 50:
            tick_interval = 5
        elif depth_range <= 100:
            tick_interval = 10
        else:
            tick_interval = 20
        
        # Draw ticks and labels
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.black)
        
        first_tick = int(depth_from / tick_interval) * tick_interval
        if first_tick < depth_from:
            first_tick += tick_interval
        
        tick_depth = first_tick
        while tick_depth <= depth_to:
            # Calculate Y position
            depth_fraction = (tick_depth - depth_from) / depth_range
            y_pos = y_top - (depth_fraction * height)
            
            # Major tick
            canvas.line(x + 20 * mm, y_pos, x + 25 * mm, y_pos)
            
            # Label
            canvas.drawRightString(x + 18 * mm, y_pos - 3, f"{tick_depth:.0f}m")
            
            tick_depth += tick_interval
        
        # Label
        canvas.setFont("Helvetica", 8)
        canvas.saveState()
        canvas.translate(x + 5 * mm, (y_top + y_bottom) / 2)
        canvas.rotate(90)
        canvas.drawCentredString(0, 0, "Depth (m)")
        canvas.restoreState()

    def _draw_strip_log_images(
        self,
        canvas: "Canvas",
        images: List[Any],
        x: float,
        y_top: float,
        y_bottom: float,
        depth_from: float,
        depth_to: float,
        selected_interval: Optional[Tuple[float, float]],
        data_manager: Optional[Any],
        visualizer: Optional[Any],
    ):
        """Draw images as a continuous strip."""
        height = y_top - y_bottom
        depth_range = depth_to - depth_from
        
        # Column background
        canvas.setFillColor(colors.HexColor("#f5f5f5"))
        canvas.rect(x, y_bottom, self.strip_log_image_width * mm, height, fill=1, stroke=0)
        
        # Draw each image
        for img in images:
            # Calculate Y position for this interval
            img_depth_from = max(img.depth_from, depth_from)
            img_depth_to = min(img.depth_to, depth_to)
            
            top_fraction = (img_depth_from - depth_from) / depth_range
            bottom_fraction = (img_depth_to - depth_from) / depth_range
            
            img_y_top = y_top - (top_fraction * height)
            img_y_bottom = y_top - (bottom_fraction * height)
            img_height = img_y_top - img_y_bottom
            
            # Note: Don't force minimum height - let images stack tightly
            # The page height calculation already accounts for proper sizing
            
            # Highlight if in selected interval
            if selected_interval:
                is_selected = (
                    img.depth_from <= selected_interval[1] and 
                    img.depth_to >= selected_interval[0]
                )
            else:
                is_selected = False
            
            if is_selected:
                canvas.setFillColor(colors.HexColor("#fff3cd"))
                canvas.rect(x, img_y_bottom, self.strip_log_image_width * mm, img_height, fill=1, stroke=0)
            
            # Load and draw image
            try:
                pil_img = self._load_image_for_pdf(img)
                if pil_img:
                    # Rotate portrait images to landscape (chip tray images should be wider than tall)
                    if pil_img.height > pil_img.width:
                        pil_img = pil_img.rotate(90, expand=True)
                    
                    # Scale to fit - image is now landscape
                    aspect = pil_img.height / pil_img.width  # Should be < 1 for landscape
                    img_width = self.strip_log_image_width * mm - 4 * mm  # Padding
                    
                    # Calculate scaled height based on landscape aspect
                    scaled_height = img_width * aspect
                    
                    # Cell height should fit the image, not the other way around
                    # But we're constrained by the pre-calculated cell height
                    if scaled_height > img_height - 2 * mm:
                        scaled_height = img_height - 2 * mm
                        img_width = scaled_height / aspect
                    
                    img_x = x + (self.strip_log_image_width * mm - img_width) / 2
                    img_y = img_y_bottom + (img_height - scaled_height) / 2
                    
                    # Save to buffer
                    img_buffer = BytesIO()
                    pil_img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    canvas.drawImage(
                        ImageReader(img_buffer),
                        img_x, img_y,
                        width=img_width, height=scaled_height,
                    )
            except Exception as e:
                self.logger.warning(f"Error drawing image {img.filename}: {e}")
            
            # Draw border
            border_color = colors.HexColor("#e74c3c") if is_selected else colors.HexColor("#cccccc")
            canvas.setStrokeColor(border_color)
            canvas.setLineWidth(2 if is_selected else 1)
            canvas.rect(x + 1 * mm, img_y_bottom, self.strip_log_image_width * mm - 2 * mm, img_height, stroke=1, fill=0)

    def _draw_viz_column(
        self,
        canvas: "Canvas",
        images: List[Any],
        x: float,
        y_top: float,
        y_bottom: float,
        depth_from: float,
        depth_to: float,
        viz_config: Dict[str, Any],
        visualizer: Optional[Any],
    ):
        """Draw a single visualization column."""
        height = y_top - y_bottom
        depth_range = depth_to - depth_from
        column_name = viz_config.get("column", "")
        color_map_name = viz_config.get("color_map", "")
        
        # Strip source suffix like "(exassay)" to get bare column name
        bare_column_name = column_name
        if " (" in column_name:
            bare_column_name = column_name.split(" (")[0]
        
        # Column background
        canvas.setFillColor(colors.HexColor("#fafafa"))
        canvas.rect(x, y_bottom, self.viz_width * mm, height, fill=1, stroke=1)
        
        # Column header
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(colors.black)
        # Truncate long names
        display_name = column_name[:10] + "..." if len(column_name) > 10 else column_name
        canvas.drawCentredString(x + self.viz_width * mm / 2, y_top + 5, display_name)
        
        # Draw values for each image
        values_found = 0
        values_missing = 0
        for img in images:
            if not hasattr(img, 'csv_data') or not img.csv_data:
                values_missing += 1
                continue
            
            # Get value - try with and without source suffix
            value = img.csv_data.get(column_name)
            if value is None:
                value = img.csv_data.get(bare_column_name)
            if value is None:
                # Try case-insensitive on both forms
                for k, v in img.csv_data.items():
                    k_lower = k.lower()
                    if k_lower == column_name.lower() or k_lower == bare_column_name.lower():
                        value = v
                        break
            
            if value is None:
                values_missing += 1
                continue
            values_found += 1
            
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue
            
            # Calculate Y position
            img_center_depth = (img.depth_from + img.depth_to) / 2
            depth_fraction = (img_center_depth - depth_from) / depth_range
            y_pos = y_top - (depth_fraction * height)
            
            # Get color from color map
            color_map = viz_config.get("color_map")
            if color_map and hasattr(color_map, 'get_color_for_value'):
                try:
                    bgr = color_map.get_color_for_value(value)
                    color = colors.HexColor(f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}")
                except:
                    color = self._get_value_color(value, "", visualizer)
            else:
                color = self._get_value_color(value, "", visualizer)
            
            # Calculate bar position to match image cell (full height, no gaps)
            img_depth_from = img.depth_from
            img_depth_to = img.depth_to
            
            top_fraction = (img_depth_from - depth_from) / depth_range
            bottom_fraction = (img_depth_to - depth_from) / depth_range
            
            bar_y_top = y_top - (top_fraction * height)
            bar_y_bottom = y_top - (bottom_fraction * height)
            bar_height = bar_y_top - bar_y_bottom
            
            # Draw bar - full width, no padding (edge-to-edge)
            canvas.setFillColor(color)
            canvas.rect(
                x,  # No horizontal padding
                bar_y_bottom,
                self.viz_width * mm,  # Full width
                bar_height,
                fill=1, stroke=0
            )
            
            # Draw value text with white background for visibility
            canvas.setFont("Helvetica-Bold", 7)
            text_y = bar_y_bottom + bar_height / 2 - 3
            
            # Draw white background behind text
            text_str = f"{value:.1f}"
            text_width = len(text_str) * 4
            canvas.setFillColor(colors.white)
            canvas.setStrokeColor(colors.white)
            canvas.rect(
                x + self.viz_width * mm / 2 - text_width / 2 - 1,
                text_y - 1,
                text_width + 2,
                10,
                fill=1, stroke=0
            )
            
            # Draw text
            canvas.setFillColor(colors.black)
            canvas.drawCentredString(x + self.viz_width * mm / 2, text_y, text_str)

        # Draw column border (left edge only)
        canvas.setStrokeColor(colors.HexColor("#cccccc"))
        canvas.setLineWidth(0.5)
        canvas.line(x, y_bottom, x, y_top)

        # Log summary for debugging (OUTSIDE the for loop)
        if values_found == 0 and len(images) > 0:
            self.logger.warning(f"Viz column '{column_name}' (bare: '{bare_column_name}'): 0/{len(images)} values found")
            # Log sample csv_data keys
            for img in images[:3]:
                if hasattr(img, 'csv_data') and img.csv_data:
                    self.logger.debug(f"  Sample csv_data keys: {list(img.csv_data.keys())[:10]}")
                    break
        else:
            self.logger.debug(f"Viz column '{column_name}': {values_found}/{len(images)} values rendered")

    def _draw_labels_column(
        self,
        canvas: "Canvas",
        images: List[Any],
        x: float,
        y_top: float,
        y_bottom: float,
        depth_from: float,
        depth_to: float,
        selected_interval: Optional[Tuple[float, float]],
    ):
        """Draw labels column with customizable fields."""
        height = y_top - y_bottom
        depth_range = depth_to - depth_from
        
        # Column width for labels - compact to avoid spillover
        label_column_width = 50 * mm
        
        for img in images:
            # Calculate Y position based on image depth range, not center
            img_depth_from = max(img.depth_from, depth_from)
            img_depth_to = min(img.depth_to, depth_to)
            
            top_fraction = (img_depth_from - depth_from) / depth_range
            bottom_fraction = (img_depth_to - depth_from) / depth_range
            
            img_y_top = y_top - (top_fraction * height)
            img_y_bottom = y_top - (bottom_fraction * height)
            img_height = img_y_top - img_y_bottom
            
            # Build label lines based on configured fields
            label_lines = []
            
            if self.label_fields.get("depth", True):
                label_lines.append(f"{img.depth_from:.1f} - {img.depth_to:.1f}m")
            
            if self.label_fields.get("hole_id", True) and hasattr(img, 'hole_id'):
                # Only add if not already in header
                pass  # hole_id is in the page header
            
            if self.label_fields.get("classification", True):
                if hasattr(img, 'classification'):
                    if hasattr(img.classification, "value"):
                        class_text = img.classification.value
                    else:
                        class_text = str(img.classification)
                    if class_text and class_text not in ("None", "", "unclassified"):
                        label_lines.append(f"Class: {class_text}")
            
            if self.label_fields.get("tags", True):
                if hasattr(img, "tags") and img.tags:
                    tags_text = ", ".join(sorted(img.tags)[:3])
                    label_lines.append(f"Tags: {tags_text}")
            
            if self.label_fields.get("comments", False):
                if hasattr(img, "comments") and img.comments:
                    comment = img.comments[:40] + "..." if len(img.comments) > 40 else img.comments
                    label_lines.append(f'"{comment}"')
            
            if self.label_fields.get("consensus", False):
                if hasattr(img, "consensus_classification") and img.consensus_classification:
                    label_lines.append(f"Consensus: {img.consensus_classification}")
            
            if self.label_fields.get("review_count", False):
                if hasattr(img, "review_count") and img.review_count:
                    label_lines.append(f"Reviews: {img.review_count}")
            
            # CSV columns
            csv_cols = self.label_fields.get("csv_columns", [])
            if csv_cols and hasattr(img, 'csv_data') and img.csv_data:
                for col in csv_cols:
                    val = img.csv_data.get(col)
                    if val is not None:
                        try:
                            label_lines.append(f"{col}: {float(val):.2f}")
                        except (ValueError, TypeError):
                            label_lines.append(f"{col}: {val}")
            
            # Draw labels - stack vertically within the image's Y range
            if label_lines:
                line_height = self.label_font_size + 2
                total_text_height = len(label_lines) * line_height
                
                # Center labels vertically within the image row
                start_y = img_y_top - (img_height - total_text_height) / 2 - line_height
                
                # Clamp to prevent going outside bounds
                start_y = min(start_y, img_y_top - 5)
                
                for i, line in enumerate(label_lines):
                    line_y = start_y - (i * line_height)
                    
                    # Skip if outside visible area
                    if line_y < y_bottom or line_y > y_top:
                        continue
                    
                    # Background rectangle with semi-transparent look
                    canvas.setFillColor(colors.HexColor("#f0f0f0"))  # Light gray background
                    canvas.rect(x, line_y - 2, label_column_width, line_height, fill=1, stroke=0)
                    
                    # Text
                    canvas.setFont("Helvetica", self.label_font_size)
                    canvas.setFillColor(self.label_text_color)
                    canvas.drawString(x + 3, line_y, line[:60])  # Truncate at 60 chars

    def _load_image_for_pdf(self, img: Any) -> Optional[PILImage.Image]:
        """Load an image for PDF rendering."""
        try:
            # Try .path attribute first (ImageInfo objects from DataCoordinator)
            if hasattr(img, 'path') and img.path:
                if isinstance(img.path, str) and os.path.exists(img.path):
                    return PILImage.open(img.path)
            
            # Try .image attribute (CompartmentImage objects)
            if hasattr(img, 'image') and img.image is not None:
                if isinstance(img.image, np.ndarray):
                    img_rgb = cv2.cvtColor(img.image, cv2.COLOR_BGR2RGB)
                    return PILImage.fromarray(img_rgb)
                elif isinstance(img.image, PILImage.Image):
                    return img.image
            
            # Try image_path
            if hasattr(img, 'image_path') and img.image_path:
                if isinstance(img.image_path, str) and os.path.exists(img.image_path):
                    return PILImage.open(img.image_path)
                elif isinstance(img.image_path, PILImage.Image):
                    return img.image_path
            
            # Try load_image method
            if hasattr(img, 'load_image'):
                img.load_image()
                if hasattr(img, 'image') and img.image is not None:
                    if isinstance(img.image, np.ndarray):
                        img_rgb = cv2.cvtColor(img.image, cv2.COLOR_BGR2RGB)
                        return PILImage.fromarray(img_rgb)
                    elif isinstance(img.image, PILImage.Image):
                        return img.image
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error loading image: {e}")
            return None

    def _get_value_color(
        self,
        value: float,
        color_map_name: str,
        visualizer: Optional[Any],
    ) -> colors.Color:
        """Get color for a value from color map."""
        # Default blue gradient based on value
        try:
            if visualizer and hasattr(visualizer, 'get_color_for_value'):
                bgr = visualizer.get_color_for_value(value, color_map_name)
                return colors.HexColor(f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}")
        except:
            pass
        
        # Fallback gradient
        normalized = min(max(value / 100, 0), 1)  # Assume 0-100 range
        r = int(normalized * 200)
        g = int((1 - normalized) * 200)
        b = 100
        return colors.HexColor(f"#{r:02x}{g:02x}{b:02x}")

    # =========================================================================
    # GRID MODE (Original functionality - kept for backward compatibility)
    # =========================================================================

    def _generate_grid_report(
        self,
        output_path: str,
        images: List[Any],
        drillhole_data_manager: Optional[Any],
        drillhole_visualizer: Optional[Any],
    ) -> bool:
        """Generate grid-style report (original functionality)."""
        try:
            # Use fixed page size for grid mode
            if self.page_size is None:
                self.page_size = A4
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=self.page_size,
                leftMargin=self.margins[0],
                rightMargin=self.margins[1],
                topMargin=self.margins[2],
                bottomMargin=self.margins[3],
                title=self.title,
            )

            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Title"],
                fontSize=16,
                textColor=colors.HexColor("#1a1a1a"),
                spaceAfter=12,
                alignment=TA_CENTER,
            )
            story.append(Paragraph(self.title, title_style))

            # Date
            if self.show_date:
                date_style = ParagraphStyle(
                    "DateStyle",
                    parent=styles["Normal"],
                    fontSize=9,
                    textColor=colors.grey,
                    alignment=TA_CENTER,
                    spaceAfter=20,
                )
                date_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                story.append(Paragraph(date_text, date_style))

            # Summary
            summary_style = ParagraphStyle(
                "SummaryStyle",
                parent=styles["Normal"],
                fontSize=10,
                spaceAfter=15,
            )
            summary_text = f"Total Images: {len(images)}"
            if self.group_by_hole:
                hole_count = len(set(img.hole_id for img in images))
                summary_text += f" | Drillholes: {hole_count}"
            story.append(Paragraph(summary_text, summary_style))
            story.append(Spacer(1, 10))

            # Group and add images
            if self.group_by_hole:
                holes_dict = {}
                for img in images:
                    if img.hole_id not in holes_dict:
                        holes_dict[img.hole_id] = []
                    holes_dict[img.hole_id].append(img)

                # Sort holes spatially if enabled
                hole_ids = list(holes_dict.keys())
                if self.spatial_grouping and self._collar_data is not None:
                    hole_ids = self.get_spatially_sorted_holes(hole_ids)
                else:
                    hole_ids = sorted(hole_ids)
                
                for hole_id in hole_ids:
                    hole_images = holes_dict[hole_id]
                    if self.show_hole_headers:
                        hole_header_style = ParagraphStyle(
                            "HoleHeader",
                            parent=styles["Heading2"],
                            fontSize=12,
                            textColor=colors.HexColor("#0066cc"),
                            spaceAfter=8,
                            spaceBefore=15,
                        )
                        story.append(Paragraph(f"Drillhole: {hole_id}", hole_header_style))

                    self._add_image_grid(
                        story, hole_images, drillhole_data_manager, drillhole_visualizer
                    )
                    
                    if self.one_hole_per_page:
                        story.append(PageBreak())
                    else:
                        story.append(Spacer(1, self.hole_spacing))
            else:
                self._add_image_grid(
                    story, images, drillhole_data_manager, drillhole_visualizer
                )

            # Build PDF
            if self.show_page_numbers:
                doc.build(story, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)
            else:
                doc.build(story)

            self.logger.info(f"✅ Grid PDF report generated: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error generating grid PDF: {e}", exc_info=True)
            return False

    def _add_image_grid(
        self,
        story: List,
        images: List[Any],
        data_manager: Optional[Any],
        visualizer: Optional[Any],
    ):
        """Add image grid to PDF story (original grid functionality)."""
        page_width = self.page_size[0] - self.margins[0] - self.margins[1]
        
        total_spacing = (self.images_per_row - 1) * self.image_spacing
        # Don't subtract viz_width from cell_width - it's added to each image internally
        cell_width = (page_width - total_spacing) / self.images_per_row
        
        # But we need to reduce cell_width if viz is enabled so composites fit
        if self.include_visualizations and self.viz_columns:
            cell_width = cell_width - self.viz_width * mm

        for i in range(0, len(images), self.images_per_row):
            row_images = images[i : i + self.images_per_row]
            
            table_data = []
            image_row = []

            for img in row_images:
                composite = self._create_composite_image(
                    img, cell_width, data_manager, visualizer
                )
                image_row.append(composite)

            while len(image_row) < self.images_per_row:
                image_row.append("")

            table_data.append(image_row)

            if self.show_depth_labels or self.show_classification_labels:
                label_row = []
                for img in row_images:
                    label_text = self._create_label_text(img)
                    label_para = Paragraph(
                        label_text,
                        ParagraphStyle(
                            "LabelStyle",
                            fontSize=self.label_font_size,
                            textColor=self.label_text_color,
                            alignment=TA_CENTER,
                            leading=self.label_font_size + 2,
                        ),
                    )
                    label_row.append(label_para)

                while len(label_row) < self.images_per_row:
                    label_row.append("")

                table_data.append(label_row)

            col_widths = [cell_width + (viz_total_width / self.images_per_row if self.include_visualizations else 0)] * self.images_per_row
            table = Table(table_data, colWidths=col_widths)

            table_style = [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]

            if self.show_depth_labels or self.show_classification_labels:
                table_style.append(("BACKGROUND", (0, 1), (-1, 1), self.label_bg_color))
                table_style.append(("TOPPADDING", (0, 1), (-1, 1), 3))
                table_style.append(("BOTTOMPADDING", (0, 1), (-1, 1), 3))

            table.setStyle(TableStyle(table_style))
            story.append(KeepTogether(table))
            story.append(Spacer(1, self.image_spacing))

    def _create_composite_image(
        self,
        img: Any,
        target_width: float,
        data_manager: Optional[Any],
        visualizer: Optional[Any],
    ) -> RLImage:
        """Create composite image (main image + visualization bars)."""
        try:
            pil_img = self._load_image_for_pdf(img)
            
            if pil_img is None:
                # Create placeholder
                placeholder = np.ones((300, 200, 3), dtype=np.uint8) * 240
                cv2.putText(placeholder, "No Image", (50, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                pil_img = PILImage.fromarray(cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB))

            # Calculate base dimensions
            aspect_ratio = pil_img.height / pil_img.width
            img_width_px = int(target_width)  # Use as pixels for compositing
            img_height_px = int(img_width_px * aspect_ratio)
            
            # Resize image
            pil_img = pil_img.resize((img_width_px, img_height_px), PILImage.Resampling.LANCZOS)
            
            # Add visualization bars if enabled
            if self.include_visualizations and self.viz_columns and hasattr(img, 'csv_data') and img.csv_data:
                pil_img = self._add_viz_bars_to_image(pil_img, img, img_height_px)
            
            img_buffer = BytesIO()
            pil_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            # Calculate final dimensions in points
            final_width = target_width
            if self.include_visualizations and self.viz_columns:
                final_width = target_width + self.viz_width * mm
            final_height = final_width * (pil_img.height / pil_img.width)
            
            return RLImage(img_buffer, width=final_width, height=final_height)

        except Exception as e:
            self.logger.error(f"Error creating composite image: {e}", exc_info=True)
            # Return minimal placeholder
            placeholder = np.ones((100, 100, 3), dtype=np.uint8) * 200
            pil_img = PILImage.fromarray(cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB))
            img_buffer = BytesIO()
            pil_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            return RLImage(img_buffer, width=100, height=100)

    def _add_viz_bars_to_image(
        self,
        pil_img: PILImage.Image,
        img: Any,
        img_height_px: int,
    ) -> PILImage.Image:
        """Add visualization bars to the right side of an image for grid mode.
        
        Matches the UI rendering in logging_review_dialog._add_data_visualizations()
        """
        from PIL import ImageDraw, ImageFont
        
        num_columns = len(self.viz_columns)
        if num_columns == 0:
            return pil_img
        
        # Calculate viz bar dimensions - distribute equally across image height
        viz_bar_width_px = int(self.viz_width * 2.83465)  # mm to pixels at ~72dpi
        col_height_px = img_height_px // num_columns
        
        # Create new image with space for viz bars
        new_width = pil_img.width + viz_bar_width_px
        composite = PILImage.new('RGB', (new_width, pil_img.height), (255, 255, 255))
        composite.paste(pil_img, (0, 0))
        
        draw = ImageDraw.Draw(composite)
        
        # Try to get a font
        try:
            font = ImageFont.truetype("arial.ttf", 10)
            font_small = ImageFont.truetype("arial.ttf", 8)
        except:
            font = ImageFont.load_default()
            font_small = font
        
        # Draw each viz column as a vertical bar
        for i, viz_config in enumerate(self.viz_columns):
            column_name = viz_config.get("column", "")
            color_map = viz_config.get("color_map")
            
            # Strip source suffix for lookup
            bare_column_name = column_name.split(" (")[0] if " (" in column_name else column_name
            
            # Get value from csv_data
            value = None
            if img.csv_data:
                value = img.csv_data.get(column_name)
                if value is None:
                    value = img.csv_data.get(bare_column_name)
                if value is None:
                    # Try case-insensitive
                    for k, v in img.csv_data.items():
                        if k.lower() == column_name.lower() or k.lower() == bare_column_name.lower():
                            value = v
                            break
            
            # Calculate bar position
            bar_x = pil_img.width
            bar_y = i * col_height_px
            bar_height = col_height_px
            
            # Get color
            color_hex = "#808080"  # Default gray
            if color_map and value is not None:
                try:
                    if hasattr(color_map, 'get_color'):
                        bgr = color_map.get_color(value)
                        color_hex = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
                    elif hasattr(color_map, 'get_color_for_value'):
                        bgr = color_map.get_color_for_value(value)
                        color_hex = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
                except:
                    pass
            
            # Draw bar
            draw.rectangle(
                [bar_x, bar_y, bar_x + viz_bar_width_px, bar_y + bar_height],
                fill=color_hex,
                outline="#cccccc"
            )
            
            # Calculate text color based on brightness
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            text_color = "white" if brightness < 128 else "black"
            
            # Draw label and value
            display_name = column_name[:8] + "..." if len(column_name) > 8 else column_name
            value_text = f"{value:.1f}" if isinstance(value, (int, float)) else (str(value) if value else "-")
            
            text_y = bar_y + bar_height // 2
            draw.text(
                (bar_x + viz_bar_width_px // 2, text_y - 6),
                display_name,
                fill=text_color,
                font=font_small,
                anchor="mm"
            )
            draw.text(
                (bar_x + viz_bar_width_px // 2, text_y + 6),
                value_text,
                fill=text_color,
                font=font,
                anchor="mm"
            )
        
        return composite

    def _create_label_text(self, img: Any) -> str:
        """Create label text for image."""
        parts = []

        if self.show_depth_labels:
            parts.append(f"{img.hole_id} | {img.depth_from:.1f}-{img.depth_to:.1f}m")

        if self.show_classification_labels:
            if hasattr(img.classification, "value"):
                class_text = img.classification.value
            else:
                class_text = str(img.classification)

            if class_text and class_text != "":
                parts.append(f"<b>{class_text}</b>")

            if hasattr(img, "tags") and img.tags:
                tag_text = ", ".join(sorted(img.tags))
                parts.append(f"[{tag_text}]")

        return " | ".join(parts) if parts else f"{img.hole_id}"

    def _add_page_number(self, canvas, doc):
        """Add page number to bottom of page."""
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(self.page_size[0] / 2, 10 * mm, text)
        canvas.restoreState()