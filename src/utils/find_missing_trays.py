#!/usr/bin/env python3
"""
Standalone script to identify missing chip tray intervals AND missing compartments.

Compares expected tray intervals (from drillhole data CSV) against actual tray images
in the Approved Originals folder to identify missing trays.

Additionally checks existing trays for missing compartment images by comparing
expected compartments (from CSV intervals) against actual compartment images
in the Approved Compartment Images folder.

Usage:
    python find_missing_trays_and_compartments.py
"""

import pandas as pd
import logging
import re
import json
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional, Any
from tkinter import Tk, filedialog
import math
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrayAndCompartmentAnalyzer:
    """Analyzes drillhole data to find missing tray intervals and compartments."""

    def __init__(self):
        # Hole data
        self.hole_max_depths: Dict[str, float] = {}
        
        # Existing trays: hole_id -> set of (from, to) tuples
        self.existing_trays: Dict[str, Set[Tuple[float, float]]] = {}
        
        # Existing compartments: hole_id -> set of depth_to values
        self.existing_compartments: Dict[str, Set[int]] = {}
        
        # Expected compartments per tray: (hole_id, from, to) -> set of depth_to values
        self.expected_compartments: Dict[Tuple[str, float, float], Set[int]] = {}
        
        # Results
        self.missing_trays: List[Dict] = []
        self.trays_with_missing_compartments: List[Dict] = []
        
        # Register data (optional)
        self.compartment_register: Dict = {}
        self.original_register: Dict = {}
        
        # CSV data for expected intervals
        self.drillhole_intervals: pd.DataFrame = None

    def load_drillhole_data(self, csv_path: str) -> bool:
        """
        Load drillhole data and calculate maximum depths per hole.
        Also stores the individual intervals for compartment checking.
        
        Args:
            csv_path: Path to drillhole data CSV
            
        Returns:
            True if successfully loaded
        """
        try:
            logger.info(f"Loading drillhole data from: {csv_path}")
            
            # Read CSV
            df = pd.read_csv(csv_path, low_memory=False)
            logger.info(f"CSV loaded, shape: {df.shape}")
            logger.info(f"CSV columns: {df.columns.tolist()}")

            # Map column names - check for various possible names (case-insensitive)
            hole_columns = ["HOLEID", "HoleID", "Hole_ID", "DHID"]
            from_columns = ["SAMPFROM", "From", "SampleFrom", "DepthFrom", "FROM", "from"]
            to_columns = ["SAMPTO", "To", "SampleTo", "DepthTo", "TO", "to"]

            # Create case-insensitive column lookup
            columns_lower = {col.lower(): col for col in df.columns}
            
            # Find which columns exist (case-insensitive)
            hole_col = next((columns_lower.get(col.lower()) for col in hole_columns if col.lower() in columns_lower), None)
            from_col = next((columns_lower.get(col.lower()) for col in from_columns if col.lower() in columns_lower), None)
            to_col = next((columns_lower.get(col.lower()) for col in to_columns if col.lower() in columns_lower), None)

            if not (hole_col and from_col and to_col):
                missing = []
                if not hole_col:
                    missing.append("HoleID (tried: " + ", ".join(hole_columns) + ")")
                if not from_col:
                    missing.append("From (tried: " + ", ".join(from_columns) + ")")
                if not to_col:
                    missing.append("To (tried: " + ", ".join(to_columns) + ")")
                logger.error(f"CSV missing required columns: {missing}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                return False

            # Rename columns to standard names
            df = df.rename(columns={
                hole_col: "HoleID",
                from_col: "From",
                to_col: "To"
            })
            logger.info(f"Mapped columns: {hole_col}->HoleID, {from_col}->From, {to_col}->To")

            # Clean and process data
            df["HoleID"] = df["HoleID"].astype(str).str.strip().str.upper()
            df["From"] = pd.to_numeric(df["From"], errors="coerce")
            df["To"] = pd.to_numeric(df["To"], errors="coerce")

            # Remove rows with invalid data
            df = df.dropna(subset=["HoleID", "From", "To"])
            logger.info(f"After cleaning: {len(df)} valid rows")

            # Store the intervals for compartment checking
            self.drillhole_intervals = df[["HoleID", "From", "To"]].copy()
            
            # Detect interval spacing
            sample_intervals = df.groupby("HoleID").apply(
                lambda x: (x["To"] - x["From"]).mode().iloc[0] if len(x) > 0 else 1
            )
            logger.info(f"Detected interval spacings: {sample_intervals.unique()}")

            # Group by HoleID and get max depth
            grouped = df.groupby("HoleID")["To"].max().reset_index()

            # Store max depths, rounded up to nearest 20
            self.hole_max_depths = {}
            for _, row in grouped.iterrows():
                hole_id = row["HoleID"]
                max_depth = row["To"]
                # Round up to nearest 20
                rounded_max_depth = math.ceil(max_depth / 20) * 20
                self.hole_max_depths[hole_id] = rounded_max_depth
                
                if rounded_max_depth != max_depth:
                    logger.debug(
                        f"Hole {hole_id}: rounded max depth from {max_depth} to {rounded_max_depth}"
                    )

            logger.info(f"Successfully loaded max depths for {len(self.hole_max_depths)} holes")
            logger.info(f"Total intervals loaded: {len(self.drillhole_intervals)}")
            
            # Log some examples
            for i, (hole_id, max_depth) in enumerate(list(self.hole_max_depths.items())[:5]):
                logger.info(f"  Example: {hole_id} -> {max_depth}m")
            
            return True

        except Exception as e:
            logger.error(f"Error loading drillhole data: {e}", exc_info=True)
            return False

    def load_registers(self, register_folder: str) -> bool:
        """
        Load JSON register files if available.
        
        Args:
            register_folder: Path to folder containing register JSON files
            
        Returns:
            True if at least one register loaded successfully
        """
        try:
            register_path = Path(register_folder)
            
            # Try to find register folder
            possible_paths = [
                register_path / "Register Data (Do not edit)",
                register_path / "Chip Tray Register" / "Register Data (Do not edit)",
                register_path,
            ]
            
            register_dir = None
            for path in possible_paths:
                if path.exists():
                    register_dir = path
                    break
            
            if not register_dir:
                logger.warning("Register folder not found")
                return False
            
            logger.info(f"Found register folder: {register_dir}")
            
            # Load compartment register
            compartment_file = register_dir / "compartment_register.json"
            if compartment_file.exists():
                with open(compartment_file, 'r', encoding='utf-8') as f:
                    self.compartment_register = json.load(f)
                logger.info(f"Loaded compartment register: {len(self.compartment_register)} entries")
            
            # Load original images register
            original_file = register_dir / "original_images_register.json"
            if original_file.exists():
                with open(original_file, 'r', encoding='utf-8') as f:
                    self.original_register = json.load(f)
                logger.info(f"Loaded original images register: {len(self.original_register)} entries")
            
            return bool(self.compartment_register or self.original_register)
            
        except Exception as e:
            logger.error(f"Error loading registers: {e}", exc_info=True)
            return False

    def scan_approved_originals(self, root_folder: str) -> bool:
        """
        Recursively scan Approved Originals folder to find existing tray intervals.
        
        Filename pattern: {HoleID}_{From}-{To}_*.jpg
        Examples:
            KM0006_0-20_Original.jpg
            KM0006_0-20_Selected_Compartments.jpg
            KM0006_40-60_Replaced.jpg
        
        Args:
            root_folder: Root folder to scan (contains project/hole subfolders)
            
        Returns:
            True if scan completed successfully
        """
        try:
            root_path = Path(root_folder)
            if not root_path.exists():
                logger.error(f"Approved Originals folder not found: {root_folder}")
                return False

            logger.info(f"Scanning Approved Originals folder: {root_folder}")

            # Pattern to match tray filenames: HoleID_From-To_Suffix.ext
            filename_pattern = re.compile(
                r'^([A-Za-z0-9_-]+)_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)',
                re.IGNORECASE
            )

            # Scan all image files recursively
            image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
            found_files = 0
            
            for image_path in root_path.rglob('*'):
                if not image_path.is_file():
                    continue
                    
                if image_path.suffix.lower() not in image_extensions:
                    continue
                
                found_files += 1
                filename = image_path.stem  # Filename without extension
                
                # Try to parse filename
                match = filename_pattern.match(filename)
                if not match:
                    logger.debug(f"Skipping file (doesn't match pattern): {image_path.name}")
                    continue
                
                hole_id = match.group(1).strip().upper()
                depth_from = float(match.group(2))
                depth_to = float(match.group(3))
                
                # Add to existing trays set
                if hole_id not in self.existing_trays:
                    self.existing_trays[hole_id] = set()
                
                self.existing_trays[hole_id].add((depth_from, depth_to))
                
                logger.debug(f"Found tray: {hole_id} {depth_from}-{depth_to}m ({image_path.name})")

            logger.info(f"Scanned {found_files} image files")
            logger.info(f"Found trays for {len(self.existing_trays)} holes")
            
            # Log some examples
            for i, (hole_id, intervals) in enumerate(list(self.existing_trays.items())[:5]):
                logger.info(f"  Example: {hole_id} has {len(intervals)} tray intervals")
            
            return True

        except Exception as e:
            logger.error(f"Error scanning Approved Originals folder: {e}", exc_info=True)
            return False

    def scan_approved_compartments(self, root_folder: str) -> bool:
        """
        Recursively scan Approved Compartment Images folder to find existing compartments.
        
        Filename pattern: {HoleID}_CC_{DepthTo:03d}_{Wet/Dry}.{ext}
        Examples:
            KM0006_CC_045_Wet.png
            KM0006_CC_045_Dry.png
            KM0006_CC_001.png (no suffix)
        
        Args:
            root_folder: Root folder containing compartment images
            
        Returns:
            True if scan completed successfully
        """
        try:
            root_path = Path(root_folder)
            if not root_path.exists():
                logger.error(f"Approved Compartments folder not found: {root_folder}")
                return False

            logger.info(f"Scanning Approved Compartments folder: {root_folder}")

            # Pattern to match compartment filenames: HoleID_CC_DepthTo_Suffix.ext
            # The depth is zero-padded to 3 digits
            filename_pattern = re.compile(
                r'^([A-Za-z0-9_-]+)_CC_(\d{3})(?:_([A-Za-z]+))?',
                re.IGNORECASE
            )

            # Scan all image files recursively
            image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
            found_files = 0
            
            for image_path in root_path.rglob('*'):
                if not image_path.is_file():
                    continue
                    
                if image_path.suffix.lower() not in image_extensions:
                    continue
                
                # Skip "With_Data" subfolder images
                if "with_data" in str(image_path).lower():
                    continue
                
                found_files += 1
                filename = image_path.stem  # Filename without extension
                
                # Try to parse filename
                match = filename_pattern.match(filename)
                if not match:
                    logger.debug(f"Skipping file (doesn't match pattern): {image_path.name}")
                    continue
                
                hole_id = match.group(1).strip().upper()
                depth_to = int(match.group(2))  # Zero-padded depth
                
                # Add to existing compartments set
                if hole_id not in self.existing_compartments:
                    self.existing_compartments[hole_id] = set()
                
                self.existing_compartments[hole_id].add(depth_to)
                
                logger.debug(f"Found compartment: {hole_id} depth_to={depth_to}m ({image_path.name})")

            logger.info(f"Scanned {found_files} compartment image files")
            logger.info(f"Found compartments for {len(self.existing_compartments)} holes")
            
            # Log some examples
            for i, (hole_id, depths) in enumerate(list(self.existing_compartments.items())[:5]):
                logger.info(f"  Example: {hole_id} has {len(depths)} compartment images")
            
            return True

        except Exception as e:
            logger.error(f"Error scanning Approved Compartments folder: {e}", exc_info=True)
            return False

    def identify_missing_trays(self) -> bool:
        """
        Identify missing tray intervals by comparing expected vs actual trays.
        
        Expected trays are generated in 20m intervals from 0 to max depth.
        
        Returns:
            True if analysis completed successfully
        """
        try:
            logger.info("Identifying missing tray intervals...")
            
            self.missing_trays = []
            total_expected = 0
            total_covered = 0
            total_missing = 0

            # Check each hole
            for hole_id, max_depth in sorted(self.hole_max_depths.items()):
                # Generate expected tray intervals (20m intervals)
                expected_intervals = []
                depth = 0
                while depth < max_depth:
                    from_depth = depth
                    to_depth = min(depth + 20, max_depth)
                    expected_intervals.append((from_depth, to_depth))
                    depth += 20
                
                total_expected += len(expected_intervals)
                
                # Get existing intervals for this hole
                existing_intervals = self.existing_trays.get(hole_id, set())
                
                # Check each expected interval for coverage
                for exp_from, exp_to in expected_intervals:
                    # Check if ANY existing tray overlaps with this expected interval
                    has_coverage = False
                    for act_from, act_to in existing_intervals:
                        if exp_from < act_to and act_from < exp_to:
                            has_coverage = True
                            break
                    
                    if has_coverage:
                        total_covered += 1
                    else:
                        total_missing += 1
                        
                        # Check register for record
                        has_register_record = self._check_original_register(hole_id, exp_from, exp_to)
                        
                        self.missing_trays.append({
                            'HoleID': hole_id,
                            'DepthFrom': exp_from,
                            'DepthTo': exp_to,
                            'Interval': f"{exp_from}-{exp_to}",
                            'MaxDepth': max_depth,
                            'Status': 'Missing',
                            'HasRegisterRecord': has_register_record,
                        })

            logger.info(f"\n{'='*60}")
            logger.info(f"MISSING TRAYS SUMMARY:")
            logger.info(f"  Total holes analyzed: {len(self.hole_max_depths)}")
            logger.info(f"  Total expected 20m intervals: {total_expected}")
            logger.info(f"  Total covered intervals: {total_covered}")
            logger.info(f"  Total missing intervals: {total_missing}")
            if total_expected > 0:
                logger.info(f"  Coverage: {total_covered/total_expected*100:.1f}%")
            logger.info(f"{'='*60}\n")

            return True

        except Exception as e:
            logger.error(f"Error identifying missing trays: {e}", exc_info=True)
            return False

    def identify_missing_compartments(self) -> bool:
        """
        Identify trays with missing compartment images.
        
        For each existing tray, compares expected compartments (from CSV intervals)
        against actual compartment images found.
        
        Returns:
            True if analysis completed successfully
        """
        try:
            logger.info("Identifying missing compartments...")
            
            if self.drillhole_intervals is None or self.drillhole_intervals.empty:
                logger.error("No drillhole interval data loaded")
                return False
            
            self.trays_with_missing_compartments = []
            total_trays_checked = 0
            total_compartments_expected = 0
            total_compartments_found = 0
            total_compartments_missing = 0

            # Process each existing tray
            for hole_id, tray_intervals in self.existing_trays.items():
                for tray_from, tray_to in tray_intervals:
                    total_trays_checked += 1
                    
                    # Get expected compartments for this tray from CSV data
                    # These are the intervals that fall within the tray range
                    mask = (
                        (self.drillhole_intervals["HoleID"] == hole_id) &
                        (self.drillhole_intervals["From"] >= tray_from) &
                        (self.drillhole_intervals["To"] <= tray_to)
                    )
                    tray_intervals_df = self.drillhole_intervals[mask]
                    
                    if tray_intervals_df.empty:
                        logger.debug(f"No CSV intervals found for {hole_id} {tray_from}-{tray_to}")
                        continue
                    
                    # Expected compartment depth_to values
                    expected_depths = set(tray_intervals_df["To"].astype(int).tolist())
                    total_compartments_expected += len(expected_depths)
                    
                    # Get existing compartments for this hole
                    existing_depths = self.existing_compartments.get(hole_id, set())
                    
                    # Find which expected compartments exist
                    found_depths = expected_depths & existing_depths
                    missing_depths = expected_depths - existing_depths
                    
                    total_compartments_found += len(found_depths)
                    total_compartments_missing += len(missing_depths)
                    
                    if missing_depths:
                        # Check register for records
                        missing_with_register = []
                        for depth in sorted(missing_depths):
                            has_record = self._check_compartment_register(hole_id, depth)
                            missing_with_register.append({
                                'depth_to': depth,
                                'has_register_record': has_record,
                            })
                        
                        self.trays_with_missing_compartments.append({
                            'HoleID': hole_id,
                            'TrayFrom': tray_from,
                            'TrayTo': tray_to,
                            'TrayInterval': f"{int(tray_from)}-{int(tray_to)}",
                            'ExpectedCount': len(expected_depths),
                            'FoundCount': len(found_depths),
                            'MissingCount': len(missing_depths),
                            'MissingDepths': sorted(missing_depths),
                            'MissingWithRegister': missing_with_register,
                            'MissingPercentage': (len(missing_depths) / len(expected_depths)) * 100,
                        })

            # Sort by most missing to least missing
            self.trays_with_missing_compartments.sort(
                key=lambda x: (-x['MissingCount'], -x['MissingPercentage'], x['HoleID'], x['TrayFrom'])
            )

            logger.info(f"\n{'='*60}")
            logger.info(f"MISSING COMPARTMENTS SUMMARY:")
            logger.info(f"  Total trays checked: {total_trays_checked}")
            logger.info(f"  Total compartments expected: {total_compartments_expected}")
            logger.info(f"  Total compartments found: {total_compartments_found}")
            logger.info(f"  Total compartments missing: {total_compartments_missing}")
            logger.info(f"  Trays with missing compartments: {len(self.trays_with_missing_compartments)}")
            if total_compartments_expected > 0:
                logger.info(f"  Compartment coverage: {total_compartments_found/total_compartments_expected*100:.1f}%")
            logger.info(f"{'='*60}\n")

            # Log top 10 worst trays
            if self.trays_with_missing_compartments:
                logger.info("Top 10 trays with most missing compartments:")
                for i, tray in enumerate(self.trays_with_missing_compartments[:10]):
                    logger.info(
                        f"  {i+1}. {tray['HoleID']} {tray['TrayInterval']}m: "
                        f"{tray['MissingCount']}/{tray['ExpectedCount']} missing "
                        f"({tray['MissingPercentage']:.0f}%) - depths: {tray['MissingDepths']}"
                    )

            return True

        except Exception as e:
            logger.error(f"Error identifying missing compartments: {e}", exc_info=True)
            return False

    def _check_original_register(self, hole_id: str, depth_from: float, depth_to: float) -> bool:
        """Check if an original image has a register record."""
        if not self.original_register:
            return False
        
        # Register keys are typically: "HOLEID_FROM-TO" or similar
        possible_keys = [
            f"{hole_id}_{int(depth_from)}-{int(depth_to)}",
            f"{hole_id}_{depth_from}-{depth_to}",
            hole_id,
        ]
        
        for key in possible_keys:
            if key in self.original_register:
                return True
        
        return False

    def _check_compartment_register(self, hole_id: str, depth_to: int) -> bool:
        """Check if a compartment has a register record."""
        if not self.compartment_register:
            return False
        
        # Compartment register may be structured as nested dict or flat
        # Try various key formats
        possible_keys = [
            f"{hole_id}_{depth_to}",
            f"{hole_id}_CC_{depth_to:03d}",
            hole_id,
        ]
        
        for key in possible_keys:
            if key in self.compartment_register:
                entry = self.compartment_register[key]
                # Check if this depth_to is recorded
                if isinstance(entry, dict):
                    if str(depth_to) in entry or depth_to in entry.get('depths', []):
                        return True
                return True
        
        # Also check if nested structure
        if hole_id in self.compartment_register:
            hole_data = self.compartment_register[hole_id]
            if isinstance(hole_data, dict):
                if str(depth_to) in hole_data:
                    return True
        
        return False

    def export_missing_trays_pdf(self, output_path: str) -> bool:
        """
        Export missing trays to PDF with alternating row colors by HoleID.
        
        Args:
            output_path: Path to output PDF file
            
        Returns:
            True if export successful
        """
        try:
            logger.info(f"Exporting missing trays to PDF: {output_path}")
            
            if not self.missing_trays:
                logger.warning("No missing trays to export")
                return False

            # Sort data by HoleID, then by DepthFrom
            sorted_trays = sorted(self.missing_trays, key=lambda x: (x['HoleID'], x['DepthFrom']))
            
            # Create PDF
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.5*inch,
                bottomMargin=0.5*inch
            )
            
            elements = []
            
            # Title
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=20,
                alignment=TA_CENTER
            )
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            title = Paragraph(f"Missing Chip Tray Intervals<br/><font size=10>{timestamp}</font>", title_style)
            elements.append(title)
            
            # Calculate rows per column and columns per page
            available_height = 10 * inch
            row_height = 0.25 * inch
            rows_per_column = int(available_height / row_height) - 1
            
            columns_per_page = 3
            rows_per_page = rows_per_column * columns_per_page
            
            # Split data into page-sized chunks
            total_rows = len(sorted_trays)
            page_number = 0
            
            for page_start in range(0, total_rows, rows_per_page):
                page_number += 1
                page_end = min(page_start + rows_per_page, total_rows)
                page_data = sorted_trays[page_start:page_end]
                
                # Split page data into columns
                columns_data = []
                for col in range(columns_per_page):
                    col_start = col * rows_per_column
                    col_end = min((col + 1) * rows_per_column, len(page_data))
                    if col_start < len(page_data):
                        columns_data.append(page_data[col_start:col_end])
                
                # Build table for this page
                page_table_data = []
                
                # Header row
                header_row = []
                for _ in range(len(columns_data)):
                    header_row.extend(['HoleID', 'From', 'To', '  '])
                page_table_data.append(header_row[:-1])
                
                # Data rows
                max_col_rows = max(len(col) for col in columns_data) if columns_data else 0
                for row_idx in range(max_col_rows):
                    data_row = []
                    for col_data in columns_data:
                        if row_idx < len(col_data):
                            tray = col_data[row_idx]
                            data_row.extend([
                                tray['HoleID'],
                                str(int(tray['DepthFrom'])),
                                str(int(tray['DepthTo'])),
                                '  '
                            ])
                        else:
                            data_row.extend(['', '', '', '  '])
                    page_table_data.append(data_row[:-1])
                
                # Create table
                col_widths = []
                for _ in range(len(columns_data)):
                    col_widths.extend([1.2*inch, 0.6*inch, 0.6*inch, 0.1*inch])
                col_widths = col_widths[:-1]
                
                table = Table(page_table_data, colWidths=col_widths, repeatRows=1)
                
                # Build table style
                table_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ]
                
                # Apply alternating colors
                for col_idx, col_data in enumerate(columns_data):
                    col_offset = col_idx * 4
                    current_hole = None
                    color_toggle = False
                    
                    for row_idx, tray in enumerate(col_data):
                        actual_row = row_idx + 1
                        if tray['HoleID'] != current_hole:
                            current_hole = tray['HoleID']
                            color_toggle = not color_toggle
                        
                        bg_color = colors.HexColor('#ecf0f1') if color_toggle else colors.white
                        table_style.append(
                            ('BACKGROUND', (col_offset, actual_row), (col_offset + 2, actual_row), bg_color)
                        )
                
                table.setStyle(TableStyle(table_style))
                elements.append(table)
                
                if page_end < total_rows:
                    elements.append(PageBreak())
            
            doc.build(elements)
            
            logger.info(f"Successfully exported {len(sorted_trays)} missing tray records to PDF")
            return True

        except Exception as e:
            logger.error(f"Error exporting missing trays to PDF: {e}", exc_info=True)
            return False

    def export_missing_compartments_pdf(self, output_path: str) -> bool:
        """
        Export missing compartments analysis to PDF.
        
        Organized by severity (most missing first), with details of which
        specific compartments are missing from each tray.
        
        Args:
            output_path: Path to output PDF file
            
        Returns:
            True if export successful
        """
        try:
            logger.info(f"Exporting missing compartments to PDF: {output_path}")
            
            if not self.trays_with_missing_compartments:
                logger.warning("No missing compartments to export")
                return False

            # Create PDF
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.5*inch,
                bottomMargin=0.5*inch
            )
            
            elements = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=10,
                alignment=TA_CENTER
            )
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            title = Paragraph(
                f"Missing Compartment Images Analysis<br/><font size=10>{timestamp}</font>",
                title_style
            )
            elements.append(title)
            
            # Summary stats
            summary_style = ParagraphStyle(
                'Summary',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#34495e'),
                spaceAfter=15,
                alignment=TA_CENTER
            )
            
            total_missing = sum(t['MissingCount'] for t in self.trays_with_missing_compartments)
            total_expected = sum(t['ExpectedCount'] for t in self.trays_with_missing_compartments)
            
            summary = Paragraph(
                f"Total: {len(self.trays_with_missing_compartments)} trays with missing compartments | "
                f"{total_missing} compartments missing of {total_expected} expected in these trays",
                summary_style
            )
            elements.append(summary)
            elements.append(Spacer(1, 10))
            
            # Section styles
            section_style = ParagraphStyle(
                'Section',
                parent=styles['Heading2'],
                fontSize=12,
                textColor=colors.HexColor('#c0392b'),
                spaceBefore=15,
                spaceAfter=10,
            )
            
            # Categorize by severity
            critical = [t for t in self.trays_with_missing_compartments if t['MissingPercentage'] >= 80]
            severe = [t for t in self.trays_with_missing_compartments if 50 <= t['MissingPercentage'] < 80]
            moderate = [t for t in self.trays_with_missing_compartments if 20 <= t['MissingPercentage'] < 50]
            minor = [t for t in self.trays_with_missing_compartments if t['MissingPercentage'] < 20]
            
            # Export each category
            categories = [
                ("CRITICAL - 80%+ Missing (Prioritize These)", critical, colors.HexColor('#c0392b')),
                ("SEVERE - 50-79% Missing", severe, colors.HexColor('#e74c3c')),
                ("MODERATE - 20-49% Missing", moderate, colors.HexColor('#f39c12')),
                ("MINOR - <20% Missing (Check Later)", minor, colors.HexColor('#27ae60')),
            ]
            
            for category_name, category_trays, header_color in categories:
                if not category_trays:
                    continue
                
                # Section header
                section_header = Paragraph(f"{category_name} ({len(category_trays)} trays)", section_style)
                elements.append(section_header)
                
                # Build table data
                table_data = [['HoleID', 'Tray', 'Missing', 'Expected', '%', 'Missing Depths']]
                
                for tray in category_trays:
                    # Format missing depths - truncate if too long
                    missing_str = ', '.join(str(d) for d in tray['MissingDepths'][:10])
                    if len(tray['MissingDepths']) > 10:
                        missing_str += f" ... (+{len(tray['MissingDepths']) - 10} more)"
                    
                    table_data.append([
                        tray['HoleID'],
                        tray['TrayInterval'],
                        str(tray['MissingCount']),
                        str(tray['ExpectedCount']),
                        f"{tray['MissingPercentage']:.0f}%",
                        missing_str,
                    ])
                
                # Create table
                col_widths = [1.0*inch, 0.8*inch, 0.6*inch, 0.7*inch, 0.5*inch, 3.5*inch]
                table = Table(table_data, colWidths=col_widths, repeatRows=1)
                
                table_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), header_color),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 1), (4, -1), 'CENTER'),
                    ('ALIGN', (5, 1), (5, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]
                
                # Alternating row colors
                for i in range(1, len(table_data)):
                    bg_color = colors.HexColor('#f8f9fa') if i % 2 == 0 else colors.white
                    table_style.append(('BACKGROUND', (0, i), (-1, i), bg_color))
                
                table.setStyle(TableStyle(table_style))
                elements.append(table)
                elements.append(Spacer(1, 15))
            
            doc.build(elements)
            
            logger.info(f"Successfully exported missing compartments analysis to PDF")
            return True

        except Exception as e:
            logger.error(f"Error exporting missing compartments to PDF: {e}", exc_info=True)
            return False

    def export_combined_csv(self, output_path: str) -> bool:
        """
        Export detailed missing compartments data to CSV for further analysis.
        
        Args:
            output_path: Path to output CSV file
            
        Returns:
            True if export successful
        """
        try:
            logger.info(f"Exporting detailed analysis to CSV: {output_path}")
            
            if not self.trays_with_missing_compartments:
                logger.warning("No missing compartments to export")
                return False
            
            rows = []
            for tray in self.trays_with_missing_compartments:
                for missing_info in tray['MissingWithRegister']:
                    rows.append({
                        'HoleID': tray['HoleID'],
                        'TrayFrom': tray['TrayFrom'],
                        'TrayTo': tray['TrayTo'],
                        'TrayInterval': tray['TrayInterval'],
                        'MissingDepthTo': missing_info['depth_to'],
                        'HasRegisterRecord': missing_info['has_register_record'],
                        'TrayExpectedCount': tray['ExpectedCount'],
                        'TrayFoundCount': tray['FoundCount'],
                        'TrayMissingCount': tray['MissingCount'],
                        'TrayMissingPercentage': tray['MissingPercentage'],
                    })
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Exported {len(rows)} missing compartment records to CSV")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}", exc_info=True)
            return False


def select_csv_file() -> str:
    """Open file picker dialog to select drillhole data CSV."""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    csv_path = filedialog.askopenfilename(
        title="Select Drillhole Data CSV",
        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return csv_path


def select_folder(title: str) -> str:
    """Open folder picker dialog."""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    folder_path = filedialog.askdirectory(title=title)
    
    root.destroy()
    return folder_path


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("Missing Chip Trays and Compartments Analyzer")
    print("="*70 + "\n")

    # Step 1: Select drillhole data CSV
    print("Step 1: Select drillhole data CSV...")
    csv_path = select_csv_file()
    
    if not csv_path:
        logger.error("No CSV file selected. Exiting.")
        return
    
    logger.info(f"Selected CSV: {csv_path}")

    # Step 2: Select Approved Originals folder
    print("\nStep 2: Select Approved Originals folder...")
    originals_folder = select_folder("Select Approved Originals Folder")
    
    if not originals_folder:
        logger.error("No Approved Originals folder selected. Exiting.")
        return
    
    logger.info(f"Selected Approved Originals: {originals_folder}")

    # Step 3: Select Approved Compartments folder
    print("\nStep 3: Select Approved Compartment Images folder...")
    compartments_folder = select_folder("Select Approved Compartment Images Folder")
    
    if not compartments_folder:
        logger.error("No Approved Compartments folder selected. Exiting.")
        return
    
    logger.info(f"Selected Approved Compartments: {compartments_folder}")

    # Step 4: Optionally select register folder
    print("\nStep 4: Optionally select Chip Tray Register folder (Cancel to skip)...")
    register_folder = select_folder("Select Chip Tray Register Folder (Cancel to skip)")
    
    if register_folder:
        logger.info(f"Selected Register folder: {register_folder}")
    else:
        logger.info("No register folder selected - skipping register checks")

    # Step 5: Analyze
    print("\nStep 5: Analyzing...")
    analyzer = TrayAndCompartmentAnalyzer()

    # Load drillhole data
    if not analyzer.load_drillhole_data(csv_path):
        logger.error("Failed to load drillhole data. Exiting.")
        return

    # Load registers if available
    if register_folder:
        analyzer.load_registers(register_folder)

    # Scan approved originals folder
    if not analyzer.scan_approved_originals(originals_folder):
        logger.error("Failed to scan Approved Originals folder. Exiting.")
        return

    # Scan approved compartments folder
    if not analyzer.scan_approved_compartments(compartments_folder):
        logger.error("Failed to scan Approved Compartments folder. Exiting.")
        return

    # Identify missing trays
    if not analyzer.identify_missing_trays():
        logger.error("Failed to identify missing trays. Exiting.")
        return

    # Identify missing compartments
    if not analyzer.identify_missing_compartments():
        logger.error("Failed to identify missing compartments. Exiting.")
        return

    # Step 6: Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(originals_folder).parent
    
    # Export missing trays PDF
    if analyzer.missing_trays:
        print(f"\nExporting {len(analyzer.missing_trays)} missing tray records...")
        trays_pdf_path = output_base / f"missing_trays_{timestamp}.pdf"
        analyzer.export_missing_trays_pdf(str(trays_pdf_path))
        print(f"  -> {trays_pdf_path}")
    else:
        print("\nNo missing trays found - all expected tray intervals are present!")

    # Export missing compartments PDF
    if analyzer.trays_with_missing_compartments:
        print(f"\nExporting {len(analyzer.trays_with_missing_compartments)} trays with missing compartments...")
        compartments_pdf_path = output_base / f"missing_compartments_{timestamp}.pdf"
        analyzer.export_missing_compartments_pdf(str(compartments_pdf_path))
        print(f"  -> {compartments_pdf_path}")
        
        # Also export detailed CSV
        csv_path = output_base / f"missing_compartments_detail_{timestamp}.csv"
        analyzer.export_combined_csv(str(csv_path))
        print(f"  -> {csv_path}")
    else:
        print("\nNo missing compartments found - all trays have complete compartment images!")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()