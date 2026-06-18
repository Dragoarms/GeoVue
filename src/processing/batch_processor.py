"""
Batch Processing Manager for automatic image processing
Handles quality checks and automatic processing without user intervention
"""

import logging
import os
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import traceback
from datetime import datetime


@dataclass
class ProcessingResult:
    """Result of processing a single image"""

    filename: str
    success: bool
    skipped: bool
    reason: Optional[str] = None
    metadata: Optional[Dict] = None
    marker_count: int = 0
    markers_in_order: bool = False
    
    # Enhanced processing details
    compartment_count: int = 0
    compartment_filenames: Optional[List[str]] = None
    interpolated_indices: Optional[List[int]] = None
    rotation_angle: Optional[float] = None
    scale_px_per_cm: Optional[float] = None
    scale_confidence: Optional[float] = None
    image_uid: Optional[str] = None
    registers_updated: Optional[List[str]] = None
    processing_timestamp: Optional[str] = None
    is_duplicate: bool = False
    duplicate_action: Optional[str] = None


@dataclass
class BatchProcessingSummary:
    """Summary of batch processing operation"""
    
    total: int = 0
    successful: int = 0
    skipped: int = 0
    failed: int = 0
    total_compartments: int = 0
    total_interpolated: int = 0
    results: List[ProcessingResult] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []


class BatchProcessor:
    """Handles automatic batch processing with quality checks"""

    def __init__(self, app, config, skip_duplicates: bool = False):
        """Initialize the batch processor

        Args:
            app: Main application instance
            config: Configuration dictionary
            skip_duplicates: If True, skip files that duplicate existing intervals.
                           If False, process duplicates for later wet/dry selection.
        """
        self.app = app
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.skip_duplicates = skip_duplicates

        # Processing state
        self.auto_mode = False
        self.current_results = []
        self.skipped_files = []
        self.cancel_requested = False

        # Quality check thresholds
        self.min_markers_required = 20
        self.required_metadata_pattern = re.compile(
            r"^([A-Z]{2,4}\d+)_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)m?.*\.(jpg|jpeg|png|tif|tiff)$",
            re.IGNORECASE,
        )

    def request_cancel(self):
        """Request cancellation of batch processing after current image completes."""
        self.cancel_requested = True
        self.logger.info("Cancellation requested - will stop after current image")

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self.cancel_requested

    def validate_filename_metadata(self, filepath: str) -> Tuple[bool, Optional[Dict]]:
        """Validate filename contains required metadata

        Args:
            filepath: Path to image file

        Returns:
            Tuple of (is_valid, metadata_dict)
        """
        filename = os.path.basename(filepath)
        match = self.required_metadata_pattern.match(filename)

        if not match:
            self.logger.warning(f"Filename lacks required metadata pattern: {filename}")
            return False, None

        hole_id = match.group(1)
        depth_from = int(match.group(2))
        depth_to = int(match.group(3))

        metadata = {
            "hole_id": hole_id,
            "depth_from": depth_from,
            "depth_to": depth_to,
            "compartment_interval": 1,  # Default - should also be int
        }

        # Infer interval from depth range
        depth_range = depth_to - depth_from
        if 18 <= depth_range <= 22:  # Close to 20m
            metadata["compartment_interval"] = 1
        elif 38 <= depth_range <= 42:  # Close to 40m (2m intervals)
            metadata["compartment_interval"] = 2

        self.logger.info(f"Extracted metadata from filename: {metadata}")
        return True, metadata

    def should_process_automatically(
        self, image_path: str
    ) -> Tuple[bool, ProcessingResult]:
        """Determine if image should be processed automatically based on filename only

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (should_process, processing_result)
        """
        filename = os.path.basename(image_path)

        # Check filename metadata
        has_metadata, metadata = self.validate_filename_metadata(image_path)

        # Create result object
        result = ProcessingResult(
            filename=filename,
            success=False,
            skipped=False,
            metadata=metadata,
            marker_count=0,  # Will be determined during actual processing
            markers_in_order=False,
        )

        # Decision logic - only check filename
        if not has_metadata:
            result.skipped = True
            result.reason = "No metadata in filename - requires manual input"
            return False, result

        # Check for duplicates and handle based on skip_duplicates setting
        if metadata and hasattr(self.app, "duplicate_handler"):
            hole_id = metadata.get("hole_id")
            depth_from = metadata.get("depth_from")
            depth_to = metadata.get("depth_to")

            if hole_id and depth_from is not None and depth_to is not None:
                is_duplicate = self.app.duplicate_handler.is_duplicate(
                    hole_id, depth_from, depth_to
                )

                if is_duplicate:
                    if self.skip_duplicates:
                        # Skip duplicates - user only wants new intervals
                        self.logger.info(
                            f"Duplicate detected for {hole_id} {depth_from}-{depth_to}m - "
                            f"skipping (skip_duplicates=True)"
                        )
                        result.skipped = True
                        result.reason = f"Duplicate interval (already exists in register)"
                        return False, result
                    else:
                        # Process duplicates - defer to QAQC for Wet/Dry resolution
                        self.logger.info(
                            f"Duplicate detected for {hole_id} {depth_from}-{depth_to}m - "
                            f"will process and defer Wet/Dry resolution to QAQC"
                        )
                        # Mark in metadata that this is a duplicate for deferred handling
                        metadata["is_duplicate"] = True
                        metadata["duplicate_mode"] = "auto_deferred"

        # All checks passed - let process_image handle marker validation
        return True, result

    def process_batch(self, image_files: List[str], progress_callback=None) -> Dict:
        """Process a batch of images automatically

        Args:
            image_files: List of image file paths
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with processing statistics
        """
        self.auto_mode = True
        self.current_results = []
        self.skipped_files = []
        self.cancel_requested = False

        stats = {
            "total": len(image_files),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "cancelled": False,
            "skipped_details": [],
        }

        self.logger.info(
            f"Starting batch processing of {len(image_files)} images in auto mode"
        )

        for idx, image_path in enumerate(image_files):
            # Check for cancellation before starting next image
            if self.cancel_requested:
                self.logger.info("Batch processing cancelled by user")
                stats["cancelled"] = True
                break

            filename = os.path.basename(image_path)

            try:
                # Update progress with simple message
                if progress_callback:
                    progress = (idx / len(image_files)) * 100
                    progress_callback(
                        f"Processing image {idx + 1} of {len(image_files)}",
                        progress,
                        filename
                    )

                # Quick pre-check of filename
                self.logger.info(f"Pre-checking image: {image_path}")

                # Check if should process automatically based on filename
                should_process, result = self.should_process_automatically(image_path)

                if not should_process:
                    # Skip this image
                    self.logger.info(f"Skipping {result.filename}: {result.reason}")
                    stats["skipped"] += 1
                    stats["skipped_details"].append(
                        {
                            "filename": result.filename,
                            "reason": result.reason,
                            "marker_count": result.marker_count,
                        }
                    )
                    self.skipped_files.append(image_path)
                    continue

                # Process the image with extracted metadata
                self.logger.info(
                    f"Auto-processing {result.filename} with metadata: {result.metadata}"
                )

                # Inject metadata for automatic processing
                self.app.metadata = result.metadata
                self.app.auto_mode = True  # Flag for skipping dialogs

                # Process the image
                process_result = self.app.process_image(image_path)

                # Check the result type
                if process_result == "quality_skip":
                    # Image failed quality checks - skip for manual review
                    stats["skipped"] += 1
                    result.skipped = True
                    result.success = False
                    result.reason = (
                        "Failed automated quality checks - needs manual review"
                    )
                    self.skipped_files.append(image_path)
                    self.logger.info(
                        f"Quality skip: {result.filename} added to manual review list"
                    )
                elif isinstance(process_result, dict):
                    # Successfully processed with detailed result
                    stats["processed"] += 1
                    result.success = True
                    
                    # Extract detailed processing information
                    result.compartment_count = process_result.get("compartment_count", 0)
                    result.compartment_filenames = process_result.get("compartment_filenames", [])
                    result.interpolated_indices = process_result.get("interpolated_indices", [])
                    result.rotation_angle = process_result.get("rotation_angle")
                    result.scale_px_per_cm = process_result.get("scale_px_per_cm")
                    result.scale_confidence = process_result.get("scale_confidence")
                    result.image_uid = process_result.get("image_uid")
                    result.registers_updated = process_result.get("registers_updated", [])
                    result.processing_timestamp = process_result.get("processing_timestamp")
                    result.is_duplicate = process_result.get("is_duplicate", False)
                    result.duplicate_action = process_result.get("duplicate_action")
                    result.marker_count = process_result.get("marker_count", 0)
                    
                    self.logger.info(
                        f"Successfully processed: {result.filename} - "
                        f"{result.compartment_count} compartments extracted"
                    )

                    # Update progress with success status
                    if progress_callback:
                        progress = ((idx + 1) / len(image_files)) * 100
                        progress_callback(
                            f"Processing image {idx + 1} of {len(image_files)}",
                            progress,
                            f"Extracted {result.compartment_count} compartments from {filename}"
                        )

                elif process_result is True:
                    # Legacy boolean result - success without details
                    stats["processed"] += 1
                    result.success = True
                    self.logger.info(f"Successfully processed: {result.filename}")

                    if progress_callback:
                        progress = ((idx + 1) / len(image_files)) * 100
                        progress_callback(
                            f"Processing image {idx + 1} of {len(image_files)}",
                            progress,
                            f"Processed {filename}"
                        )
                else:
                    # Processing failed for other reasons
                    stats["failed"] += 1
                    result.success = False
                    result.reason = "Processing error"
                    self.logger.error(f"Processing failed: {result.filename}")

                    if progress_callback:
                        progress = ((idx + 1) / len(image_files)) * 100
                        progress_callback(
                            f"Processing image {idx + 1} of {len(image_files)}",
                            progress,
                            f"FAILED: {filename} - skipping"
                        )

                self.current_results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {str(e)}")
                self.logger.error(traceback.format_exc())
                stats["failed"] += 1

                result = ProcessingResult(
                    filename=os.path.basename(image_path),
                    success=False,
                    skipped=False,
                    reason=str(e),
                )
                self.current_results.append(result)

        # Reset auto mode
        self.auto_mode = False
        self.app.auto_mode = False

        self.logger.info(f"Batch processing complete: {stats}")
        return stats

    def get_processing_report(self) -> str:
        """Generate a comprehensive summary report of batch processing"""
        if not self.current_results:
            return "No processing results available"

        report = []
        report.append("=" * 80)
        report.append("BATCH PROCESSING REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Statistics
        total = len(self.current_results)
        successful = sum(1 for r in self.current_results if r.success)
        skipped = sum(1 for r in self.current_results if r.skipped)
        failed = total - successful - skipped
        total_compartments = sum(r.compartment_count for r in self.current_results if r.success)
        total_interpolated = sum(
            len(r.interpolated_indices) for r in self.current_results 
            if r.success and r.interpolated_indices
        )

        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Images:              {total}")
        report.append(f"Successfully Processed:    {successful}")
        report.append(f"Skipped (Quality Issues):  {skipped}")
        report.append(f"Failed:                    {failed}")
        report.append(f"Total Compartments:        {total_compartments}")
        if total_interpolated > 0:
            report.append(f"Interpolated Boundaries:   {total_interpolated}")
        report.append("")

        # Successful processing details
        if successful > 0:
            report.append("=" * 80)
            report.append("SUCCESSFULLY PROCESSED")
            report.append("=" * 80)

            for result in self.current_results:
                if result.success:
                    report.append("")
                    report.append(f"{result.filename}")
                    report.append("-" * 60)
                    
                    # Metadata
                    if result.metadata:
                        hole_id = result.metadata.get("hole_id", "Unknown")
                        depth_from = result.metadata.get("depth_from", "?")
                        depth_to = result.metadata.get("depth_to", "?")
                        report.append(f"  Hole ID:        {hole_id}")
                        report.append(f"  Depth Range:    {depth_from}-{depth_to}m")
                    
                    # Processing details
                    report.append(f"  Markers Found:  {result.marker_count}")
                    report.append(f"  Compartments:   {result.compartment_count}")
                    
                    if result.rotation_angle is not None:
                        report.append(f"  Rotation:       {result.rotation_angle:.2f}°")
                    
                    if result.scale_px_per_cm is not None:
                        confidence = result.scale_confidence or 0
                        report.append(f"  Scale:          {result.scale_px_per_cm:.2f} px/cm ({confidence:.0f}% confidence)")
                    
                    # Interpolation
                    if result.interpolated_indices:
                        interp_str = ", ".join(str(i+1) for i in result.interpolated_indices)
                        report.append(f"  Interpolated:   Compartments {interp_str}")
                    
                    # Duplicate handling
                    if result.is_duplicate:
                        action = result.duplicate_action or "pending QAQC"
                        report.append(f"  Duplicate:      Yes ({action})")
                    
                    # Image UID
                    if result.image_uid:
                        report.append(f"  Image UID:      {result.image_uid[:8]}...")
                    
                    # Registers updated
                    if result.registers_updated:
                        registers = ", ".join(result.registers_updated)
                        report.append(f"  Registers:      {registers}")
                    
                    # Compartment files
                    if result.compartment_filenames:
                        report.append(f"  Saved Files:")
                        # Group by suffix for cleaner display
                        for i, fname in enumerate(result.compartment_filenames[:5]):
                            report.append(f"    - {os.path.basename(fname)}")
                        if len(result.compartment_filenames) > 5:
                            remaining = len(result.compartment_filenames) - 5
                            report.append(f"    ... and {remaining} more")

        # Details of skipped files
        if skipped > 0:
            report.append("")
            report.append("=" * 80)
            report.append("SKIPPED FILES (Require Manual Review)")
            report.append("=" * 80)

            # Group by reason
            duplicate_skips = []
            metadata_skips = []
            quality_skips = []

            for result in self.current_results:
                if result.skipped:
                    reason = result.reason or ""
                    if "Duplicate" in reason:
                        duplicate_skips.append(result)
                    elif "metadata" in reason.lower():
                        metadata_skips.append(result)
                    else:
                        quality_skips.append(result)

            if duplicate_skips:
                report.append("")
                report.append("Duplicates Found:")
                for result in duplicate_skips:
                    report.append(f"  [SKIP] {result.filename}")

            if metadata_skips:
                report.append("")
                report.append("Missing Metadata:")
                for result in metadata_skips:
                    report.append(f"  [WARN] {result.filename}")

            if quality_skips:
                report.append("")
                report.append("Quality Issues:")
                for result in quality_skips:
                    report.append(f"  [WARN] {result.filename}")
                    if result.marker_count > 0:
                        report.append(f"      Markers found: {result.marker_count}")

        # Details of failed files
        if failed > 0:
            report.append("")
            report.append("=" * 80)
            report.append("FAILED FILES")
            report.append("=" * 80)
            for result in self.current_results:
                if not result.success and not result.skipped:
                    report.append(f"")
                    report.append(f"  [FAIL] {result.filename}")
                    report.append(f"     Error: {result.reason}")

        report.append("")
        report.append("=" * 80)
        return "\n".join(report)