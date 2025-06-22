"""
Register Synchronizer for updating registers based on OneDrive folder contents.

This module scans the OneDrive approved compartment images and original images
folders to ensure the registers are up-to-date with actual files.

Author: George Symonds
Created: 2025
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime
import pandas as pd
import cv2
from collections import defaultdict

from utils.json_register_manager import JSONRegisterManager


class RegisterSynchronizer:
    """Synchronizes register data with actual files in OneDrive folders."""
    
    def __init__(self, file_manager, config, progress_callback: Optional[Callable] = None):
        """
        Initialize the register synchronizer.
        
        Args:
            file_manager: FileManager instance
            config: Configuration dictionary
            progress_callback: Optional callback for progress updates (message, percentage)
        """
        self.file_manager = file_manager
        self.config = config
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)
        self.json_manager = None
        self.compartment_updates = []
        self.original_updates = []

        self._cached_paths = {
            # Shared paths
            'approved_folder': None,
            'processed_originals': None,
            'rejected_folder': None,
            'register_data': None,
            'approved_compartments': None,
            'review_compartments': None,
            # ===================================================
            # ADDED: Local paths
            # ===================================================
            'local_approved_folder': None,
            'local_processed_originals': None,
            'local_rejected_folder': None,
            'local_approved_compartments': None,
            'local_review_compartments': None
        }
        self._cache_paths()
            
    def _cache_paths(self):
        """Cache all required paths at initialization."""
        try:
            self._cached_paths['approved_folder'] = self.file_manager.get_shared_path('approved_originals')
            self._cached_paths['processed_originals'] = self.file_manager.get_shared_path('processed_originals')
            self._cached_paths['rejected_folder'] = self.file_manager.get_shared_path('rejected_originals')
            self._cached_paths['register_data'] = self.file_manager.get_shared_path('register_data')
            self._cached_paths['approved_compartments'] = self.file_manager.get_shared_path('approved_compartments')
            self._cached_paths['review_compartments'] = self.file_manager.get_shared_path('review_compartments')
            self._cached_paths['local_approved_folder'] = Path(self.file_manager.dir_structure.get('approved_originals'))
            self._cached_paths['local_processed_originals'] = Path(self.file_manager.dir_structure.get('processed_originals'))
            self._cached_paths['local_rejected_folder'] = Path(self.file_manager.dir_structure.get('rejected_originals'))
            self._cached_paths['local_approved_compartments'] = Path(self.file_manager.dir_structure.get('approved_compartments'))
            self._cached_paths['local_review_compartments'] = Path(self.file_manager.dir_structure.get('temp_review'))
            
            self.logger.info("Cached both shared and local folder paths for synchronization")
            for key, path in self._cached_paths.items():
                if path:
                    self.logger.debug(f"  {key}: {path}")
        except Exception as e:
            self.logger.error(f"Error caching paths: {e}")
        
    def set_json_manager(self, base_path: str):
        """Set up JSON manager for register operations."""
        try:
            self.json_manager = JSONRegisterManager(base_path, self.logger)
            self.logger.info("JSON manager initialized for register synchronization")
        except Exception as e:
            self.logger.error(f"Failed to initialize JSON manager: {e}")
            self.json_manager = None  
    



    def _perform_batch_updates(self):
        """Perform all batch updates at once."""
        if not self.json_manager:
            self.logger.warning("No JSON manager available for batch updates")
            return
        
        # ===================================================
        # NEW: Batch update compartments
        # ===================================================
        if self.compartment_updates:
            self.logger.info(f"Performing batch update of {len(self.compartment_updates)} compartments")
            
            # Use the batch update method
            successful = self.json_manager.batch_update_compartments(self.compartment_updates)
            self.logger.info(f"Successfully updated {successful} compartment entries")
        
        # ===================================================
        # NEW: Batch update originals
        # ===================================================
        if self.original_updates:
            self.logger.info(f"Performing batch update of {len(self.original_updates)} original images")
            
            # For originals, we need to update one by one (no batch method yet)
            # But we can still benefit from single file lock/unlock
            successful = 0
            for update in self.original_updates:
                try:
                    if self.json_manager.update_original_image(**update):
                        successful += 1
                except Exception as e:
                    self.logger.error(f"Error updating original: {e}")
            
            self.logger.info(f"Successfully updated {successful} original image entries")

    def synchronize_all(self) -> Dict:
        """
        Synchronize all registers with OneDrive folders.
        
        Returns:
            Dictionary with synchronization results
        """
        results = {
            'success': True,
            'compartments_added': 0,
            'missing_compartments': 0,
            'originals_added': 0,
            'originals_updated': 0,  # NEW
            'error': None
        }
        
        try:
            # Synchronize compartment images
            self._report_progress("Scanning approved compartment images...", 10)
            comp_results = self._sync_compartment_images()
            results['compartments_added'] = comp_results['added']
            
            # Synchronize original images
            self._report_progress("Scanning original images...", 40)
            orig_results = self._sync_original_images()
            results['originals_added'] = orig_results['added']
            results['originals_updated'] = orig_results.get('updated', 0)  # NEW
            
            # Validate existing entries
            self._report_progress("Validating existing entries...", 60)
            validation_results = self.validate_existing_entries()
            results['missing_files_compartments'] = validation_results['missing_compartments']
            results['missing_files_originals'] = validation_results['missing_originals']

            # Calculate missing hex colors
            self._report_progress("Calculating missing hex colors...", 75)
            hex_results = self.calculate_missing_hex_colors()
            results['hex_colors_calculated'] = hex_results['calculated']
            results['hex_colors_failed'] = hex_results['failed']
            
            # Sync wet/dry filenames
            self._report_progress("Syncing wet/dry filenames...", 80)
            wet_dry_results = self._sync_wet_dry_filenames()
            results['files_renamed'] = wet_dry_results['renamed']
            results['rename_failed'] = wet_dry_results['failed']
            if wet_dry_results['errors']:
                if 'errors' not in results:
                    results['errors'] = []
                results['errors'].extend(wet_dry_results['errors'])

            # Check for missing compartments
            self._report_progress("Checking for missing compartments...", 70)
            missing_results = self._check_missing_compartments()
            results['missing_compartments'] = missing_results['missing']
            
            self._report_progress("Synchronization complete", 100)

            # Batch update
            self._report_progress("Writing updates to register...", 90)
            self._perform_batch_updates()
            
            self._report_progress("Synchronization complete", 100)
            
            
        except Exception as e:
            self.logger.error(f"Synchronization error: {str(e)}")
            results['success'] = False
            results['error'] = str(e)
            
        return results
    
    def _report_progress(self, message: str, percentage: float):
        """Report progress if callback is available."""
        if self.progress_callback:
            self.progress_callback(message, percentage)
        
    def _sync_compartment_images(self) -> Dict:
        """
        Synchronize compartment images with register.
        
        Returns:
            Dictionary with results
        """
        results = {'added': 0, 'errors': []}
        
        # Get existing entries from register
        if self.json_manager:
            existing_df = self.json_manager.get_compartment_data()
            existing_keys = set()
            if not existing_df.empty:
                existing_keys = set(
                    (row['HoleID'], row['From'], row['To'])
                    for _, row in existing_df.iterrows()
                )
        else:
            existing_keys = set()
        
        # ===================================================
        # ADDED: Helper function to detect interval for a hole
        # ===================================================
        def detect_interval_for_hole(hole_id: str) -> int:
            """Detect the compartment interval for a specific hole based on existing data."""
            if existing_df.empty:
                return self.config.get('compartment_interval', 1)
            
            # Get existing compartments for this hole
            hole_data = existing_df[existing_df['HoleID'] == hole_id].copy()
            if hole_data.empty:
                return self.config.get('compartment_interval', 1)
            
            # Sort by To depth
            hole_data = hole_data.sort_values('To')
            to_values = hole_data['To'].tolist()
            
            if len(to_values) < 2:
                return self.config.get('compartment_interval', 1)
            
            # Calculate intervals between consecutive compartments
            intervals = []
            for i in range(1, len(to_values)):
                interval = to_values[i] - to_values[i-1]
                if interval > 0:  # Only positive intervals
                    intervals.append(interval)
            
            if not intervals:
                return self.config.get('compartment_interval', 1)
            
            # Find the most common interval (mode)
            from collections import Counter
            interval_counts = Counter(intervals)
            most_common_interval = interval_counts.most_common(1)[0][0]
            
            self.logger.debug(f"Detected interval for {hole_id}: {most_common_interval}m")
            return int(most_common_interval)
        
        # Scan for compartment images
        pattern = re.compile(r'^([A-Z]{2}\d{4})_CC_(\d{3,4})(?:_(Wet|Dry))?\.(?:png|jpg|jpeg|tiff|tif)$', re.IGNORECASE)
        
        # ===================================================
        # CHANGED: Check multiple locations
        # ===================================================
        locations_to_scan = [
            ('Found', self._cached_paths.get('approved_compartments')),
            ('For Review', self._cached_paths.get('review_compartments')),
            ('Found', self._cached_paths.get('local_approved_compartments')),
            ('For Review', self._cached_paths.get('local_review_compartments'))
        ]
        
        # Track found compartments to avoid duplicates
        found_compartments = set()
        
        for photo_status, base_path in locations_to_scan:
            if not base_path or not base_path.exists():
                continue
                
            self.logger.info(f"Scanning {photo_status} folder: {base_path}")
            
            # Scan through project folders
            for project_folder in os.listdir(str(base_path)):
                project_path = base_path / project_folder
                if not project_path.is_dir():
                    continue
                    
                # Scan through hole folders
                for hole_folder in os.listdir(str(project_path)):
                    hole_path = project_path / hole_folder
                    if not hole_path.is_dir():
                        continue
                    
                    # Scan for compartment images
                    for filename in os.listdir(str(hole_path)):
                        match = pattern.match(filename)
                        if match:
                            hole_id = match.group(1)
                            depth_to = int(match.group(2))
                            wet_dry = match.group(3) if match.group(3) else None
                            
                            # Skip if we've already found this compartment in a higher priority location
                            comp_key = (hole_id, depth_to)
                            if comp_key in found_compartments:
                                continue
                            
                            # ===================================================
                            # CHANGED: Detect interval based on existing data
                            # ===================================================
                            interval = detect_interval_for_hole(hole_id)
                            depth_from = depth_to - interval
                            
                            # Check if already in register
                            key = (hole_id, depth_from, depth_to)
                            if key not in existing_keys:
                                # Determine final photo status
                                if photo_status == 'For Review':
                                    final_status = 'For Review'
                                elif wet_dry:
                                    final_status = f'OK_{wet_dry}'
                                else:
                                    final_status = photo_status
                                
                                # Add to batch instead of immediate update
                                self.compartment_updates.append({
                                    'hole_id': hole_id,
                                    'depth_from': depth_from,
                                    'depth_to': depth_to,
                                    'photo_status': final_status,
                                    'approved_by': 'System Sync',
                                    'comments': f'Auto-imported from {filename} in {photo_status} folder'
                                })
                                
                                results['added'] += 1
                                existing_keys.add(key)  # Prevent duplicates in this run
                                found_compartments.add(comp_key)  # Mark as found
                                self.logger.info(f"Queued compartment: {hole_id} {depth_from}-{depth_to}m (Status: {final_status})")
        
        return results



    def _sync_original_images(self) -> Dict:
        """
        Synchronize original images with register.
        
        Returns:
            Dictionary with results
        """
        results = {'added': 0, 'updated': 0, 'errors': []}
        
        processed_path = self._cached_paths['processed_originals']
        if not processed_path or not os.path.exists(processed_path):
            self.logger.warning("Processed originals folder not found")
            return results
        
        # Get existing entries with their current file counts and filenames
        existing_entries = {}
        if self.json_manager:
            existing_df = self.json_manager.get_original_image_data()
            if not existing_df.empty:
                for _, row in existing_df.iterrows():
                    if pd.notna(row['Depth_From']) and pd.notna(row['Depth_To']):
                        key = (row['HoleID'], int(row['Depth_From']), int(row['Depth_To']))
                        existing_entries[key] = row.to_dict()
        
        # Dictionary to collect all files for each hole/depth combination
        original_files_by_key = {}
        
        # Updated pattern to capture optional numeric suffix
        pattern = re.compile(r'^([A-Z]{2}\d{4})_(\d+)-(\d+)_Original(?:_(\d+))?\.(?:png|jpg|jpeg|tiff|tif)$', re.IGNORECASE)
        
        # Scan folders
        for project_folder in os.listdir(processed_path):
            project_path = os.path.join(processed_path, project_folder)
            if not os.path.isdir(project_path):
                continue
                
            for hole_folder in os.listdir(project_path):
                hole_path = os.path.join(project_path, hole_folder)
                if not os.path.isdir(hole_path):
                    continue
                
                for filename in os.listdir(hole_path):
                    match = pattern.match(filename)
                    if match:
                        hole_id = match.group(1)
                        depth_from = int(match.group(2))
                        depth_to = int(match.group(3))
                        
                        key = (hole_id, depth_from, depth_to)
                        
                        # Collect all filenames for this key
                        if key not in original_files_by_key:
                            original_files_by_key[key] = []
                        original_files_by_key[key].append(filename)
                    
                    elif '_UPLOADED' in filename:
                        self.logger.warning(f"Found unexpected _UPLOADED suffix in OneDrive file: {filename}")
        
        # Process collected files
        for key, filenames in original_files_by_key.items():
            hole_id, depth_from, depth_to = key
            
            # Sort filenames to get the primary one first
            filenames.sort()
            primary_filename = filenames[0]
            file_count = len(filenames)
            all_filenames = ', '.join(filenames) if file_count > 1 else None
            
            if key in existing_entries:
                existing = existing_entries[key]
                existing_file_count = existing.get('File_Count', 1)
                existing_all_filenames = existing.get('All_Filenames', '')
                
    
                # FIXED: Only update if something actually changed
    
                # Check if there are actual changes
                has_changes = False
                
                # Check file count
                if existing_file_count != file_count:
                    has_changes = True
                    
                # Check filenames list
                if all_filenames != existing_all_filenames:
                    has_changes = True
                    
                # Check if File_Count field is missing (legacy data)
                if 'File_Count' not in existing:
                    has_changes = True
                
                if has_changes:
                    # Update existing entry
                    self.original_updates.append({
                        'hole_id': hole_id,
                        'depth_from': depth_from,
                        'depth_to': depth_to,
                        'original_filename': primary_filename,
                        'is_approved': True,
                        'upload_success': True,
                        'uploaded_by': existing.get('Uploaded_By', 'System Sync'),
                        'comments': f'Updated file count from {existing_file_count} to {file_count}',
                        'file_count': file_count,
                        'all_filenames': all_filenames
                    })
                    
                    results['updated'] += 1
                    self.logger.info(f"Queued update for original: {hole_id} {depth_from}-{depth_to}m (files: {existing_file_count} -> {file_count})")
                else:
                    self.logger.debug(f"No changes for: {hole_id} {depth_from}-{depth_to}m")
            else:
                # New entry
                self.original_updates.append({
                    'hole_id': hole_id,
                    'depth_from': depth_from,
                    'depth_to': depth_to,
                    'original_filename': primary_filename,
                    'is_approved': True,
                    'upload_success': True,
                    'uploaded_by': 'System Sync',
                    'comments': f'Auto-imported from OneDrive. File count: {file_count}',
                    'file_count': file_count,
                    'all_filenames': all_filenames
                })
                
                results['added'] += 1
                self.logger.info(f"Queued new original: {hole_id} {depth_from}-{depth_to}m ({file_count} files)")
        
        # Also check rejected folder
        rejected_path = self._cached_paths['rejected_folder']
        if rejected_path and os.path.exists(rejected_path):
            self._scan_rejected_folder(rejected_path, existing_entries, results)
        
        return results

    def validate_existing_entries(self) -> Dict:
        """
        Validate that files referenced in register still exist.
        Mark entries with missing files.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'missing_compartments': 0,
            'missing_originals': 0,
            'errors': []
        }
        
        try:
            # Check compartment images
            self._report_progress("Validating compartment image paths...", 20)
            comp_results = self._validate_compartment_images()
            results['missing_compartments'] = comp_results['missing']
            
            # Check original images
            self._report_progress("Validating original image paths...", 50)
            orig_results = self._validate_original_images()
            results['missing_originals'] = orig_results['missing']
            
            self._report_progress("Validation complete", 100)
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            results['errors'].append(str(e))
            
        return results
    
    def _validate_compartment_images(self) -> Dict:
        """Check if compartment images still exist."""
        results = {'missing': 0, 'updates': []}
        
        if not self.json_manager:
            return results
            
        # Get all compartment entries
        compartments_df = self.json_manager.get_compartment_data()
        if compartments_df.empty:
            return results
            
        approved_path = self._cached_paths['approved_folder']
        if not approved_path:
            self.logger.warning("Cannot validate - approved folder not found")
            return results
            
        for _, row in compartments_df.iterrows():
            # Skip if already marked as missing
            if row.get('Photo_Status') == 'MISSING_FILE':
                continue
                
            hole_id = row['HoleID']
            depth_to = int(row['To'])
            
            # Build expected path
            project_code = hole_id[:2].upper()
            expected_patterns = [
                f"{hole_id}_CC_{depth_to:03d}.png",
                f"{hole_id}_CC_{depth_to:03d}_Wet.png",
                f"{hole_id}_CC_{depth_to:03d}_Dry.png"
            ]
            
            # Check if any expected file exists
            file_found = False
            hole_path = os.path.join(approved_path, project_code, hole_id)
            
            if os.path.exists(hole_path):
                for pattern in expected_patterns:
                    if any(pattern.lower() in f.lower() for f in os.listdir(hole_path)):
                        file_found = True
                        break
                        
            if not file_found and row.get('Photo_Status') == 'Found':
                # File is missing but register shows it exists
                self.compartment_updates.append({
                    'hole_id': hole_id,
                    'depth_from': int(row['From']),
                    'depth_to': depth_to,
                    'photo_status': 'MISSING_FILE',
                    'approved_by': 'System Validation',
                    'comments': f'File not found during validation on {datetime.now().strftime("%Y-%m-%d")}'
                })
                results['missing'] += 1
                self.logger.warning(f"Missing file for compartment: {hole_id} {row['From']}-{depth_to}m")
                
        return results
    
    def _validate_original_images(self) -> Dict:
        """Check if original images still exist."""
        results = {'missing': 0, 'updates': []}
        
        if not self.json_manager:
            return results
            
        # Get all original image entries
        originals_df = self.json_manager.get_original_image_data()
        if originals_df.empty:
            return results
            
        processed_path = self._cached_paths['processed_originals']
        rejected_path = self._cached_paths['rejected_folder']
        
        if not processed_path:
            self.logger.warning("Cannot validate - processed folder not found")
            return results
            
        for _, row in originals_df.iterrows():
            hole_id = row['HoleID']
            depth_from = int(row['Depth_From'])
            depth_to = int(row['Depth_To'])
            
            # Check both approved and rejected paths
            file_found = False
            
            # Check processed folder
            if processed_path:
                project_code = hole_id[:2].upper()
                hole_path = os.path.join(processed_path, project_code, hole_id)
                if os.path.exists(hole_path):
                    pattern = f"{hole_id}_{depth_from}-{depth_to}_Original"
                    if any(pattern in f for f in os.listdir(hole_path)):
                        file_found = True
                        
            # Check rejected folder if not found
            if not file_found and rejected_path:
                project_code = hole_id[:2].upper()
                hole_path = os.path.join(rejected_path, project_code, hole_id)
                if os.path.exists(hole_path):
                    pattern = f"{hole_id}_{depth_from}-{depth_to}_Rejected"
                    if any(pattern in f for f in os.listdir(hole_path)):
                        file_found = True
                        
            if not file_found:
                # Mark as missing
                self.original_updates.append({
                    'hole_id': hole_id,
                    'depth_from': depth_from,
                    'depth_to': depth_to,
                    'original_filename': row.get('Original_Filename', 'Unknown'),
                    'is_approved': row.get('Approved_Upload_Status') == 'Uploaded',
                    'upload_success': False,
                    'uploaded_by': 'System Validation',
                    'comments': f'File not found during validation on {datetime.now().strftime("%Y-%m-%d")}. Previous status: {row.get("Approved_Upload_Status", "Unknown")}',
                    'file_count': 0,
                    'all_filenames': None
                })
                results['missing'] += 1
                self.logger.warning(f"Missing original file: {hole_id} {depth_from}-{depth_to}m")
                
        return results

    def _sync_wet_dry_filenames(self) -> Dict:
        """
        Rename files to include _Wet or _Dry suffix based on register data.
        
        Returns:
            Dictionary with results
        """
        results = {'renamed': 0, 'failed': 0, 'errors': []}
        
        approved_path = self.file_manager.get_shared_path('approved_compartments', create_if_missing=False)
        if not approved_path or not approved_path.exists():
            self.logger.warning("Approved compartment images folder not found")
            return results
        
        # Get compartment data from register
        if not self.json_manager:
            return results
            
        compartments_df = self.json_manager.get_compartment_data()
        if compartments_df.empty:
            return results
        
        # Filter for entries with wet/dry status
        wet_dry_entries = compartments_df[
            compartments_df['Photo_Status'].isin(['OK_Wet', 'OK_Dry'])
        ]
        
        if wet_dry_entries.empty:
            self.logger.info("No entries with wet/dry status to sync")
            return results
        
        self.logger.info(f"Found {len(wet_dry_entries)} entries with wet/dry status to check")
        
        # Process each entry
        for _, row in wet_dry_entries.iterrows():
            hole_id = row['HoleID']
            depth_to = int(row['To'])
            status = row['Photo_Status']
            
            # Determine expected suffix
            expected_suffix = "_Wet" if status == "OK_Wet" else "_Dry"
            
            # Check actual file
            project_code = hole_id[:2].upper()

            # Use Path objects for path manipulation

            hole_path = approved_path / project_code / hole_id
            
            if not hole_path.exists():
                continue
            
            # Look for compartment file
            found_file = None
            needs_rename = False
            
            for filename in os.listdir(str(hole_path)):
                # Check if this is the compartment we're looking for
                if f"{hole_id}_CC_{depth_to:03d}" in filename:
                    found_file = filename
                    
                    # Check if it already has correct suffix
                    if expected_suffix in filename:
                        # Already has correct suffix
                        break
                    else:
                        # Needs renaming
                        needs_rename = True
                        break
            
            if found_file and needs_rename:
                # Prepare paths for renaming
                old_path = str(hole_path / found_file)
                
                # Extract base name and extension
                base_pattern = f"{hole_id}_CC_{depth_to:03d}"
                
                # Remove any existing _Wet or _Dry suffix
                new_name = found_file.replace("_Wet", "").replace("_Dry", "")
                
                # Find the extension
                name_parts = new_name.split('.')
                if len(name_parts) > 1:
                    base_name = '.'.join(name_parts[:-1])
                    extension = name_parts[-1]
                    
                    # Build new filename
                    new_filename = f"{base_pattern}{expected_suffix}.{extension}"
                    new_path = str(hole_path / new_filename)
                    
        
                    # Use FileManager's safe rename method
        
                    if self.file_manager.rename_file_safely(old_path, new_path):
                        results['renamed'] += 1
                        self.logger.info(f"Successfully renamed: {found_file} -> {new_filename}")
                    else:
                        results['failed'] += 1
                        error_msg = f"Failed to rename {found_file}"
                        results['errors'].append(error_msg)
                        self.logger.error(error_msg)
        
        return results

    def calculate_missing_hex_colors(self) -> Dict:
        """
        Calculate average hex colors for compartments missing this data.
        
        Returns:
            Dictionary with calculation results
        """
        results = {
            'calculated': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            if not self.json_manager:
                return results
                
            # Get compartment data
            compartments_df = self.json_manager.get_compartment_data()
            if compartments_df.empty:
                return results
            # Check if Average_Hex_Color column exists, if not create it
            if 'Average_Hex_Color' not in compartments_df.columns:
                compartments_df['Average_Hex_Color'] = None
                self.logger.info("Average_Hex_Color column not found, will calculate for all valid compartments")
                
            # Find entries missing hex color
            # First check which rows have valid photo status
            valid_statuses = compartments_df['Photo_Status'].isin(['OK_Wet', 'OK_Dry', 'Found', 'Wet', 'Dry'])
            
            # Then check which have missing hex color (None, NaN, or empty string)
            missing_color = (
                compartments_df['Average_Hex_Color'].isna() | 
                (compartments_df['Average_Hex_Color'] == '') |
                (compartments_df['Average_Hex_Color'].isnull())
            )
            
            # Combine conditions
            missing_hex = compartments_df[valid_statuses & missing_color]
                        
            if missing_hex.empty:
                self.logger.info("No compartments missing hex color data")
                return results
                
            self._report_progress(f"Found {len(missing_hex)} compartments needing hex color calculation", 10)
            
            # Group by hole for efficient processing
            hex_updates = []
            holes_processed = 0
            total_holes = len(missing_hex['HoleID'].unique())
            
            for hole_id in missing_hex['HoleID'].unique():
                hole_entries = missing_hex[missing_hex['HoleID'] == hole_id]
                holes_processed += 1
                progress = 10 + (holes_processed / total_holes * 80)
                self._report_progress(f"Processing colors for {hole_id}...", progress)
                
                for _, row in hole_entries.iterrows():
                    depth_to = int(row['To'])
                    
                    # Find the image file
                    image_path = self._find_compartment_image(hole_id, depth_to)
                    
                    if image_path:
                        # Calculate average color
                        hex_color = self._calculate_average_hex_color(image_path)
                        
                        if hex_color:
                            hex_updates.append({
                                'hole_id': hole_id,
                                'depth_from': int(row['From']),
                                'depth_to': depth_to,
                                'average_hex_color': hex_color
                            })
                            results['calculated'] += 1
                        else:
                            results['failed'] += 1
                            self.logger.warning(f"Failed to calculate color for {hole_id} depth {depth_to}")
                    else:
                        results['failed'] += 1
                        self.logger.warning(f"Image not found for {hole_id} depth {depth_to}")
            
            # Batch update the register
            if hex_updates:
                self._report_progress("Updating register with hex colors...", 95)
                updated = self.json_manager.batch_update_compartment_colors(hex_updates)
                
                if updated < len(hex_updates):
                    results['errors'].append(f"Only {updated}/{len(hex_updates)} updates succeeded")
                    
            self._report_progress("Hex color calculation complete", 100)
            
        except Exception as e:
            self.logger.error(f"Error calculating hex colors: {str(e)}")
            results['errors'].append(str(e))
            
        return results
    
    def _find_compartment_image(self, hole_id: str, depth_to: int) -> Optional[str]:
        """Find the compartment image file path in both shared and local folders."""
        project_code = hole_id[:2].upper()
        
        # Look for compartment image with various suffixes
        patterns = [
            f"{hole_id}_CC_{depth_to:03d}_Wet",
            f"{hole_id}_CC_{depth_to:03d}_Dry",
            f"{hole_id}_CC_{depth_to:03d}"
        ]
        
        # ===================================================
        # CHANGED: Check multiple locations in priority order
        # ===================================================
        locations_to_check = [
            ('shared_approved', self._cached_paths.get('approved_compartments')),
            ('shared_review', self._cached_paths.get('review_compartments')),
            ('local_approved', self._cached_paths.get('local_approved_compartments')),
            ('local_review', self._cached_paths.get('local_review_compartments'))
        ]
        
        for location_name, base_path in locations_to_check:
            if base_path and base_path.exists():
                hole_path = base_path / project_code / hole_id
                
                if hole_path.exists():
                    for pattern in patterns:
                        for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                            filename = pattern + ext
                            file_path = hole_path / filename
                            if file_path.exists():
                                self.logger.debug(f"Found compartment image in {location_name}: {file_path}")
                                return str(file_path)
                                
        return None


    def _calculate_average_hex_color(self, image_path: str) -> Optional[str]:
        """
        Calculate the average hex color of an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Hex color string (e.g., '#A5B3C1') or None if failed
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Calculate mean color for each channel
            mean_color = img_rgb.mean(axis=(0, 1))
            
            # Convert to integers
            r, g, b = [int(c) for c in mean_color]
            
            # Convert to hex
            hex_color = f'#{r:02x}{g:02x}{b:02x}'.upper()
            
            return hex_color
            
        except Exception as e:
            self.logger.error(f"Error calculating hex color for {image_path}: {str(e)}")
            return None


    def _scan_rejected_folder(self, rejected_path: str, existing_entries: dict, results: dict):
            """Scan rejected folder for original images."""
            # Dictionary to collect all files for each hole/depth combination
            rejected_files_by_key = {}
            
            # Updated pattern to capture optional numeric suffix
            pattern = re.compile(r'^([A-Z]{2}\d{4})_(\d+)-(\d+)_Rejected(?:_(\d+))?\.(?:png|jpg|jpeg|tiff|tif)$', re.IGNORECASE)
            
            for project_folder in os.listdir(rejected_path):
                project_path = os.path.join(rejected_path, project_folder)
                if not os.path.isdir(project_path):
                    continue
                    
                for hole_folder in os.listdir(project_path):
                    hole_path = os.path.join(project_path, hole_folder)
                    if not os.path.isdir(hole_path):
                        continue
                    
                    for filename in os.listdir(hole_path):
                        match = pattern.match(filename)
                        if match:
                            hole_id = match.group(1)
                            depth_from = int(match.group(2))
                            depth_to = int(match.group(3))
                            file_number = match.group(4)  # May be None for first file
                            
                            key = (hole_id, depth_from, depth_to)
                            
                            # Collect all filenames for this key
                            if key not in rejected_files_by_key:
                                rejected_files_by_key[key] = []
                            rejected_files_by_key[key].append(filename)
                        
                        elif '_UPLOADED' in filename:
                            self.logger.warning(f"Found unexpected _UPLOADED suffix in OneDrive rejected file: {filename}")
            
            # Process collected files
            for key, filenames in rejected_files_by_key.items():
                hole_id, depth_from, depth_to = key
                
                # Sort filenames to get the primary one first
                filenames.sort()
                primary_filename = filenames[0]
                file_count = len(filenames)
                all_filenames = ', '.join(filenames) if file_count > 1 else None
                
                if key in existing_entries:
                    existing = existing_entries[key]
                    existing_file_count = existing.get('File_Count', 1)
                    
                    # Check if file count has changed
                    if existing_file_count != file_count or 'File_Count' not in existing:
                        # Update existing entry
                        self.original_updates.append({
                            'hole_id': hole_id,
                            'depth_from': depth_from,
                            'depth_to': depth_to,
                            'original_filename': primary_filename,
                            'is_approved': False,  # Rejected
                            'upload_success': True,
                            'uploaded_by': existing.get('Uploaded_By', 'System Sync'),
                            'comments': f'Updated file count from {existing_file_count} to {file_count} (rejected)',
                            'file_count': file_count,
                            'all_filenames': all_filenames
                        })
                        
                        results['updated'] += 1
                        self.logger.info(f"Updated rejected original: {hole_id} {depth_from}-{depth_to}m (files: {existing_file_count} -> {file_count})")
                else:
                    # New entry
                    self.original_updates.append({
                        'hole_id': hole_id,
                        'depth_from': depth_from,
                        'depth_to': depth_to,
                        'original_filename': primary_filename,
                        'is_approved': False,  # Rejected
                        'upload_success': True,
                        'uploaded_by': 'System Sync',
                        'comments': f'Auto-imported from rejected folder. File count: {file_count}',
                        'file_count': file_count,
                        'all_filenames': all_filenames
                    })
                    
                    results['added'] += 1
                    self.logger.info(f"Added rejected original: {hole_id} {depth_from}-{depth_to}m ({file_count} files)")

    def _check_missing_compartments(self) -> Dict:
            """
            Check for missing compartments based on original image ranges.
            
            Returns:
                Dictionary with results
            """
            results = {'missing': 0, 'errors': []}
            
            if not self.json_manager:
                return results
            
            # Get original images data
            originals_df = self.json_manager.get_original_image_data()
            if originals_df.empty:
                return results
            
            # Get compartment data
            compartments_df = self.json_manager.get_compartment_data()
            
            # Build set of existing compartments
            existing_compartments = set()
            if not compartments_df.empty:
                for _, row in compartments_df.iterrows():
                    hole_id = row['HoleID']
                    depth_to = row['To']
                    existing_compartments.add((hole_id, depth_to))
            
            # Check each original image range
            interval = self.config.get('compartment_interval', 1)
            
            for _, orig_row in originals_df.iterrows():
                hole_id = orig_row['HoleID']
                depth_from = int(orig_row['Depth_From'])
                depth_to = int(orig_row['Depth_To'])
                
                # Calculate expected compartments
                current_depth = depth_from
                while current_depth < depth_to:
                    comp_depth_to = current_depth + interval
                    
                    # Check if this compartment exists
                    if (hole_id, comp_depth_to) not in existing_compartments:
            
                        # FIXED: Add to batch instead of immediate update
            
                        self.compartment_updates.append({
                            'hole_id': hole_id,
                            'depth_from': current_depth,
                            'depth_to': comp_depth_to,
                            'photo_status': 'Missing',
                            'approved_by': 'System Check',
                            'comments': f'Missing compartment detected from original range {depth_from}-{depth_to}m'
                        })
                        
                        results['missing'] += 1
                        existing_compartments.add((hole_id, comp_depth_to))
                        self.logger.info(f"Queued missing compartment: {hole_id} {current_depth}-{comp_depth_to}m")
                    
                    current_depth = comp_depth_to
            
            return results