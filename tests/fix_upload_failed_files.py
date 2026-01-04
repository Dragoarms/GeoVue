#!/usr/bin/env python3
"""
Fix files incorrectly marked as UPLOAD_FAILED when upload actually succeeded.
This utility renames files by removing the _UPLOAD_FAILED suffix if the 
corresponding file exists in the shared folder.
"""

import os
import sys
from pathlib import Path

def fix_upload_failed_files(local_dir, shared_dir):
    """Fix incorrectly marked UPLOAD_FAILED files."""
    
    local_path = Path(local_dir)
    shared_path = Path(shared_dir)
    
    if not local_path.exists():
        print(f"ERROR: Local directory does not exist: {local_dir}")
        return
        
    if not shared_path.exists():
        print(f"ERROR: Shared directory does not exist: {shared_dir}")
        return
    
    print(f"Checking for UPLOAD_FAILED files in: {local_dir}")
    print(f"Verifying against shared folder: {shared_dir}")
    print("-" * 80)
    
    # Find all UPLOAD_FAILED files
    failed_files = list(local_path.glob("*_UPLOAD_FAILED.*"))
    
    if not failed_files:
        print("No UPLOAD_FAILED files found.")
        return
    
    fixed_count = 0
    
    for failed_file in failed_files:
        # Calculate what the shared file should be named
        expected_shared_name = failed_file.name.replace("_UPLOAD_FAILED", "")
        expected_shared_file = shared_path / expected_shared_name
        
        # Check if file exists in shared folder
        if expected_shared_file.exists():
            # Calculate correct local name (UPLOADED instead of UPLOAD_FAILED)
            correct_local_name = failed_file.name.replace("_UPLOAD_FAILED", "_UPLOADED")
            correct_local_file = failed_file.parent / correct_local_name
            
            try:
                # Rename the file
                failed_file.rename(correct_local_file)
                print(f"FIXED: {failed_file.name} -> {correct_local_name}")
                fixed_count += 1
            except Exception as e:
                print(f"ERROR: Could not rename {failed_file.name}: {e}")
        else:
            print(f"SKIP: {failed_file.name} - no corresponding file in shared folder")
    
    print("-" * 80)
    print(f"Fixed {fixed_count} files")

if __name__ == "__main__":
    # Your specific case
    local_dir = r"C:\GeoVue Chip Tray Photos\Processed Original Images\Approved Originals\BA\BA0008"
    shared_dir = r"C:\GEOVUE SHARED FOLDER\Processed Original Images\Approved Originals\BA\BA0008"
    
    print("Upload Failed File Fix Utility")
    print("=" * 80)
    
    fix_upload_failed_files(local_dir, shared_dir)