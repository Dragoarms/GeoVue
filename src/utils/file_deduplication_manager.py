"""
UID-Aware File Deduplication Manager

Handles deduplication while preserving UID relationships between original and compartment images.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


class UIDDeduplicationManager:
    """Manages UID-aware deduplication of files across multiple directories."""

    def __init__(self, file_manager, logger=None):
        self.file_manager = file_manager
        self.logger = logger or logging.getLogger(__name__)

    def find_and_remove_duplicates(self, dry_run=True) -> Dict[str, any]:
        """
        Find and optionally remove duplicate files while preserving UID relationships.

        Args:
            dry_run: If True, only report what would be done without making changes

        Returns:
            Dictionary with deduplication results
        """
        results = {
            "duplicates_found": 0,
            "files_removed": 0,
            "bytes_saved": 0,
            "uid_conflicts": 0,
            "uid_chain_breaks": 0,
            "duplicate_groups": [],
            "uid_conflict_groups": [],
            "errors": [],
        }

        try:
            # Get all compartment directories to scan
            directories = self._get_compartment_directories()

            # Build UID-aware file map
            uid_file_map, orphaned_files = self._build_uid_file_map(directories)

            # Find content duplicates within same UID groups
            content_duplicates = self._find_content_duplicates_by_uid(uid_file_map)

            # Find UID conflicts (same filename pattern, different UIDs)
            uid_conflicts = self._find_uid_conflicts(uid_file_map)

            results["duplicate_groups"] = content_duplicates
            results["uid_conflict_groups"] = uid_conflicts
            results["duplicates_found"] = sum(
                len(group) - 1 for group in content_duplicates
            )
            results["uid_conflicts"] = len(uid_conflicts)

            if not dry_run:
                # Remove content duplicates (safe - same UID)
                for group in content_duplicates:
                    removed_files, bytes_saved = self._remove_duplicates_from_group(
                        group
                    )
                    results["files_removed"] += removed_files
                    results["bytes_saved"] += bytes_saved

                # Log UID conflicts but don't auto-remove (needs manual review)
                for conflict_group in uid_conflicts:
                    self.logger.warning(
                        f"UID conflict detected: {[str(f) for f in conflict_group]}"
                    )

            return results

        except Exception as e:
            self.logger.error(f"Error in UID-aware deduplication: {e}")
            results["errors"].append(str(e))
            return results

    def _build_uid_file_map(
        self, directories: List[Path]
    ) -> Tuple[Dict[str, Dict[str, List[Path]]], List[Path]]:
        """
        Build a map of UID -> content_hash -> file_paths.

        Uses a two-pass approach for efficiency:
        1. First pass: Group files by name pattern (fast - filename only)
        2. Second pass: Only do expensive UID/MD5 on files with duplicate patterns

        Returns:
            Tuple of (uid_file_map, orphaned_files)
        """
        uid_file_map = defaultdict(lambda: defaultdict(list))
        orphaned_files = []

        # First pass: Group files by base pattern (fast - just filename parsing)
        pattern_groups = defaultdict(list)

        for directory in directories:
            if not directory.exists():
                continue

            for file_path in directory.rglob("*.png"):
                if self._is_compartment_file(file_path):
                    pattern = self._extract_compartment_pattern(file_path)
                    if pattern:
                        pattern_groups[pattern].append(file_path)

            for file_path in directory.rglob("*.jpg"):
                if self._is_compartment_file(file_path):
                    pattern = self._extract_compartment_pattern(file_path)
                    if pattern:
                        pattern_groups[pattern].append(file_path)

        # Second pass: Only do expensive UID/MD5 checks on potential duplicates
        potential_duplicates = [
            files for files in pattern_groups.values() if len(files) > 1
        ]

        if potential_duplicates:
            self.logger.info(
                f"Found {len(potential_duplicates)} pattern groups with potential duplicates, "
                f"checking {sum(len(f) for f in potential_duplicates)} files"
            )

            for file_group in potential_duplicates:
                for file_path in file_group:
                    uid = self.file_manager.extract_uid_from_any_image(str(file_path))
                    if uid:
                        content_hash = self._calculate_md5(file_path)
                        uid_file_map[uid][content_hash].append(file_path)
                    else:
                        orphaned_files.append(file_path)
        else:
            self.logger.info("No potential duplicates found - skipping expensive UID/MD5 checks")

        return uid_file_map, orphaned_files

    def _find_content_duplicates_by_uid(
        self, uid_file_map: Dict[str, Dict[str, List[Path]]]
    ) -> List[List[Path]]:
        """Find content duplicates within the same UID group (safe to deduplicate)."""
        duplicate_groups = []

        for uid, content_map in uid_file_map.items():
            for content_hash, file_paths in content_map.items():
                if len(file_paths) > 1:
                    duplicate_groups.append(file_paths)

        return duplicate_groups

    def _find_uid_conflicts(
        self, uid_file_map: Dict[str, Dict[str, List[Path]]]
    ) -> List[List[Path]]:
        """Find files with same compartment pattern but different UIDs (needs manual review)."""
        # Group files by compartment pattern (hole_id + compartment_depth + moisture)
        pattern_groups = defaultdict(list)

        for uid, content_map in uid_file_map.items():
            for content_hash, file_paths in content_map.items():
                for file_path in file_paths:
                    pattern = self._extract_compartment_pattern(file_path)
                    if pattern:
                        pattern_groups[pattern].append((uid, file_path))

        # Find conflicts (same pattern, different UIDs)
        conflicts = []
        for pattern, uid_file_list in pattern_groups.items():
            uids = set(uid for uid, _ in uid_file_list)
            if len(uids) > 1:
                conflict_files = [file_path for _, file_path in uid_file_list]
                conflicts.append(conflict_files)

        return conflicts

    def _extract_compartment_pattern(self, file_path: Path) -> Optional[str]:
        """Extract the base compartment pattern (hole_id + depth + moisture) from filename."""
        import re

        # Pattern to match compartment files
        pattern = r"([A-Z]{2}\d{4})_CC_(\d+)(?:_(Wet|Dry))?(?:_temp|_new|_review|_UPLOADED|_UPLOAD_FAILED)?\.(?:png|jpg)$"
        match = re.match(pattern, file_path.name, re.IGNORECASE)

        if match:
            hole_id, depth, moisture = match.groups()
            moisture = moisture or "unknown"
            return f"{hole_id}_CC_{depth}_{moisture}"

        return None

    def _get_compartment_directories(self) -> List[Path]:
        """Get all directories that contain compartment images."""
        directories = []

        # Add local directories
        for key in ["approved_compartments", "temp_review"]:
            path = self.file_manager.dir_structure.get(key)
            if path and os.path.exists(path):
                directories.append(Path(path))

        # Add shared directories if different from local
        for key in ["approved_compartments", "review_compartments"]:
            shared_path = self.file_manager.get_shared_path(key)
            if shared_path and os.path.exists(shared_path):
                shared_path_obj = Path(shared_path)
                if shared_path_obj not in directories:
                    directories.append(shared_path_obj)

        return directories

    def _is_compartment_file(self, file_path: Path) -> bool:
        """Check if file is a compartment image based on naming pattern."""
        import re

        pattern = r"([A-Z]{2}\d{4})_CC_(\d+)(?:_temp|_new|_review|_Wet|_Dry|_UPLOADED|_UPLOAD_FAILED)?\.(?:png|tiff|jpg)$"
        return bool(re.match(pattern, file_path.name, re.IGNORECASE))

    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _remove_duplicates_from_group(self, group: List[Path]) -> Tuple[int, int]:
        """Remove duplicates from a group, keeping the best copy."""
        if len(group) <= 1:
            return 0, 0

        # Sort by priority - keep the "best" copy
        sorted_group = self._sort_by_priority(group)
        keep_file = sorted_group[0]
        remove_files = sorted_group[1:]

        files_removed = 0
        bytes_saved = 0

        for file_path in remove_files:
            try:
                file_size = file_path.stat().st_size
                file_path.unlink()
                files_removed += 1
                bytes_saved += file_size
                self.logger.info(f"Removed duplicate (same UID): {file_path}")
            except Exception as e:
                self.logger.error(f"Could not remove duplicate {file_path}: {e}")

        self.logger.info(f"Kept best copy (UID preserved): {keep_file}")
        return files_removed, bytes_saved

    def _sort_by_priority(self, files: List[Path]) -> List[Path]:
        """Sort files by priority - best copies first."""

        def priority_score(file_path: Path) -> int:
            score = 0
            name = file_path.name

            # Prefer files with moisture suffix over temp files
            if "_Wet" in name or "_Dry" in name:
                score += 100

            # Prefer approved over review locations
            if "Approved" in str(file_path):
                score += 50
            elif "Review" in str(file_path):
                score += 25

            # Prefer uploaded files
            if "_UPLOADED" in name:
                score += 10

            # Penalize temp files
            if "_temp" in name:
                score -= 50

            # Penalize failed uploads
            if "_UPLOAD_FAILED" in name:
                score -= 25

            return score

        return sorted(files, key=priority_score, reverse=True)
