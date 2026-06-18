"""
Review Export Manager - Handles saving review data.

This module manages exporting review data back to the JSON register
and other formats.
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class ReviewExportManager:
    """
    Manages export and save operations for review data.
    
    Handles saving to JSON register and exporting to other formats.
    """
    
    def __init__(self, json_manager):
        """
        Initialize export manager.
        
        Args:
            json_manager: JSONRegisterManager instance
        """
        self.json_manager = json_manager
        self.logger = logging.getLogger(__name__)
        self.current_user = os.getenv("USERNAME", "Unknown")
    
    def save_reviews(
        self, 
        images: List,
        classification_field: str = "Classification"
    ) -> Dict[str, int]:
        """
        Save all modified reviews to JSON register.
        
        Args:
            images: List of CompartmentImage objects
            classification_field: Field name for classification (default: "Classification")
            
        Returns:
            Dict with statistics: {"saved": int, "skipped": int, "errors": int}
        """
        stats = {"saved": 0, "skipped": 0, "errors": 0}
        
        try:
            # Collect reviews to save
            reviews_to_save = []
            
            for img in images:
                # Skip if no classification
                if not img.classification or img.classification == "Unassigned":
                    stats["skipped"] += 1
                    continue
                
                # Skip if unchanged
                if not img.has_changed():
                    stats["skipped"] += 1
                    continue
                
                # Build review data
                review_data = {
                    "hole_id": img.hole_id,
                    "depth_from": int(img.depth_from),
                    "depth_to": int(img.depth_to),
                    classification_field: img.classification,
                    "Comments": img.comments,
                    "reviewed_by": self.current_user,
                    "compartment_uid": img.compartment_uid
                }
                
                reviews_to_save.append(review_data)
            
            # Batch save to JSON register
            if reviews_to_save:
                save_stats = self.json_manager.batch_update_compartment_reviews(
                    reviews_to_save
                )
                
                stats["saved"] = save_stats.get("updated", 0) + save_stats.get("created", 0)
                stats["errors"] = save_stats.get("failed", 0)
                
                # Update image states to mark as saved
                for img in images:
                    if img.has_changed():
                        img.original_classification = img.classification
                        img.original_comments = img.comments
                        img._has_saved_classification = True
                
                self.logger.info(
                    f"Saved {stats['saved']} reviews, "
                    f"skipped {stats['skipped']}, "
                    f"errors {stats['errors']}"
                )
            else:
                self.logger.info("No reviews to save")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error saving reviews: {e}")
            stats["errors"] += 1
            return stats
    
    def save_single_review(
        self,
        img,
        classification_field: str = "Classification"
    ) -> bool:
        """
        Save a single image's review.
        
        Args:
            img: CompartmentImage object
            classification_field: Field name for classification
            
        Returns:
            True if successful
        """
        try:
            if not img.classification or img.classification == "Unassigned":
                return False
            
            success = self.json_manager.update_compartment_review(
                hole_id=img.hole_id,
                depth_from=int(img.depth_from),
                depth_to=int(img.depth_to),
                reviewed_by=self.current_user,
                comments=img.comments,
                compartment_uid=img.compartment_uid,
                **{classification_field: img.classification}
            )
            
            if success:
                img.original_classification = img.classification
                img.original_comments = img.comments
                img._has_saved_classification = True
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving single review for {img.filename}: {e}")
            return False
    
    def export_to_csv(
        self,
        images: List,
        output_path: str,
        include_unreviewed: bool = False
    ) -> bool:
        """
        Export reviews to CSV file.
        
        Args:
            images: List of CompartmentImage objects
            output_path: Path to output CSV file
            include_unreviewed: Include images without reviews
            
        Returns:
            True if successful
        """
        try:
            import pandas as pd
            
            # Build rows
            rows = []
            for img in images:
                # Skip unreviewed if not including them
                if not include_unreviewed:
                    if not img.classification or img.classification == "Unassigned":
                        continue
                
                row = {
                    "Filename": img.filename,
                    "HoleID": img.hole_id,
                    "From": img.depth_from,
                    "To": img.depth_to,
                    "Classification": img.classification,
                    "Moisture": img.moisture_status or "",
                    "Comments": img.comments,
                    "Reviewed_By": img.classified_by or self.current_user,
                    "Review_Date": img.classified_date or datetime.now().isoformat()
                }
                
                # Add CSV data fields
                for key, value in img.csv_data.items():
                    if key not in row:
                        row[key] = value
                
                rows.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Exported {len(rows)} reviews to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def export_statistics(
        self,
        images: List,
        output_path: str
    ) -> bool:
        """
        Export classification statistics to CSV.
        
        Args:
            images: List of CompartmentImage objects
            output_path: Path to output CSV file
            
        Returns:
            True if successful
        """
        try:
            import pandas as pd
            from collections import Counter
            
            # Overall statistics
            total = len(images)
            classified = sum(
                1 for img in images 
                if img.classification and img.classification != "Unassigned"
            )
            unassigned = total - classified
            
            # Classification breakdown
            classifications = Counter()
            for img in images:
                if img.classification and img.classification != "Unassigned":
                    classifications[img.classification] += 1
            
            # Build rows
            rows = [
                {"Category": "Total Images", "Count": total},
                {"Category": "Classified", "Count": classified},
                {"Category": "Unassigned", "Count": unassigned},
                {"Category": "", "Count": ""},  # Separator
            ]
            
            # Add classification breakdown
            for classification, count in classifications.most_common():
                rows.append({
                    "Category": f"  {classification}",
                    "Count": count
                })
            
            # Create DataFrame and save
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Exported statistics to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting statistics: {e}")
            return False
    
    def get_save_summary(self, images: List) -> Dict:
        """
        Get summary of what would be saved.
        
        Args:
            images: List of CompartmentImage objects
            
        Returns:
            Dict with summary info
        """
        to_save = 0
        unchanged = 0
        unreviewed = 0
        
        for img in images:
            if not img.classification or img.classification == "Unassigned":
                unreviewed += 1
            elif img.has_changed():
                to_save += 1
            else:
                unchanged += 1
        
        return {
            "to_save": to_save,
            "unchanged": unchanged,
            "unreviewed": unreviewed,
            "total": len(images)
        }
