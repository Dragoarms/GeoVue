"""
Peer Review Manager - Manages multi-user review operations.

This module handles loading and analyzing reviews from multiple users,
detecting conflicts, and calculating consensus classifications.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter
from pathlib import Path


logger = logging.getLogger(__name__)


class PeerReviewManager:
    """
    Manages multi-user review data and conflict detection.
    
    Integrates with JSONRegisterManager to load reviews from all users
    and provide peer review functionality.
    """
    
    def __init__(self, json_manager):
        """
        Initialize peer review manager.
        
        Args:
            json_manager: JSONRegisterManager instance
        """
        self.json_manager = json_manager
        self.logger = logging.getLogger(__name__)
        
        # Multi-user review storage
        self.all_user_reviews: Dict[Tuple, List[Dict]] = {}  # (hole, from, to) -> [reviews]
        self.current_user = os.getenv("USERNAME", "Unknown")
        
        # Statistics
        self.total_reviewers = 0
        self.conflict_count = 0
    
    def load_all_user_reviews(self, images: List) -> bool:
        """
        Load reviews from ALL users for the given images.
        
        This reads from all user-specific JSON files and groups reviews
        by compartment.
        
        Args:
            images: List of CompartmentImage objects
            
        Returns:
            True if successful
        """
        try:
            # Get DataFrame with all user data
            # This includes a 'User_Source' or 'Reviewed_By' column
            all_reviews_df = self.json_manager.get_all_compartment_data()
            
            if all_reviews_df.empty:
                self.logger.info("No review data found from other users")
                return False
            
            # Group reviews by compartment
            self.all_user_reviews.clear()
            
            for _, row in all_reviews_df.iterrows():
                hole_id = row.get("HoleID")
                depth_from = int(row.get("From", 0))
                depth_to = int(row.get("To", 0))
                
                key = (hole_id, depth_from, depth_to)
                
                # Get reviewer name
                reviewer = row.get("Reviewed_By", "Unknown")
                
                # Skip if this is current user (we already loaded their reviews)
                if reviewer.lower() == self.current_user.lower():
                    continue
                
                # Create review dict
                review = {
                    "HoleID": hole_id,
                    "From": depth_from,
                    "To": depth_to,
                    "Classification": (
                        row.get("Classification") or
                        row.get("Lithology") or
                        row.get("Rock_Type")
                    ),
                    "Comments": row.get("Comments", ""),
                    "Reviewed_By": reviewer,
                    "Review_Date": row.get("Review_Date", ""),
                    "Review_Number": row.get("Review_Number", 1)
                }
                
                # Add to group
                if key not in self.all_user_reviews:
                    self.all_user_reviews[key] = []
                
                self.all_user_reviews[key].append(review)
            
            # Calculate statistics
            all_reviewers = set()
            for reviews in self.all_user_reviews.values():
                for review in reviews:
                    all_reviewers.add(review["Reviewed_By"])
            
            self.total_reviewers = len(all_reviewers)
            self.conflict_count = sum(1 for reviews in self.all_user_reviews.values() if self._has_conflicts(reviews))
            
            self.logger.info(
                f"Loaded peer reviews from {self.total_reviewers} users, "
                f"{len(self.all_user_reviews)} compartments, "
                f"{self.conflict_count} conflicts"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading all user reviews: {e}")
            return False
    
    def get_peer_reviews(self, img) -> List[Dict]:
        """
        Get peer reviews for a specific image (excluding current user).
        
        Args:
            img: CompartmentImage object
            
        Returns:
            List of review dicts from other users
        """
        key = (img.hole_id, int(img.depth_from), int(img.depth_to))
        return self.all_user_reviews.get(key, [])
    
    def get_all_reviews_for_image(self, img) -> List[Dict]:
        """
        Get ALL reviews for an image (including current user if available).
        
        Args:
            img: CompartmentImage object
            
        Returns:
            List of all review dicts
        """
        reviews = self.get_peer_reviews(img).copy()
        
        # Add current user's review if it exists
        if img.classification and img.classification != "Unassigned":
            reviews.append({
                "HoleID": img.hole_id,
                "From": int(img.depth_from),
                "To": int(img.depth_to),
                "Classification": img.classification,
                "Comments": img.comments,
                "Reviewed_By": self.current_user,
                "Review_Date": img.classified_date,
                "Review_Number": 1
            })
        
        return reviews
    
    def get_consensus_classification(self, img) -> Optional[str]:
        """
        Get consensus classification from peer reviews.
        Returns the most common classification, or None if no consensus.
        
        Args:
            img: CompartmentImage object
            
        Returns:
            Most common classification or None
        """
        all_reviews = self.get_all_reviews_for_image(img)
        
        if not all_reviews:
            return None
        
        # Count classifications
        classifications = [
            r["Classification"] 
            for r in all_reviews 
            if r.get("Classification")
        ]
        
        if not classifications:
            return None
        
        # Get most common
        counter = Counter(classifications)
        most_common = counter.most_common(1)[0]
        
        # Return only if there's a clear majority (>50%)
        if most_common[1] > len(classifications) / 2:
            return most_common[0]
        
        return None
    
    def has_peer_reviews(self, img) -> bool:
        """Check if image has reviews from other users"""
        return len(self.get_peer_reviews(img)) > 0
    
    def has_current_user_review(self, img) -> bool:
        """Check if current user has reviewed this image"""
        return img.classification and img.classification != "Unassigned"
    
    def has_review_conflicts(self, img) -> bool:
        """
        Check if current user's review conflicts with peer reviews.
        
        Args:
            img: CompartmentImage object
            
        Returns:
            True if there are conflicting classifications
        """
        if not self.has_current_user_review(img):
            return False
        
        peer_reviews = self.get_peer_reviews(img)
        if not peer_reviews:
            return False
        
        # Check if any peer review disagrees with current user
        user_classification = img.classification
        
        for review in peer_reviews:
            peer_classification = review.get("Classification")
            if peer_classification and peer_classification != user_classification:
                return True
        
        return False
    
    def _has_conflicts(self, reviews: List[Dict]) -> bool:
        """Check if a list of reviews has conflicts"""
        if len(reviews) <= 1:
            return False
        
        classifications = [
            r["Classification"] 
            for r in reviews 
            if r.get("Classification")
        ]
        
        # No conflict if everyone agrees
        return len(set(classifications)) > 1
    
    def get_images_needing_review(self, images: List) -> List:
        """
        Get images that have peer reviews but no review from current user.
        
        Args:
            images: List of CompartmentImage objects
            
        Returns:
            Filtered list of images needing review
        """
        needs_review = []
        
        for img in images:
            # Has peer reviews but current user hasn't reviewed
            if self.has_peer_reviews(img) and not self.has_current_user_review(img):
                needs_review.append(img)
        
        return needs_review
    
    def get_images_with_peer_reviews_of_user(self, images: List) -> List:
        """
        Get images where current user has reviewed AND others have also reviewed.
        Used for the "my reviews reviewed" mode.
        
        Args:
            images: List of CompartmentImage objects
            
        Returns:
            Filtered list of images
        """
        result = []
        
        for img in images:
            # Current user has reviewed AND there are peer reviews
            if self.has_current_user_review(img) and self.has_peer_reviews(img):
                result.append(img)
        
        return result
    
    def get_conflict_images(self, images: List) -> List:
        """
        Get images with conflicting reviews.
        
        Args:
            images: List of CompartmentImage objects
            
        Returns:
            List of images with conflicts
        """
        conflicts = []
        
        for img in images:
            if self.has_review_conflicts(img):
                conflicts.append(img)
        
        return conflicts
    
    def get_peer_review_summary(self, img) -> Dict:
        """
        Get summary statistics for peer reviews of an image.
        
        Args:
            img: CompartmentImage object
            
        Returns:
            Dict with summary info
        """
        peer_reviews = self.get_peer_reviews(img)
        
        if not peer_reviews:
            return {
                "total_reviews": 0,
                "reviewers": [],
                "classifications": {},
                "consensus": None,
                "has_conflict": False
            }
        
        # Count classifications
        classifications = Counter()
        reviewers = []
        
        for review in peer_reviews:
            reviewer = review.get("Reviewed_By", "Unknown")
            reviewers.append(reviewer)
            
            classification = review.get("Classification")
            if classification:
                classifications[classification] += 1
        
        # Check for consensus
        consensus = None
        if len(classifications) == 1:
            consensus = list(classifications.keys())[0]
        
        # Check for conflict
        has_conflict = len(classifications) > 1
        
        return {
            "total_reviews": len(peer_reviews),
            "reviewers": reviewers,
            "classifications": dict(classifications),
            "consensus": consensus,
            "has_conflict": has_conflict
        }
