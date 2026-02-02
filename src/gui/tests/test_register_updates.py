"""Test script for register update functionality including UIDs"""

import sys
import os

# Add the src directory to path (go up 2 levels from tests to src)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(
    os.path.dirname(current_dir)
)  # Go up from tests to gui to src
sys.path.insert(0, src_dir)

import logging
from pathlib import Path
from datetime import datetime
from utils.json_register_manager import JSONRegisterManager
from core.file_manager import FileManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_register_updates():
    """Test all register update operations including UID handling"""

    # Initialize managers
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    file_manager = FileManager(base_dir=test_dir)
    json_manager = JSONRegisterManager(file_manager=file_manager)

    # Test data
    test_hole = "XX0001"
    test_depth_from = 0
    test_depth_to = 1
    test_uid = "test-uid-12345"

    print("Testing Register Updates with UIDs\n" + "=" * 50)

    # Test 1: Update compartment review WITH UID
    print("\n1. Testing compartment review update with UID...")
    success = json_manager.update_compartment_review(
        hole_id=test_hole,
        depth_from=test_depth_from,
        depth_to=test_depth_to,
        comments="Test comment with UID",
        classification="BIFf",
        compartment_uid=test_uid,
        moisture_status="Dry",
    )
    print(f"   Result: {'✓ Success' if success else '✗ Failed'}")

    # Test 2: Verify UID was saved
    print("\n2. Verifying UID was saved in review register...")
    reviews = json_manager.get_all_compartment_reviews(
        test_hole, test_depth_from, test_depth_to
    )
    if reviews:
        saved_uid = reviews[0].get("compartment_uid") or reviews[0].get(
            "Compartment_UID"
        )
        if saved_uid == test_uid:
            print(f"   Result: ✓ UID correctly saved: {saved_uid}")
        else:
            print(
                f"   Result: ✗ UID mismatch - Expected: {test_uid}, Found: {saved_uid}"
            )
    else:
        print("   Result: ✗ No reviews found")

    # Test 3: Update without UID (should preserve existing)
    print("\n3. Testing update preserves existing UID...")
    success = json_manager.update_compartment_review(
        hole_id=test_hole,
        depth_from=test_depth_from,
        depth_to=test_depth_to,
        comments="Updated comment without UID",
    )
    reviews = json_manager.get_all_compartment_reviews(
        test_hole, test_depth_from, test_depth_to
    )
    if reviews:
        preserved_uid = reviews[0].get("compartment_uid") or reviews[0].get(
            "Compartment_UID"
        )
        if preserved_uid == test_uid:
            print(f"   Result: ✓ UID preserved: {preserved_uid}")
        else:
            print(f"   Result: ✗ UID changed to: {preserved_uid}")

    # Test 4: Check compartment register
    print("\n4. Checking compartment register for UID...")
    comp_data = json_manager.get_compartment(test_hole, test_depth_from, test_depth_to)
    if comp_data:
        comp_uid = comp_data.get("Source_Image_UID")
        print(f"   Compartment register UID: {comp_uid}")
    else:
        print("   No compartment record found")

    # Test 5: Visual indicator test - check if classification was saved
    print("\n5. Testing visual indicator data...")
    if reviews:
        classification = reviews[0].get("Classification") or reviews[0].get(
            "classification"
        )
        print(f"   Classification saved: {classification}")
        print(f"   Review date: {reviews[0].get('Review_Date', 'Not found')}")
        print(f"   Reviewed by: {reviews[0].get('Reviewed_By', 'Not found')}")

    print("\n" + "=" * 50)
    print("Test complete!")

    # Show file locations
    print(f"\nGenerated files in: {test_dir}")
    for json_file in test_dir.glob("*.json"):
        print(f"  - {json_file.name}")


if __name__ == "__main__":
    test_register_updates()
