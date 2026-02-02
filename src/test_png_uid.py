"""
Test PNG UID Embedding and Extraction
Run this to verify the UID system is working correctly.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Just import the methods we need without ConfigManager
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def embed_uid_in_png_test(image_path: str, uid: str) -> str:
    """Test UID embedding in PNG."""
    img = PILImage.open(image_path)
    png_info = PngInfo()
    png_info.add_text("SourceUID", uid)
    png_info.add_text("Description", f"Source Image UID: {uid}")
    
    # Save with metadata
    img.save(image_path, pnginfo=png_info)
    logger.info(f"Embedded UID {uid} in PNG")
    return uid

def extract_uid_from_png_test(png_path: str) -> str:
    """Test UID extraction from PNG."""
    img = PILImage.open(png_path)
    if "SourceUID" in img.info:
        return img.info["SourceUID"]
    elif "Description" in img.info:
        desc = img.info["Description"]
        if desc.startswith("Source Image UID: "):
            return desc[18:]
    return None

def test_png_uid_embedding():
    """Test PNG UID embedding and extraction."""
    
    print("=" * 80)
    print("PNG UID EMBEDDING TEST")
    print("=" * 80)
    
    # Test 1: Create a test PNG image
    print("\n1. Creating test PNG image...")
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:] = (100, 150, 200)  # Blue-ish color
    
    test_path = Path("test_uid_image.png")
    cv2.imwrite(str(test_path), test_image)
    print(f"   ✓ Created test image: {test_path}")
    
    # Test 2: Embed UID
    print("\n2. Embedding UID in PNG...")
    test_uid = "test-uuid-12345-67890-abcdef"
    embedded_uid = embed_uid_in_png_test(str(test_path), test_uid)
    print(f"   ✓ Embedded UID: {embedded_uid}")
    
    # Test 3: Extract UID
    print("\n3. Extracting UID from PNG...")
    extracted_uid = extract_uid_from_png_test(str(test_path))
    
    if extracted_uid:
        print(f"   ✓ Extracted UID: {extracted_uid}")
        
        # Test 4: Verify match
        print("\n4. Verifying UID match...")
        if extracted_uid == test_uid:
            print(f"   ✅ SUCCESS! UIDs match")
        else:
            print(f"   ❌ FAIL! UIDs don't match")
            print(f"      Expected: {test_uid}")
            print(f"      Got:      {extracted_uid}")
    else:
        print(f"   ❌ FAIL! Could not extract UID")
    
    # Test 5: Test on actual compartment image
    print("\n5. Testing on actual compartment image...")
    print("   Enter path to a compartment PNG (or press Enter to skip):")
    comp_path = input("   Path: ").strip()
    
    if comp_path and Path(comp_path).exists():
        print(f"\n   Testing: {comp_path}")
        comp_uid = extract_uid_from_png_test(comp_path)
        if comp_uid:
            print(f"   ✓ Found UID: {comp_uid}")
        else:
            print(f"   ⚠️  No UID found in this image")
            print(f"      This means the image was processed before UID implementation")
    else:
        print("   Skipped actual image test")
    
    # Cleanup
    print("\n6. Cleanup...")
    if test_path.exists():
        test_path.unlink()
        print(f"   ✓ Deleted test image")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_png_uid_embedding()