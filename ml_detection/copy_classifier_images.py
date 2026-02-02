"""
Copy classifier images to a single folder for Google Drive upload.
"""
import shutil
from pathlib import Path

def main():
    # Destination
    dest = Path(__file__).parent / "classifier_images"
    dest.mkdir(exist_ok=True)

    # Sources
    sources = [
        Path(r"C:\Users\georg\Pictures\Shared folder EX\Extracted Compartment Images\Approved Compartment Images"),
        Path(r"C:\GeoVue Chip Tray Photos\empty"),
    ]

    print("Copying classifier images...")
    copied = 0

    for src in sources:
        if src.exists():
            print(f"  From: {src}")
            for img in src.rglob("*.png"):
                dst = dest / img.name
                if not dst.exists():
                    shutil.copy2(img, dst)
                    copied += 1

    print(f"\nCopied {copied} images to: {dest}")
    print("\nNext: Upload ml_detection folder to Google Drive")

if __name__ == "__main__":
    main()
