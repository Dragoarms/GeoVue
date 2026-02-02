"""
Prepare YOLO Dataset from GeoVue Compartment Corner Data

This script converts the compartment corner coordinates from GeoVue's JSON format
into YOLO format for training an object detection model.

YOLO format: class_id center_x center_y width height (all normalized 0-1)
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CompartmentBox:
    """Represents a compartment bounding box"""
    image_filename: str
    compartment_number: int
    top_left_x: int
    top_left_y: int
    top_right_x: int
    top_right_y: int
    bottom_right_x: int
    bottom_right_y: int
    bottom_left_x: int
    bottom_left_y: int

    def to_yolo_format(self, img_width: int, img_height: int) -> str:
        """
        Convert to YOLO format: class_id center_x center_y width height
        All values normalized to 0-1
        """
        # Calculate bounding box from corner coordinates
        x_coords = [self.top_left_x, self.top_right_x, self.bottom_right_x, self.bottom_left_x]
        y_coords = [self.top_left_y, self.top_right_y, self.bottom_right_y, self.bottom_left_y]

        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Calculate center and dimensions
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # Normalize to 0-1
        center_x_norm = center_x / img_width
        center_y_norm = center_y / img_height
        width_norm = width / img_width
        height_norm = height / img_height

        # Clamp values to valid range
        center_x_norm = max(0, min(1, center_x_norm))
        center_y_norm = max(0, min(1, center_y_norm))
        width_norm = max(0, min(1, width_norm))
        height_norm = max(0, min(1, height_norm))

        # Class 0 = cell/compartment
        return f"0 {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}"


def load_all_corner_data(register_dir: Path) -> Dict[str, List[CompartmentBox]]:
    """Load all compartment corners from JSON files, grouped by image filename"""
    image_boxes: Dict[str, List[CompartmentBox]] = {}

    json_files = list(register_dir.glob("compartment_corners_*.json"))
    logger.info(f"Found {len(json_files)} corner data files")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            for record in data:
                filename = record.get('Original_Filename', '')
                if not filename or filename == 'EXAMPLE_0-20_Original.jpg':
                    continue

                # Skip records with invalid coordinates
                if record.get('Top_Left_X', 0) == 0 and record.get('Top_Left_Y', 0) == 0:
                    continue

                box = CompartmentBox(
                    image_filename=filename,
                    compartment_number=record.get('Compartment_Number', 0),
                    top_left_x=record.get('Top_Left_X', 0),
                    top_left_y=record.get('Top_Left_Y', 0),
                    top_right_x=record.get('Top_Right_X', 0),
                    top_right_y=record.get('Top_Right_Y', 0),
                    bottom_right_x=record.get('Bottom_Right_X', 0),
                    bottom_right_y=record.get('Bottom_Right_Y', 0),
                    bottom_left_x=record.get('Bottom_Left_X', 0),
                    bottom_left_y=record.get('Bottom_Left_Y', 0),
                )

                if filename not in image_boxes:
                    image_boxes[filename] = []
                image_boxes[filename].append(box)

        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")

    logger.info(f"Loaded annotations for {len(image_boxes)} unique images")
    return image_boxes


def find_image_files(images_dir: Path) -> Dict[str, Path]:
    """Build a mapping of image filenames to their full paths"""
    image_map = {}

    for img_path in images_dir.rglob("*_Original.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Map both the exact filename and lowercase version
            image_map[img_path.name] = img_path
            image_map[img_path.name.lower()] = img_path

    logger.info(f"Found {len(image_map) // 2} original images on disk")
    return image_map


def get_image_dimensions(img_path: Path) -> Tuple[int, int]:
    """Get image width and height"""
    with Image.open(img_path) as img:
        return img.size  # (width, height)


def create_yolo_dataset(
    corner_data: Dict[str, List[CompartmentBox]],
    image_map: Dict[str, Path],
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    max_images: Optional[int] = None
):
    """
    Create YOLO dataset structure:
    output_dir/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/
        dataset.yaml
    """
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Find matching images (have both annotations and image files)
    matched_images = []
    for filename, boxes in corner_data.items():
        # Try exact match first, then case-insensitive
        img_path = image_map.get(filename) or image_map.get(filename.lower())
        if img_path and img_path.exists():
            matched_images.append((filename, boxes, img_path))

    logger.info(f"Found {len(matched_images)} images with matching annotations")

    if max_images and len(matched_images) > max_images:
        random.shuffle(matched_images)
        matched_images = matched_images[:max_images]
        logger.info(f"Limited to {max_images} images")

    # Shuffle and split
    random.shuffle(matched_images)
    n_total = len(matched_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        'train': matched_images[:n_train],
        'val': matched_images[n_train:n_train + n_val],
        'test': matched_images[n_train + n_val:]
    }

    stats = {'train': 0, 'val': 0, 'test': 0, 'total_boxes': 0}

    for split_name, split_data in splits.items():
        for filename, boxes, img_path in split_data:
            try:
                # Get image dimensions
                img_width, img_height = get_image_dimensions(img_path)

                # Create unique output filename (handle duplicates from different folders)
                base_name = img_path.stem
                out_img_name = f"{base_name}{img_path.suffix}"
                out_label_name = f"{base_name}.txt"

                # Copy image
                dest_img = output_dir / 'images' / split_name / out_img_name
                if not dest_img.exists():
                    shutil.copy2(img_path, dest_img)

                # Create label file
                label_lines = []
                for box in boxes:
                    yolo_line = box.to_yolo_format(img_width, img_height)
                    label_lines.append(yolo_line)

                dest_label = output_dir / 'labels' / split_name / out_label_name
                with open(dest_label, 'w') as f:
                    f.write('\n'.join(label_lines))

                stats[split_name] += 1
                stats['total_boxes'] += len(boxes)

            except Exception as e:
                logger.warning(f"Error processing {filename}: {e}")

    # Create dataset.yaml
    yaml_content = f"""# GeoVue Chip Tray Cell Detection Dataset
# Auto-generated for YOLOv8 training

path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: cell

# Dataset statistics
# Train images: {stats['train']}
# Val images: {stats['val']}
# Test images: {stats['test']}
# Total bounding boxes: {stats['total_boxes']}
"""

    with open(output_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)

    logger.info(f"Dataset created successfully!")
    logger.info(f"  Train: {stats['train']} images")
    logger.info(f"  Val: {stats['val']} images")
    logger.info(f"  Test: {stats['test']} images")
    logger.info(f"  Total boxes: {stats['total_boxes']}")

    return stats


def main():
    # Configuration - UPDATE THESE PATHS
    SHARED_FOLDER = Path(r"C:\Users\georg\Pictures\Shared folder EX")
    REGISTER_DIR = SHARED_FOLDER / "Chip Tray Register" / "Register Data (Do not edit)"
    IMAGES_DIR = SHARED_FOLDER / "Processed Original Images" / "Approved Originals"
    OUTPUT_DIR = Path(__file__).parent / "yolo_dataset"

    # Set random seed for reproducibility
    random.seed(42)

    logger.info("=" * 60)
    logger.info("GeoVue YOLO Dataset Preparation")
    logger.info("=" * 60)

    # Load corner data
    logger.info("\nStep 1: Loading corner annotations...")
    corner_data = load_all_corner_data(REGISTER_DIR)

    # Find image files
    logger.info("\nStep 2: Scanning for image files...")
    image_map = find_image_files(IMAGES_DIR)

    # Create dataset
    logger.info("\nStep 3: Creating YOLO dataset...")
    stats = create_yolo_dataset(
        corner_data=corner_data,
        image_map=image_map,
        output_dir=OUTPUT_DIR,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        max_images=None  # Set to a number to limit dataset size for testing
    )

    logger.info("\nDone! Dataset ready at: " + str(OUTPUT_DIR))
    logger.info("Next step: Run train_yolo.py to train the model")


if __name__ == "__main__":
    main()
