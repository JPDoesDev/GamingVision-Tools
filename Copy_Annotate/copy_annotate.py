"""
Copy Annotate - Static Annotation Copier

Copies annotations for static screen elements (HUD, etc.) from a master
annotation file to all images in a dataset. Useful for elements that are
always in the same screen position.

Usage: python copy_annotate.py [config_path]
Default config path: ./config.json
"""

import json
import os
import shutil
import sys


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file, ignoring comment fields."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return {k: v for k, v in config.items() if not k.startswith('_comment')}


def load_classes(classes_path: str) -> list[str]:
    """Load class names from a text file (one class per line)."""
    classes = []
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    classes.append(line)
    return classes


def load_annotations(label_path: str) -> list[tuple[int, float, float, float, float]]:
    """Load YOLO format annotations from a label file."""
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_idx = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append((class_idx, x_center, y_center, width, height))
    return annotations


def save_annotations(label_path: str, annotations: list[tuple[int, float, float, float, float]]) -> None:
    """Save annotations in YOLO format."""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, 'w') as f:
        for ann in annotations:
            class_idx, x, y, w, h = ann
            f.write(f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def find_annotation_file(directory: str) -> str | None:
    """Find the annotation .txt file in the master directory (not classes.txt)."""
    for filename in os.listdir(directory):
        if filename.endswith('.txt') and filename.lower() != 'classes.txt':
            return os.path.join(directory, filename)
    return None


def main():
    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else './config.json'

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Resolve paths relative to config file location
    config_dir = os.path.dirname(os.path.abspath(config_path))

    def resolve_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(config_dir, path))

    master_dir = resolve_path(config['master_annotation_dir'])
    images_dir = resolve_path(config['images_dir'])
    labels_dir = resolve_path(config['labels_dir'])
    classes_to_copy = config['classes_to_copy']

    # Validate master directory
    if not os.path.exists(master_dir):
        print(f"Error: Master annotation directory not found: {master_dir}")
        sys.exit(1)

    # Find and validate master files
    master_classes_path = os.path.join(master_dir, 'classes.txt')
    if not os.path.exists(master_classes_path):
        print(f"Error: classes.txt not found in master directory: {master_dir}")
        sys.exit(1)

    master_annotation_path = find_annotation_file(master_dir)
    if not master_annotation_path:
        print(f"Error: No annotation .txt file found in master directory: {master_dir}")
        sys.exit(1)

    # Validate images directory
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    # Load master classes
    master_classes = load_classes(master_classes_path)
    print(f"Loaded {len(master_classes)} classes from master")

    # Validate classes to copy exist in master
    master_class_to_idx = {name: idx for idx, name in enumerate(master_classes)}
    for cls in classes_to_copy:
        if cls not in master_class_to_idx:
            print(f"Error: Class '{cls}' not found in master classes.txt")
            print(f"Available classes: {master_classes}")
            sys.exit(1)

    # Load master annotations
    master_annotations = load_annotations(master_annotation_path)
    print(f"Loaded {len(master_annotations)} annotations from: {os.path.basename(master_annotation_path)}")

    # Filter to only annotations for classes we want to copy
    classes_to_copy_indices = {master_class_to_idx[cls] for cls in classes_to_copy}
    annotations_to_copy = [ann for ann in master_annotations if ann[0] in classes_to_copy_indices]

    if not annotations_to_copy:
        print(f"Error: No annotations found for selected classes: {classes_to_copy}")
        sys.exit(1)

    print(f"Found {len(annotations_to_copy)} annotation(s) to copy for classes: {classes_to_copy}")

    # Target classes.txt path - will always be copied from master
    target_classes_path = os.path.join(labels_dir, 'classes.txt')

    # Get list of images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]

    if not image_files:
        print(f"No images found in: {images_dir}")
        sys.exit(0)

    # Count existing label files
    existing_labels = sum(
        1 for f in image_files
        if os.path.exists(os.path.join(labels_dir, os.path.splitext(f)[0] + '.txt'))
    )

    # Display summary and confirmation
    print("\n" + "=" * 60)
    print("COPY ANNOTATE SUMMARY")
    print("=" * 60)
    print(f"\nMASTER ANNOTATION:")
    print(f"  Directory:    {master_dir}")
    print(f"  Annotation:   {os.path.basename(master_annotation_path)}")
    print(f"  Classes:      {len(master_classes)} total")
    print(f"\nTARGET DATASET:")
    print(f"  Images dir:   {images_dir}")
    print(f"  Labels dir:   {labels_dir}")
    print(f"  Images:       {len(image_files)} to process")
    print(f"  Existing:     {existing_labels} label file(s) already exist")
    print(f"\nCLASSES TO COPY:")
    for cls in classes_to_copy:
        idx = master_class_to_idx[cls]
        # Find the annotation for this class
        ann = next((a for a in annotations_to_copy if a[0] == idx), None)
        if ann:
            print(f"  [{idx}] {cls}: ({ann[1]:.3f}, {ann[2]:.3f}, {ann[3]:.3f}, {ann[4]:.3f})")
    print(f"\nOPERATION:")
    print(f"  - Copy {len(annotations_to_copy)} annotation(s) to {len(image_files)} image(s)")
    print(f"  - Skip if class already exists in label file")
    print(f"  - Copy master classes.txt to: {target_classes_path}")
    print("\n" + "=" * 60)

    # Confirmation prompt
    while True:
        response = input("\nProceed with annotation copy? (y/n): ").strip().lower()
        if response in ('y', 'yes'):
            break
        elif response in ('n', 'no'):
            print("Aborted by user.")
            sys.exit(0)
        else:
            print("Please enter 'y' or 'n'.")

    # Create labels directory if needed
    os.makedirs(labels_dir, exist_ok=True)

    # Always copy master classes.txt to target to ensure indices match
    shutil.copy2(master_classes_path, target_classes_path)
    print(f"\nCopied classes.txt to: {target_classes_path}")

    # Process each image
    print(f"\nProcessing {len(image_files)} images...")

    total_added = 0
    images_modified = 0

    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        # Load existing annotations
        existing_annotations = load_annotations(label_path)
        existing_class_indices = {ann[0] for ann in existing_annotations}

        # Add annotations that don't already exist
        new_annotations = list(existing_annotations)
        added_count = 0

        for ann in annotations_to_copy:
            class_idx = ann[0]
            if class_idx not in existing_class_indices:
                new_annotations.append(ann)
                existing_class_indices.add(class_idx)
                added_count += 1

        # Save if we added anything (or create new file)
        if added_count > 0 or not existing_annotations:
            save_annotations(label_path, new_annotations)

        if added_count > 0:
            total_added += added_count
            images_modified += 1
            print(f"  {img_file}: Added {added_count} annotation(s)")
        else:
            print(f"  {img_file}: Skipped (classes already present)")

    print(f"\nComplete!")
    print(f"  Images processed: {len(image_files)}")
    print(f"  Images modified: {images_modified}")
    print(f"  Total annotations added: {total_added}")


if __name__ == '__main__':
    main()
