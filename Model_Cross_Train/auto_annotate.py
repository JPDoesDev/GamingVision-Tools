"""
Model Cross Trainer - Auto Annotator

Uses an existing ONNX model to auto-annotate images for cross-training.
Supports class mapping between source and target datasets.

Usage: python auto_annotate.py [config_path]
Default config path: ./config.json
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file, ignoring comment fields."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Filter out comment keys
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


def save_classes(classes_path: str, classes: list[str]) -> None:
    """Save class names to a text file."""
    os.makedirs(os.path.dirname(classes_path), exist_ok=True)
    with open(classes_path, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")


def get_model_input_size(session: ort.InferenceSession) -> tuple[int, int]:
    """Auto-detect model input size from ONNX model."""
    input_shape = session.get_inputs()[0].shape
    # Shape is typically [batch, channels, height, width]
    if len(input_shape) == 4:
        height = input_shape[2]
        width = input_shape[3]
        # Handle dynamic dimensions
        if isinstance(height, str) or height is None:
            height = 640  # Default fallback
        if isinstance(width, str) or width is None:
            width = 640
        return int(height), int(width)
    raise ValueError(f"Unexpected input shape: {input_shape}")


def preprocess_image(image: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int], tuple[float, float], tuple[int, int]]:
    """
    Preprocess image for YOLO model inference with letterboxing.

    Returns:
        - preprocessed image tensor
        - original size (height, width)
        - scale factors (scale_h, scale_w)
        - padding (pad_h, pad_w)
    """
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = input_size

    # Calculate scale to fit image in target size (letterbox)
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create letterbox canvas
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

    # Calculate padding
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    # Place resized image on canvas
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    # Convert BGR to RGB
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Normalize and transpose to NCHW format
    tensor = canvas.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, (orig_h, orig_w), scale, (pad_h, pad_w)


def postprocess_detections(
    outputs: np.ndarray,
    orig_size: tuple[int, int],
    input_size: tuple[int, int],
    scale: float,
    padding: tuple[int, int],
    num_classes: int,
    confidence_threshold: float
) -> list[dict]:
    """
    Post-process YOLO model outputs to get detections.

    Handles both YOLOv5/v8 output formats:
    - [batch, num_preds, 4 + num_classes] (v8 style, no objectness)
    - [batch, num_preds, 5 + num_classes] (v5 style, with objectness)
    - [batch, 4 + num_classes, num_preds] (transposed variants)
    """
    # Remove batch dimension
    output = outputs[0]

    # Handle transposed format [4+classes, num_preds] -> [num_preds, 4+classes]
    if output.shape[0] < output.shape[1] and output.shape[0] == (4 + num_classes):
        output = output.T
    elif output.shape[0] < output.shape[1] and output.shape[0] == (5 + num_classes):
        output = output.T

    detections = []
    orig_h, orig_w = orig_size
    input_h, input_w = input_size
    pad_h, pad_w = padding

    # Determine format based on shape
    if output.shape[-1] == 4 + num_classes:
        # YOLOv8 format: [x_center, y_center, width, height, class_scores...]
        has_objectness = False
    elif output.shape[-1] == 5 + num_classes:
        # YOLOv5 format: [x_center, y_center, width, height, objectness, class_scores...]
        has_objectness = True
    else:
        print(f"Warning: Unexpected output shape {output.shape}, expected last dim to be {4 + num_classes} or {5 + num_classes}")
        return detections

    for pred in output:
        if has_objectness:
            x, y, w, h, obj_conf = pred[:5]
            class_scores = pred[5:]
            # Multiply objectness by class scores
            class_scores = class_scores * obj_conf
        else:
            x, y, w, h = pred[:4]
            class_scores = pred[4:]

        # Get best class
        class_idx = int(np.argmax(class_scores))
        confidence = float(class_scores[class_idx])

        if confidence < confidence_threshold:
            continue

        # Convert from input coords to original image coords
        # Remove padding
        x = x - pad_w
        y = y - pad_h
        # Scale back to original size
        x = x / scale
        y = y / scale
        w = w / scale
        h = h / scale

        # Convert to normalized coordinates (0-1)
        x_norm = x / orig_w
        y_norm = y / orig_h
        w_norm = w / orig_w
        h_norm = h / orig_h

        # Clip to valid range
        x_norm = np.clip(x_norm, 0, 1)
        y_norm = np.clip(y_norm, 0, 1)
        w_norm = np.clip(w_norm, 0, 1)
        h_norm = np.clip(h_norm, 0, 1)

        detections.append({
            'class_idx': class_idx,
            'confidence': confidence,
            'x_center': float(x_norm),
            'y_center': float(y_norm),
            'width': float(w_norm),
            'height': float(h_norm)
        })

    return detections


def apply_nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Apply Non-Maximum Suppression to filter overlapping detections."""
    if not detections:
        return detections

    # Group by class
    class_detections = {}
    for det in detections:
        cls = det['class_idx']
        if cls not in class_detections:
            class_detections[cls] = []
        class_detections[cls].append(det)

    result = []
    for cls, dets in class_detections.items():
        # Sort by confidence (descending)
        dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while dets:
            best = dets.pop(0)
            keep.append(best)

            # Filter remaining detections
            remaining = []
            for det in dets:
                if compute_iou(best, det) < iou_threshold:
                    remaining.append(det)
            dets = remaining

        result.extend(keep)

    return result


def compute_iou(det1: dict, det2: dict) -> float:
    """Compute Intersection over Union between two detections."""
    # Convert center format to corner format
    x1_min = det1['x_center'] - det1['width'] / 2
    y1_min = det1['y_center'] - det1['height'] / 2
    x1_max = det1['x_center'] + det1['width'] / 2
    y1_max = det1['y_center'] + det1['height'] / 2

    x2_min = det2['x_center'] - det2['width'] / 2
    y2_min = det2['y_center'] - det2['height'] / 2
    x2_max = det2['x_center'] + det2['width'] / 2
    y2_max = det2['y_center'] + det2['height'] / 2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    # Union
    area1 = det1['width'] * det1['height']
    area2 = det2['width'] * det2['height']
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


def load_existing_annotations(label_path: str) -> list[tuple[int, float, float, float, float]]:
    """Load existing YOLO format annotations from a label file."""
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

    model_path = resolve_path(config['model_path'])
    model_classes_path = resolve_path(config['model_classes_path'])
    images_dir = resolve_path(config['images_dir'])
    labels_dir = resolve_path(config['labels_dir'])
    target_classes_path = resolve_path(config['target_classes_path'])

    classes_to_detect = config['classes_to_detect']
    confidence_threshold = config.get('confidence_threshold', 0.3)
    use_nms = config.get('use_nms', False)
    nms_iou_threshold = config.get('nms_iou_threshold', 0.5)

    # Validate paths
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(model_classes_path):
        print(f"Error: Model classes file not found: {model_classes_path}")
        sys.exit(1)

    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    # Load source model classes
    source_classes = load_classes(model_classes_path)
    print(f"Loaded {len(source_classes)} source model classes")

    # Validate classes to detect exist in source model
    source_class_to_idx = {name: idx for idx, name in enumerate(source_classes)}
    for cls in classes_to_detect:
        if cls not in source_class_to_idx:
            print(f"Error: Class '{cls}' not found in source model. Available classes:")
            for idx, name in enumerate(source_classes):
                print(f"  {idx}: {name}")
            sys.exit(1)

    print(f"Classes to detect: {classes_to_detect}")

    # Load or initialize target classes
    target_classes = load_classes(target_classes_path)
    print(f"Loaded {len(target_classes)} existing target classes")

    # Build class mapping: source class name -> target class index
    # Add missing classes to target if needed
    class_mapping = {}  # source_class_name -> target_class_idx
    target_modified = False

    for cls in classes_to_detect:
        if cls in target_classes:
            target_idx = target_classes.index(cls)
            print(f"  Class '{cls}': using existing target index {target_idx}")
        else:
            target_idx = len(target_classes)
            target_classes.append(cls)
            target_modified = True
            print(f"  Class '{cls}': added to target as index {target_idx}")
        class_mapping[cls] = target_idx

    # Save updated target classes if modified
    if target_modified:
        save_classes(target_classes_path, target_classes)
        print(f"Saved updated target classes to: {target_classes_path}")

    # Load ONNX model (CPU mode)
    print(f"\nLoading ONNX model: {model_path}")
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)

    # Get model input size
    input_size = get_model_input_size(session)
    print(f"Model input size: {input_size[1]}x{input_size[0]} (WxH)")

    input_name = session.get_inputs()[0].name

    # Get list of images to process
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

    # Display summary and confirmation prompt
    print("\n" + "=" * 60)
    print("AUTO-ANNOTATION SUMMARY")
    print("=" * 60)
    print(f"\nSOURCE MODEL:")
    print(f"  Model:        {os.path.basename(model_path)}")
    print(f"  Input size:   {input_size[1]}x{input_size[0]} (WxH)")
    print(f"  Classes:      {len(source_classes)} total")
    print(f"\nTARGET DATASET:")
    print(f"  Images dir:   {images_dir}")
    print(f"  Labels dir:   {labels_dir}")
    print(f"  Images:       {len(image_files)} to process")
    print(f"  Existing:     {existing_labels} label file(s) already exist")
    print(f"\nDETECTION SETTINGS:")
    print(f"  Confidence:   {confidence_threshold}")
    print(f"  NMS:          {'Enabled (IoU: ' + str(nms_iou_threshold) + ')' if use_nms else 'Disabled'}")
    print(f"\nCLASS MAPPING:")
    for cls in classes_to_detect:
        source_idx = source_class_to_idx[cls]
        target_idx = class_mapping[cls]
        print(f"  '{cls}': source[{source_idx}] -> target[{target_idx}]")
    print(f"\nOPERATION:")
    print(f"  - Detect: {classes_to_detect}")
    print(f"  - Will append to existing labels (skip if class already annotated)")
    if target_modified:
        print(f"  - New classes will be added to: {target_classes_path}")
    print("\n" + "=" * 60)

    # Prompt for confirmation
    while True:
        response = input("\nProceed with auto-annotation? (y/n): ").strip().lower()
        if response in ('y', 'yes'):
            break
        elif response in ('n', 'no'):
            print("Aborted by user.")
            sys.exit(0)
        else:
            print("Please enter 'y' or 'n'.")

    print(f"\nProcessing {len(image_files)} images...")

    # Create labels directory if needed
    os.makedirs(labels_dir, exist_ok=True)

    # Process each image
    total_detections = 0
    images_with_detections = 0

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  Warning: Could not load image: {img_file}")
            continue

        # Preprocess
        tensor, orig_size, scale, padding = preprocess_image(image, input_size)

        # Run inference
        outputs = session.run(None, {input_name: tensor})

        # Post-process
        detections = postprocess_detections(
            outputs[0],
            orig_size,
            input_size,
            scale,
            padding,
            len(source_classes),
            confidence_threshold
        )

        # Apply NMS if enabled
        if use_nms and detections:
            detections = apply_nms(detections, nms_iou_threshold)

        # Filter to only classes we want to detect
        source_detect_indices = {source_class_to_idx[cls] for cls in classes_to_detect}
        detections = [d for d in detections if d['class_idx'] in source_detect_indices]

        # Load existing annotations
        existing_annotations = load_existing_annotations(label_path)
        existing_class_indices = {ann[0] for ann in existing_annotations}

        # Add new detections (only if class not already present)
        new_annotations = list(existing_annotations)
        added_count = 0

        for det in detections:
            source_class_name = source_classes[det['class_idx']]
            target_class_idx = class_mapping[source_class_name]

            # Skip if this class already exists in annotations
            if target_class_idx in existing_class_indices:
                continue

            new_annotations.append((
                target_class_idx,
                det['x_center'],
                det['y_center'],
                det['width'],
                det['height']
            ))
            existing_class_indices.add(target_class_idx)
            added_count += 1

        # Save if we have annotations
        if new_annotations:
            save_annotations(label_path, new_annotations)

        if added_count > 0:
            total_detections += added_count
            images_with_detections += 1
            print(f"  {img_file}: Added {added_count} new annotation(s)")
        else:
            print(f"  {img_file}: No new detections")

    print(f"\nComplete!")
    print(f"  Images processed: {len(image_files)}")
    print(f"  Images with new detections: {images_with_detections}")
    print(f"  Total annotations added: {total_detections}")


if __name__ == '__main__':
    main()
