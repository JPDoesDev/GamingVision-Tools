# Model Cross Trainer - Auto Annotator

A Python tool that uses existing ONNX computer vision models to auto-annotate images for new datasets. This enables "cross-training" by jumpstarting new model training with detections from models trained on similar content.

## Use Case

You have a trained object detection model for Game A that detects classes like `waypoint`, `interact`, `enemy`, etc. You want to train a new model for Game B, which has visually similar elements. Instead of manually annotating everything from scratch, this tool:

1. Runs the Game A model on Game B screenshots
2. Automatically generates YOLO-format annotations for selected classes
3. Handles class index mapping between source and target datasets

## Requirements

- Python 3.10+
- Dependencies:
  ```
  pip install onnxruntime opencv-python numpy
  ```

## Project Structure

```
Model_Cross_Train/
├── auto_annotate.py          # Main script
├── config.json               # Configuration file
├── README.md
├── crosstrain_model/         # Source model(s)
│   ├── model.onnx
│   └── model.txt             # Class names (one per line)
└── new_training_capture/     # Target dataset
    ├── images/
    │   ├── screenshot_001.jpg
    │   └── screenshot_002.jpg
    └── labels/
        └── classes.txt       # Target class names
```

## Configuration

Edit `config.json` to configure the tool. All paths are relative to the config file location.

### Required Settings

| Setting | Description |
|---------|-------------|
| `model_path` | Path to the ONNX model file |
| `model_classes_path` | Path to source model's classes.txt |
| `images_dir` | Directory containing images to annotate |
| `labels_dir` | Output directory for annotation files |
| `target_classes_path` | Path to target dataset's classes.txt |
| `classes_to_detect` | List of class names to detect |

### Optional Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `confidence_threshold` | `0.3` | Minimum confidence (0.0-1.0) for detections |
| `use_nms` | `false` | Enable Non-Maximum Suppression |
| `nms_iou_threshold` | `0.5` | IoU threshold for NMS (if enabled) |

### Example Configuration

```json
{
    "model_path": "./crosstrain_model/arc_raiders_model.onnx",
    "model_classes_path": "./crosstrain_model/arc_raiders_model.txt",
    "images_dir": "./new_training_capture/images",
    "labels_dir": "./new_training_capture/labels",
    "target_classes_path": "./new_training_capture/labels/classes.txt",
    "classes_to_detect": ["interact", "waypoint", "enemy"],
    "confidence_threshold": 0.3,
    "use_nms": false,
    "nms_iou_threshold": 0.5
}
```

## Usage

### Basic Usage

```bash
python auto_annotate.py
```

This uses `./config.json` in the current directory.

### Custom Config Path

```bash
python auto_annotate.py path/to/config.json
```

### Example Output

```
Loading config from: ./config.json
Loaded 21 source model classes
Classes to detect: ['interact', 'waypoint']
Loaded 0 existing target classes
  Class 'interact': added to target as index 0
  Class 'waypoint': added to target as index 1
Saved updated target classes to: ./new_training_capture/labels/classes.txt

Loading ONNX model: ./crosstrain_model/model.onnx
Model input size: 1440x1440 (WxH)

============================================================
AUTO-ANNOTATION SUMMARY
============================================================

SOURCE MODEL:
  Model:        model.onnx
  Input size:   1440x1440 (WxH)
  Classes:      21 total

TARGET DATASET:
  Images dir:   ./new_training_capture/images
  Labels dir:   ./new_training_capture/labels
  Images:       7 to process
  Existing:     0 label file(s) already exist

DETECTION SETTINGS:
  Confidence:   0.3
  NMS:          Disabled

CLASS MAPPING:
  'interact': source[5] -> target[0]
  'waypoint': source[9] -> target[1]

OPERATION:
  - Detect: ['interact', 'waypoint']
  - Will append to existing labels (skip if class already annotated)
  - New classes will be added to: ./new_training_capture/labels/classes.txt

============================================================

Proceed with auto-annotation? (y/n): y

Processing 7 images...
  screenshot_001.jpg: Added 2 new annotation(s)
  screenshot_002.jpg: No new detections
  screenshot_003.jpg: Added 1 new annotation(s)

Complete!
  Images processed: 7
  Images with new detections: 2
  Total annotations added: 3
```

## Class Index Mapping

The tool automatically handles class index differences between source and target models.

### Scenario 1: Empty Target classes.txt

If the target `classes.txt` is empty, detected classes are added in the order specified in `classes_to_detect`:

```
# classes_to_detect: ["interact", "waypoint"]
# Result in target classes.txt:
interact    # index 0
waypoint    # index 1
```

### Scenario 2: Existing Target Classes

If the target already has classes, the tool maps to existing indices:

```
# Existing target classes.txt:
enemy       # index 0
loot        # index 1
waypoint    # index 2

# classes_to_detect: ["interact", "waypoint"]
# Result:
# - "waypoint" detections use index 2 (existing)
# - "interact" added as index 3 (new)
```

## Annotation Behavior

### Appending to Existing Labels

If a label file already exists for an image, the tool:
- Loads existing annotations
- Only adds detections for classes **not already present**
- Preserves all existing annotations

This prevents duplicate annotations and assumes existing labels are correct.

### Output Format

Annotations are saved in YOLO format:

```
class_index x_center y_center width height
```

All coordinates are normalized (0.0 to 1.0):

```
0 0.576006 0.650247 0.047052 0.033602
1 0.587476 0.413059 0.024317 0.054011
```

## Supported Formats

### Images
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.webp`

### Models
- ONNX format (`.onnx`)
- YOLOv5/v8 style outputs
- Auto-detects input resolution

## Workflow

1. **Prepare source model**: Place your trained ONNX model and its classes.txt in `crosstrain_model/`

2. **Capture target images**: Add screenshots/images to `new_training_capture/images/`

3. **Configure**: Edit `config.json` with appropriate paths and classes to detect

4. **Run auto-annotation**:
   ```bash
   python auto_annotate.py
   ```

5. **Review annotations**: Use your preferred annotation tool to verify and correct the auto-generated labels

6. **Train new model**: Use the annotated dataset to train your new model

## Tips

- **Start with low confidence**: Use `0.3` or lower initially, then review results. It's easier to delete false positives than to add missed detections.

- **Choose transferable classes**: Focus on classes that look similar across games (UI elements, generic markers, common shapes).

- **Verify before training**: Always review auto-generated annotations before training. This tool accelerates annotation, but human verification is essential.

- **Incremental annotation**: Run the tool multiple times with different source models to detect different classes.
