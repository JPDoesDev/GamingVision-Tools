# Copy Annotate - Static Annotation Copier

A Python tool that copies annotations for static screen elements (HUD, UI, etc.) from a master annotation file to all images in a dataset. Useful for elements that are always in the same screen position across all screenshots.

## Use Case

Many games have HUD elements (health bars, ammo counters, minimaps) that appear in the exact same screen position in every frame. Instead of manually annotating these elements on hundreds of images, you can:

1. Annotate them once in a master file
2. Use this tool to copy those annotations to all images

## Requirements

- Python 3.10+
- No additional dependencies (uses only standard library)

## Project Structure

```
Copy_Annotate/
├── copy_annotate.py       # Main script
├── config.json            # Configuration file
├── README.md
├── Master_Annotation/     # Master reference files
│   ├── classes.txt        # Class definitions
│   └── reference.txt      # Master annotation file (any name)
├── images/                # Images to annotate
│   ├── screenshot_001.jpg
│   └── screenshot_002.jpg
└── labels/                # Output annotations
    ├── classes.txt        # Copied from master (always)
    ├── screenshot_001.txt
    └── screenshot_002.txt
```

## Setup

1. Create a `Master_Annotation/` folder with:
   - `classes.txt` - One class name per line
   - An annotation `.txt` file with YOLO format annotations for static elements

2. Place your images in the `images/` folder

3. Edit `config.json` to select which classes to copy

## Configuration

```json
{
    "master_annotation_dir": "./Master_Annotation",
    "images_dir": "./images",
    "labels_dir": "./labels",
    "classes_to_copy": ["hud-ammo", "hud-hp", "minimap"]
}
```

| Setting | Description |
|---------|-------------|
| `master_annotation_dir` | Directory containing classes.txt and master annotation file |
| `images_dir` | Directory containing images to annotate |
| `labels_dir` | Output directory for annotation files |
| `classes_to_copy` | List of class names to copy (must exist in master) |

## Usage

```bash
python copy_annotate.py                     # uses ./config.json
python copy_annotate.py path/to/config.json # custom config
```

### Example Output

```
============================================================
COPY ANNOTATE SUMMARY
============================================================

MASTER ANNOTATION:
  Directory:    ./Master_Annotation
  Annotation:   reference.txt
  Classes:      6 total

TARGET DATASET:
  Images dir:   ./images
  Labels dir:   ./labels
  Images:       50 to process
  Existing:     0 label file(s) already exist

CLASSES TO COPY:
  [1] hud-ammo: (0.849, 0.931, 0.169, 0.065)
  [2] hud-hp: (0.149, 0.934, 0.177, 0.040)

OPERATION:
  - Copy 2 annotation(s) to 50 image(s)
  - Skip if class already exists in label file
  - Copy master classes.txt to: ./labels/classes.txt

============================================================

Proceed with annotation copy? (y/n): y

Processing 50 images...
  screenshot_001.jpg: Added 2 annotation(s)
  screenshot_002.jpg: Added 2 annotation(s)
  ...

Complete!
  Images processed: 50
  Images modified: 50
  Total annotations added: 100
```

## Behavior

### Class Index Handling

The tool always copies the master `classes.txt` to the target labels directory, ensuring class indices are consistent. This means:
- The master `classes.txt` is the source of truth
- Annotations use the same indices as defined in master
- No index remapping or validation needed

### Skipping Existing Annotations

When processing each image:
- If a label file already contains an annotation for a class, that class is skipped
- Other classes are still added
- This allows incremental annotation without duplicates

### Output Format

Annotations are saved in YOLO format:
```
class_index x_center y_center width height
```

All coordinates are normalized (0.0 to 1.0).

## Workflow

1. **Capture reference screenshot**: Take one screenshot with all static HUD elements visible

2. **Create master annotation**: Annotate the static elements in your annotation tool, save as YOLO format

3. **Set up Master_Annotation folder**: Copy the `classes.txt` and annotation file

4. **Configure**: Edit `config.json` with classes to copy

5. **Run**: Execute the script to copy annotations to all images

6. **Verify**: Spot-check a few images in your annotation tool

## Tips

- Only include truly static elements - elements that move or disappear should be annotated normally
- The master annotation file can have any name (the script finds any `.txt` that isn't `classes.txt`)
- Run the tool again after adding new images - it will skip already-annotated files
- The master `classes.txt` is always copied to the target, overwriting any existing file - ensure your master has all classes you need
