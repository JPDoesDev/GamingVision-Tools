# GamingVision Tools

A collection of Python tools for accelerating computer vision dataset annotation, specifically designed for gaming applications.

## Tools

### Model Cross Trainer

**Location:** `Model_Cross_Train/`

Uses an existing trained ONNX model to auto-annotate images for a new dataset. Enables "cross-training" by leveraging models trained on similar games to jumpstart annotation of new games.

**Key Features:**
- Run ONNX models in CPU mode for detection
- Select specific classes to detect
- Automatic class index mapping between source and target datasets
- Appends to existing annotations without duplicates
- Configurable confidence threshold and NMS

**Use Case:** You have a trained model for Game A that detects `waypoint`, `interact`, `enemy`. Game B has similar visual elements. Use the Game A model to auto-annotate Game B screenshots, then refine manually.

```bash
cd Model_Cross_Train
python auto_annotate.py
```

---

### Copy Annotate

**Location:** `Copy_Annotate/`

Copies annotations for static screen elements from a master annotation file to all images. For HUD elements that appear in the exact same position in every frame.

**Key Features:**
- One-time annotation of static elements, applied to all images
- Copies master classes.txt to ensure consistent indices
- Skips images that already have the selected classes
- No external dependencies

**Use Case:** Health bars, ammo counters, minimaps, and other HUD elements that never move can be annotated once and copied to hundreds of images instantly.

```bash
cd Copy_Annotate
python copy_annotate.py
```

---

## Requirements

- Python 3.10+
- For Model Cross Trainer:
  ```
  pip install onnxruntime opencv-python numpy
  ```
- For Copy Annotate: No additional dependencies

## Quick Start

1. Clone or download this repository
2. Navigate to the tool directory
3. Edit `config.json` with your paths and settings
4. Run the Python script
5. Review the summary and confirm with `y`

Each tool displays a detailed summary before executing and requires confirmation.

## Project Structure

```
GamingVision-Tools/
├── README.md
├── Model_Cross_Train/
│   ├── auto_annotate.py
│   ├── config.json
│   ├── README.md
│   ├── crosstrain_model/      # Source ONNX models
│   └── new_training_capture/  # Target dataset
│       ├── images/
│       └── labels/
└── Copy_Annotate/
    ├── copy_annotate.py
    ├── config.json
    ├── README.md
    ├── Master_Annotation/     # Reference annotations
    ├── images/
    └── labels/
```

## Common Workflow

1. **Capture screenshots** from a new game
2. **Use Model Cross Trainer** to auto-annotate with a similar game's model
3. **Use Copy Annotate** to add static HUD element annotations
4. **Review and refine** annotations in your annotation tool
5. **Train** your new model with the annotated dataset
