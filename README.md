# AutoLabel

A fast, single-object video annotation tool. The tool runs a short, adaptive analysis of global (camera) and local (object) motion to learn stable features and scale priors. It then tracks the object frame-by-frame to maintain a relatively tight bounding box even under camera motion.

## Overview

AutoLabel provides two complementary CLI tools for video object annotation:
- **Labeling Tool**: Interactive frame-by-frame annotation with intelligent object tracking
- **Validation Tool**: Quality assurance visualization of annotations


## Quick Start

```bash
# Clone the repository
git clone https://github.com/rexlee2/autolabel.git
cd autolabel

# Install dependencies
pip install -r requirements.txt

# Run the labeling tool
python -m autolabel.cli assets/sample_video.mp4

# Validate annotations
python -m validation.cli assets/sample_video.mp4
```

## Installation

### Prerequisites
- Python 3.8+
- OpenCV with contrib modules (for CSRT tracker)
- Linux/macOS/Windows with GUI support

### Setup Instructions

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Dependencies
```
opencv-contrib-python>=4.5.0
numpy>=1.19.0
typer>=0.9.0
pyyaml>=5.4.0
```

## Usage

### Program 1: Labeling Tool

```bash
# Basic usage with default output
python -m autolabel.cli assets/sample_video.mp4

# Specify output file
python -m autolabel.cli assets/sample_video.mp4 --out assets/sample_video.annotations

```


### Program 2: Validation Tool

```bash
# Validate with default annotation file <path_to_video>/<video_name>.annotations
python -m validation.cli assets/sample_video.mp4

# Specify annotation file explicitly
python -m validation.cli assets/sample_video.mp4 --ann assets/sample_video.annotations

# Headless mode (generates output video)
python -m validation.cli assets/sample_video.mp4 --headless --out assets/sample_video_validation.mp4
```

## Annotation File Format

Each line corresponds to one video frame:

```
V 250 150 50 75    # Visible: V <x_center> <y_center> <width> <height>
S -1 -1 -1 -1      # Skipped: S -1 -1 -1 -1
I -1 -1 -1 -1      # Invisible: I -1 -1 -1 -1
```

## Architecture & Design Decisions


#### **1. File Organization**
```
.
├── autolabel/
│   ├── __init__.py
│   ├── cli.py           # Main labeling tool entry point
│   ├── ui_utils.py      # Shared utilities (UI, frame reading, geometry)
│   ├── tracker.py       # Object tracking with CSRT
│   └── tracker.yaml     # Tracker configuration
├── validation/
│   ├── __init__.py
│   └── cli.py           # Validation tool
├── assets/
│   └── sample_video.mp4
├── requirements.txt
├── setup.py
└── README.md
```


#### **2. CSRT Tracker Selection**

**Why CSRT over alternatives?**
- **Accuracy**: Superior to KCF/MOSSE for precise bounding boxes
- **Robustness**: Handles scale changes and partial occlusions
- **Speed**: Fast enough for interactive use
- **No additional dependency**: Zero additional dependencies 

**Alternatives considered:**
- **ByteTrack/SORT/DeepSORT**: Detection-based trackers that require an object detector (YOLO, etc.) to provide bounding boxes each frame. **Issue**: Cannot track arbitrary objects not in the detector's training classes such as "something that looks like helicoper landing pad". When tested, ByteTrack failed to track user-defined objects that YOLO couldn't recognize, making it unsuitable for open-vocabulary annotation.
- **KCF (Kernelized Correlation Filter)**: Faster than CSRT but struggles with scale changes and rotation. Would require manual scale adjustment in many frames.
- **MOSSE**: Extremely fast but poor accuracy, especially with appearance changes or occlusions.
- **MedianFlow**: Good for smooth, predictable motion but fails with fast movement or occlusions.


#### **3. User Interface Design**

**Features implemented:**
- Dynamic window resizing with image aspect ratio preservation
- DPI awareness for high-resolution displays
- Graceful fallback to manual labeling if tracker fails
- Fills incomplete annotations with "Skips" so that it can be completed later  


## Configuration

### Tracker Parameters (tracker.yaml)

```yaml
csrt_params:
  filter_lr: 0.02        # Learning rate for filter updates
  scale_lr: 0.2          # Scale adaptation rate
  psr_threshold: 0.045   # Peak-to-sidelobe ratio for quality
```

Tune the three key parameters first for given video:
- **Higher filter_lr**: Faster adaptation, less stable
- **Lower psr_threshold**: More permissive tracking
- **More scales**: Better scale handling, slower performance


## Known Limitations

1. **Tracking quality**: Limited tracking in casess where object changes shape/size quickly
2. **Insufficient testing**: Not yet tested on variety of videos and annotation tasks
3. **Single object only**: Not designed for multi-object tracking

## Future Improvements?

- [ ] Improve tracking quality for a wide range of annotation tasks
- [ ] Resuming annotation from previous session
- [ ] Text-based initial bounding-box
- [ ] Multi-object tracking support


## AI Usages

**Tools used**: ChatGPT-5 and Claude

**How AI was utilized**:
1. **Code refactoring**: Improving code readability and structure
2. **Poor/weak code identification**: Poor or weak code identification
3. **Documentation**: Generation of initial README.md


## License
MIT License - See LICENSE file for details

