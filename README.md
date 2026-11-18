# ME416 Hair Cutting Robot — 3D Skull Detection & SDF Pipeline

Multi-view 3D head and skull reconstruction from four handheld smartphone photos (`Front.jpg`, `LeftSide.jpg`, `RightSide.jpg`, and `Back.jpg`), generating a metric 3D skull mesh (`.stl`) with a scalp safety buffer and a compact `64×64×64` Signed Distance Field (`.bin`) for real-time microcontroller cutter boundary enforcement.

---

## Project Overview & Pipeline

1. **Multi-View Landmark & Anchor Alignment (`fit_skull.py`):**
   - Extracts 2D facial landmarks from `Front.jpg`, `LeftSide.jpg`, and `RightSide.jpg` using MediaPipe FaceLandmarker (`face_landmarker.task`).
   - Fits a symmetric, 20-component FLAME 3D head/skull model using perspective pinhole camera projection and per-view neck articulation (handling handheld selfie head turns).
2. **Metric Calibration & Safety Buffer:**
   - Uses a standard CR80 ID card (`85.60 mm` wide) held against the forehead in `Front.jpg` to scale the 3D mesh into real-world millimeters.
   - Applies a smooth `+3.0 mm` outward safety offset across the scalp (`FLAME_masks.pkl["scalp"]`).
3. **Microcontroller Boundary Export:**
   - Exports `output/safe_skull_boundary.stl` for CAD / 3D zone segmentation.
   - Voxelizes the skull boundary into a `64×64×64` `int16` Signed Distance Field (`output/skull_sdf_64x64x64.bin`, `512 KB`) and metadata (`output/skull_sdf_meta.json`) for embedded lookup on an ESP32-S3.

---

## Quick Start

### 1. Clone & Install Dependencies
```bash
git clone [https://github.com/JulioR789/ME416HairCuttingRobotSkullDetection.git](https://github.com/JulioR789/ME416HairCuttingRobotSkullDetection.git)
cd ME416HairCuttingRobotSkullDetection
pip install -r requirements.txt
```

### 2. Run Reconstruction on Sample or New Photos
Place four images inside `captures/` (`Front.jpg`, `LeftSide.jpg`, `RightSide.jpg`, `Back.jpg`) or use the included sample images, then run:
```bash
python fit_skull.py
```
*(Note: `model/face_landmarker.task` downloads automatically on the first run.)*

### 3. Interactive Calibration Clicks During Execution
When prompted by the OpenCV windows, click the two requested reference points and press **any key** to advance:
- **`LeftSide.jpg` & `RightSide.jpg`:** Click **(1)** the front ear notch (tragus, right in front of the ear canal) and **(2)** the back of the neck where the lower hairline ends.
- **`Back.jpg`:** Click **(1)** the outer middle edge of the left ear and **(2)** the outer middle edge of the right ear.
- **`Front.jpg`:** Click **(1)** the far-left edge and **(2)** the far-right edge of the ID card (`85.60 mm`).

---

## Repository Structure

```text
ME416HairCuttingRobotSkullDetection/
├── fit_skull.py             # Main multi-view 3D skull fitting, overlay renderer, and SDF exporter
├── requirements.txt         # Python dependencies (torch, mediapipe, opencv-python, trimesh, scipy, numpy)
├── captures/                # Input smartphone images (Front.jpg, LeftSide.jpg, RightSide.jpg, Back.jpg)
├── model/                   # FLAME template (generic_model.pkl), masks (FLAME_masks.pkl), & MediaPipe embedding
└── output/                  # Generated STL mesh, 512 KB SDF binary, JSON metadata, and 4-view debug overlays
```

---

## Next Stage: Cutting Zone Segmentation *(In Progress)*
- Partition the fitted scalp mesh into **Top/Crown**, **Left Side**, **Right Side**, **Back/Occipital**, and **No-Cut** regions (face, ears, neck).
- Map per-zone target hair lengths (`mm`) to maximum cutter extension bounds in the SDF lookup table.

---

## Model Attribution & Citations (FLAME)

This project builds upon the **FLAME** (*Faces Learned with an Articulated Model and Expressions*) 3D head model and PyTorch framework developed by the Max Planck Institute for Intelligent Systems.

- **FLAME Project Page & License:** [http://flame.is.tue.mpg.de/](http://flame.is.tue.mpg.de/) (Available under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/)).
- **Landmark Embedding:** [RingNet Project](https://github.com/soubhiksanyal/RingNet/tree/master/flame_model).

```bibtex
@article{FLAME:SiggraphAsia2017,
  title   = {Learning a model of facial shape and expression from {4D} scans},
  author  = {Li, Tianye and Bolkart, Timo and Black, Michael. J. and Li, Hao and Romero, Javier},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
  volume  = {36},
  number  = {6},
  year    = {2017},
  url     = {[https://doi.org/10.1145/3130800.3130813](https://doi.org/10.1145/3130800.3130813)}
}
```

```bibtex
@inproceedings{RingNet:CVPR:2019,
  title     = {Learning to Regress 3D Face Shape and Expression from an Image without 3D Supervision},
  author    = {Sanyal, Soubhik and Bolkart, Timo and Feng, Haiwen and Black, Michael},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  month     = jun,
  year      = {2019}
}
```
