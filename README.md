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

## Cutting Zones (Flowbee)

`zones.py` runs at the end of `fit_skull.py` and splits the fitted scalp into haircut zones, each with its own target hair length (Flowbee spacer / actuator setting). Boundaries follow the fitted anatomy:

| id | Zone | Boundary | Default |
|---|---|---|---|
| 0 | `NO_CUT` | face, forehead, ears, neck below the nape hairline (plane through the clicked nape point and both ear lobes) | no cut |
| 1 | `TOP` | surface tilted > 25° upward (parietal ridge) + front hairline | 25 mm |
| 2 / 3 | `LEFT` / `RIGHT` | subject's left / right, in front of the back edge of the ear | 12 mm |
| 4 | `BACK` | behind the ears, above tragus height | 12 mm |
| 5 | `NAPE` | behind the ears, below tragus height (taper) | 6 mm |
| 6 | `EAR_GUARD` | 10 mm band around each ear | no cut |

Edit **`zones_config.json`** (created on first run) to change lengths, enable/disable zones, or move boundaries (`top_min_elevation_deg`, `back_plane_offset_mm`, `nape_top_offset_mm`, `ear_guard_mm`). Lengths are blended over `blend_mm` across zone borders so the actuator steps smoothly.

> Note: `LeftSide.jpg` (nose pointing image-right) shows the subject's **right** side, so it is coloured `RIGHT`.

**Outputs** (same 64³ grid, origin, step and index order as the SDF — `flat = ix*N*N + iy*N + iz`):
- `output/skull_zone_64x64x64.bin` — `uint8` zone id per voxel (nearest scalp surface)
- `output/skull_length_64x64x64.bin` — `uint8` blended target length in mm, `255` = do not cut
- `output/skull_zones_meta.json` — zone table, areas, landmarks
- `output/skull_zones.ply` — colour-coded mesh (MeshLab / Blender)
- `output/zones_{front,left,right,back}.jpg` — zone overlays for checking boundaries

**Firmware lookup per cutter position:** `sdf <= 0` → retract; `length == 255` → trimmer off; otherwise set the spacer/actuator to `length` mm.

Re-run without the click windows using saved clicks: `python fit_skull.py --clicks output/clicks.json`.

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
