import os
import sys
import types
import json
import pickle
import urllib.request
import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import trimesh
from scipy.spatial import cKDTree

import zones

# ==========================================
# 0. MOCK CHUMPY SO LEGACY .PKL LOADS CLEANLY
# ==========================================
class _DummyChumpyArray:
    def __setstate__(self, state):
        if isinstance(state, dict) and "x" in state:
            self.x = state["x"]
        elif isinstance(state, tuple):
            for item in state:
                if isinstance(item, dict) and "x" in item:
                    self.x = item["x"]
                    break

    def __array__(self, dtype=None):
        return np.asarray(self.x, dtype=dtype)

_chumpy_mock = types.ModuleType("chumpy")
_chumpy_ch = types.ModuleType("chumpy.ch")
_chumpy_ch.Ch = _DummyChumpyArray
_chumpy_mock.ch = _chumpy_ch
_chumpy_mock.Ch = _DummyChumpyArray
sys.modules["chumpy"] = _chumpy_mock
sys.modules["chumpy.ch"] = _chumpy_ch

# ==========================================
# 1. CONFIGURATION & FILE PATHS
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "model/generic_model.pkl"
MASKS_PATH = "model/FLAME_masks.pkl"
MP_EMBEDDING_PATH = "model/mediapipe_landmark_embedding.npz"
FACE_LANDMARKER_TASK = "model/face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

IMAGE_PATHS = {
    "front": "captures/Front.jpg",
    "left":  "captures/LeftSide.jpg",
    "right": "captures/RightSide.jpg",
    "back":  "captures/Back.jpg",
}

CARD_REAL_WIDTH_MM = 85.60
SAFETY_BUFFER_MM = 3.0
NUM_SHAPE_PCS = 20
BETA_CLAMP = 1.8
SDF_GRID_SIZE = 64
SDF_DOMAIN_MM = 300.0  # scalp is ~260 mm tall; leave clearance above the crown for the cutter


# ==========================================
# 2. PURE-PYTORCH FLAME MODEL WITH NECK JOINT
# ==========================================
class PurePyTorchFLAMESkull:
    def __init__(self, model_path, masks_path, num_pcs, device):
        with open(model_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        with open(masks_path, "rb") as f:
            masks = pickle.load(f, encoding="latin1")

        self.v_template = torch.tensor(np.asarray(data["v_template"], dtype=np.float32), device=device)
        shapedirs_np = np.asarray(data["shapedirs"], dtype=np.float32)[:, :, :num_pcs]
        self.shapedirs = torch.tensor(shapedirs_np, device=device)
        self.faces = np.asarray(data["f"], dtype=np.int64)
        self.faces_tensor = torch.tensor(self.faces, device=device)
        self.scalp_indices = np.asarray(masks["scalp"], dtype=np.int64)

        self.left_ear_idx = np.asarray(masks["left_ear"], dtype=np.int64)
        self.right_ear_idx = np.asarray(masks["right_ear"], dtype=np.int64)

        v_np = np.asarray(data["v_template"], dtype=np.float32)

        # Front-most ear notch (tragus) vertices for side-view clicks
        self.left_tragus_idx = int(self.left_ear_idx[np.argmax(v_np[self.left_ear_idx, 2])])
        self.right_tragus_idx = int(self.right_ear_idx[np.argmax(v_np[self.right_ear_idx, 2])])

        # Outermost lateral ear rim vertices for rear-view clicks
        self.left_outer_ear_idx = int(self.left_ear_idx[np.argmax(np.abs(v_np[self.left_ear_idx, 0]))])
        self.right_outer_ear_idx = int(self.right_ear_idx[np.argmax(np.abs(v_np[self.right_ear_idx, 0]))])

        # Posterior nape-of-neck hairline vertex
        nape_candidates = np.where(
            (v_np[:, 1] < -0.05) & (v_np[:, 1] > -0.09) &
            (np.abs(v_np[:, 0]) < 0.015) & (v_np[:, 2] < -0.04)
        )[0]
        self.nape_idx = int(nape_candidates[np.argmin(v_np[nape_candidates, 2])])

        # Precompute nearest-neighbor X-mirror indices to enforce natural skull symmetry
        mirrored_template = v_np.copy()
        mirrored_template[:, 0] *= -1.0
        sym_tree = cKDTree(v_np)
        _, mirror_idx = sym_tree.query(mirrored_template, k=1)
        self.mirror_idx = torch.tensor(mirror_idx, dtype=torch.long, device=device)

        # FLAME skinning weights & joint regressor for natural neck twist/tilt
        weights_np = np.asarray(data["weights"], dtype=np.float32)
        self.head_skin_weight = torch.tensor(weights_np[:, 1:2], device=device)
        j_reg = data["J_regressor"]
        if hasattr(j_reg, "toarray"):
            j_reg = j_reg.toarray()
        self.j_regressor = torch.tensor(np.asarray(j_reg, dtype=np.float32), device=device)

        # Smooth falloff weight [0..1] around scalp vertices for safety shell
        tree = cKDTree(v_np[self.scalp_indices])
        dists, _ = tree.query(v_np, k=1)
        self.smooth_scalp_weights = np.clip(1.0 - (dists / 0.030), 0.0, 1.0).astype(np.float32)

    def forward(self, shape_params, head_rel_neck_rot=None):
        verts = self.v_template + torch.einsum("vcp,p->vc", self.shapedirs, shape_params)

        # Symmetrize X/Y/Z across left & right hemispheres so off-center selfies never skew the head
        verts_mirrored = verts[self.mirror_idx].clone()
        verts_mirrored[:, 0] *= -1.0
        verts = 0.5 * (verts + verts_mirrored)

        if head_rel_neck_rot is not None:
            joints = self.j_regressor @ verts
            pivot = joints[1]
            R_head = rodrigues_matrix(head_rel_neck_rot)
            verts_rotated = (verts - pivot) @ R_head.T + pivot
            verts = self.head_skin_weight * verts_rotated + (1.0 - self.head_skin_weight) * verts
        return verts


# ==========================================
# 3. LOAD MEDIAPIPE BARYCENTRIC EMBEDDING
# ==========================================
def load_mediapipe_embedding(npz_path, device):
    data = np.load(npz_path, allow_pickle=True, encoding="latin1")
    mp_indices = data["landmark_indices"].astype(np.int64)
    face_idx = torch.tensor(data["lmk_face_idx"].astype(np.int64), device=device)
    bary_coords = torch.tensor(data["lmk_b_coords"].astype(np.float32), device=device)
    return mp_indices, face_idx, bary_coords


def compute_embedded_landmarks_3d(vertices, faces_tensor, face_idx, bary_coords):
    selected_faces = faces_tensor[face_idx]
    tri_verts = vertices[selected_faces, :]
    return torch.einsum("lij,li->lj", tri_verts, bary_coords)


# ==========================================
# 4. 2D LANDMARK EXTRACTION
# ==========================================
def ensure_face_landmarker_model():
    if not os.path.exists(FACE_LANDMARKER_TASK):
        os.makedirs(os.path.dirname(FACE_LANDMARKER_TASK), exist_ok=True)
        print(f"Downloading MediaPipe FaceLandmarker model to {FACE_LANDMARKER_TASK}...")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_TASK)


def extract_2d_landmarks(image_path, mp_indices, view_name, detector):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if not result.face_landmarks:
        return None, None, (h, w), img

    all_lms = result.face_landmarks[0]
    pts_2d = np.zeros((len(mp_indices), 2), dtype=np.float32)
    weights = np.ones(len(mp_indices), dtype=np.float32)

    nose_x = all_lms[1].x
    brow_y = all_lms[10].y

    FACE_OVAL_INDICES = {
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    }

    for i, mp_idx in enumerate(mp_indices):
        lm = all_lms[mp_idx]
        pts_2d[i] = [lm.x * w, lm.y * h]

        if view_name == "front":
            if lm.y < (brow_y - 0.01):
                weights[i] = 0.0
        elif view_name in ("left", "right"):
            if mp_idx in FACE_OVAL_INDICES and mp_idx != 152:
                weights[i] = 0.0
            elif view_name == "left" and lm.x < (nose_x - 0.14):
                weights[i] = 0.0
            elif view_name == "right" and lm.x > (nose_x + 0.14):
                weights[i] = 0.0
            else:
                weights[i] = 0.65

    return pts_2d, weights, (h, w), img


# ==========================================
# 5. INTERACTIVE POINT CLICKING HELPERS
# ==========================================
CLICKS_IN = {}   # clicks loaded via --clicks (skips the OpenCV windows)
CLICKS_OUT = {}  # every click made this run, saved to output/clicks.json


def click_points(img_path, prompt_title, num_points=2, key=None):
    key = key or img_path
    if key in CLICKS_IN:
        pts = [np.array(c, dtype=np.float32) for c in CLICKS_IN[key]]
        print(f"\n[CLICKS] Reusing saved points for '{key}': {[p.tolist() for p in pts]}")
        CLICKS_OUT[key] = [p.tolist() for p in pts]
        return pts

    img = cv2.imread(img_path)
    display_img = img.copy()
    clicks = []

    max_h = 850
    scale_display = 1.0
    if display_img.shape[0] > max_h:
        scale_display = max_h / display_img.shape[0]
        display_img = cv2.resize(display_img, (0, 0), fx=scale_display, fy=scale_display)

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < num_points:
            clicks.append((x / scale_display, y / scale_display))
            cv2.circle(display_img, (x, y), 6, (0, 255, 0), -1)
            if len(clicks) == 2:
                pt1 = (int(clicks[0][0] * scale_display), int(clicks[0][1] * scale_display))
                pt2 = (int(clicks[1][0] * scale_display), int(clicks[1][1] * scale_display))
                cv2.line(display_img, pt1, pt2, (0, 255, 0), 2)
            cv2.imshow(prompt_title, display_img)

    cv2.imshow(prompt_title, display_img)
    cv2.setMouseCallback(prompt_title, mouse_cb)
    print(f"\n[ACTION REQUIRED] {prompt_title}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicks) != num_points:
        raise ValueError(f"Please click exactly {num_points} point(s) before pressing a key.")
    CLICKS_OUT[key] = [[float(c[0]), float(c[1])] for c in clicks]
    os.makedirs("output", exist_ok=True)
    with open("output/clicks.json", "w") as f:
        json.dump(CLICKS_OUT, f, indent=2)
    return [np.array(c, dtype=np.float32) for c in clicks]


# ==========================================
# 6. PERSPECTIVE PINHOLE CAMERA PROJECTION
# ==========================================
def rodrigues_matrix(rot_vec):
    # Smooth norm (no NaN gradient at 0). K must be built with torch.stack, not
    # torch.tensor: torch.tensor copies values and silently detaches them from
    # autograd, which froze the rotation axis and any rotation starting at zero.
    theta = torch.sqrt(torch.sum(rot_vec ** 2) + 1e-12)
    k = rot_vec / theta
    zero = torch.zeros((), dtype=rot_vec.dtype, device=rot_vec.device)
    K = torch.stack([
        torch.stack([zero, -k[2], k[1]]),
        torch.stack([k[2], zero, -k[0]]),
        torch.stack([-k[1], k[0], zero]),
    ])
    return torch.eye(3, dtype=rot_vec.dtype, device=rot_vec.device) + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def project_perspective(points_3d, focal_px, rot_vec, trans_3d, principal_pt):
    R = rodrigues_matrix(rot_vec)
    return project_perspective_matrix(points_3d, focal_px, R, trans_3d, principal_pt)


def project_perspective_matrix(points_3d, focal_px, R_mat, trans_3d, principal_pt):
    cam_pts = points_3d @ R_mat.T + trans_3d
    depth = torch.clamp(-cam_pts[:, 2], min=0.25)
    u = focal_px * (cam_pts[:, 0] / depth) + principal_pt[0]
    v = -focal_px * (cam_pts[:, 1] / depth) + principal_pt[1]
    return torch.stack([u, v], dim=-1)


def compute_initial_perspective_camera(lm3d_init, pts_2d, weights, yaw_rad, img_size):
    h, w = img_size
    focal_px = torch.tensor([float(h) * 0.88], dtype=torch.float32, device=lm3d_init.device)
    principal_pt = torch.tensor([w / 2.0, h / 2.0], dtype=torch.float32, device=lm3d_init.device)

    with torch.no_grad():
        rot_init = torch.tensor([0.0, yaw_rad, 0.0], dtype=torch.float32, device=lm3d_init.device)
        R = rodrigues_matrix(rot_init)
        rot_3d = lm3d_init @ R.T

        mask = weights > 0.1
        p3 = rot_3d[mask]
        p2 = pts_2d[mask]

        std_3d = torch.norm(p3[:, :2] - p3[:, :2].mean(dim=0), dim=1).mean() + 1e-6
        std_2d = torch.norm(p2 - p2.mean(dim=0), dim=1).mean()

        z0 = (focal_px * std_3d) / std_2d
        p2_mean = p2.mean(dim=0)
        p3_mean = p3.mean(dim=0)

        tx = ((p2_mean[0] - principal_pt[0]) * z0 / focal_px) - p3_mean[0]
        ty = (-(p2_mean[1] - principal_pt[1]) * z0 / focal_px) - p3_mean[1]
        tz = -z0 - p3_mean[2]

        trans_init = torch.tensor([tx.item(), ty.item(), tz.item()], dtype=torch.float32, device=lm3d_init.device)

    return (
        focal_px,
        rot_init.clone().detach().requires_grad_(True),
        trans_init.clone().detach().requires_grad_(True),
        principal_pt,
    )


# ==========================================
# 7. FULL 3D MESH OVERLAY RENDERER
# ==========================================
def save_mesh_overlay_visualization(img, vertices_3d, faces_np, focal_px, R_mat, trans_3d, principal_pt, out_path):
    with torch.no_grad():
        proj_verts = project_perspective_matrix(vertices_3d, focal_px, R_mat, trans_3d, principal_pt).cpu().numpy()
        cam_pts = (vertices_3d @ R_mat.T + trans_3d).cpu().numpy()

    overlay = img.copy()
    pts_int = np.round(proj_verts).astype(np.int32)

    # Hidden-surface removal (back-face culling + painter's ID buffer) so the
    # far side of the head (face, eyeballs, mouth) doesn't show through.
    _, visible = zones.visible_face_ids(cam_pts, pts_int, faces_np, img.shape)

    wire_layer = img.copy()
    for tri in faces_np[visible]:
        p0, p1, p2 = pts_int[tri[0]], pts_int[tri[1]], pts_int[tri[2]]
        cv2.line(wire_layer, tuple(p0), tuple(p1), (255, 200, 0), 1, cv2.LINE_AA)
        cv2.line(wire_layer, tuple(p1), tuple(p2), (255, 200, 0), 1, cv2.LINE_AA)
        cv2.line(wire_layer, tuple(p2), tuple(p0), (255, 200, 0), 1, cv2.LINE_AA)

    blended = cv2.addWeighted(wire_layer, 0.45, overlay, 0.55, 0)
    cv2.imwrite(out_path, blended)


# ==========================================
# 8. LOW-MEMORY SDF EXPORTER FOR ESP32-S3
# ==========================================
def export_esp32_sdf(mesh, out_bin_path, out_meta_path, focus_indices=None):
    # Centre the grid on the scalp (the cutting region), not the whole mesh:
    # FLAME's neck/shoulder flare makes the full mesh ~315 mm tall, so a
    # bbox-centred 260 mm cube would clip the crown.
    focus = mesh.vertices if focus_indices is None else mesh.vertices[focus_indices]
    center = 0.5 * (focus.min(axis=0) + focus.max(axis=0))
    half = SDF_DOMAIN_MM / 2.0
    if np.any(np.abs(focus - center) > half):
        print(f"[WARNING] Scalp extends beyond the {SDF_DOMAIN_MM:.0f} mm SDF domain; increase SDF_DOMAIN_MM.")
    lin = np.linspace(-half, half, SDF_GRID_SIZE, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin + center[0], lin + center[1], lin + center[2], indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

    surf_pts, face_ids = trimesh.sample.sample_surface(mesh, 60000)
    surf_normals = mesh.face_normals[face_ids]

    tree = cKDTree(surf_pts)
    dists, idx = tree.query(grid_pts, k=1)

    vecs = grid_pts - surf_pts[idx]
    signs = np.where(np.sum(vecs * surf_normals[idx], axis=1) >= 0.0, 1.0, -1.0)
    sdf_mm = dists * signs

    sdf_int16 = np.clip(np.round(sdf_mm * 10.0), -32767, 32767).astype(np.int16)
    sdf_int16.tofile(out_bin_path)

    meta = {
        "grid_size": SDF_GRID_SIZE,
        "domain_mm": SDF_DOMAIN_MM,
        "voxel_step_mm": round(SDF_DOMAIN_MM / (SDF_GRID_SIZE - 1), 4),
        "origin_min_mm": np.round(center - half, 3).tolist(),
        "centroid_mm": np.round(center, 3).tolist(),
        "unit_scale_mm": 0.1,
    }
    with open(out_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Exported 512 KB ESP32-S3 SDF lookup table: {out_bin_path}")
    return meta


# ==========================================
# 9. MAIN MULTI-VIEW FITTING PIPELINE
# ==========================================
def fit_multiview_flame():
    print(f"Running on device: {DEVICE}")
    ensure_face_landmarker_model()

    base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_TASK)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
    )

    flame = PurePyTorchFLAMESkull(MODEL_PATH, MASKS_PATH, NUM_SHAPE_PCS, DEVICE)
    mp_indices, lmk_face_idx, lmk_b_coords = load_mediapipe_embedding(MP_EMBEDDING_PATH, DEVICE)

    # Near-profile shots often fall below the 0.4 threshold; retry those at 0.2
    fallback_options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
    )

    view_data = {}
    with mp_vision.FaceLandmarker.create_from_options(options) as detector, \
         mp_vision.FaceLandmarker.create_from_options(fallback_options) as fallback_detector:
        for view_name in ["front", "left", "right"]:
            path = IMAGE_PATHS[view_name]
            if not os.path.exists(path):
                print(f"[WARNING] Missing {path}, skipping {view_name} view.")
                continue
            pts, weights, (h, w), raw_img = extract_2d_landmarks(path, mp_indices, view_name, detector)
            if pts is None:
                print(f"[INFO] No face in {path} at confidence 0.4, retrying at 0.2...")
                pts, weights, (h, w), raw_img = extract_2d_landmarks(path, mp_indices, view_name, fallback_detector)
            if pts is not None:
                view_data[view_name] = {
                    "pts": torch.tensor(pts, dtype=torch.float32, device=DEVICE),
                    "weights": torch.tensor(weights, dtype=torch.float32, device=DEVICE),
                    "size": (h, w),
                    "img": raw_img,
                    "ear_pt": None,
                    "nape_pt": None,
                    "tragus_idx": None,
                    "neck_rot": torch.zeros(3, dtype=torch.float32, device=DEVICE, requires_grad=True),
                }
            else:
                print(f"[WARNING] Face landmarks not detected in {path}.")

    # Click (1) Front Ear Notch and (2) Back of Neck at Bottom Hairline
    for side in ["left", "right"]:
        if side in view_data:
            [pt_ear, pt_nape] = click_points(
                IMAGE_PATHS[side],
                f"{side.capitalize()}: Click (1) Front Ear Notch, then (2) Back of Neck Hairline, then ANY KEY",
                num_points=2,
                key=side,
            )
            view_data[side]["ear_pt"] = torch.tensor(pt_ear, dtype=torch.float32, device=DEVICE)
            view_data[side]["nape_pt"] = torch.tensor(pt_nape, dtype=torch.float32, device=DEVICE)

    active_shape = torch.zeros(NUM_SHAPE_PCS, dtype=torch.float32, device=DEVICE, requires_grad=True)

    init_yaws = {"front": 0.0, "left": 1.45, "right": -1.45}
    with torch.no_grad():
        lm3d_zero = compute_embedded_landmarks_3d(
            flame.forward(active_shape), flame.faces_tensor, lmk_face_idx, lmk_b_coords
        )

    cam_params = {}
    for v, data in view_data.items():
        f_px, r_vec, t_3d, p_pt = compute_initial_perspective_camera(
            lm3d_zero, data["pts"], data["weights"], init_yaws[v], data["size"]
        )
        cam_params[v] = {"focal": f_px, "rot": r_vec, "trans": t_3d, "pp": p_pt}

        if data["ear_pt"] is not None:
            with torch.no_grad():
                R = rodrigues_matrix(r_vec)
                v_rot = flame.v_template @ R.T
                z_left = v_rot[flame.left_tragus_idx, 2]
                z_right = v_rot[flame.right_tragus_idx, 2]
                data["tragus_idx"] = flame.left_tragus_idx if z_left > z_right else flame.right_tragus_idx

    def compute_multiview_loss():
        loss = 0.0
        for v, data in view_data.items():
            verts_v = flame.forward(active_shape, head_rel_neck_rot=data["neck_rot"])
            lm3d = compute_embedded_landmarks_3d(verts_v, flame.faces_tensor, lmk_face_idx, lmk_b_coords)

            proj = project_perspective(
                lm3d, cam_params[v]["focal"], cam_params[v]["rot"], cam_params[v]["trans"], cam_params[v]["pp"]
            )
            diff_sq = torch.sum((proj - data["pts"]) ** 2, dim=-1)
            loss += torch.sum(diff_sq * data["weights"])

            if data["ear_pt"] is not None and data["tragus_idx"] is not None:
                t_idx = data["tragus_idx"]
                ear_proj = project_perspective(
                    verts_v[t_idx:t_idx + 1], cam_params[v]["focal"], cam_params[v]["rot"], cam_params[v]["trans"], cam_params[v]["pp"]
                )[0]
                loss += 10.0 * torch.sum((ear_proj - data["ear_pt"]) ** 2)

            if data["nape_pt"] is not None:
                nape_proj = project_perspective(
                    verts_v[flame.nape_idx:flame.nape_idx + 1],
                    cam_params[v]["focal"], cam_params[v]["rot"], cam_params[v]["trans"], cam_params[v]["pp"]
                )[0]
                # Gentle weight so nape follows neck angle without pushing the back of the skull out
                loss += 2.5 * torch.sum((nape_proj - data["nape_pt"]) ** 2)

            loss += 4.0 * torch.sum(data["neck_rot"] ** 2)
        return loss

    # Stage 1: Perspective Camera + Neck Twist/Pitch Alignment
    opt_cam = torch.optim.Adam([
        {"params": [cam_params[v]["trans"] for v in cam_params], "lr": 0.005},
        {"params": [cam_params[v]["rot"] for v in cam_params],   "lr": 0.012},
        {"params": [view_data[v]["neck_rot"] for v in view_data], "lr": 0.008},
    ])

    print("\nStage 1: Aligning 3D perspective camera poses + neck articulation...")
    for step in range(450):
        opt_cam.zero_grad()
        loss = compute_multiview_loss()
        loss.backward()
        opt_cam.step()

    # Stage 2: Constrained Symmetric 20-PC Skull Shape + Camera Refinement
    opt_joint = torch.optim.Adam([
        {"params": [active_shape], "lr": 0.007},
        {"params": [cam_params[v]["trans"] for v in cam_params], "lr": 0.0015},
        {"params": [cam_params[v]["rot"] for v in cam_params],   "lr": 0.003},
        {"params": [view_data[v]["neck_rot"] for v in view_data], "lr": 0.003},
    ])

    print("Stage 2: Fitting symmetric 3D FLAME skull shape...")
    for step in range(500):
        opt_joint.zero_grad()
        reproj_loss = compute_multiview_loss()
        # Slightly stronger regularization prevents posterior skull ballooning on casual photos
        reg_loss = 30.0 * torch.sum(active_shape ** 2)
        total_loss = reproj_loss + reg_loss
        total_loss.backward()
        opt_joint.step()

        with torch.no_grad():
            active_shape.clamp_(-BETA_CLAMP, BETA_CLAMP)

        if step % 100 == 0:
            print(f"Step {step:03d} | Reproj Loss: {reproj_loss.item():.1f} | Reg Loss: {reg_loss.item():.1f}")

    # ==========================================
    # 10. SAVE ALL 4 FULL 3D MESH OVERLAYS (FRONT, LEFT, RIGHT, BACK)
    # ==========================================
    os.makedirs("output", exist_ok=True)
    with torch.no_grad():
        neutral_vertices = flame.forward(active_shape, head_rel_neck_rot=None)

        for v, data in view_data.items():
            verts_v = flame.forward(active_shape, head_rel_neck_rot=data["neck_rot"])
            lm3d = compute_embedded_landmarks_3d(verts_v, flame.faces_tensor, lmk_face_idx, lmk_b_coords)
            R_v = rodrigues_matrix(cam_params[v]["rot"])

            save_mesh_overlay_visualization(
                data["img"],
                verts_v,
                flame.faces,
                cam_params[v]["focal"],
                R_v,
                cam_params[v]["trans"],
                cam_params[v]["pp"],
                f"output/overlay_{v}.jpg"
            )

            vis = data["img"].copy()
            proj = project_perspective_matrix(
                lm3d, cam_params[v]["focal"], R_v, cam_params[v]["trans"], cam_params[v]["pp"]
            ).cpu().numpy()
            target = data["pts"].cpu().numpy()
            w_np = data["weights"].cpu().numpy()

            for i in range(len(proj)):
                if w_np[i] > 0.1:
                    cv2.circle(vis, (int(target[i, 0]), int(target[i, 1])), 4, (0, 255, 0), -1)
                    cv2.circle(vis, (int(proj[i, 0]), int(proj[i, 1])), 4, (0, 0, 255), -1)

            if data["ear_pt"] is not None and data["tragus_idx"] is not None:
                t_idx = data["tragus_idx"]
                et = data["ear_pt"].cpu().numpy()
                ep = project_perspective_matrix(
                    verts_v[t_idx:t_idx + 1],
                    cam_params[v]["focal"], R_v, cam_params[v]["trans"], cam_params[v]["pp"]
                )[0].cpu().numpy()
                cv2.circle(vis, (int(et[0]), int(et[1])), 7, (0, 255, 255), -1)
                cv2.circle(vis, (int(ep[0]), int(ep[1])), 7, (255, 0, 255), -1)

            if data["nape_pt"] is not None:
                nt = data["nape_pt"].cpu().numpy()
                np_proj = project_perspective_matrix(
                    verts_v[flame.nape_idx:flame.nape_idx + 1],
                    cam_params[v]["focal"], R_v, cam_params[v]["trans"], cam_params[v]["pp"]
                )[0].cpu().numpy()
                cv2.circle(vis, (int(nt[0]), int(nt[1])), 7, (255, 255, 0), -1)
                cv2.circle(vis, (int(np_proj[0]), int(np_proj[1])), 7, (0, 165, 255), -1)

            cv2.imwrite(f"output/debug_fit_{v}.jpg", vis)

    # Generate 4th overlay for Back.jpg matching the outer middle ear edges
    back_cam = None
    if os.path.exists(IMAGE_PATHS["back"]):
        back_img = cv2.imread(IMAGE_PATHS["back"])
        bh, bw, _ = back_img.shape
        [back_l_ear, back_r_ear] = click_points(
            IMAGE_PATHS["back"],
            "Back: Click OUTER MIDDLE edge of LEFT ear, then RIGHT ear, then ANY KEY",
            num_points=2,
            key="back",
        )
        with torch.no_grad():
            f_back = torch.tensor([float(bh) * 0.88], dtype=torch.float32, device=DEVICE)
            pp_back = torch.tensor([bw / 2.0, bh / 2.0], dtype=torch.float32, device=DEVICE)

            dx_ear = float(back_r_ear[0] - back_l_ear[0])
            dy_ear = float(back_r_ear[1] - back_l_ear[1])
            # Correct camera roll direction in image coordinates (+Y points downward)
            roll_back = float(np.arctan2(dy_ear, dx_ear))

            r_yaw = rodrigues_matrix(torch.tensor([0.0, float(np.pi), 0.0], dtype=torch.float32, device=DEVICE))
            r_roll = rodrigues_matrix(torch.tensor([0.0, 0.0, roll_back], dtype=torch.float32, device=DEVICE))
            R_b = r_roll @ r_yaw
            v_b = neutral_vertices @ R_b.T

            e0 = v_b[flame.left_outer_ear_idx]
            e1 = v_b[flame.right_outer_ear_idx]
            left_3d, right_3d = (e0, e1) if e0[0] < e1[0] else (e1, e0)

            ear_span_3d = torch.abs(right_3d[0] - left_3d[0]).item()
            ear_span_2d = float(np.linalg.norm(back_r_ear - back_l_ear))
            z_back = (f_back.item() * ear_span_3d) / max(ear_span_2d, 1.0)

            mid_2d = 0.5 * (back_l_ear + back_r_ear)
            mid_3d = 0.5 * (left_3d + right_3d)
            tx_b = ((mid_2d[0] - pp_back[0].item()) * z_back / f_back.item()) - mid_3d[0].item()
            ty_b = (-(mid_2d[1] - pp_back[1].item()) * z_back / f_back.item()) - mid_3d[1].item()
            tz_b = -z_back - mid_3d[2].item()
            t_back = torch.tensor([tx_b, ty_b, tz_b], dtype=torch.float32, device=DEVICE)

            save_mesh_overlay_visualization(
                back_img, neutral_vertices, flame.faces, f_back, R_b, t_back, pp_back, "output/overlay_back.jpg"
            )
            back_cam = {"img": back_img, "focal": f_back, "R": R_b, "trans": t_back, "pp": pp_back}

    print("\nSaved all 4 diagnostic overlays:")
    print("  - output/overlay_front.jpg")
    print("  - output/overlay_left.jpg")
    print("  - output/overlay_right.jpg")
    print("  - output/overlay_back.jpg")

    # ==========================================
    # 11. METRIC SCALING, SMOOTH SCALP OFFSET & EXPORTS
    # ==========================================
    c_left, c_right = click_points(
        IMAGE_PATHS["front"],
        "Front: Click LEFT edge then RIGHT edge of ID Card, then press ANY KEY",
        num_points=2,
        key="card",
    )
    pixel_dist = np.linalg.norm(c_left - c_right)
    mm_per_pixel = CARD_REAL_WIDTH_MM / pixel_dist

    with torch.no_grad():
        # The card's mm-per-pixel only holds at the card's own depth, i.e. the
        # forehead surface right under it. Find the mesh vertices that project
        # onto the card and take the nearest one to the camera (the visible
        # surface the card rests on). The old code took the scalp's *farthest*
        # point (.min() of camera z = back of the head), which inflated the scale.
        fp = cam_params["front"]
        verts_front = flame.forward(active_shape, head_rel_neck_rot=view_data["front"]["neck_rot"])
        R_front = rodrigues_matrix(fp["rot"])
        cam_verts = verts_front @ R_front.T + fp["trans"]
        depths = -cam_verts[:, 2]
        proj_front = project_perspective_matrix(verts_front, fp["focal"], R_front, fp["trans"], fp["pp"])
        card_mid = torch.tensor(0.5 * (c_left + c_right), dtype=torch.float32, device=DEVICE)
        under_card = torch.norm(proj_front - card_mid, dim=1) < 0.25 * float(pixel_dist)
        if under_card.any():
            card_depth = depths[under_card].min().item()
        else:
            print("[WARNING] Card midpoint does not land on the mesh; using nearest scalp depth.")
            card_depth = depths[flame.scalp_indices].min().item()
        focal_front = fp["focal"].item()
        mm_per_flame_unit = (focal_front / card_depth) * mm_per_pixel
        print(f"Card: {pixel_dist:.1f} px wide -> {mm_per_pixel:.4f} mm/px at depth {card_depth:.4f} FLAME units")

    print(f"Scale factor: {mm_per_flame_unit:.1f} mm per FLAME unit")

    verts_raw = neutral_vertices.detach().cpu().numpy()
    verts_mm = verts_raw * mm_per_flame_unit
    verts_mm -= verts_mm.mean(axis=0)

    mesh = trimesh.Trimesh(vertices=verts_mm, faces=flame.faces, process=False)
    normals = mesh.vertex_normals
    mesh.vertices += normals * (flame.smooth_scalp_weights[:, None] * SAFETY_BUFFER_MM)

    stl_path = "output/safe_skull_boundary.stl"
    mesh.export(stl_path)
    print(f"\nSUCCESS: Exported smooth metric 3D skull mesh to: {stl_path}")
    print(f"Fitted Head Dimensions (Width x Height x Depth in mm): {mesh.extents.round(1)}")

    sdf_meta = export_esp32_sdf(mesh, "output/skull_sdf_64x64x64.bin", "output/skull_sdf_meta.json",
                                focus_indices=flame.scalp_indices)

    # ==========================================
    # 12. FLOWBEE CUTTING ZONES (see zones.py / zones_config.json)
    # ==========================================
    print("\nSegmenting Flowbee cutting zones...")
    zone_cfg = zones.load_zone_config()
    seg = zones.segment_zones(np.asarray(mesh.vertices), flame, zone_cfg)
    zones.export_zone_outputs(seg, zone_cfg, sdf_meta)

    # Zone overlays on each photo, using the same per-view poses as the fit
    with torch.no_grad():
        views = []
        for v, data in view_data.items():
            verts_v = flame.forward(active_shape, head_rel_neck_rot=data["neck_rot"])
            cp = cam_params[v]
            views.append((v, data["img"], verts_v, cp["focal"], rodrigues_matrix(cp["rot"]), cp["trans"], cp["pp"]))
        if back_cam is not None:
            views.append(("back", back_cam["img"], neutral_vertices, back_cam["focal"], back_cam["R"],
                          back_cam["trans"], back_cam["pp"]))
        for v, img, verts_v, focal, R_v, t_v, pp_v in views:
            sub = torch.tensor(seg["S"] @ verts_v.cpu().numpy().astype(np.float64),
                               dtype=torch.float32, device=DEVICE)
            pts2d = project_perspective_matrix(sub, focal, R_v, t_v, pp_v).cpu().numpy()
            cam_pts = (sub @ R_v.T + t_v).cpu().numpy()
            zones.save_zone_overlay(img, cam_pts, pts2d, seg, zone_cfg, f"output/zones_{v}.jpg")
    print("Saved zone overlays: " + ", ".join(f"output/zones_{v[0]}.jpg" for v in views))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-view FLAME skull fit + ESP32 SDF export")
    parser.add_argument("--clicks", help="Reuse calibration clicks from a JSON file (e.g. output/clicks.json) "
                                         "instead of opening the click windows")
    args = parser.parse_args()
    if args.clicks:
        with open(args.clicks) as f:
            CLICKS_IN.update(json.load(f))
    fit_multiview_flame()