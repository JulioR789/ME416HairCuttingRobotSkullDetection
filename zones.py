"""
Cutting-zone segmentation for the Flowbee trimmer.

Splits the fitted, metric skull into haircut regions, each with its own target
hair length (Flowbee spacer / actuator setting):

    0 NO_CUT     face, forehead, neck below the nape hairline (never cut)
    1 TOP        crown + front hairline (where the head curves over the top)
    2 LEFT       subject's left side, in front of the back of the ear
    3 RIGHT      subject's right side
    4 BACK       behind the ears, above the occipital/tragus line
    5 NAPE       back, below the tragus line down to the hairline (taper)
    6 EAR_GUARD  band around each ear (default: no cut, Flowbee suction)

Boundaries are anchored on anatomy of the fitted mesh (ears, tragus, nape
hairline you clicked, surface curvature), so they adapt to each head. All
thresholds and lengths live in zones_config.json, which is created with
defaults on first run.

Outputs (same 64^3 grid as the SDF, same origin/step/index order):
    output/skull_zone_64x64x64.bin     uint8 zone id of the nearest scalp surface
    output/skull_length_64x64x64.bin   uint8 target hair length in mm, blended
                                       across zone boundaries; 255 = do not cut
    output/skull_zones_meta.json       zone table + grid description
    output/skull_zones.ply             colour-coded mesh for MeshLab/Blender
    output/zones_<view>.jpg            zone overlays on each photo
"""
import json
import os

import cv2
import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.spatial import cKDTree

ZONE_CONFIG_PATH = "zones_config.json"
NO_CUT_LENGTH = 255

DEFAULT_ZONE_CONFIG = {
    "_help": (
        "Haircut zones for the Flowbee. length_mm = target hair length (spacer setting). "
        "cut=false means the trimmer must stay off in that zone. Angles in degrees, distances in mm."
    ),
    "top_min_elevation_deg": 25.0,
    "normal_smoothing_mm": 20.0,
    "front_max_azimuth_deg": 50.0,
    "back_plane_offset_mm": 0.0,
    "nape_top_offset_mm": 0.0,
    "ear_guard_mm": 10.0,
    "blend_mm": 15.0,
    "zones": {
        "NO_CUT":    {"id": 0, "cut": False, "length_mm": None, "color": "#3c3c3c"},
        "TOP":       {"id": 1, "cut": True,  "length_mm": 25.0, "color": "#ffc800"},
        "LEFT":      {"id": 2, "cut": True,  "length_mm": 12.0, "color": "#00c850"},
        "RIGHT":     {"id": 3, "cut": True,  "length_mm": 12.0, "color": "#0096ff"},
        "BACK":      {"id": 4, "cut": True,  "length_mm": 12.0, "color": "#c800c8"},
        "NAPE":      {"id": 5, "cut": True,  "length_mm": 6.0,  "color": "#ff3232"},
        "EAR_GUARD": {"id": 6, "cut": False, "length_mm": None, "color": "#ffffff"},
    },
}


def load_zone_config(path=ZONE_CONFIG_PATH):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(DEFAULT_ZONE_CONFIG, f, indent=2)
        print(f"Created default zone config: {path} (edit lengths/thresholds there)")
    with open(path) as f:
        user = json.load(f)
    cfg = json.loads(json.dumps(DEFAULT_ZONE_CONFIG))
    cfg.update({k: v for k, v in user.items() if k != "zones"})
    for name, z in user.get("zones", {}).items():
        cfg["zones"].setdefault(name, {}).update(z)
    return cfg


def _hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# ------------------------------------------------------------------
# Midpoint subdivision as a reusable linear map, so the same labels
# apply to the neutral mesh and to every posed/projected copy of it.
# ------------------------------------------------------------------
def build_subdivision(faces, n_verts, levels=2):
    S = sp.identity(n_verts, format="csr", dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = n_verts
    for _ in range(levels):
        nf = len(f)
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        uniq, inv = np.unique(edges, axis=0, return_inverse=True)
        inv = inv.reshape(3, nf).T
        m01, m12, m20 = (n + inv[:, 0], n + inv[:, 1], n + inv[:, 2])
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        f = np.concatenate([
            np.stack([a, m01, m20], 1),
            np.stack([b, m12, m01], 1),
            np.stack([c, m20, m12], 1),
            np.stack([m01, m12, m20], 1),
        ])
        ne = len(uniq)
        rows = np.concatenate([np.arange(n), n + np.repeat(np.arange(ne), 2)])
        cols = np.concatenate([np.arange(n), uniq.ravel()])
        vals = np.concatenate([np.ones(n), np.full(2 * ne, 0.5)])
        step = sp.csr_matrix((vals, (rows, cols)), shape=(n + ne, n))
        S = step @ S
        n += ne
    return S.tocsr(), f


# ------------------------------------------------------------------
# Zone labelling
# ------------------------------------------------------------------
def segment_zones(verts_mm, flame, cfg, levels=2):
    """Label every face of the subdivided mesh with a zone id.

    verts_mm: (V,3) fitted neutral mesh in mm, FLAME axes (x = subject's left,
    y = up, z = forward). Returns dict with subdivision map, faces, labels,
    lengths and the landmarks used.
    """
    zid = {name: z["id"] for name, z in cfg["zones"].items()}
    V = len(verts_mm)
    S, sub_faces = build_subdivision(flame.faces, V, levels)
    sub_v = S @ verts_mm

    scalp_ind = np.zeros(V); scalp_ind[flame.scalp_indices] = 1.0
    ear_ind = np.zeros(V); ear_ind[flame.left_ear_idx] = 1.0; ear_ind[flame.right_ear_idx] = 1.0
    sub_scalp = S @ scalp_ind
    sub_ear = S @ ear_ind

    # --- anatomical landmarks (fitted mesh) ---
    tragus = 0.5 * (verts_mm[flame.left_tragus_idx] + verts_mm[flame.right_tragus_idx])
    ears = [verts_mm[flame.left_ear_idx], verts_mm[flame.right_ear_idx]]
    ear_back_z = float(np.mean([np.percentile(e[:, 2], 2) for e in ears])) + cfg["back_plane_offset_mm"]
    lobes = [e[np.argmin(e[:, 1])] for e in ears]
    nape = verts_mm[flame.nape_idx]
    crown = verts_mm[flame.scalp_indices][np.argmax(verts_mm[flame.scalp_indices, 1])]

    # Lower hairline: plane through the nape hairline point and both ear lobes
    plane_n = np.cross(lobes[0] - nape, lobes[1] - nape)
    plane_n /= np.linalg.norm(plane_n)
    if np.dot(crown - nape, plane_n) < 0:
        plane_n = -plane_n

    mesh = trimesh.Trimesh(sub_v, sub_faces, process=False)
    cen = mesh.triangles_center
    f_scalp = sub_scalp[sub_faces].mean(axis=1)

    # Smoothed surface direction: raw FLAME triangle normals are bumpy and give
    # sawtooth zone borders, so average vertex normals over a ~2 cm patch.
    base = trimesh.Trimesh(verts_mm, flame.faces, process=False)
    vn = base.vertex_normals
    r = cfg["normal_smoothing_mm"]
    if r > 0:
        nbrs = cKDTree(verts_mm).query_ball_point(verts_mm, r=r)
        vn = np.stack([vn[nb].mean(axis=0) for nb in nbrs])
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    nrm = (S @ vn)[sub_faces].mean(axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12

    elev = np.degrees(np.arcsin(np.clip(nrm[:, 1], -1, 1)))
    azim = np.degrees(np.arctan2(nrm[:, 0], nrm[:, 2]))  # 0 = facing forward

    ear_pts = sub_v[sub_ear > 0.5]
    d_ear, _ = cKDTree(ear_pts).query(cen)

    cut = (f_scalp >= 0.5) & ((cen - nape) @ plane_n > 0.0)
    top = (elev > cfg["top_min_elevation_deg"]) | (
        (np.abs(azim) < cfg["front_max_azimuth_deg"]) & (cen[:, 2] > tragus[2]))
    back = cen[:, 2] < ear_back_z
    low = cen[:, 1] < tragus[1] + cfg["nape_top_offset_mm"]

    zone = np.full(len(sub_faces), zid["NO_CUT"], dtype=np.uint8)
    zone[cut & ~top & ~back & (cen[:, 0] >= 0)] = zid["LEFT"]
    zone[cut & ~top & ~back & (cen[:, 0] < 0)] = zid["RIGHT"]
    zone[cut & ~top & back & ~low] = zid["BACK"]
    zone[cut & ~top & back & low] = zid["NAPE"]
    zone[cut & top] = zid["TOP"]
    zone[cut & (d_ear < cfg["ear_guard_mm"])] = zid["EAR_GUARD"]

    # --- target length per face, blended across cut-zone boundaries ---
    by_id = {z["id"]: z for z in cfg["zones"].values()}
    cuttable = np.array([bool(by_id[i]["cut"]) for i in zone])
    raw_len = np.array([by_id[i]["length_mm"] if by_id[i]["cut"] else 0.0 for i in zone], dtype=np.float64)
    length = np.full(len(zone), float(NO_CUT_LENGTH))
    idx = np.where(cuttable)[0]
    if len(idx):
        if cfg["blend_mm"] > 0:
            sigma = cfg["blend_mm"] / 2.0
            tree = cKDTree(cen[idx])
            areas = mesh.area_faces[idx]
            nbrs = tree.query_ball_point(cen[idx], r=cfg["blend_mm"])
            for k, nb in enumerate(nbrs):
                nb = np.asarray(nb)
                w = areas[nb] * np.exp(-np.sum((cen[idx[nb]] - cen[idx[k]]) ** 2, axis=1) / (2 * sigma ** 2))
                length[idx[k]] = np.sum(w * raw_len[idx[nb]]) / np.sum(w)
        else:
            length[idx] = raw_len[idx]

    return {
        "S": S, "faces": sub_faces, "verts": sub_v, "zone": zone, "length": length,
        "landmarks": {
            "tragus_mid": tragus.round(1).tolist(), "ear_back_z": round(ear_back_z, 1),
            "nape": nape.round(1).tolist(), "lobes": [l.round(1).tolist() for l in lobes],
        },
    }


# ------------------------------------------------------------------
# Hidden-surface helper shared by the wireframe and zone overlays
# ------------------------------------------------------------------
def visible_face_ids(cam_pts, pts_int, faces, img_shape):
    """Per-pixel front-most face id (-1 = background) and a per-face visibility mask.

    cam_pts: (V,3) camera-space points (camera at origin looking down -Z);
    pts_int: (V,2) integer pixel coords. Uses back-face culling + a painter's
    algorithm ID buffer so the far side of the head never shows through.
    """
    tri = cam_pts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    facing = np.einsum("ij,ij->i", normals, -tri.mean(axis=1)) > 0.0

    h, w = img_shape[:2]
    id_buf = np.full((h, w), -1, dtype=np.int32)
    depth = -tri[:, :, 2].mean(axis=1)
    for f_idx in np.argsort(-depth):  # farthest first
        cv2.fillConvexPoly(id_buf, pts_int[faces[f_idx]], int(f_idx))
    id_buf[(id_buf >= 0) & ~facing[np.maximum(id_buf, 0)]] = -1
    seen = np.zeros(len(faces), dtype=bool)
    ids = np.unique(id_buf)
    seen[ids[ids >= 0]] = True
    return id_buf, facing & seen


def save_zone_overlay(img, cam_pts, pts2d, seg, cfg, out_path, alpha=0.45):
    pts_int = np.round(pts2d).astype(np.int32)
    id_buf, _ = visible_face_ids(cam_pts, pts_int, seg["faces"], img.shape)

    palette = np.zeros((256, 3), dtype=np.uint8)
    for z in cfg["zones"].values():
        palette[z["id"]] = _hex_to_bgr(z["color"])
    no_cut_id = cfg["zones"]["NO_CUT"]["id"]

    face_zone = seg["zone"]
    mask = id_buf >= 0
    zone_px = np.full(id_buf.shape, no_cut_id, dtype=np.uint8)
    zone_px[mask] = face_zone[id_buf[mask]]
    paint = mask & (zone_px != no_cut_id)

    out = img.copy()
    color = palette[zone_px]
    out[paint] = (alpha * color[paint] + (1 - alpha) * img[paint]).astype(np.uint8)

    # Zone boundary lines (where the actuator setting changes)
    edges = np.zeros(id_buf.shape, dtype=np.uint8)
    edges[:, 1:] |= (zone_px[:, 1:] != zone_px[:, :-1]) & (paint[:, 1:] | paint[:, :-1])
    edges[1:, :] |= (zone_px[1:, :] != zone_px[:-1, :]) & (paint[1:, :] | paint[:-1, :])
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    out[edges > 0] = (255, 255, 255)

    # Legend
    scale = max(img.shape[0] / 1000.0, 0.6)
    y = int(30 * scale)
    for name, z in sorted(cfg["zones"].items(), key=lambda kv: kv[1]["id"]):
        label = f"{name}: {z['length_mm']:.0f} mm" if z["cut"] else f"{name}: no cut"
        cv2.rectangle(out, (int(10 * scale), y - int(16 * scale)), (int(30 * scale), y + int(2 * scale)),
                      _hex_to_bgr(z["color"]), -1)
        cv2.putText(out, label, (int(38 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale,
                    (0, 0, 0), int(4 * scale), cv2.LINE_AA)
        cv2.putText(out, label, (int(38 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale,
                    (255, 255, 255), max(int(1.5 * scale), 1), cv2.LINE_AA)
        y += int(26 * scale)
    cv2.imwrite(out_path, out)


# ------------------------------------------------------------------
# Exports
# ------------------------------------------------------------------
def export_zone_outputs(seg, cfg, sdf_meta, out_dir="output"):
    """Write zone/length voxel grids aligned with the SDF grid, plus PLY + JSON."""
    n = sdf_meta["grid_size"]
    step = sdf_meta["voxel_step_mm"]
    origin = np.asarray(sdf_meta["origin_min_mm"], dtype=np.float64)
    lin = np.linspace(0.0, sdf_meta["domain_mm"], n)
    gx, gy, gz = np.meshgrid(lin + origin[0], lin + origin[1], lin + origin[2], indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

    mesh = trimesh.Trimesh(seg["verts"], seg["faces"], process=False)
    surf_pts, fid = trimesh.sample.sample_surface(mesh, 200000, seed=0)
    _, near = cKDTree(surf_pts).query(grid_pts, k=1)
    zone_grid = seg["zone"][fid[near]].astype(np.uint8)
    len_grid = np.clip(np.round(seg["length"][fid[near]]), 0, NO_CUT_LENGTH).astype(np.uint8)

    zone_path = os.path.join(out_dir, f"skull_zone_{n}x{n}x{n}.bin")
    len_path = os.path.join(out_dir, f"skull_length_{n}x{n}x{n}.bin")
    zone_grid.tofile(zone_path)
    len_grid.tofile(len_path)

    colors = np.zeros((len(seg["faces"]), 4), dtype=np.uint8)
    colors[:, 3] = 255
    for z in cfg["zones"].values():
        b, g, r = _hex_to_bgr(z["color"])
        colors[seg["zone"] == z["id"]] = (r, g, b, 255)
    mesh.visual.face_colors = colors
    mesh.export(os.path.join(out_dir, "skull_zones.ply"))

    areas = mesh.area_faces
    meta = {
        "grid": {k: sdf_meta[k] for k in ("grid_size", "domain_mm", "voxel_step_mm", "origin_min_mm")},
        "index_order": "flat = ix*N*N + iy*N + iz (x fastest-varying last), same as the SDF",
        "zone_file": os.path.basename(zone_path),
        "length_file": os.path.basename(len_path),
        "length_units": "mm, uint8; 255 = do not cut",
        "zones": {
            name: {
                "id": z["id"], "cut": z["cut"], "length_mm": z["length_mm"], "color": z["color"],
                "surface_area_cm2": round(float(areas[seg["zone"] == z["id"]].sum()) / 100.0, 1),
            }
            for name, z in sorted(cfg["zones"].items(), key=lambda kv: kv[1]["id"])
        },
        "blend_mm": cfg["blend_mm"],
        "landmarks_mm": seg["landmarks"],
    }
    with open(os.path.join(out_dir, "skull_zones_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Exported zone grid {zone_path} and length grid {len_path} ({zone_grid.nbytes // 1024} KB each)")
    for name, z in meta["zones"].items():
        setting = f"{z['length_mm']:.0f} mm" if z["cut"] else "no cut"
        print(f"  {name:10s} id={z['id']}  {setting:>7s}  {z['surface_area_cm2']:7.1f} cm^2")
    return meta
