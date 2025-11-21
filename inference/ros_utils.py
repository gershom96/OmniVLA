import json
import math
import numpy as np
import cv2 
from typing import Optional, Tuple

def load_calibration(json_path: str, fx: float, fy: float, cx: float, cy: float, mode: str = "jackal"):
    """
    Builds:
      K (3x3), dist=None, T_cam_from_base (4x4)
    from tf.json with H_cam_bl: pitch(deg), x,y,z.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    entry = data.get(mode, None)
    if entry is None or "H_cam_bl" not in entry:
        raise ValueError(f"Missing '{mode}' in {json_path}")

    h = entry["H_cam_bl"]
    roll = math.radians(float(h["roll"]))
    xt, yt, zt = float(h["x"]), float(h["y"]), float(h["z"])

    # Rotation about +y (camera pitched down is positive pitch if y up/right-handed)
    Ry = np.array([
        [ 0.0, math.sin(roll), math.cos(roll)],
        [-1.0, 0.0, 0.0],
        [0.0, -math.cos(roll),  math.sin(roll)]
    ], dtype=np.float64)

    T_base_from_cam = np.eye(4, dtype=np.float64)
    T_base_from_cam[:3, :3] = Ry
    T_base_from_cam[:3, 3]  = np.array([xt, yt, 0.25], dtype=np.float64)
    # T_base_from_cam[:3, 3]  = np.array([xt, yt, zt], dtype=np.float64)

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    dist = None  # explicitly no distortion
    return K, dist, T_base_from_cam

def make_offset_paths(traj_b: np.ndarray,
                      theta_samples: np.ndarray, 
                      offsets: np.ndarray):
    """Generate left/right offset paths for trajectory following."""

    x = traj_b[:, 0]
    y = traj_b[:, 1]
    # normal to heading (x-forward, y-left):
    n_x = -np.sin(theta_samples)
    n_y =  np.cos(theta_samples)

    xL = x + offsets * n_x
    yL = y + offsets * n_y
    xR = x - offsets * n_x
    yR = y - offsets * n_y

    z = np.zeros_like(x)
    left_o  = np.stack([xL, yL, z], axis=1)
    right_o = np.stack([xR, yR, z], axis=1)

    return left_o, right_o

def create_yaws_from_path(path_b: np.ndarray) -> np.ndarray:
    """Create yaw angles (radians) from a base_link path."""
    deltas = np.diff(path_b[:, :2], axis=0)  # (N-1, 2)
    yaws = np.arctan2(deltas[:, 1], deltas[:, 0])  # (N-1,)
    # Append last yaw to maintain same length
    if len(yaws) > 0:
        yaws = np.concatenate([yaws, yaws[-1:]], axis=0)
    else:
        yaws = np.array([0.0])
    return yaws

def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points; returns Nx3."""
    assert T.shape == (4, 4)
    N = P.shape[0]
    Ph = np.hstack([P, np.ones((N, 1))])
    Qh = (T @ Ph.T).T
    return Qh[:, :3]

def make_corridor_polygon(traj_b: np.ndarray,
                          theta_samples: np.ndarray,
                          width_m: float, 
                          bridge_pts: int = 15):
    """
    Given centerline (N,3) and heading samples (N,), create left/right offsets and a closed polygon.
    width_m: robot width; offsets at ±width_m/2.
    Returns:
      left_b, right_b: (N,3) in base_link
      poly_b: (2N,3) polygon points (left forward then right backward)
    """
    d = width_m * 0.5
    x = traj_b[:, 0]
    y = traj_b[:, 1]
    # normal to heading (x-forward, y-left):
    n_x = -np.sin(theta_samples)
    n_y =  np.cos(theta_samples)

    xL = x + d * n_x
    yL = y + d * n_y
    xR = x - d * n_x
    yR = y - d * n_y

    z = np.zeros_like(x)
    left_b  = np.stack([xL, yL, z], axis=1)
    right_b = np.stack([xR, yR, z], axis=1)

    # left_b = left_b[left_b[:, 0]>0]
    # right_b = right_b[right_b[:, 0]>0]

    if bridge_pts > 0:
        bx = np.linspace(xL[-1], xR[-1], bridge_pts)
        by = np.linspace(yL[-1], yR[-1], bridge_pts)
        bridge_end = np.stack([bx, by, np.zeros_like(bx)], axis=1)
    # Build polygon: left (0→N-1) + right (N-1→0)
    poly_b = np.vstack([left_b, bridge_end,right_b[::-1]])
    return left_b, right_b, poly_b

def project_clip(poly_b_xyz: np.ndarray, T_cam_from_base, K, dist, H: int, W: int,
                  smooth_first=True) -> np.ndarray:
    """base→cam→image, then clip to bottom and (optionally) densify first segment. Returns [x,y] float."""
    poly_b_xyz[:,2] = 0.0
    poly_c = transform_points(T_cam_from_base, poly_b_xyz)       # (N,3)
    pts_xy = project_points_cam(K, dist, poly_c)                 # (N,2) [x,y]
    
    if pts_xy.size == 0:
        return pts_xy

    # print(pts_xy)
    if pts_xy.shape[0] > 1:
        pts_xy = clip_to_bottom_xy(pts_xy, H)
    # print(pts_xy)

    # if smooth_first:
    #     pts_xy = densify_first_segment_xy(pts_xy, px_step=2.0)

    # clamp to bounds to be safe

    if pts_xy.size == 0:
        return pts_xy
    
    pts_xy[:,0] = np.clip(pts_xy[:,0], 0, W-1)
    pts_xy[:,1] = np.clip(pts_xy[:,1], 0, H-1)
 
    return pts_xy

def clean_2d(arr, W, H, max_jump_px=300):
    # keep finite + in-bounds
    arr = arr[np.isfinite(arr).all(axis=1)]
    arr = arr[(arr[:,0]>=0)&(arr[:,0]<W)&(arr[:,1]>=0)&(arr[:,1]<H)]
    if len(arr) < 2:
        return arr
    # cut at first large jump to avoid across-screen segments
    jumps = np.linalg.norm(np.diff(arr,axis=0),axis=1)
    bad = np.where(jumps < max_jump_px)
    return arr if len(bad)==0 else arr[bad]

def clip_to_bottom_xy(poly_xy: np.ndarray, img_h: int) -> np.ndarray:
    """Clip a [x,y] polyline to the bottom scanline y=img_h-1, inserting the exact intersection."""
    if poly_xy is None or len(poly_xy) == 0:
        return poly_xy
    pts = poly_xy.astype(float, copy=True)
    yb = float(img_h - 2)

    # Already starts on the bottom row?
    if abs(pts[0,1] - yb) < 1e-6:
        return pts

    # Find first adjacent segment that spans yb and insert intersection
    for i in range(len(pts)-1):
        y0, y1 = pts[i,1], pts[i+1,1]
        if y0 == y1:
            if abs(y0 - yb) < 1e-6:
                return pts[i:].copy()
            continue
        if (y0 - yb) * (y1 - yb) <= 0:  # spans scanline
            # print(y0, y1, pts[i,0], pts[i+1,0])
            t = (yb - y0) / (y1 - y0)
            # t = max(0.0, min(1.0, t))
            x = pts[i,0] + t * (pts[i+1,0] - pts[i,0])
            inter = np.array([[x, yb]], dtype=float)
            return np.vstack([inter, pts[i+1:]])
    else:
        return pts

def project_points_cam(K: np.ndarray, dist, P_cam: np.ndarray) -> np.ndarray:
    """Project Nx3 camera-frame points to pixels. No distortion if dist is None."""
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    pts2d, _ = cv2.projectPoints(P_cam.astype(np.float64), rvec, tvec, K, None)
    return pts2d.reshape(-1, 2)

def draw_polyline(img: np.ndarray, pts2d: np.ndarray, thickness: int, color):
    H, W = img.shape[:2]
    poly = []
    for (uu, vv) in pts2d:
        ui, vi = int(round(uu)), int(round(vv))
        if 0 <= ui < W and 0 <= vi < H:
            poly.append((ui, vi))
    for i in range(len(poly) - 1):
        cv2.line(img, poly[i], poly[i + 1], color, thickness, lineType=cv2.LINE_AA)

def draw_corridor(img: np.ndarray, poly_2d: np.ndarray, left_2d: np.ndarray, right_2d: np.ndarray,
                  fill_alpha: float = 0.35,
                  fill_color = (0,0,255),   # BGR
                  edge_color = (0,0,200),
                  edge_thickness: int = 2,):
    H, W = img.shape[:2]
    # Clip to image bounds
    def clip_pts(uv, polygon=False):
        pts = []
        for (u,v) in uv:
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                pts.append([ui, vi])
        if polygon and len(pts) >= 3:
            # Ensure polygon is closed
            if pts[0] != pts[-1]:
                pts.append(pts[0])

        return np.array(pts, dtype=np.int32)

    poly = clip_pts(poly_2d, polygon=True)
    L = clip_pts(left_2d)
    R = clip_pts(right_2d)

    if len(poly) >= 3:
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], fill_color)
        img[:] = cv2.addWeighted(overlay, fill_alpha, img, 1.0 - fill_alpha, 0)

    if len(L) >= 2:
        cv2.polylines(img, [L], isClosed=False, color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)
    if len(R) >= 2:
        cv2.polylines(img, [R], isClosed=False, color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)

def make_corridor_polygon_from_cam_lines(left_c: np.ndarray,
                          right_c: np.ndarray, 
                          bridge_pts: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if bridge_pts > 0:
        bx = np.linspace(left_c[-1][0], right_c[-1][0], bridge_pts)
        by = np.linspace(left_c[-1][1], right_c[-1][1], bridge_pts)
        bridge_end = np.stack([bx, by], axis=1)
    # Build polygon: left (0→N-1) + right (N-1→0)
    poly_c = np.vstack([left_c, bridge_end, right_c[::-1]])
    return poly_c
