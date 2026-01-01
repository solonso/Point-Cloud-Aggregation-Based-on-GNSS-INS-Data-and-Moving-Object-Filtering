#!/usr/bin/env python3
"""
Name: Nwafor Solomon Chibuzo
Nuptun ID: UITK8W
Task 1: Aggregate nuScenes LIDAR_TOP frames into the global frame and visualize.

The nuScenes devkit transforms
(`LidarPointCloud` + `transform_matrix`) and show an Open3D window. No
annotations are used here.

Run from repo root:
    python scripts/task1.py --scene scene-0103 --max-frames 40
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "nuscenes"


def main():
    parser = argparse.ArgumentParser(description="Aggregate and visualize LIDAR_TOP frames.")
    parser.add_argument("--scene", default="scene-0103", help="Scene name from scene.json (default: scene-0103)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick tests")
    parser.add_argument("--no-view", action="store_true", help="Skip Open3D visualization")
    parser.add_argument("--color", choices=["gray", "height", "intensity"], default="height", help="Color scheme for visualization")
    parser.add_argument("--point-size", type=float, default=1.5, help="Open3D point size")
    parser.add_argument("--save-pcd", action="store_true", help="Also save aggregated cloud to output/<scene>.pcd with colors")
    args = parser.parse_args()

    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "nuScenes devkit (nuscenes-devkit) and pyquaternion are required for this script. "
            f"Error: {exc}"
        ) from exc

    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)
    scene = next((s for s in nusc.scene if s["name"] == args.scene), None)
    if scene is None:
        raise SystemExit(f"Scene {args.scene} not found.")

    frames = []
    intensities = []
    num_points = 0
    sample_token = scene["first_sample_token"]
    frame_idx = 0

    while sample_token:
        sample = nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd = nusc.get("sample_data", lidar_token)
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ego = nusc.get("ego_pose", sd["ego_pose_token"])

        pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))
        sensor_to_ego = transform_matrix(calib["translation"], Quaternion(calib["rotation"]), inverse=False)
        ego_to_global = transform_matrix(ego["translation"], Quaternion(ego["rotation"]), inverse=False)
        pc.transform(ego_to_global @ sensor_to_ego)

        xyz = pc.points[:3, :].T
        intens = pc.points[3, :]
        frames.append(xyz)
        intensities.append(intens)
        num_points += xyz.shape[0]
        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"[{args.scene}] processed {frame_idx} frames ({num_points} pts)")
        if args.max_frames and frame_idx >= args.max_frames:
            break

        sample_token = sample["next"]

    print(f"[{args.scene}] done. Frames: {len(frames)}, points: {num_points}")
    if not frames:
        raise SystemExit("No frames aggregated.")

    cloud = np.vstack(frames)
    intensity = np.concatenate(intensities)

    if args.no_view:
        return

    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed; install open3d to visualize, or rerun with --no-view.")
        return

    # Choose colors
    if args.color == "height":
        z = cloud[:, 2]
        z_min, z_max = np.percentile(z, [1, 99])  # robust range
        span = max(z_max - z_min, 1e-3)
        norm = np.clip((z - z_min) / span, 0, 1)

        def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
            """Vectorized HSV->RGB for 0-1 inputs."""
            i = np.floor(h * 6).astype(int)
            f = h * 6 - i
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            i_mod = i % 6
            r = np.choose(i_mod, [v, q, p, p, t, v])
            g = np.choose(i_mod, [t, v, v, q, p, p])
            b = np.choose(i_mod, [p, p, t, v, v, q])
            return np.stack([r, g, b], axis=-1)

        hues = 0.66 * (1 - norm)  # blue high, red low
        colors = hsv_to_rgb(hues, np.ones_like(hues) * 0.8, np.ones_like(hues) * 0.9)
    elif args.color == "intensity":
        lo, hi = np.percentile(intensity, [1, 99])
        span = max(hi - lo, 1e-3)
        norm = np.clip((intensity - lo) / span, 0, 1)
        colors = np.stack([norm, norm, norm], axis=-1)
    else:  # gray
        colors = np.full((cloud.shape[0], 3), 0.85)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"taask1_{args.scene}.pcd"
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"Saved PCD to {out_path}")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Aggregated {args.scene}")
    vis.add_geometry(pcd)
    ropt = vis.get_render_option()
    ropt.point_size = args.point_size
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
