#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import cv2
from glob import glob
import os

def main():
    result_dir = 'outputs/HM_SLAM_autonome_systeme/060525_SLAM_1_2_video'
    
    # Load intrinsics
    c = np.load(f'{result_dir}/intrinsics.npy')
    intrinsic = o3d.core.Tensor([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1]], dtype=o3d.core.Dtype.Float64)
    print(f"Intrinsics: {intrinsic}")
    
    # Load trajectory
    poses = np.loadtxt(f'{result_dir}/traj_full.txt')
    print(f"Trajectory shape: {poses.shape}")
    print(f"First pose: {poses[0]}")
    
    # Get depth and color files
    depth_files = sorted(glob(f'{result_dir}/renders/depth_after_opt/*'))
    color_files = sorted(glob(f'{result_dir}/renders/image_after_opt/*'))
    
    print(f"Found {len(depth_files)} depth files and {len(color_files)} color files")
    
    # Test loading first depth image
    if depth_files:
        depth_path = depth_files[0]
        print(f"Testing depth file: {depth_path}")
        
        # Test with OpenCV
        depth_cv = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        print(f"OpenCV depth shape: {depth_cv.shape}, dtype: {depth_cv.dtype}")
        print(f"OpenCV depth min/max: {np.min(depth_cv)}, {np.max(depth_cv)}")
        print(f"OpenCV non-zero pixels: {np.sum(depth_cv > 0)}")
        
        # Test with Open3D
        try:
            depth_o3d = o3d.t.io.read_image(depth_path)
            print(f"Open3D depth shape: {depth_o3d.shape}, dtype: {depth_o3d.dtype}")
            depth_array = depth_o3d.numpy()
            print(f"Open3D depth array min/max: {np.min(depth_array)}, {np.max(depth_array)}")
            print(f"Open3D non-zero pixels: {np.sum(depth_array > 0)}")
        except Exception as e:
            print(f"Error loading with Open3D: {e}")
    
    # Test with different depth scales
    test_scales = [1.0, 1000.0, 5000.0, 6553.5]
    for scale in test_scales:
        depth_scaled = depth_cv / scale
        print(f"Scale {scale}: depth range {np.min(depth_scaled[depth_scaled > 0]):.6f} - {np.max(depth_scaled):.6f}")

if __name__ == "__main__":
    main()
