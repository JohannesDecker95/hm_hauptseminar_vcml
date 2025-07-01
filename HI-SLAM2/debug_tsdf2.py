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
    print(f"Loaded intrinsics: {c}")
    
    # Check the calibration file
    calib_data = np.loadtxt('calib/hm_slam.txt')
    print(f"Calibration file: {calib_data}")
    
    # Load trajectory
    poses = np.loadtxt(f'{result_dir}/traj_full.txt')
    print(f"Trajectory shape: {poses.shape}")
    
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
        
        # Test with Open3D - fix the attribute error
        try:
            depth_o3d = o3d.t.io.read_image(depth_path)
            print(f"Open3D depth shape: {depth_o3d.get_min_bound()}, {depth_o3d.get_max_bound()}")
            
            # Try to get numpy representation
            depth_o3d_cpu = depth_o3d.cpu()
            depth_array = depth_o3d_cpu.as_tensor().numpy()
            print(f"Open3D depth array shape: {depth_array.shape}, dtype: {depth_array.dtype}")
            print(f"Open3D depth array min/max: {np.min(depth_array)}, {np.max(depth_array)}")
            print(f"Open3D non-zero pixels: {np.sum(depth_array > 0)}")
        except Exception as e:
            print(f"Error loading with Open3D: {e}")
    
    # Test pose conversion
    from scipy.spatial.transform import Rotation as R
    
    def to_se3_matrix(pvec):
        pose = np.eye(4)
        pose[:3, :3] = R.from_quat(pvec[4:]).as_matrix()
        pose[:3, 3] = pvec[1:4]
        return pose
    
    test_pose = to_se3_matrix(poses[0])
    print(f"First pose matrix:")
    print(test_pose)
    
    # Check if pose is reasonable (not too far from origin)
    translation = test_pose[:3, 3]
    print(f"Translation magnitude: {np.linalg.norm(translation)}")

if __name__ == "__main__":
    main()
