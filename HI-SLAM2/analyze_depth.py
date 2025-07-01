import cv2
import numpy as np
import os
from glob import glob

# Analyze depth images for HM_SLAM_autonome_systeme dataset
depth_dir = "outputs/HM_SLAM_autonome_systeme/060525_SLAM_1_2_video/renders/depth_after_opt"
depth_files = sorted(glob(f"{depth_dir}/*.png"))

print(f"Found {len(depth_files)} depth images")

if len(depth_files) > 0:
    # Sample a few depth images to understand the value range
    sample_files = depth_files[::50]  # Sample every 50th file
    if len(sample_files) > 10:
        sample_files = sample_files[:10]  # Limit to 10 samples
    
    all_stats = []
    
    for i, depth_file in enumerate(sample_files):
        try:
            # Read depth image (16-bit)
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
            if depth is not None:
                # Get statistics
                non_zero_mask = depth > 0
                if np.any(non_zero_mask):
                    min_val = np.min(depth[non_zero_mask])
                    max_val = np.max(depth[non_zero_mask])
                    mean_val = np.mean(depth[non_zero_mask])
                    std_val = np.std(depth[non_zero_mask])
                    
                    stats = {
                        'file': os.path.basename(depth_file),
                        'min': min_val,
                        'max': max_val,
                        'mean': mean_val,
                        'std': std_val,
                        'non_zero_pixels': np.sum(non_zero_mask),
                        'total_pixels': depth.size
                    }
                    all_stats.append(stats)
                    
                    print(f"File {i+1}: {os.path.basename(depth_file)}")
                    print(f"  Min: {min_val:.3f}, Max: {max_val:.3f}, Mean: {mean_val:.3f}, Std: {std_val:.3f}")
                    print(f"  Non-zero pixels: {np.sum(non_zero_mask)}/{depth.size} ({100*np.sum(non_zero_mask)/depth.size:.1f}%)")
                else:
                    print(f"File {i+1}: {os.path.basename(depth_file)} - All pixels are zero!")
            else:
                print(f"File {i+1}: {os.path.basename(depth_file)} - Failed to read!")
                
        except Exception as e:
            print(f"Error reading {depth_file}: {e}")
    
    if all_stats:
        # Overall statistics
        all_mins = [s['min'] for s in all_stats]
        all_maxs = [s['max'] for s in all_stats]
        all_means = [s['mean'] for s in all_stats]
        
        print(f"\n=== OVERALL STATISTICS ===")
        print(f"Global min depth: {min(all_mins):.3f}")
        print(f"Global max depth: {max(all_maxs):.3f}")
        print(f"Average mean depth: {np.mean(all_means):.3f}")
        print(f"Depth range: {min(all_mins):.3f} - {max(all_maxs):.3f}")
        
        # Suggest appropriate depth_scale
        if max(all_maxs) < 10:
            print(f"\nSUGGESTED: Depth values appear to be in meters")
            print(f"Try depth_scale=1.0 or depth_scale=1000.0")
        elif max(all_maxs) < 100:
            print(f"\nSUGGESTED: Depth values appear to be in decimeters or scaled meters")
            print(f"Try depth_scale=10.0 or depth_scale=100.0")
        elif max(all_maxs) > 1000:
            print(f"\nSUGGESTED: Depth values appear to be in millimeters")
            print(f"Try depth_scale=1000.0 or depth_scale=5000.0")
        else:
            print(f"\nSUGGESTED: Depth values in intermediate range")
            print(f"Try depth_scale=100.0 or depth_scale=500.0")
            
        # Also suggest voxel_size based on depth range
        depth_range = max(all_maxs) - min(all_mins)
        suggested_voxel = depth_range / 100  # Aim for ~100 voxels across depth range
        print(f"SUGGESTED voxel_size: {suggested_voxel:.4f} (depth_range / 100)")
        
else:
    print("No depth files found!")
