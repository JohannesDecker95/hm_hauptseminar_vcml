import os
import argparse
import numpy as np
import open3d as o3d
import cv2
import time
from glob import glob
from tqdm import trange
from scipy.spatial.transform import Rotation as R


def to_se3_matrix(pvec):
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(pvec[4:]).as_matrix()
    pose[:3, 3] = pvec[1:4]
    return pose


def load_intrinsic_extrinsic(result, stamps):
    c = np.load(f'{result}/intrinsics.npy')
    
    # Read first image to get actual dimensions
    depth_files = sorted(glob(f'{result}/renders/depth_after_opt/*.png'))
    if depth_files:
        first_depth = cv2.imread(depth_files[0], cv2.IMREAD_ANYDEPTH)
        height, width = first_depth.shape
        print(f"Detected image dimensions: {width}x{height}")
    else:
        # Fallback dimensions
        width, height = 640, 480
        print(f"Warning: No depth files found, using default dimensions: {width}x{height}")
    
    # Create intrinsic matrix with correct dimensions
    intrinsic = o3d.core.Tensor([
        [c[0], 0, c[2]], 
        [0, c[1], c[3]], 
        [0, 0, 1]
    ], dtype=o3d.core.Dtype.Float64)
    
    poses = np.loadtxt(f'{result}/traj_full.txt')
    poses = [np.linalg.inv(to_se3_matrix(poses[int(s)])) for s in stamps]
    poses = list(map(lambda x: o3d.core.Tensor(x, dtype=o3d.core.Dtype.Float64), poses))
    return intrinsic, poses, (width, height)


def integrate(depth_file_names, color_file_names, intrinsic, extrinsic, img_dims, args):
    n_files = len(depth_file_names)
    device = o3d.core.Device('cpu:0')
    
    print(f"Using device: {device}")
    print(f"Image dimensions: {img_dims}")

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=args.voxel_size,
        block_count=30000,
        device=device
    )

    start = time.time()
    processed_frames = 0

    pbar = trange(n_files, desc="TSDF Integration")
    for i in pbar:
        try:
            pbar.set_description(f"TSDF Integration, frame {i+1}/{n_files}")
            
            depth_path = depth_file_names[i]
            color_path = color_file_names[i]
            
            if not os.path.exists(depth_path) or not os.path.exists(color_path):
                continue
                
            # Read depth and check validity
            dep_cv = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) 
            if dep_cv is None:
                continue
                
            # Scale depth values
            dep_cv = dep_cv.astype(np.float32) / args.depth_scale
            if dep_cv.min() >= args.depth_max:
                continue

            # Load images for Open3D
            depth = o3d.t.io.read_image(depth_path).to(device)
            color = o3d.t.io.read_image(color_path).to(device)
            
            # Check dimensions match
            if depth.shape[0] != img_dims[1] or depth.shape[1] != img_dims[0]:
                print(f"Warning: Depth image shape {depth.shape} doesn't match expected {img_dims[::-1]}")
                continue
                
            pose = extrinsic[i]

            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth, intrinsic, pose, args.depth_scale, args.depth_max)

            vbg.integrate(frustum_block_coords, depth, color, intrinsic, pose, args.depth_scale, args.depth_max)
            processed_frames += 1
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue

    dt = time.time() - start
    print(f"TSDF integration took {dt:.2f} seconds, processed {processed_frames}/{n_files} frames")
    return vbg, processed_frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Corrected TSDF integration with proper dimension handling')
    parser.add_argument('--result', type=str, required=True, help='Path to the result folder')
    parser.add_argument('--voxel_size', type=float, default=0.008, help='Voxel size')
    parser.add_argument('--depth_scale', type=float, default=6553.5, help='Depth scale')
    parser.add_argument('--depth_max', type=float, default=4.0, help='Maximum depth')
    parser.add_argument('--weight', type=float, default=[2.0], nargs='+', help='Weight threshold')
    args = parser.parse_args()

    depth_file_names = sorted(glob(f'{args.result}/renders/depth_after_opt/*.png'))
    color_file_names = sorted(glob(f'{args.result}/renders/image_after_opt/*.jpg'))
    
    if not depth_file_names:
        print(f"Error: No depth files found in {args.result}/renders/depth_after_opt/")
        exit(1)
    if not color_file_names:
        print(f"Error: No color files found in {args.result}/renders/image_after_opt/")
        exit(1)
    
    # Ensure same number of files
    min_files = min(len(depth_file_names), len(color_file_names))
    depth_file_names = depth_file_names[:min_files]
    color_file_names = color_file_names[:min_files]
    
    stamps = [float(os.path.basename(i)[:-4]) for i in color_file_names]
    print(f"Processing {len(depth_file_names)} depth and {len(color_file_names)} color images")

    try:
        intrinsic, extrinsic, img_dims = load_intrinsic_extrinsic(args.result, stamps)
        vbg, processed_frames = integrate(depth_file_names, color_file_names, intrinsic, extrinsic, img_dims, args)

        if processed_frames == 0:
            print("Error: No frames were successfully processed")
            exit(1)

        for w in args.weight:
            out = f'{args.result}/tsdf_mesh_w{w:.1f}.ply'
            
            if os.path.exists(out):
                print(f"Mesh file {out} already exists, skipping")
                continue
                
            print(f"Extracting mesh with weight threshold {w}")
            try:
                mesh = vbg.extract_triangle_mesh(weight_threshold=w)
                
                if not mesh.has_vertices():
                    print("Warning: Extracted mesh has no vertices")
                    continue
                    
                mesh_legacy = mesh.to_legacy()
                
                if len(mesh_legacy.vertices) == 0:
                    print("Warning: Mesh has no vertices after conversion to legacy format")
                    continue
                    
                success = o3d.io.write_triangle_mesh(out, mesh_legacy)
                if success:
                    print(f"Successfully saved TSDF mesh to {out} with {len(mesh_legacy.vertices)} vertices and {len(mesh_legacy.triangles)} triangles")
                else:
                    print("Failed to write mesh file")
                    
            except Exception as e:
                print(f"Error during mesh extraction: {e}")
                # Create a minimal placeholder
                with open(out, 'w') as f:
                    f.write("ply\nformat ascii 1.0\ncomment TSDF extraction failed\nelement vertex 0\nelement face 0\nend_header\n")
        
        print("TSDF integration completed")
        
    except Exception as e:
        print(f"Fatal error during TSDF integration: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
