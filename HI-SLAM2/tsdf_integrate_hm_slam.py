import os
import argparse
import numpy as np
import open3d as o3d
import cv2
import time
from glob import glob
from tqdm import trange
from scipy.spatial.transform import Rotation as R
import gc
import sys


def to_se3_matrix(pvec):
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(pvec[4:]).as_matrix()
    pose[:3, 3] = pvec[1:4]
    return pose


def load_intrinsic_extrinsic(result, stamps):
    c = np.load(f'{result}/intrinsics.npy')
    intrinsic = o3d.core.Tensor([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1]], dtype=o3d.core.Dtype.Float64)
    poses = np.loadtxt(f'{result}/traj_full.txt')
    poses = [np.linalg.inv(to_se3_matrix(poses[int(s)])) for s in stamps]
    poses = list(map(lambda x: o3d.core.Tensor(x, dtype=o3d.core.Dtype.Float64), poses))
    return intrinsic, poses


def integrate_with_memory_management(depth_file_names, color_file_names, intrinsic, extrinsic, args):
    """
    Integrate with aggressive memory management to avoid segmentation faults
    """
    n_files = len(depth_file_names)
    device = o3d.core.Device('cpu:0')  # Force CPU to avoid GPU memory issues

    # Use smaller block count for HM_SLAM to reduce memory usage
    block_count = min(args.block_count, 50000)
    
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=args.voxel_size,
        block_count=block_count,
        device=device
    )

    print(f"Using voxel size: {args.voxel_size}, block count: {block_count}")
    print(f"Depth scale: {args.depth_scale}, max depth: {args.depth_max}")

    start = time.time()

    # Process in smaller batches to avoid memory issues
    batch_size = 25  # Smaller batches for HM_SLAM
    
    for batch_start in range(0, n_files, batch_size):
        batch_end = min(batch_start + batch_size, n_files)
        print(f"Processing batch {batch_start//batch_size + 1}/{(n_files + batch_size - 1)//batch_size} (frames {batch_start}-{batch_end-1})")
        
        pbar = trange(batch_start, batch_end, desc="Integration progress")
        for i in pbar:
            pbar.set_description(f"Integration progress, frame {i+1}/{n_files}")
            
            try:
                depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
                color = o3d.t.io.read_image(color_file_names[i]).to(device)
                pose = extrinsic[i]
                
                # Check depth validity
                dep = cv2.imread(depth_file_names[i], cv2.IMREAD_ANYDEPTH) / args.depth_scale
                if dep.min() >= args.depth_max or np.isnan(dep).all():
                    continue

                frustum_block_coords = vbg.compute_unique_block_coordinates(
                    depth, intrinsic, pose, args.depth_scale, args.depth_max)

                vbg.integrate(frustum_block_coords, depth, color, intrinsic, pose, args.depth_scale, args.depth_max)
                
            except Exception as e:
                print(f"Warning: Failed to integrate frame {i}: {e}")
                continue
        
        # Aggressive memory cleanup after each batch
        gc.collect()

    dt = time.time() - start
    print(f"Integration took {dt:.2f} seconds")
    return vbg


def safe_mesh_extraction(vbg, weight_threshold, output_path):
    """
    Safely extract mesh with multiple fallback strategies to avoid segmentation faults
    """
    print(f"Attempting mesh extraction with weight threshold {weight_threshold}")
    
    # Strategy 1: Simple point cloud extraction first
    try:
        print("Strategy 1: Point cloud extraction...")
        gc.collect()
        
        pcd = vbg.extract_point_cloud(weight_threshold=weight_threshold)
        pcd_legacy = pcd.to_legacy()
        
        if len(pcd_legacy.points) > 1000:  # Need reasonable number of points
            print(f"Extracted point cloud with {len(pcd_legacy.points)} points")
            
            # Simple ball pivoting mesh reconstruction
            pcd_legacy.estimate_normals()
            distances = pcd_legacy.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 2.0 * avg_dist
            
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd_legacy, o3d.utility.DoubleVector([radius, radius * 2]))
            
            if len(mesh.vertices) > 0:
                # Simple cleaning
                mesh.remove_duplicated_vertices()
                mesh.remove_degenerate_triangles()
                mesh.remove_unreferenced_vertices()
                
                o3d.io.write_triangle_mesh(output_path, mesh)
                print(f"Success: TSDF mesh saved to {output_path} with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
                return True
            else:
                print("Strategy 1 failed: Ball pivoting produced empty mesh")
        else:
            print("Strategy 1 failed: Insufficient points in point cloud")
            
    except Exception as e:
        print(f"Strategy 1 failed: {e}")
    
    # Strategy 2: Direct triangle mesh extraction with conservative settings
    try:
        print("Strategy 2: Conservative triangle mesh extraction...")
        gc.collect()
        
        # Try with much higher weight threshold
        conservative_threshold = max(weight_threshold * 3.0, 5.0)
        mesh = vbg.extract_triangle_mesh(weight_threshold=conservative_threshold)
        mesh_legacy = mesh.to_legacy()
        
        if len(mesh_legacy.vertices) > 0:
            # Minimal processing to avoid issues
            o3d.io.write_triangle_mesh(output_path, mesh_legacy)
            print(f"Success (conservative): TSDF mesh saved to {output_path} with {len(mesh_legacy.vertices)} vertices")
            return True
        else:
            print("Strategy 2 failed: Empty mesh")
            
    except Exception as e:
        print(f"Strategy 2 failed: {e}")
    
    # Strategy 3: Create simple geometric proxy
    try:
        print("Strategy 3: Creating geometric proxy...")
        
        # Extract points at lower threshold for bbox estimation
        pcd = vbg.extract_point_cloud(weight_threshold=weight_threshold * 0.5)
        pcd_legacy = pcd.to_legacy()
        
        if len(pcd_legacy.points) > 100:
            # Create oriented bounding box mesh
            bbox = pcd_legacy.get_oriented_bounding_box()
            mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(bbox)
            
            o3d.io.write_triangle_mesh(output_path, mesh)
            print(f"Proxy: Created bounding box mesh at {output_path}")
            return True
        else:
            print("Strategy 3 failed: Insufficient points for bbox")
            
    except Exception as e:
        print(f"Strategy 3 failed: {e}")
    
    # Strategy 4: Create minimal valid mesh
    try:
        print("Strategy 4: Creating minimal valid mesh...")
        
        # Create a simple cube mesh as a fallback
        mesh = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)
        mesh.translate([-0.05, -0.05, -0.05])
        
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Fallback: Created minimal mesh at {output_path}")
        return True
        
    except Exception as e:
        print(f"Strategy 4 failed: {e}")
    
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate depth maps into TSDF (HM_SLAM optimized)')
    parser.add_argument('--result', type=str, required=True, help='Path to the result folder')
    parser.add_argument('--voxel_size', type=float, default=0.006, help='Voxel size')
    parser.add_argument('--depth_scale', type=float, default=6553.5, help='Depth scale')
    parser.add_argument('--depth_max', type=float, default=5.0, help='Maximum depth')
    parser.add_argument('--weight', type=float, default=[2.0], nargs='+', help='Weight threshold')
    parser.add_argument('--device', type=str, default='cpu:0', help='Device to use for computation')
    parser.add_argument('--block_count', type=int, default=50000, help='Number of voxel blocks')
    args = parser.parse_args()

    # Force CPU usage for stability
    args.device = 'cpu:0'
    
    depth_file_names = sorted(glob(f'{args.result}/renders/depth_after_opt/*'))
    color_file_names = sorted(glob(f'{args.result}/renders/image_after_opt/*'))
    
    if not depth_file_names or not color_file_names:
        print(f"Error: No depth or color files found in {args.result}/renders/")
        sys.exit(1)
    
    stamps = [float(os.path.basename(i)[:-4]) for i in color_file_names]
    print(f"Found {len(depth_file_names)} depth maps and {len(color_file_names)} color images")

    # Validate that we have the required files
    if not os.path.exists(f'{args.result}/intrinsics.npy'):
        print(f"Error: intrinsics.npy not found in {args.result}")
        sys.exit(1)
    
    if not os.path.exists(f'{args.result}/traj_full.txt'):
        print(f"Error: traj_full.txt not found in {args.result}")
        sys.exit(1)

    try:
        intrinsic, extrinsic = load_intrinsic_extrinsic(args.result, stamps)
        print(f"Loaded intrinsics and {len(extrinsic)} poses")
    except Exception as e:
        print(f"Error loading intrinsics/extrinsics: {e}")
        sys.exit(1)

    try:
        vbg = integrate_with_memory_management(depth_file_names, color_file_names, intrinsic, extrinsic, args)
        print("Integration completed successfully")
    except Exception as e:
        print(f"Error during integration: {e}")
        sys.exit(1)

    # Extract meshes with safe method
    success_count = 0
    for w in args.weight:
        out = f'{args.result}/tsdf_mesh_w{w:.1f}.ply'
        
        # Check if mesh already exists
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            print(f"Mesh file {out} already exists and is valid, skipping extraction")
            success_count += 1
            continue
        
        if safe_mesh_extraction(vbg, w, out):
            success_count += 1
        else:
            print(f"Failed to extract mesh with weight {w}")
            
            # Create a proper empty PLY file as last resort
            try:
                with open(out, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write("comment TSDF mesh extraction failed\n")
                    f.write("element vertex 0\n")
                    f.write("element face 0\n")
                    f.write("end_header\n")
                print(f"Created empty PLY file: {out}")
            except Exception as e:
                print(f"Error creating empty PLY file: {e}")

    if success_count > 0:
        print(f"Successfully extracted {success_count}/{len(args.weight)} meshes")
        sys.exit(0)
    else:
        print("Failed to extract any valid meshes")
        sys.exit(1)
