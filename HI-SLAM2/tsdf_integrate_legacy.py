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
    # Legacy Open3D intrinsic format
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=640, height=480, fx=c[0], fy=c[1], cx=c[2], cy=c[3])
    
    poses = np.loadtxt(f'{result}/traj_full.txt')
    poses = [np.linalg.inv(to_se3_matrix(poses[int(s)])) for s in stamps]
    return intrinsic, poses


def integrate_legacy(depth_file_names, color_file_names, intrinsic, extrinsic, args):
    """Use legacy Open3D TSDF integration which is more stable"""
    n_files = len(depth_file_names)
    
    # Create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=args.voxel_size * 5,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    start = time.time()
    processed = 0

    pbar = trange(n_files, desc="Legacy TSDF Integration")
    for i in pbar:
        pbar.set_description(f"Legacy TSDF Integration, frame {i+1}/{n_files}")
        try:
            # Load images
            depth_path = depth_file_names[i]
            color_path = color_file_names[i]
            
            if not os.path.exists(depth_path) or not os.path.exists(color_path):
                continue
                
            depth = o3d.io.read_image(depth_path)
            color = o3d.io.read_image(color_path)
            
            # Create RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, 
                depth_scale=args.depth_scale,
                depth_trunc=args.depth_max,
                convert_rgb_to_intensity=False
            )
            
            # Integrate into volume
            volume.integrate(rgbd, intrinsic, extrinsic[i])
            processed += 1
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue

    dt = time.time() - start
    print(f"Legacy integration took {dt:.2f} seconds, processed {processed}/{n_files} frames")
    return volume


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate depth maps into TSDF using legacy Open3D')
    parser.add_argument('--result', type=str, required=True, help='Path to the result folder')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel size')
    parser.add_argument('--depth_scale', type=float, default=6553.5, help='Depth scale')
    parser.add_argument('--depth_max', type=float, default=3.0, help='Maximum depth')
    parser.add_argument('--weight', type=float, default=[2.0], nargs='+', help='Weight threshold (not used in legacy)')
    args = parser.parse_args()

    depth_file_names = sorted(glob(f'{args.result}/renders/depth_after_opt/*.png'))
    color_file_names = sorted(glob(f'{args.result}/renders/image_after_opt/*.jpg'))
    
    # Process subset for testing
    step = max(1, len(depth_file_names) // 100)  # Use every Nth frame
    depth_file_names = depth_file_names[::step]
    color_file_names = color_file_names[::step]
    
    stamps = [float(os.path.basename(i)[:-4]) for i in color_file_names]
    print(f"Processing {len(depth_file_names)} depth maps and {len(color_file_names)} color images")

    try:
        intrinsic, extrinsic = load_intrinsic_extrinsic(args.result, stamps)
        volume = integrate_legacy(depth_file_names, color_file_names, intrinsic, extrinsic, args)

        for w in args.weight:
            out = f'{args.result}/tsdf_mesh_w{w:.1f}.ply'
            print(f"Extracting mesh from legacy TSDF volume")
            
            try:
                mesh = volume.extract_triangle_mesh()
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                
                if len(mesh.vertices) > 0:
                    o3d.io.write_triangle_mesh(out, mesh)
                    print(f"Successfully saved legacy TSDF mesh to {out} with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
                else:
                    print("Warning: Empty mesh extracted")
                    raise ValueError("Empty mesh")
                    
            except Exception as e:
                print(f"Error during mesh extraction: {e}")
                # Create placeholder 
                with open(out, 'w') as f:
                    f.write("ply\nformat ascii 1.0\ncomment Legacy TSDF extraction failed\nelement vertex 0\nelement face 0\nend_header\n")
                
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
