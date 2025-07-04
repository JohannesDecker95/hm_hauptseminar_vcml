import argparse
import random
import sys
import traceback
import trimesh
import torch
import open3d as o3d
import numpy as np
from evaluate_3d_reconstruction import run_evaluation
from tqdm import tqdm
sys.path.append('.')



import torch
import trimesh
from scipy.spatial import cKDTree as KDTree


def normalize(x):
    return x / np.linalg.norm(x)


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(float))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp


def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transformation = reg_p2p.transformation
    return transformation


def check_proj(points, W, H, fx, fy, cx, cy, c2w):
    """
    Check if points can be projected into the camera view.

    Returns:
        bool: True if there are points can be projected

    """
    c2w = c2w.copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    points = torch.from_numpy(points).cuda().clone()
    w2c = np.linalg.inv(c2w)
    w2c = torch.from_numpy(w2c).cuda().float()
    K = torch.from_numpy(
        np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).cuda()
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
    homo_points = torch.cat(
        [points, ones], dim=1).reshape(-1, 4, 1).cuda().float()  # (N, 4)
    cam_cord_homo = w2c@homo_points  # (N, 4, 1)=(4,4)*(N, 4, 1)
    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
    cam_cord[:, 0] *= -1
    uv = K.float()@cam_cord.float()
    z = uv[:, -1:]+1e-5
    uv = uv[:, :2]/z
    uv = uv.float().squeeze(-1).cpu().numpy()
    edge = 10
    mask = (0 <= -z[:, 0, 0].cpu().numpy()) & (uv[:, 0] < W -
                                               edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)
    return mask.sum() > 0


def calc_3d_metric(rec_meshfile, gt_meshfile, align=True):
    """
    3D reconstruction metric.

    """
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        mesh_rec = mesh_rec.apply_transform(transformation)

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(
        gt_pc_tri.vertices, rec_pc_tri.vertices)
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %
    return {"accuracy":accuracy_rec, "completion":completion_rec, "completion_ratio":completion_ratio_rec}

def get_cam_position(gt_meshfile):
    mesh_gt = trimesh.load(gt_meshfile)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents[2] *= 0.7
    extents[1] *= 0.7
    extents[0] *= 0.3
    transform = np.linalg.inv(to_origin)
    transform[2, 3] += 0.4
    return extents, transform


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def calc_2d_metric(rec_meshfile, gt_meshfile,align=True, n_imgs=1000):
    """
    2D reconstruction metric, depth L1 loss.

    """
    H = 500
    W = 500
    focal = 300
    fx = focal
    fy = focal
    cx = H/2.0-0.5
    cy = W/2.0-0.5

    gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    unseen_gt_pointcloud_file = gt_meshfile.replace('.ply', '_pc_unseen.npy')
    pc_unseen = np.load(unseen_gt_pointcloud_file)
    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    # get vacant area inside the room
    extents, transform = get_cam_position(gt_meshfile)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H, visible=False)
    vis.get_render_option().mesh_show_back_face = True
    errors = []
    for i in tqdm(range(n_imgs)):
        while True:
            # sample view, and check if unseen region is not inside the camera view
            # if inside, then needs to resample
            up = [0, 0, -1]
            origin = trimesh.sample.volume_rectangular(
                extents, 1, transform=transform)
            origin = origin.reshape(-1)
            tx = round(random.uniform(-10000, +10000), 2)
            ty = round(random.uniform(-10000, +10000), 2)
            tz = round(random.uniform(-10000, +10000), 2)
            # will be normalized, so sample from range [0.0,1.0]
            target = [tx, ty, tz]
            target = np.array(target)-np.array(origin)
            c2w = viewmatrix(target, up, origin)
            tmp = np.eye(4)
            tmp[:3, :] = c2w  # sample translations
            c2w = tmp
            # if unseen points are projected into current view (c2w)
            seen = check_proj(pc_unseen, W, H, fx, fy, cx, cy, c2w)
            if (~seen):
                break

        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array

        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W, H, fx, fy, cx, cy)

        ctr = vis.get_view_control()
        ctr.set_constant_z_far(20)
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.add_geometry(gt_mesh, reset_bounding_box=True,)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        gt_depth = vis.capture_depth_float_buffer(True)
        gt_depth = np.asarray(gt_depth)
        vis.remove_geometry(gt_mesh, reset_bounding_box=True,)

        vis.add_geometry(rec_mesh, reset_bounding_box=True,)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        ours_depth = vis.capture_depth_float_buffer(True)
        ours_depth = np.asarray(ours_depth)
        vis.remove_geometry(rec_mesh, reset_bounding_box=True,)

        # filter missing surfaces where depth is 0
        if (ours_depth > 0).sum() > 0:
            errors += [np.abs(gt_depth[ours_depth > 0] -
                              ours_depth[ours_depth > 0]).mean()]
        else:
            continue

    errors = np.array(errors)
    return {'depth l1': errors.mean()*100}

def eval_recon(rec_mesh, gt_mesh, eval_2d, eval_3d, align=True):
    result = {}
    try:
        if eval_3d:
            pred_ply = rec_mesh.split('/')[-1]
            last_slash_index = rec_mesh.rindex('/')
            path_to_pred_ply = rec_mesh[:last_slash_index]
            
            # Adaptive distance threshold based on mesh scales
            try:
                import open3d as o3d
                import numpy as np
                
                # Load meshes to determine appropriate threshold
                recon_mesh_o3d = o3d.io.read_triangle_mesh(rec_mesh)
                gt_mesh_o3d = o3d.io.read_triangle_mesh(gt_mesh)
                
                # Calculate extents
                recon_bbox = recon_mesh_o3d.get_axis_aligned_bounding_box()
                gt_bbox = gt_mesh_o3d.get_axis_aligned_bounding_box()
                
                recon_extent = np.array(recon_bbox.max_bound) - np.array(recon_bbox.min_bound)
                gt_extent = np.array(gt_bbox.max_bound) - np.array(gt_bbox.min_bound)
                
                # Use 1% of the maximum extent as threshold
                max_extent = max(np.max(recon_extent), np.max(gt_extent))
                adaptive_threshold = max(0.05, max_extent * 0.01)  # At least 0.05, but scale with mesh size
                
                print(f"Adaptive distance threshold: {adaptive_threshold:.4f} (mesh extents: recon={np.max(recon_extent):.2f}, gt={np.max(gt_extent):.2f})")
                
            except Exception as e:
                print(f"Failed to compute adaptive threshold: {e}, using default 0.05")
                adaptive_threshold = 0.05
            
            try:
                result_3d = run_evaluation(pred_ply, path_to_pred_ply, gt_mesh.split("/")[-1][:-4],
                                           distance_thresh=adaptive_threshold, full_path_to_gt_ply=gt_mesh, icp_align=align)
                result = dict(result.items() | result_3d.items())
                print(result['mean precision'], result['mean recall'], result['recall'])
            except ZeroDivisionError:
                print("Warning: Division by zero in 3D evaluation (both precision and recall are zero)")
                print("This typically indicates the mesh has no valid correspondences with ground truth")
                # Create default result with zero values
                result_3d = {
                    'mean precision': 0.0,
                    'mean recall': 0.0,
                    'recall': 0.0,
                    'precision': 0.0,
                    'f1': 0.0
                }
                result = dict(result.items() | result_3d.items())
                print(0.0, 0.0, 0.0)
            except Exception as e:
                print(f"Error in 3D evaluation: {e}")
                # Create default result with zero values
                result_3d = {
                    'mean precision': 0.0,
                    'mean recall': 0.0,
                    'recall': 0.0,
                    'precision': 0.0,
                    'f1': 0.0
                }
                result = dict(result.items() | result_3d.items())
                print(0.0, 0.0, 0.0)

        if eval_2d:
            result_2d = calc_2d_metric(
                rec_mesh, gt_mesh, align=align, n_imgs=10)
            result = dict(result.items() | result_2d.items())
            print(result_2d)
    except Exception as e:
        traceback.print_exception(e)

    return result    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the reconstruction.'
    )    
    parser.add_argument("rec_mesh", type=str, help="Path to config file.")
    parser.add_argument("gt_mesh", type=str, help="Path to config file.")
    parser.add_argument("--eval_2d", action="store_true")
    parser.add_argument("--eval_3d", action="store_true")
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    result = eval_recon(args.rec_mesh, args.gt_mesh, args.eval_2d, args.eval_3d)

    if args.save:
        open(args.save, 'w').write(f'{result}')
