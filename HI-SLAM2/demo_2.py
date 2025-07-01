import os    # nopep8
import sys   # nopep8
sys.path.append(os.path.join(os.path.dirname(__file__), 'hislam2'))   # nopep8
sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty/lietorch/examples/core'))   # nopep8
import time
import torch
import cv2
import re
import os
import argparse
import numpy as np
import lietorch
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

from tqdm import tqdm
from torch.multiprocessing import Process, Queue
from hi2 import Hi2
from data_readers.tum import TUMStream


def mono_stream_legacy(queue, imagedir, calib, undistort=False, cropborder=False, start=0, length=100000):
    """ Legacy image generator """
    RES = 341 * 640
    calib = np.loadtxt(calib, delimiter=" ")
    K = np.array([[calib[0], 0, calib[2]],[0, calib[1], calib[3]],[0,0,1]])
    image_list = sorted(os.listdir(imagedir))[start:start+length]
    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        intrinsics = torch.tensor(calib[:4])
        if len(calib) > 4 and undistort:
            image = cv2.undistort(image, K, calib[4:])
        if cropborder > 0:
            image = image[cropborder:-cropborder, cropborder:-cropborder]
            intrinsics[2:] -= cropborder
        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((RES) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((RES) / (h0 * w0)))
        h1 = h1 - h1 % 8
        w1 = w1 - w1 % 8
        image = cv2.resize(image, (w1, h1))
        image = torch.as_tensor(image).permute(2, 0, 1)
        intrinsics[[0,2]] *= (w1 / w0)
        intrinsics[[1,3]] *= (h1 / h0)
        is_last = (t == len(image_list)-1)
        queue.put((t, image[None], intrinsics[None], is_last))
    time.sleep(10)


def show_image(image, depth_prior, depth, normal):
    from util.utils import colorize_np
    image = image[[2,1,0]].permute(1, 2, 0).cpu().numpy()
    depth = colorize_np(np.concatenate((depth_prior.cpu().numpy(), depth.cpu().numpy()), axis=1), range=(0, 4))
    normal = normal.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('rgb / prior normal / aligned prior depth / JDSA depth', np.concatenate((image / 255.0, (normal[...,[2,1,0]]+1.)/2., depth), axis=1)[::2,::2])
    cv2.waitKey(1)


def tum_stream(queue, datapath, start=0, length=100000):
    """ TUM dataset image generator using proper timestamp associations """
    RES = 341 * 640

    # Create TUM data stream
    tum_dataset = TUMStream(datapath, frame_rate=-1)
    
    # Extract timestamps from rgb.txt for proper evaluation
    rgb_file = os.path.join(datapath, 'rgb.txt')
    timestamps = []
    if os.path.exists(rgb_file):
        with open(rgb_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    timestamp = float(line.split()[0])
                    timestamps.append(timestamp)
    
    # Limit processing to the specified range
    end_idx = min(start + length, len(tum_dataset))
    
    for t in range(start, end_idx):
        if t >= len(tum_dataset):
            break
            
        # Get data from TUM dataset
        index, image, depth, pose, intrinsics = tum_dataset[t]
        
        # Convert to expected format
        image = image.byte()  # Convert from float to byte if needed
        h0, w0 = image.shape[1], image.shape[2]
        
        # Resize image to target resolution
        h1 = int(h0 * np.sqrt((RES) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((RES) / (h0 * w0)))
        h1 = h1 - h1 % 8
        w1 = w1 - w1 % 8
        
        # Resize image and adjust intrinsics
        image_resized = torch.nn.functional.interpolate(
            image[None].float(), size=(h1, w1), mode='bilinear', align_corners=False
        )[0].byte()
        
        # Adjust intrinsics for resizing
        intrinsics_adjusted = intrinsics.clone()
        intrinsics_adjusted[0] *= (w1 / w0)  # fx
        intrinsics_adjusted[1] *= (h1 / h0)  # fy
        intrinsics_adjusted[2] *= (w1 / w0)  # cx
        intrinsics_adjusted[3] *= (h1 / h0)  # cy

        is_last = (t == end_idx - 1)
        queue.put((t - start, image_resized[None], intrinsics_adjusted[None], is_last))

    time.sleep(10)


def save_trajectory(hi2, traj_full, datapath, output, start=0):
    t = hi2.video.counter.value
    tstamps = hi2.video.tstamp[:t]
    poses_wc = lietorch.SE3(hi2.video.poses[:t]).inv().data
    np.save("{}/intrinsics.npy".format(output), hi2.video.intrinsics[0].cpu().numpy()*8)

    # Load timestamps from TUM rgb.txt file
    rgb_file = os.path.join(datapath, 'rgb.txt')
    tstamps_full = []
    if os.path.exists(rgb_file):
        with open(rgb_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    timestamp = float(line.split()[0])
                    tstamps_full.append(timestamp)
        tstamps_full = np.array(tstamps_full)[start:][..., np.newaxis]
    else:
        # Check for traj_tum.txt file (HM_SLAM format)
        traj_tum_file = os.path.join(datapath, 'traj_tum.txt')
        if os.path.exists(traj_tum_file):
            with open(traj_tum_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        timestamp = float(line.split()[0])
                        tstamps_full.append(timestamp)
            tstamps_full = np.array(tstamps_full)[start:][..., np.newaxis]
        else:
            # Fallback to old method if neither rgb.txt nor traj_tum.txt exists
            imagedir = os.path.join(datapath, 'colors') if os.path.isdir(os.path.join(datapath, 'colors')) else \
                      os.path.join(datapath, 'images') if os.path.isdir(os.path.join(datapath, 'images')) else datapath
            image_files = sorted(os.listdir(imagedir))[start:]
            tstamps_fallback = []
            for x in image_files:
                numbers = re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)
                if numbers:
                    tstamps_fallback.append(float(numbers[-1]))
                else:
                    # If no numbers found, use sequential numbering
                    tstamps_fallback.append(float(len(tstamps_fallback)))
            tstamps_full = np.array(tstamps_fallback)[..., np.newaxis]
    
    # Ensure indices are within bounds of tstamps_full
    indices = tstamps.cpu().numpy().astype(int)
    max_index = len(tstamps_full) - 1
    valid_indices = np.clip(indices, 0, max_index)
    
    # Check if any indices were clipped and warn
    if np.any(indices != valid_indices):
        print(f"Warning: Some keyframe indices ({np.max(indices)}) exceed available timestamps ({max_index}). Clipping to valid range.")
    
    tstamps_kf = tstamps_full[valid_indices]
    ttraj_kf = np.concatenate([tstamps_kf, poses_wc.cpu().numpy()], axis=1)
    np.savetxt(f"{output}/traj_kf.txt", ttraj_kf)                     #  for evo evaluation 
    
    if traj_full is not None:
        # Ensure dimensions match for concatenation
        traj_len = len(traj_full)
        tstamps_len = len(tstamps_full)
        
        if traj_len <= tstamps_len:
            # Use the first traj_len timestamps
            tstamps_for_traj = tstamps_full[:traj_len]
            ttraj_full = np.concatenate([tstamps_for_traj, traj_full], axis=1)
            np.savetxt(f"{output}/traj_full.txt", ttraj_full)
        else:
            # More trajectory points than timestamps - generate sequential timestamps
            print(f"Warning: More trajectory points ({traj_len}) than timestamps ({tstamps_len}). Generating sequential timestamps.")
            sequential_tstamps = np.arange(traj_len).reshape(-1, 1).astype(float)
            ttraj_full = np.concatenate([sequential_tstamps, traj_full], axis=1)
            np.savetxt(f"{output}/traj_full.txt", ttraj_full)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="path to TUM dataset directory (containing rgb.txt, depth.txt, etc.)", default=None)
    parser.add_argument("--imagedir", type=str, help="path to image directory (for backward compatibility)", default=None)
    parser.add_argument("--calib", type=str, help="path to calibration file (optional for TUM datasets)", default=None)
    parser.add_argument("--config", type=str, help="path to configuration file")
    parser.add_argument("--output", default='outputs/demo', help="path to save output")
    parser.add_argument("--gtdepthdir", type=str, default=None, help="optional for evaluation, assumes 16-bit depth scaled by 6553.5")

    parser.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "pretrained_models/droid.pth"))
    parser.add_argument("--buffer", type=int, default=-1, help="number of keyframes to buffer (default: 1/10 of total frames)")
    parser.add_argument("--undistort", action="store_true", help="undistort images if calib file contains distortion parameters")
    parser.add_argument("--cropborder", type=int, default=0, help="crop images to remove black border")

    parser.add_argument("--droidvis", action="store_true")
    parser.add_argument("--gsvis", action="store_true")

    parser.add_argument("--start", type=int, default=0, help="start frame")
    parser.add_argument("--length", type=int, default=100000, help="number of frames to process")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Determine if we're using TUM dataset or legacy mode
    use_tum_loader = False
    if args.datapath and os.path.exists(os.path.join(args.datapath, 'rgb.txt')):
        use_tum_loader = True
        datapath = args.datapath
        if not args.gtdepthdir:
            args.gtdepthdir = os.path.join(datapath, 'depth')
    elif args.imagedir:
        # Legacy mode - use imagedir
        datapath = os.path.dirname(args.imagedir) if args.imagedir else args.datapath
        if not datapath:
            raise ValueError("Either --datapath (for TUM dataset) or --imagedir must be provided")
    else:
        raise ValueError("Either --datapath (for TUM dataset) or --imagedir must be provided")

    hi2 = None
    queue = Queue(maxsize=8)
    
    # Choose the appropriate data loader
    if use_tum_loader:
        print(f"Using TUM data loader for dataset: {datapath}")
        reader = Process(target=tum_stream, args=(queue, datapath, args.start, args.length))
        # Count total frames from TUM dataset
        tum_dataset = TUMStream(datapath, frame_rate=-1)
        N = min(len(tum_dataset) - args.start, args.length)
    else:
        print(f"Using legacy data loader for images: {args.imagedir}")
        reader = Process(target=mono_stream_legacy, args=(queue, args.imagedir, args.calib, args.undistort, args.cropborder, args.start, args.length))
        N = len(os.listdir(args.imagedir))
    
    reader.start()

    args.buffer = min(1000, N // 10 + 150) if args.buffer < 0 else args.buffer
    pbar = tqdm(range(N), desc="Processing keyframes")
    
    while 1:
        (t, image, intrinsics, is_last) = queue.get()
        pbar.update()

        if hi2 is None:
            args.image_size = [image.shape[2], image.shape[3]]
            hi2 = Hi2(args)

        hi2.track(t, image, intrinsics=intrinsics, is_last=is_last)

        if args.droidvis and hi2.video.tstamp[hi2.video.counter.value-1] == t:
            from geom.ba import get_prior_depth_aligned
            index = hi2.video.counter.value-2
            depth_prior, _ = get_prior_depth_aligned(hi2.video.disps_prior_up[index][None].cuda(), hi2.video.dscales[index][None])
            show_image(image[0], 1./depth_prior.squeeze(), 1./hi2.video.disps_up[index], hi2.video.normals[index])
        pbar.set_description(f"Processing keyframe {hi2.video.counter.value} gs {hi2.gs.gaussians._xyz.shape[0]}")

        if is_last:
            pbar.close()
            break

    reader.join()

    traj = hi2.terminate()
    save_trajectory(hi2, traj, datapath, args.output, start=args.start)

    print("Done")
