<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">HI-SLAM2: Geometry-Aware Gaussian SLAM for Fast Monocular Scene Reconstruction</h1>
  <p align="center">
    <a href="https://www.ifp.uni-stuttgart.de/en/institute/team/Zhang-00004/" target="_blank"><strong>Wei Zhang</strong></a>
    ·
    <a href="https://cvg.cit.tum.de/members/cheq" target="_blank"><strong>Qing Cheng</strong></a>
    ·
    <a href="https://www.ifp.uni-stuttgart.de/en/institute/team/Skuddis/" target="_blank"><strong>David Skuddis</strong></a>
    ·
    <a href="https://www.niclas-zeller.de/" target="_blank"><strong>Niclas Zeller</strong></a>
    ·
    <a href="https://cvg.cit.tum.de/members/cremers" target="_blank"><strong>Daniel Cremers</strong></a>
    ·
    <a href="https://www.ifp.uni-stuttgart.de/en/institute/team/Haala-00001/" target="_blank"><strong>Norbert Haala</strong></a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2411.17982">Paper</a> | <a href="https://hi-slam2.github.io/">Project Page</a></h3>
  <div align="center"></div>
</p>

## Hardware
The experiments performed with HI-SLAM2 for the seminar paper were run on an NVIDIA RTX 4090 GPU that has a Compute Capability of 8.9. If you want to use another GPU, it is recommended to use a NVIDIA GPU with a Compute Capability of 8.9 (see https://developer.nvidia.com/cuda-gpus).

## Getting Started
1. Create a new Conda environment and then activate it. Please note that we use the PyTorch version compiled by CUDA 11.8 in the `environment.yaml` file.
```Bash
conda env create -f environment.yaml
conda activate hislam2
```

2. Compile the CUDA kernel extensions (takes about 10 minutes). Please note that this process assume you have CUDA 11 installed, not 12. To look into the installed CUDA version, you can run `nvcc --version` in the terminal.
```Bash
python setup.py install
```

3. Download the pretrained weights of Omnidata models for generating depth and normal priors
```Bash
wget https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt -P pretrained_models
wget https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt -P pretrained_models
```

## Data Preparation
### TUM RGB-D and HM dataset
Download the preprocessed data for the TUM RGB-D and HM datasets. Depending on whether the Replica dataset was downloaded already, the 'data' directory exists, and you have to unzip the downloaded data in another directory and manually copy the directories contained in the data.zip into the 'data directory.

Download the data.zip (24.7 GB) from the following sources (two links are provided for redundancy):

https://cold1.gofile.io/download/direct/fa62c81f-d6de-4e20-9aec-04f1709ccf8d/data.zip

or

https://drive.google.com/file/d/10RgVmq_TUWlN7kjVnw_JIQ7rYjDIpo3m/view?usp=sharing

### Replica
Download and prepare the Replica dataset by running
```Bash
bash scripts/download_replica.sh
python scripts/preprocess_replica.py
```
where the data is converted to the expected format and put to `data/Replica` folder.

Finally the 'data' directory should be structured as follows:

<details>
  <summary>[Directory structure of 'data' (click to expand)]</summary>

```bash
  .
  └── data
        ├── HM_SLAM_autonome_systeme
        ├── Replica
        └── TUM_RGBD
```
</details>

## Run Demo
After preparing the Replica dataset, you can run HI-SLAM2 for a demo. It takes about 2 minutes to run the demo on an Nvidia RTX 4090 GPU. The result will be saved in the `outputs/room0` folder including the estimated camera poses, the Gaussian map, and the renderings. To visualize the constructing process of the Gaussian map, using the `--gsvis` flag. To visualize the intermediate results e.g. estimated depth and point cloud, using the `--droidvis` flag.
```bash
python demo.py \
--imagedir data/Replica/room0/colors \
--calib calib/replica.txt \
--config config/replica_config.yaml \
--output outputs/room0 \
[--gsvis] # Optional: Enable Gaussian map display
[--droidvis] # Optional: Enable point cloud display
```
To generate the TSDF mesh from the reconstructed Gaussian map, you can run
```bash
python tsdf_integrate.py --result outputs/room0 --voxel_size 0.01 --weight 2
```

## Run Evaluation
### Replica
Run the following script to automate the evaluation process on all sequences of the Replica dataset. It will evaluate the tracking error, rendering quality, and reconstruction accuracy.
```bash
python scripts/run_replica.py
```

## Run your own data
HI-SLAM2 supports casual video recordings from smartphone or camera (demo above with iPhone 15). To use your own video data, we provide a preprocessing script that extracts individual frames from your video and runs COLMAP to automatically estimate camera intrinsics. Run the preprocessing with:
```bash
python scripts/preprocess_owndata.py PATH_TO_YOUR_VIDEO PATH_TO_OUTPUT_DIR
```
once the intrinsics are obtained, you can run HI-SLAM2 by using the following command:
```bash
python demo.py \
--imagedir PATH_TO_OUTPUT_DIR/images \
--calib PATH_TO_OUTPUT_DIR/calib.txt \
--config config/owndata_config.yaml \
--output outputs/owndata \
--undistort --droidvis --gsvis
```
there are some other command line arguments you can use:
- `--undistort` undistort the image if distortion parameters are provided in the calib file
- `--droidvis` visualize the point cloud map and the intermediate results
- `--gsvis` visualize the Gaussian map
- `--buffer` max number of keyframes to pre-allocate memory for (default: 10% of total frames). 
  Increase this if you encounter the error: `IndexError: index X is out of bounds for dimension 0 with size X`. 
- `--start` start frame index (default: from the first frame)
- `--length` number of frames to process (default: all frames)

## Outpus
The outputs that were generated while working on this seminar paper can also be downloaded. If some modeling was performed already, the outputs.zip file can't be unzipped in this directory, because the 'outputs' directory already exists.

Download the outputs.zip (8.7 GB) from the following sources (two links are provided for redundancy):

https://cold1.gofile.io/download/direct/90f9d99f-1ac3-4f2e-9671-b2d431977fb5/outputs.zip

or

https://drive.google.com/file/d/11-wohn1Sr3aQaLlVeWymi1TsxDfiGVDv/view?usp=sharing
