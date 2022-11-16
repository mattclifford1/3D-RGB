# 3D-RGB -- Full Pipeline of Camera movement for videos
Currently estimates on a per frame basis (no temporal information used). This will lead to temporal inconsistencies for some videos.

## Instalation
First clone this repo eg:
```
git clone https://github.com/mattclifford1/3D-RGB.git
```

Setup the Pipeline environment by installing all the dependant packages and downloading all the models using
```
cd setup
set_up_with_conda.sh
cd ..
```


## Pipeline
To run the entire method , place videos in [vids/input](vids/input) and use the script [run_pipeline.sh](run_pipeline.sh) to start the pipeline eg:
```
./run_pipeline xxx
```
where xxx is the vid prefix (project folder) name of vid (without .mp4).

This will:
  - extract video into frames
  - estimate depth for each frame using MiDas
  - estimate the LDI (layered depth image) using
  - then reconstruct the video using new camera postions

### Adjusting camera parameters
Use the code configuration file [configs/desk.yml](configs/desk.yml) to adjust the pipeline.

In particular adjust:
 - x_shift_range: (left/right movement)
 - y_shift_range: (up/down movement)
 - z_shift_range: (forwards/backwards)
 - traj_types: ([camera path type](https://github.com/mattclifford1/3D-RGB/blob/c1f09fade7db451d5dd03c4831519019109bc94b/utils.py#L29) - 'straight-line'/'double-straight-line'/'pan-right'/'pan-left'/'control')
 - video_postfix: (name added to video file)
