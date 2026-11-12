# Monocular 3D Scene Reconstruction
## dev/creator=tubakhxn

Research & Educational Purposes Only.

Pipeline: RGB frame -> **Depth Anything V2 (Small)** pretrained inference -> RGB-D back-projection -> 3D point cloud -> 3-panel video.

| Panel | Content |
|---|---|
| Top left | Original dashcam frame |
| Bottom left | AI-predicted depth (TURBO colormap, warm = near, cool = far) |
| Right | 3D point cloud (RGB colours kept), X/Y/Z axes, ground grid, slowly swaying virtual camera |

## Run
```
pip install -r requirements.txt        # needs ffmpeg on PATH
python main.py                         # input/original_dashcam.mp4 -> output/*.mp4
python main.py --preview 120           # single composite PNG -> cache/
```
Outputs: `output/3d_scene_reconstruction.mp4` (main), `depth_visualization.mp4`, `point_cloud_visualization.mp4`.

## What is real
- Depth maps come from actual inference of the pretrained Depth Anything V2 **Small** weights
  (`models/depth_anything_v2_small.onnx`, ONNX export of the official model, run on CPU with ONNX Runtime).
- Every point in the cloud is a back-projected pixel (x=(u-cx)z/f, y=(v-cy)z/f) coloured from the RGB frame.
- Processing rate is 10 fps (331 frames); the 30 fps output is produced by cross-blending consecutive rendered frames.

## Assumptions and limitations (please read)
- **Depth is relative, not metric.** The model outputs relative inverse depth. It is normalised (2nd..99.5th percentile with
  temporal smoothing) and mapped to an *assumed* range of 4–60 units. Axes are in these relative units — not metres, not LiDAR.
- **Camera intrinsics are assumed** (no calibration available): horizontal FOV 90 deg, square pixels, principal point at image centre.
  The video is a screen recording, so true lens parameters are unknown; wrong FOV distorts the geometry.
- Each frame's cloud is built independently (no ego-motion / SLAM fusion), so the cloud is a per-frame reconstruction, not a map.
- Filtering: car-hood rows (bottom 30%), sky / far-plane pixels, median-filtered depth, and "flying pixel" removal at depth discontinuities.
  Thin structures (wires, poles) and the stair-stepped road rows are artefacts of single-image depth.
- The point cloud is rendered with a custom NumPy z-buffer splatting renderer (Open3D offscreen rendering is unreliable on
  headless CPU-only machines); it renders real 3D geometry from a virtual camera but is not an Open3D window.
- Inference used ONNX Runtime instead of PyTorch, because it was the only route to the official-architecture weights in the
  restricted build environment. Optional YOLO detection was **not** included.
