# Gaussian Splatting (15-474 Neural Graphics, A2)

2D and 3D Gaussian splatting in PyTorch: fitting images with 2D Gaussians (P1-P5), projecting 3D Gaussians into a camera (P6), reconstructing a multi-view scene (P7), densification in 3D (P8), and novel-view evaluation (P9). Everything lives in `test.py`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch pillow numpy matplotlib
```

The code picks a device automatically (CUDA, then Apple MPS, then CPU). If MPS is slow or errors, force CPU by setting `device = "cpu"` near the top of `test.py`.

## Data layout

Run everything from the `A2/` folder:

```
A2/
  test.py
  coffee.png  astronaut.png  cat.png      # P5 images (resized to 128x128)
  spheres/
    cameras.json                          # K, frames, val_frames, height, width
    <frame images referenced by cameras.json>
```

## Running the experiments

`test.py` ends with bare calls to the experiments. Replace them with this dispatcher at the bottom of the file, so each experiment runs on demand:

```python
if __name__ == "__main__":
    import sys
    {"p5": p5, "p7": run_p7, "p8": run_p8, "p9": run_p9}[sys.argv[1]]()
```

Then:

```bash
python test.py p5   # 2D fitting sweep
python test.py p7   # 3D reconstruction, fixed 4000 Gaussians
python test.py p8   # 3D densification + comparison with P7
python test.py p9   # novel-view evaluation and orbit
```

P8 needs `p7_params.pt` and P9 needs `p8_params.pt`, so run **p7, then p8, then p9**.

| Experiment | Command | What it does | Outputs |
|---|---|---|---|
| P5 | `python test.py p5` | Fits coffee, astronaut and cat with N = 256, 1024, 4096 Gaussians (2000 steps each) and prints a PSNR table | `<image>_N<N>.png`, `psnr_vs_N.png` |
| P7 | `python test.py p7` | Fits 4000 random-init 3D Gaussians to the `spheres` training views (1500 steps) and reports mean train PSNR | `p7_train_renders.png`, `p7_params.pt` |
| P8 | `python test.py p8` | Trains with densification (1000 -> toward 4000 Gaussians; pass every 200 steps until step 1000) and compares train/held-out PSNR against P7 | `p8_params.pt`, `p8_heldout_compare.png` (plain / densified / ground truth) |
| P9 | `python test.py p9` | Reports train vs held-out PSNR for the densified fit and renders a 12-frame orbit | `p9_orbit.gif`, `p9_orbit_grid.png` |

P4 (2D densification) is implemented in `fit_image_densified` but has no driver in the dispatcher. It uses the slower loop-based `render`, so run it only on a small image.

## Implementation notes

- **Renderers:** `render` (loop, reference), `render_fast` (vectorised, used for P5) and `render_tiled` (16x16 tiles, 3-sigma truncation, used for P7-P9). `render_tiled` matches `render_fast` to within about 5e-5 mean pixel difference on random scenes, with gradients within 1%.
- **3D pipeline:** `covariance_3d` (scale + quaternion) -> `project_gaussian` (pinhole projection with Jacobian) -> `render_view` (depth sort, near-plane cull, 0.3 px^2 covariance dilation) -> `render_tiled`.
- **Densification (P8):** prune opacity < 0.005; clone small and split large Gaussians among the top 15% by mean position-gradient (averaged over views where visible); size threshold 0.06 in world units; Adam moments are carried over for surviving Gaussians and zeroed for new ones.
- **Orbit (P9):** the orbit axis is the normal of the plane fitted to the training camera centres. If the orbit looks tilted, hard-code the axis in `orbit_cams`.

## Reproducibility

P5, P7 and P8 use fixed seeds (`seed=0`). Numbers can still differ slightly across devices (CPU vs MPS vs CUDA), so expect small differences rather than exact matches.

Results from my run (Apple MPS):

| Fit | Gaussians | Train PSNR (dB) | Held-out PSNR (dB) |
|---|---|---|---|
| Plain P7 | 4000 | 23.68 | 16.28 |
| Densified P8 | 2014 | 24.17 | 16.71 |

## Runtime (rough)

P7 took 11.5 minutes on a 2-core CPU sandbox, and the P5 sweep was estimated at about 10 minutes with the tiled renderer. Neither has been timed on every device. P8 and P9 are similar in cost to P7.