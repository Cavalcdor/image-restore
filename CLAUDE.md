# Project Memory

## Assignment
This repository is for an image restoration/reconstruction coursework task. The submission target is `main.py` only. The notebook and experimental files are useful context, but the platform reads the functions in `main.py`.

Required public API:

```python
noise_mask_image(img, noise_ratio=[0.8, 0.4, 0.6])
get_noise_mask(noise_img)
restore_image(noise_img, size=4)
```

Keep the signatures unchanged. Return arrays should preserve input shape, use `np.double`, and stay in `[0, 1]`.

## Environment
Use the local `env` at `D:\MyHub\env`:

```powershell
D:\MyHub\env\Scripts\python.exe -m py_compile main.py
D:\MyHub\env\Scripts\python.exe run_ablation.py
```

Avoid adding dependencies that are not already available. In particular, do not require `skimage` for submission code.

## Current Best Algorithm
`main.py` contains the current best submission algorithm, selected from the ablation experiment. It is not identical to `main_modify_2.py`.

Best variant: `fixed_window_ycrcb`.

Main ideas:
- exact mask generation per RGB channel and per row;
- zero/near-zero pixels are missing pixels;
- known nonzero pixels must remain unchanged;
- each RGB channel is first restored using Gaussian-weighted local quadratic ridge regression;
- regression features are `[dx, dy, dx^2, dy^2, dx*dy, 1]`;
- fixed small-window priority performed best in ablation;
- YCrCb chroma correction is retained to reduce color artifacts;
- final output is clipped to `[0, 1]`.

The full adaptive-window version in `main_modify_2.py` is useful for comparison, but its platform-style metrics were worse than fixed-window YCrCb.

## Ablation Results
The reproducible ablation script is `run_ablation.py`.

Latest useful output directory:

```text
results/ablation/20260709_152732
```

Generated-mask-noise averages from that run:

| Method | Mean L2 | Mean SSIM | Mean Cosine |
|---|---:|---:|---:|
| `quadratic_gaussian_main_modify` | ~13.40 | ~0.8543 | ~0.9974 |
| `adaptive_rgb_no_ycrcb` | ~14.59 | ~0.7934 | ~0.9970 |
| `fixed_window_ycrcb` | ~10.88 | ~0.8976 | ~0.9983 |
| `full_adaptive_ycrcb` | ~11.71 | ~0.8594 | ~0.9980 |

Conclusion: YCrCb chroma correction is the biggest win. Texture-adaptive window selection was a reasonable experiment, but fixed small windows score better on current platform-like zero-mask noise.

## Files
- `main.py`: final platform submission.
- `main_modify.py`: Gaussian-weighted quadratic ridge variant.
- `main_modify_2.py`: adaptive-window + YCrCb experimental variant.
- `run_ablation.py`: ablation runner that writes CSV, JSON summaries, restored images, and comparison grids.
- `A.png`: primary local test image.
- `samples/`: sample clean/noisy image pairs.
- `results/ablation/`: generated experiment results; normally keep untracked.

## Git State Notes
Recent important commits:

```text
252ce46 feat: use best ablation restoration in main
d336f53 test: add restoration ablation experiment
d3a9d19 feat: add adaptive ycrcb restoration variant
32b775c feat: add weighted quadratic regression variant
d0e73e3 fix: sync submission main.py implementation
```

Known unrelated local state at context capture time:
- `.gitignore` modified;
- the generated program-report PDF with a Chinese filename is deleted;
- `results/ablation/` untracked.

Do not stage, revert, or delete those unless explicitly asked.

## Working Rules
- Use `grep_search` or `file_search` for searching.
- Use `replace_string_in_file` / `multi_replace_string_in_file` for edits.
- Use `D:\MyHub\env` for validation:
  ```powershell
  D:\MyHub\env\Scripts\python.exe -m py_compile main.py
  D:\MyHub\env\Scripts\python.exe run_ablation.py
  ```
- Commit after important changes.
- Do not commit generated ablation images unless the user explicitly requests it.
