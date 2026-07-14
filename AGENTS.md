# Project Context for Agents

## Goal
- This is an image restoration/reconstruction assignment.
- The platform only needs `main.py`; keep its public functions stable:
  - `noise_mask_image(img, noise_ratio=[0.8, 0.4, 0.6])`
  - `get_noise_mask(noise_img)`
  - `restore_image(noise_img, size=4)`
- Outputs must remain `np.double`, same shape as input, values clipped to `[0, 1]`.
- Use `D:\MyHub\env` for validation:
  - `D:\MyHub\env\Scripts\python.exe ...`

## Current Best Submission
- `main.py` is the current best platform submission file.
- It was updated from the ablation winner, not from the full `main_modify_2.py`.
- Current best method:
  - exact per-row/per-channel zero mask generation in `noise_mask_image`;
  - missing pixels detected by `<= 1e-12`;
  - RGB channels restored first with Gaussian-weighted local quadratic ridge regression;
  - features are `[dx, dy, dx^2, dy^2, dx*dy, 1]`;
  - fixed small-window priority is used instead of texture-adaptive window selection;
  - YCrCb chroma correction is retained: keep RGB-recovered luminance structure and replace/smooth Cr/Cb;
  - known nonzero pixels are written back unchanged.
- Important result: the ablation best was `fixed_window_ycrcb`, not full `main_modify_2.py`.

## Important Files
- `main.py`: final platform-ready implementation.
- `main_modify.py`: earlier Gaussian-weighted quadratic regression variant.
- `main_modify_2.py`: adaptive-window + YCrCb experimental variant.
- `run_ablation.py`: reproducible ablation script.
- `results/ablation/<timestamp>/`: generated ablation outputs; do not commit these large result directories unless explicitly requested.
- `A.png`, `samples/`: evaluation images.
- the grading markdown file with a Chinese filename: scoring rules.
- the third lab lecture PDF with a Chinese filename: reference material.

## Ablation Summary
- The latest full ablation output used:
  - `results/ablation/20260709_152732`
- Key generated-mask-noise averages:
  - `quadratic_gaussian_main_modify`: L2 about `13.40`, SSIM about `0.8543`.
  - `adaptive_rgb_no_ycrcb`: L2 about `14.59`, SSIM about `0.7934`.
  - `fixed_window_ycrcb`: L2 about `10.88`, SSIM about `0.8976`.
  - `full_adaptive_ycrcb`: L2 about `11.71`, SSIM about `0.8594`.
- Interpretation:
  - YCrCb chroma correction is strongly helpful.
  - Texture-adaptive windows were not best for platform-style zero-mask noise.
  - Fixed small-window priority gives the best measured score for the current platform-like tests.

## Useful Commands
- Syntax check:
  - `D:\MyHub\MyProject(Gitted)\cv-ml-env\Scripts\python.exe -m py_compile main.py`
- Run ablation:
  - `D:\MyHub\MyProject(Gitted)\cv-ml-env\Scripts\python.exe run_ablation.py`
- Check git state:
  - `git status --short`

## Git Notes
- Recent relevant commits:
  - `252ce46 feat: use best ablation restoration in main`
  - `d336f53 test: add restoration ablation experiment`
  - `d3a9d19 feat: add adaptive ycrcb restoration variant`
  - `32b775c feat: add weighted quadratic regression variant`
  - `d0e73e3 fix: sync submission main.py implementation`
- At the time this context was written, unrelated/uncommitted local state included:
  - modified `.gitignore`;
  - deleted the generated program-report PDF with a Chinese filename;
  - untracked `results/ablation/`.
- Do not revert or stage those unless the user asks.

## Agent Workflow
- Prefer `rg` for search.
- Use `apply_patch` for manual edits.
- Do not rewrite or simplify the algorithm casually; small changes can move platform metrics significantly.
- After important code changes, commit once with a focused message, per the user's standing request.
