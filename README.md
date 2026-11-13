[English](README.md) | [简体中文](README.zh-CN.md)

# Image Restore

[![License](https://img.shields.io/badge/License-See%20Rights%20Notice-64748B?style=flat-square)](LICENSE)

![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square) ![Library: OpenCV / NumPy](https://img.shields.io/badge/Library-OpenCV%20%2F%20NumPy-5C3EE8?style=flat-square) ![Type: Course Project](https://img.shields.io/badge/Type-Course%20Project-64748B?style=flat-square)

Image restoration using **locally weighted ridge regression** and **YCrCb chroma correction**, developed for an Artificial Intelligence and Big Data practice course.

## Overview

Pixels are randomly set to zero independently per RGB channel and row, with missing ratios of 80% for red, 40% for green and 60% for blue. Gaussian-weighted local quadratic ridge regression reconstructs the image from sparse known pixels, followed by YCrCb chroma correction to reduce color artifacts.

## Algorithm

| Stage | Method |
| --- | --- |
| Corruption | Independent per-row, per-channel random zero masks |
| RGB restoration | Three-window cascade of Gaussian-weighted quadratic ridge regression |
| Chroma correction | Re-estimation of Cr/Cb in YCrCb space |
| Luminance blending | 25% inpainting + 65% regression + 10% local mean |

## Files

| Path | Purpose |
| --- | --- |
| `main.py` | Final submission and best-performing implementation |
| `main_modify.py` | Gaussian-weighted quadratic regression variant |
| `main_modify_2.py` | Adaptive-window and YCrCb experimental variant |
| `run_ablation.py` | Ablation experiments |
| `main.ipynb` | Notebook experiments |
| `A.png` | Main evaluation image |
| `samples/` | Test images, including clean/corrupted pairs |
| `results/` | Experiment outputs |

## Quick start

Python 3.12+, `numpy` and `opencv-python` are required. The maintainer's shared environment is at `D:\MyHub\env`; its setup is documented in `D:\MyHub\env\README.md`.

```powershell
D:\MyHub\env\Scripts\python.exe main.py
D:\MyHub\env\Scripts\python.exe run_ablation.py
D:\MyHub\env\Scripts\python.exe -m py_compile main.py
```

## Public API

```python
noise_mask_image(img, noise_ratio=[0.8, 0.4, 0.6])
get_noise_mask(noise_img)
restore_image(noise_img, size=4)
```

## Ablation results

| Method | L2 ↓ | SSIM ↑ | Cosine ↑ |
| --- | ---: | ---: | ---: |
| **`fixed_window_ycrcb`** | **10.88** | **0.8976** | **0.9983** |
| `full_adaptive_ycrcb` | 11.71 | 0.8594 | 0.9980 |
| `quadratic_gaussian_main_modify` | 13.40 | 0.8543 | 0.9974 |
| `adaptive_rgb_no_ycrcb` | 14.59 | 0.7934 | 0.9970 |

The fixed-window YCrCb variant is the best result in these experiments; the fully adaptive variant is not the final submission.

## License and rights

This fork has **no repository-wide open-source license**. See the [rights and licensing notice](LICENSE). Its [upstream repository](https://github.com/2301601594/Image-Restore) also had no repository-wide license when checked on 2026-11-13. This fork does not grant rights over upstream code, course materials, images or datasets. Original contributions retain their respective authors' rights unless separately licensed.
