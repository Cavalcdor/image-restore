# Image Restore — 图像恢复重建

> 基于**局部加权岭回归**与**YCrCb 色度校正**的图像恢复算法。
>
> 人工智能与大数据综合实践课程项目

## 项目概述

对 RGB 图像逐通道、逐行随机置零部分像素（R: 80%, G: 40%, B: 60%），利用高斯加权局部二次岭回归从稀疏已知像素中恢复完整图像，再通过 YCrCb 色度校正减少色彩伪影。

## 核心算法

| 阶段 | 方法 |
|------|------|
| **噪声生成** | 按通道/行独立随机选取像素置零 |
| **RGB 恢复** | 三窗口级联高斯加权二次岭回归 |
| **色度校正** | YCrCb 空间重估 Cr/Cb 通道 |
| **融合策略** | 亮度: 25% inpaint + 65% 回归 + 10% 局部均值 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | **最终提交版** — 最优算法实现 |
| `main_modify.py` | 高斯加权二次回归变体 |
| `main_modify_2.py` | 自适应窗口 + YCrCb 实验变体 |
| `run_ablation.py` | 消融实验脚本 |
| `main.ipynb` | Jupyter Notebook 实验记录 |
| `A.png` | 主要测试图像 |
| `samples/` | 样本测试图像（含干净/噪声对） |
| `results/` | 实验结果输出 |

## 快速开始

```powershell
# 1. 使用统一环境运行（环境位于 D:\MyHub\env）
D:\MyHub\env\Scripts\python.exe main.py

# 2. 运行消融实验
D:\MyHub\env\Scripts\python.exe run_ablation.py

# 3. 语法检查
D:\MyHub\env\Scripts\python.exe -m py_compile main.py
```

## 公共 API

```python
# 生成受损图像
noise_mask_image(img, noise_ratio=[0.8, 0.4, 0.6])

# 获取噪声掩码
get_noise_mask(noise_img)

# 恢复图像
restore_image(noise_img, size=4)
```

## 消融实验结果

| 方法 | L2 ↓ | SSIM ↑ | Cosine ↑ |
|------|:----:|:------:|:--------:|
| `fixed_window_ycrcb` **(最优)** | **10.88** | **0.8976** | **0.9983** |
| `full_adaptive_ycrcb` | 11.71 | 0.8594 | 0.9980 |
| `quadratic_gaussian_main_modify` | 13.40 | 0.8543 | 0.9974 |
| `adaptive_rgb_no_ycrcb` | 14.59 | 0.7934 | 0.9970 |

## 依赖

- Python 3.12+
- `numpy`
- `opencv-python`

> 环境配置详见 `D:\MyHub\env\README.md`
