import numpy as np
import cv2


def noise_mask_image(img, noise_ratio=[0.8, 0.4, 0.6]):
    """
    根据题目要求生成受损图片
    :param img: cv2 读取图片,而且通道数顺序为 RGB
    :param noise_ratio: 噪声比率，类型是 List,，内容:[r 上的噪声比率,g 上的噪声比率,b 上的噪声比率]
                        默认值分别是 [0.8,0.4,0.6]
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array,
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
    img = np.asarray(img, dtype=np.double)
    if img.ndim == 2:
        work_img = img[:, :, np.newaxis]
        squeeze_channel = True
    elif img.ndim == 3:
        work_img = img
        squeeze_channel = False
    else:
        raise ValueError("img must be a 2-D or 3-D image array")

    height, width, channels = work_img.shape
    ratios = np.asarray(noise_ratio, dtype=np.double).reshape(-1)
    if ratios.size == 0:
        ratios = np.zeros(channels, dtype=np.double)
    elif ratios.size == 1:
        ratios = np.repeat(ratios, channels)
    elif ratios.size < channels:
        ratios = np.pad(ratios, (0, channels - ratios.size), mode="edge")
    ratios = np.clip(ratios[:channels], 0.0, 1.0)

    mask = np.ones_like(work_img, dtype=np.double)
    for channel in range(channels):
        zero_count = int(round(width * float(ratios[channel])))
        if zero_count <= 0:
            continue
        if zero_count >= width:
            mask[:, :, channel] = 0.0
            continue
        for row in range(height):
            zero_cols = np.random.choice(width, size=zero_count, replace=False)
            mask[row, zero_cols, channel] = 0.0

    noise_img = work_img * mask
    if squeeze_channel:
        noise_img = noise_img[:, :, 0]

    return noise_img.astype(np.double)


def get_noise_mask(noise_img):
    """
    获取噪声图像，一般为 np.array
    :param noise_img: 带有噪声的图片
    :return: 噪声图像矩阵
    """
    return np.array(noise_img != 0, dtype="double")


def restore_image(noise_img, size=4):
    """
    使用 你最擅长的算法模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    arr = np.asarray(noise_img, dtype=np.double)
    if arr.ndim == 2:
        work_img = arr[:, :, np.newaxis]
        squeeze_channel = True
    elif arr.ndim == 3:
        work_img = arr
        squeeze_channel = False
    else:
        raise ValueError("noise_img must be a 2-D or 3-D image array")

    res_img = np.copy(work_img)
    height, width, channels = work_img.shape
    radius = max(1, int(size))
    radii = [radius, max(radius * 2, radius + 1), max(radius * 4, radius + 2)]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.double)

    def _box_sum(values, r):
        kernel_size = (2 * int(r) + 1, 2 * int(r) + 1)
        return cv2.boxFilter(
            values.astype(np.double),
            ddepth=-1,
            ksize=kernel_size,
            normalize=False,
            borderType=cv2.BORDER_REFLECT,
        )

    for channel in range(channels):
        plane = work_img[:, :, channel]
        missing = plane <= 1e-12
        if not np.any(missing):
            continue

        known = ~missing
        global_mean = float(np.mean(plane[known])) if np.any(known) else 0.0

        mask8 = (missing.astype(np.uint8) * 255)
        plane8 = np.clip(plane * 255.0, 0, 255).astype(np.uint8)
        inpaint = cv2.inpaint(plane8, mask8, 3, cv2.INPAINT_TELEA).astype(np.double) / 255.0

        known_f = known.astype(np.double)
        value_f = np.where(known, plane, 0.0)
        small_count = _box_sum(known_f, radii[0])
        small_sum = _box_sum(value_f, radii[0])
        large_count = _box_sum(known_f, radii[-1])
        large_sum = _box_sum(value_f, radii[-1])
        local_mean = np.full_like(plane, global_mean, dtype=np.double)
        local_mean = np.where(large_count > 0, large_sum / np.maximum(large_count, 1.0), local_mean)
        local_mean = np.where(small_count > 0, small_sum / np.maximum(small_count, 1.0), local_mean)

        regression = np.copy(local_mean)
        fitted = np.zeros_like(missing, dtype=bool)
        weighted_x = known_f * xx
        weighted_y = known_f * yy
        weighted_xx = known_f * xx * xx
        weighted_yy = known_f * yy * yy
        weighted_xy = known_f * xx * yy
        weighted_zx = value_f * xx
        weighted_zy = value_f * yy

        for r in radii:
            n = _box_sum(known_f, r)
            target = missing & (~fitted) & (n >= 4)
            coords = np.argwhere(target)
            if coords.size == 0:
                continue

            sx = _box_sum(weighted_x, r)
            sy = _box_sum(weighted_y, r)
            sxx = _box_sum(weighted_xx, r)
            syy = _box_sum(weighted_yy, r)
            sxy = _box_sum(weighted_xy, r)
            sz = _box_sum(value_f, r)
            sxz = _box_sum(weighted_zx, r)
            syz = _box_sum(weighted_zy, r)

            chunk = 100000
            for start in range(0, len(coords), chunk):
                part = coords[start:start + chunk]
                ii = part[:, 0]
                jj = part[:, 1]
                nn = n[ii, jj]
                x0 = jj.astype(np.double)
                y0 = ii.astype(np.double)

                sx_c = sx[ii, jj] - x0 * nn
                sy_c = sy[ii, jj] - y0 * nn
                sxx_c = sxx[ii, jj] - 2 * x0 * sx[ii, jj] + x0 * x0 * nn
                syy_c = syy[ii, jj] - 2 * y0 * sy[ii, jj] + y0 * y0 * nn
                sxy_c = sxy[ii, jj] - x0 * sy[ii, jj] - y0 * sx[ii, jj] + x0 * y0 * nn
                sxz_c = sxz[ii, jj] - x0 * sz[ii, jj]
                syz_c = syz[ii, jj] - y0 * sz[ii, jj]

                lam = 1e-3 * np.maximum(nn, 1.0)
                mat = np.zeros((len(part), 3, 3), dtype=np.double)
                rhs = np.zeros((len(part), 3), dtype=np.double)
                mat[:, 0, 0] = sxx_c + lam
                mat[:, 0, 1] = sxy_c
                mat[:, 0, 2] = sx_c
                mat[:, 1, 0] = sxy_c
                mat[:, 1, 1] = syy_c + lam
                mat[:, 1, 2] = sy_c
                mat[:, 2, 0] = sx_c
                mat[:, 2, 1] = sy_c
                mat[:, 2, 2] = nn + lam * 1e-3
                rhs[:, 0] = sxz_c
                rhs[:, 1] = syz_c
                rhs[:, 2] = sz[ii, jj]

                pred = np.einsum("nij,nj->ni", np.linalg.inv(mat), rhs)[:, 2]
                regression[ii, jj] = np.clip(pred, 0.0, 1.0)
                fitted[ii, jj] = True

        fused = 0.55 * inpaint + 0.35 * regression + 0.10 * local_mean
        restored_plane = np.copy(plane)
        restored_plane[missing] = fused[missing]
        res_img[:, :, channel] = restored_plane

    res_img = np.clip(res_img, 0.0, 1.0).astype(np.double)
    if squeeze_channel:
        res_img = res_img[:, :, 0]

    return res_img
