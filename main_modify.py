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
    高斯加权局部岭回归 + 二次特征进行图像恢复。
    使用特征 [dx, dy, dx^2, dy^2, dx*dy, 1] 拟合缺失像素邻域，
    最终融合权重为 0.35 inpaint / 0.55 regression / 0.10 mean。
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

    def _box_sum(values, r):
        kernel_size = (2 * int(r) + 1, 2 * int(r) + 1)
        return cv2.boxFilter(
            values.astype(np.double),
            ddepth=-1,
            ksize=kernel_size,
            normalize=False,
            borderType=cv2.BORDER_REFLECT,
        )

    def _quadratic_feature_kernels(r):
        r = int(r)
        coords = np.arange(-r, r + 1, dtype=np.double)
        dx, dy = np.meshgrid(coords / max(r, 1), coords / max(r, 1))
        sigma = 0.55
        weight = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        features = [dx, dy, dx * dx, dy * dy, dx * dy, np.ones_like(dx)]
        return weight, features

    def _weighted_sum(values, kernel):
        return cv2.filter2D(
            values.astype(np.double),
            ddepth=-1,
            kernel=kernel.astype(np.double),
            borderType=cv2.BORDER_REFLECT,
        )

    for channel in range(channels):
        plane = work_img[:, :, channel]
        missing = plane <= 1e-12
        if not np.any(missing):
            continue

        known = ~missing
        global_mean = float(np.mean(plane[known])) if np.any(known) else 0.0

        mask8 = missing.astype(np.uint8) * 255
        plane8 = np.clip(plane * 255.0, 0, 255).astype(np.uint8)
        inpaint = cv2.inpaint(plane8, mask8, 3, cv2.INPAINT_TELEA).astype(np.double) / 255.0

        known_f = known.astype(np.double)
        value_f = np.where(known, plane, 0.0)

        small_weight, _ = _quadratic_feature_kernels(radii[0])
        large_weight, _ = _quadratic_feature_kernels(radii[-1])
        small_count = _weighted_sum(known_f, small_weight)
        small_sum = _weighted_sum(value_f, small_weight)
        large_count = _weighted_sum(known_f, large_weight)
        large_sum = _weighted_sum(value_f, large_weight)
        local_mean = np.full_like(plane, global_mean, dtype=np.double)
        local_mean = np.where(large_count > 1e-12, large_sum / np.maximum(large_count, 1e-12), local_mean)
        local_mean = np.where(small_count > 1e-12, small_sum / np.maximum(small_count, 1e-12), local_mean)

        regression = np.copy(local_mean)
        fitted = np.zeros_like(missing, dtype=bool)

        for r in radii:
            raw_count = _box_sum(known_f, r)
            target = missing & (~fitted) & (raw_count >= 8)
            coords = np.argwhere(target)
            if coords.size == 0:
                continue

            weight, features = _quadratic_feature_kernels(r)
            weighted_count = _weighted_sum(known_f, weight)

            rhs_maps = [_weighted_sum(value_f, weight * feature) for feature in features]
            mat_maps = []
            for i in range(6):
                row = []
                for j in range(6):
                    row.append(_weighted_sum(known_f, weight * features[i] * features[j]))
                mat_maps.append(row)

            chunk = 40000
            for start in range(0, len(coords), chunk):
                part = coords[start:start + chunk]
                ii = part[:, 0]
                jj = part[:, 1]

                mat = np.empty((len(part), 6, 6), dtype=np.double)
                rhs = np.empty((len(part), 6), dtype=np.double)
                for i in range(6):
                    rhs[:, i] = rhs_maps[i][ii, jj]
                    for j in range(6):
                        mat[:, i, j] = mat_maps[i][j][ii, jj]

                lam = 1e-3 * np.maximum(weighted_count[ii, jj], 1.0)
                diag = np.arange(6)
                mat[:, diag, diag] += lam[:, np.newaxis]
                mat[:, 5, 5] -= lam * 0.999

                pred = np.einsum("nij,nj->ni", np.linalg.inv(mat), rhs)[:, 5]
                regression[ii, jj] = np.clip(pred, 0.0, 1.0)
                fitted[ii, jj] = True

        fused = 0.35 * inpaint + 0.55 * regression + 0.10 * local_mean
        restored_plane = np.copy(plane)
        restored_plane[missing] = fused[missing]
        res_img[:, :, channel] = restored_plane

    res_img = np.clip(res_img, 0.0, 1.0).astype(np.double)
    if squeeze_channel:
        res_img = res_img[:, :, 0]

    return res_img
