import numpy as np
import cv2


def noise_mask_image(img, noise_ratio=[0.8, 0.4, 0.6]):
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
    return np.array(noise_img != 0, dtype="double")


def restore_image(noise_img, size=4):
    arr = np.asarray(noise_img, dtype=np.double)
    if arr.ndim == 2:
        work_img = arr[:, :, np.newaxis]
        squeeze_channel = True
    elif arr.ndim == 3:
        work_img = arr
        squeeze_channel = False
    else:
        raise ValueError("noise_img must be a 2-D or 3-D image array")

    height, width, channels = work_img.shape
    radius = max(1, int(size))
    radii = [radius, max(radius * 2, radius + 1), max(radius * 4, radius + 2)]
    eps = 1e-12

    kernel_cache = {}

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
        if r in kernel_cache:
            return kernel_cache[r]
        coords = np.arange(-r, r + 1, dtype=np.double)
        dx, dy = np.meshgrid(coords / max(r, 1), coords / max(r, 1))
        sigma = 0.55
        weight = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        features = [dx, dy, dx * dx, dy * dy, dx * dy, np.ones_like(dx)]
        kernel_cache[r] = (weight, features)
        return kernel_cache[r]

    def _weighted_sum(values, kernel):
        return cv2.filter2D(
            values.astype(np.double),
            ddepth=-1,
            kernel=kernel.astype(np.double),
            borderType=cv2.BORDER_REFLECT,
        )

    def _texture_map(guide):
        guide = np.clip(guide.astype(np.double), 0.0, 1.0)
        blur = cv2.GaussianBlur(guide, (0, 0), 1.2, borderType=cv2.BORDER_REFLECT)
        gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
        grad = np.sqrt(gx * gx + gy * gy)
        mean = cv2.boxFilter(guide, -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT)
        mean2 = cv2.boxFilter(guide * guide, -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT)
        var = np.maximum(mean2 - mean * mean, 0.0)

        grad_scale = float(np.percentile(grad, 95)) + 1e-12
        var_scale = float(np.percentile(var, 95)) + 1e-12
        grad_n = np.clip(grad / grad_scale, 0.0, 1.0)
        var_n = np.clip(var / var_scale, 0.0, 1.0)
        return np.clip(0.7 * grad_n + 0.3 * var_n, 0.0, 1.0)

    def _preferred_radius(texture, mode):
        pref = np.full(texture.shape, 2, dtype=np.int32)
        if mode == "y":
            pref[texture > 0.18] = 1
            pref[texture > 0.38] = 0
        elif mode == "chroma":
            pref[texture > 0.45] = 1
        else:
            pref[texture > 0.20] = 1
            pref[texture > 0.42] = 0
        return pref

    def _restore_plane(plane, missing, guide_texture, mode):
        plane = np.asarray(plane, dtype=np.double)
        missing = np.asarray(missing, dtype=bool)
        if not np.any(missing):
            return np.clip(plane, 0.0, 1.0).astype(np.double)

        known = ~missing
        global_mean = float(np.mean(plane[known])) if np.any(known) else 0.5

        plane_for_inpaint = np.where(known, plane, 0.0)
        mask8 = missing.astype(np.uint8) * 255
        plane8 = np.clip(plane_for_inpaint * 255.0, 0, 255).astype(np.uint8)
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

        preferred = _preferred_radius(guide_texture, mode)
        regression = np.copy(local_mean)
        fitted = np.zeros_like(missing, dtype=bool)
        min_samples = 8 if mode != "chroma" else 10

        for radius_index, r in enumerate(radii):
            raw_count = _box_sum(known_f, r)
            target = missing & (~fitted) & (preferred <= radius_index) & (raw_count >= min_samples)
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

                lam_scale = 2e-3 if mode == "chroma" else 1e-3
                lam = lam_scale * np.maximum(weighted_count[ii, jj], 1.0)
                diag = np.arange(6)
                mat[:, diag, diag] += lam[:, np.newaxis]
                mat[:, 5, 5] -= lam * 0.999

                pred = np.einsum("nij,nj->ni", np.linalg.inv(mat), rhs)[:, 5]
                regression[ii, jj] = np.clip(pred, 0.0, 1.0)
                fitted[ii, jj] = True

        regression = np.where(fitted, regression, local_mean)
        if mode == "y":
            fused = 0.25 * inpaint + 0.65 * regression + 0.10 * local_mean
        elif mode == "chroma":
            fused = 0.20 * inpaint + 0.35 * regression + 0.45 * local_mean
        else:
            fused = 0.35 * inpaint + 0.55 * regression + 0.10 * local_mean

        restored = np.copy(plane)
        restored[missing] = fused[missing]
        return np.clip(restored, 0.0, 1.0).astype(np.double)

    missing_rgb = work_img <= eps
    rgb_direct = np.copy(work_img)

    if channels == 1:
        guide_seed = np.copy(work_img[:, :, 0])
        miss = missing_rgb[:, :, 0]
        if np.any(miss):
            mask8 = miss.astype(np.uint8) * 255
            plane8 = np.clip(guide_seed * 255.0, 0, 255).astype(np.uint8)
            guide_seed = cv2.inpaint(plane8, mask8, 3, cv2.INPAINT_TELEA).astype(np.double) / 255.0
        texture = _texture_map(guide_seed)
        out = _restore_plane(work_img[:, :, 0], miss, texture, "rgb")
        return out if squeeze_channel else out[:, :, np.newaxis]

    for channel in range(channels):
        plane = work_img[:, :, channel]
        missing = missing_rgb[:, :, channel]
        if not np.any(missing):
            continue
        mask8 = missing.astype(np.uint8) * 255
        plane8 = np.clip(plane * 255.0, 0, 255).astype(np.uint8)
        seed = cv2.inpaint(plane8, mask8, 3, cv2.INPAINT_TELEA).astype(np.double) / 255.0
        texture = _texture_map(seed)
        rgb_direct[:, :, channel] = _restore_plane(plane, missing, texture, "rgb")

    if channels < 3:
        return np.clip(rgb_direct, 0.0, 1.0).astype(np.double)

    rgb3 = np.clip(rgb_direct[:, :, :3], 0.0, 1.0)
    ycc_seed = cv2.cvtColor(rgb3.astype(np.float32), cv2.COLOR_RGB2YCrCb).astype(np.double)

    complete_known = np.all(~missing_rgb[:, :, :3], axis=2)
    ycc_known_src = cv2.cvtColor(np.clip(work_img[:, :, :3], 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2YCrCb).astype(np.double)
    ycc_work = np.copy(ycc_seed)
    ycc_work[complete_known] = ycc_known_src[complete_known]

    texture_y = _texture_map(ycc_seed[:, :, 0])
    ycc_restored = np.empty_like(ycc_work)
    ycc_restored[:, :, 0] = _restore_plane(ycc_work[:, :, 0], ~complete_known, texture_y, "y")
    ycc_restored[:, :, 1] = _restore_plane(ycc_work[:, :, 1], ~complete_known, texture_y, "chroma")
    ycc_restored[:, :, 2] = _restore_plane(ycc_work[:, :, 2], ~complete_known, texture_y, "chroma")

    direct_ycc = cv2.cvtColor(np.clip(rgb_direct[:, :, :3], 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2YCrCb).astype(np.double)
    mixed_ycc = np.copy(direct_ycc)
    mixed_ycc[:, :, 1] = ycc_restored[:, :, 1]
    mixed_ycc[:, :, 2] = ycc_restored[:, :, 2]
    rgb_mixed = cv2.cvtColor(np.clip(mixed_ycc, 0.0, 1.0).astype(np.float32), cv2.COLOR_YCrCb2RGB).astype(np.double)

    res_img = np.copy(rgb_direct)
    res_img[:, :, :3] = np.clip(rgb_mixed, 0.0, 1.0)
    res_img[~missing_rgb] = work_img[~missing_rgb]
    res_img = np.clip(res_img, 0.0, 1.0).astype(np.double)

    return res_img
