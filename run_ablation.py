import csv
import importlib
import json
import time
import types
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "results" / "ablation" / time.strftime("%Y%m%d_%H%M%S")
IMG_DIR = OUT_ROOT / "images"
CMP_DIR = OUT_ROOT / "comparison"

SIZE = 4
IMAGE_SIZE = (150, 150)
RATIOS = [[0.8, 0.4, 0.6], [0.4, 0.6, 0.8]]
SEEDS = [7, 42, 2026]
METHOD_CONFIGS = {
    "degraded": "no restoration baseline",
    "telea_only": "cv2 Telea inpainting only",
    "local_mean_only": "box-filtered local mean only",
    "linear_ridge_main": "main.py linear ridge regression [dx, dy, 1]",
    "quadratic_gaussian_main_modify": "main_modify.py gaussian weighted quadratic ridge",
    "adaptive_rgb_no_ycrcb": "main_modify_2.py adaptive RGB restoration, YCrCb disabled",
    "fixed_window_ycrcb": "main_modify_2.py fixed small window, YCrCb chroma correction enabled",
    "full_adaptive_ycrcb": "main_modify_2.py adaptive window and YCrCb chroma correction enabled",
}


def read_rgb(path, resize=True):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.double) / 255.0
    if resize:
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return img


def save_rgb(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(path.suffix, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"failed to encode image: {path}")
    encoded.tofile(str(path))


def ssim_simple(a, b):
    a = np.asarray(a, dtype=np.double)
    b = np.asarray(b, dtype=np.double)
    scores = []
    for c in range(a.shape[2]):
        x, y = a[:, :, c], b[:, :, c]
        ux = cv2.GaussianBlur(x, (7, 7), 1.5)
        uy = cv2.GaussianBlur(y, (7, 7), 1.5)
        vx = cv2.GaussianBlur(x * x, (7, 7), 1.5) - ux * ux
        vy = cv2.GaussianBlur(y * y, (7, 7), 1.5) - uy * uy
        vxy = cv2.GaussianBlur(x * y, (7, 7), 1.5) - ux * uy
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * ux * uy + c1) * (2 * vxy + c2)) / (
            (ux * ux + uy * uy + c1) * (vx + vy + c2)
        )
        scores.append(float(np.mean(ssim_map)))
    return float(np.mean(scores))


def metrics(restored, clean, degraded):
    diff = restored - clean
    l2 = float(np.linalg.norm(diff.ravel()))
    mse = float(np.mean(diff * diff))
    psnr = float(99.0 if mse <= 1e-15 else 10.0 * np.log10(1.0 / mse))
    cosine = float(
        np.dot(restored.ravel(), clean.ravel())
        / (np.linalg.norm(restored.ravel()) * np.linalg.norm(clean.ravel()) + 1e-12)
    )
    known = degraded > 1e-12
    known_diff = 0.0
    if np.any(known):
        known_diff = float(np.max(np.abs(restored[known] - degraded[known])))
    return {
        "l2_error": l2,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim_simple(restored, clean),
        "cosine": cosine,
        "known_pixel_max_diff": known_diff,
        "finite": bool(np.isfinite(restored).all()),
    }


def telea_only(noise_img):
    out = np.copy(noise_img)
    for c in range(out.shape[2]):
        plane = out[:, :, c]
        missing = plane <= 1e-12
        if not np.any(missing):
            continue
        mask8 = missing.astype(np.uint8) * 255
        plane8 = np.clip(plane * 255.0, 0, 255).astype(np.uint8)
        out[:, :, c] = cv2.inpaint(plane8, mask8, 3, cv2.INPAINT_TELEA).astype(np.double) / 255.0
    out[noise_img > 1e-12] = noise_img[noise_img > 1e-12]
    return np.clip(out, 0.0, 1.0)


def local_mean_only(noise_img, radius=4):
    out = np.copy(noise_img)
    ksize = (2 * radius + 1, 2 * radius + 1)
    for c in range(out.shape[2]):
        plane = noise_img[:, :, c]
        missing = plane <= 1e-12
        if not np.any(missing):
            continue
        known = (~missing).astype(np.double)
        values = np.where(missing, 0.0, plane)
        count = cv2.boxFilter(known, -1, ksize, normalize=False, borderType=cv2.BORDER_REFLECT)
        total = cv2.boxFilter(values, -1, ksize, normalize=False, borderType=cv2.BORDER_REFLECT)
        global_mean = float(np.mean(plane[~missing])) if np.any(~missing) else 0.5
        mean = np.where(count > 0, total / np.maximum(count, 1.0), global_mean)
        out[:, :, c][missing] = mean[missing]
    return np.clip(out, 0.0, 1.0)


def load_modified_main_modify_2(name, disable_ycrcb=False, fixed_window=False):
    src = (ROOT / "main_modify_2.py").read_text(encoding="utf-8")

    if fixed_window:
        start = src.index("    def _preferred_radius(texture, mode):")
        end = src.index("    def _restore_plane(plane, missing, guide_texture, mode):")
        replacement = (
            "    def _preferred_radius(texture, mode):\n"
            "        return np.zeros(texture.shape, dtype=np.int32)\n\n"
        )
        src = src[:start] + replacement + src[end:]

    if disable_ycrcb:
        start = src.index("    rgb3 = np.clip(rgb_direct[:, :, :3], 0.0, 1.0)")
        end = src.index("    return res_img", start) + len("    return res_img")
        replacement = (
            "    res_img = np.clip(rgb_direct, 0.0, 1.0).astype(np.double)\n"
            "    res_img[~missing_rgb] = work_img[~missing_rgb]\n"
            "    return res_img"
        )
        src = src[:start] + replacement + src[end:]

    module = types.ModuleType(name)
    exec(compile(src, f"<{name}>", "exec"), module.__dict__)
    return module


def build_methods():
    main = importlib.import_module("main")
    main_modify = importlib.import_module("main_modify")
    main_modify_2 = importlib.import_module("main_modify_2")
    adaptive_rgb = load_modified_main_modify_2("adaptive_rgb_no_ycrcb", disable_ycrcb=True)
    fixed_ycc = load_modified_main_modify_2("fixed_window_ycrcb", fixed_window=True)

    return {
        "degraded": lambda x: np.copy(x),
        "telea_only": telea_only,
        "local_mean_only": lambda x: local_mean_only(x, SIZE),
        "linear_ridge_main": lambda x: main.restore_image(x, SIZE),
        "quadratic_gaussian_main_modify": lambda x: main_modify.restore_image(x, SIZE),
        "adaptive_rgb_no_ycrcb": lambda x: adaptive_rgb.restore_image(x, SIZE),
        "fixed_window_ycrcb": lambda x: fixed_ycc.restore_image(x, SIZE),
        "full_adaptive_ycrcb": lambda x: main_modify_2.restore_image(x, SIZE),
    }


def make_grid(case_dir, case_id, methods):
    imgs = []
    labels = ["clean"] + list(methods.keys())
    for label in labels:
        img_path = case_dir / f"{label}.png"
        data = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        tile = cv2.resize(img, (150, 150))
        cv2.putText(tile, label[:20], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(tile, label[:20], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        imgs.append(tile)
    grid = np.concatenate(imgs, axis=1)
    CMP_DIR.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", grid)
    if not ok:
        raise OSError(f"failed to encode comparison image: {case_id}")
    encoded.tofile(str(CMP_DIR / f"{case_id}_grid.png"))


def iter_cases(noise_func):
    clean_a = read_rgb(ROOT / "A.png")
    for ratio in RATIOS:
        for seed in SEEDS:
            np.random.seed(seed)
            degraded = noise_func(clean_a, ratio)
            ratio_id = "_".join(str(int(r * 10)).zfill(2) for r in ratio)
            yield f"A_ratio_{ratio_id}_seed_{seed}", clean_a, degraded, {
                "source": "A.png",
                "noise_ratio": ratio,
                "seed": seed,
                "generated_noise": True,
            }

    sample_pairs = [
        ("forest", "forest.png", "forest_random_noise.png"),
        ("mona_lisa", "mona_lisa.png", "mona_lisa_random_noise.png"),
        ("potala_palace", "potala_palace.png", "potala_palace_random_noise.png"),
        ("the_school_of_athens", "the_school_of_athens.png", "the_school_of_athens_random_noise.png"),
        ("xihu", "xihu.png", "xihu_random_noise.png"),
    ]
    for case_id, clean_name, noisy_name in sample_pairs:
        clean = read_rgb(ROOT / "samples" / clean_name)
        degraded = read_rgb(ROOT / "samples" / noisy_name)
        yield f"sample_{case_id}", clean, degraded, {
            "source": clean_name,
            "noise_source": noisy_name,
            "generated_noise": False,
        }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    methods = build_methods()
    noise_module = importlib.import_module("main_modify_2")
    rows = []
    case_params = []

    for case_id, clean, degraded, meta in iter_cases(noise_module.noise_mask_image):
        case_dir = IMG_DIR / case_id
        save_rgb(case_dir / "clean.png", clean)
        save_rgb(case_dir / "degraded.png", degraded)
        case_params.append({"case_id": case_id, **meta})

        for method_name, method in methods.items():
            start = time.perf_counter()
            restored = method(degraded)
            runtime = time.perf_counter() - start
            save_rgb(case_dir / f"{method_name}.png", restored)

            row = {
                "case_id": case_id,
                "method": method_name,
                "runtime_sec": runtime,
                **metrics(restored, clean, degraded),
                **meta,
            }
            rows.append(row)

        make_grid(case_dir, case_id, methods)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with (OUT_ROOT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    params = {
        "size": SIZE,
        "image_size": IMAGE_SIZE,
        "ratios": RATIOS,
        "seeds": SEEDS,
        "methods": list(methods.keys()),
        "method_configs": METHOD_CONFIGS,
        "cases": case_params,
        "output_dir": str(OUT_ROOT),
    }
    (OUT_ROOT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    def summarize(vals):
        return {
            "mean_l2": float(np.mean([r["l2_error"] for r in vals])),
            "mean_ssim": float(np.mean([r["ssim"] for r in vals])),
            "mean_cosine": float(np.mean([r["cosine"] for r in vals])),
            "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in vals])),
        }

    summary = {}
    for method in methods:
        vals = [r for r in rows if r["method"] == method]
        summary[method] = summarize(vals)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_by_group = {}
    for group_name, generated_noise in [("generated_mask_noise", True), ("sample_random_noise", False)]:
        summary_by_group[group_name] = {}
        for method in methods:
            vals = [
                r
                for r in rows
                if r["method"] == method and bool(r["generated_noise"]) == generated_noise
            ]
            summary_by_group[group_name][method] = summarize(vals)
    (OUT_ROOT / "summary_by_group.json").write_text(
        json.dumps(summary_by_group, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Ablation output:", OUT_ROOT)
    for method, item in summary.items():
        print(method, item)
    print("Generated mask noise summary:")
    for method, item in summary_by_group["generated_mask_noise"].items():
        print(method, item)


if __name__ == "__main__":
    main()
