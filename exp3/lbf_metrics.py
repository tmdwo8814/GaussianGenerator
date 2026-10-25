"""Full-population LBF measurements; no renderer, training, or target RGB."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from exp3.scale_opacity import PrioritySample


SCHEMA = 1
ACTIVE = 1 / 255
ALPHAS = (0.05, 0.10, 0.25)
PERCENTILES = (80, 90, 95)
FATES = ("inactive", "invalid_reference", "smaller_only", "more_opaque_only",
         "smaller_and_more_opaque", "remains_lbf")


def manifest_hash(scenes):
    return hashlib.sha256(json.dumps(scenes, ensure_ascii=False).encode()).hexdigest()


def save_npz(path, **arrays):
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def renderer_focals(intrinsics, image_shape):
    """Pixel focal implied by this repo's get_fov -> symmetric projection path."""
    inverse = np.linalg.inv(np.asarray(intrinsics, dtype=np.float64))
    rays = np.array([[0, 0.5, 1], [1, 0.5, 1], [0.5, 0, 1], [0.5, 1, 1]])
    directions = rays[None] @ inverse.transpose(0, 2, 1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    cosine = np.stack(((directions[:, 0] * directions[:, 1]).sum(-1),
                       (directions[:, 2] * directions[:, 3]).sum(-1)), axis=-1)
    fov = np.arccos(np.clip(cosine, -1, 1))
    height, width = image_shape
    focal = np.array([width, height]) / (2 * np.tan(fov / 2))
    if not np.isfinite(focal).all() or np.any(focal <= 0):
        raise ValueError("Invalid renderer focal length")
    return focal


def projected_geometry(means, covariance, opacity, log_s_max, extrinsics, intrinsics, image_shape):
    """Origin-view s_max*fx/z and unfiltered perspective-Jacobian ellipse area.

    The reference camera is the slot's source view, even when its mean moves.
    A nonpositive origin depth is excluded and counted; no camera reassignment.
    Area is potential 1-sigma ellipse area, not visible union/composited coverage.
    """
    means, covariance = np.asarray(means, dtype=np.float64), np.asarray(covariance, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    height, width = image_shape
    views, count = len(extrinsics), len(means)
    if (means.shape != (count, 3) or covariance.shape != (count, 3, 3)
            or opacity.shape != (count,) or count != views * height * width
            or intrinsics.shape != (views, 3, 3)):
        raise ValueError("Expected view-major, one-slot-per-pixel Gaussian/camera tensors")
    if not all(np.isfinite(a).all() for a in (means, covariance, opacity, extrinsics, intrinsics)):
        raise ValueError("Non-finite Gaussian/camera inputs")
    if np.any((opacity < 0) | (opacity > 1)):
        raise ValueError("Invalid opacity")
    active = opacity >= ACTIVE
    if np.asarray(log_s_max).shape != (int(active.sum()),):
        raise ValueError("Supply world log scales for all active slots, in slot order")
    world_scale = np.full(count, np.nan)
    world_scale[active] = np.power(10.0, log_s_max)
    if not np.isfinite(world_scale[active]).all() or np.any(world_scale[active] <= 0):
        raise ValueError("Invalid active world scales")
    w2c = np.linalg.inv(extrinsics)
    focal = renderer_focals(intrinsics, image_shape)
    ref_cam = np.arange(count) // (height * width)
    positions = np.empty((count, 3), dtype=np.float64)
    all_behind = np.ones(count, dtype=bool)
    for view in range(views):
        camera_points = means @ w2c[view, :3, :3].T + w2c[view, :3, 3]
        all_behind &= camera_points[:, 2] <= 0
        own = ref_cam == view
        positions[own] = camera_points[own]
    valid = positions[:, 2] > 0
    eligible = active & valid
    s_px, area = np.full(count, np.nan), np.full(count, np.nan)
    for view in range(views):
        own = eligible & (ref_cam == view)
        points = positions[own]
        z = points[:, 2]
        fx, fy = focal[view]
        s_px[own] = world_scale[own] * fx / z
        rotation = w2c[view, :3, :3]
        camera_cov = rotation @ covariance[own] @ rotation.T
        jacobian = np.zeros((len(points), 2, 3))
        jacobian[:, 0, 0], jacobian[:, 1, 1] = fx / z, fy / z
        jacobian[:, 0, 2] = -fx * points[:, 0] / z**2
        jacobian[:, 1, 2] = -fy * points[:, 1] / z**2
        screen_cov = jacobian @ camera_cov @ jacobian.transpose(0, 2, 1)
        eigenvalues = np.linalg.eigvalsh((screen_cov + screen_cov.transpose(0, 2, 1)) * 0.5)
        tolerance = 32 * np.finfo(np.float32).eps * np.maximum(
            np.abs(eigenvalues).max(axis=1), np.finfo(np.float64).tiny,
        )
        if np.any(eigenvalues[:, 0] < -tolerance):
            raise ValueError("Projected covariance is not positive semidefinite")
        area[own] = np.pi * np.sqrt(np.maximum(eigenvalues[:, 0], 0)) * np.sqrt(
            np.maximum(eigenvalues[:, 1], 0))
    if not np.isfinite(s_px[eligible]).all() or not np.isfinite(area[eligible]).all():
        raise ValueError("Non-finite projected size/area")
    return {"s_world": world_scale, "s_px": s_px, "area_px": area, "opacity": opacity,
            "valid_ref": valid, "all_behind": all_behind, "z_ref": positions[:, 2],
            "f_ref": focal[ref_cam, 0], "ref_cam": ref_cam.astype(np.int16)}


def eligible_mask(data, space):
    active = data["opacity"] >= ACTIVE
    return active & data["valid_ref"] if space == "projected" else active


def size_values(data, space):
    return data["s_px"] if space == "projected" else data["s_world"]


def cache_path(directory, name, index):
    return Path(directory) / "lbf_cache" / name / f"{index:06d}.npz"


class LBFCollector:
    def __init__(self, directory, name, scenes, seed, capacity):
        self.directory, self.name, self.scenes, self.seed = Path(directory), name, scenes, seed
        (self.directory / "lbf_cache" / name).mkdir(parents=True, exist_ok=True)
        self.sampler = PrioritySample(capacity)
        self.n_projected = self.n_world = self.seen = 0

    def add(self, index, scene, means, covariance, opacity, log_s_max, extrinsics, intrinsics, image_shape):
        data = projected_geometry(means, covariance, opacity, log_s_max,
                                  extrinsics, intrinsics, image_shape)
        save_npz(cache_path(self.directory, self.name, index), **data, scene=np.array(scene),
                 camera_extrinsics=extrinsics, camera_intrinsics=intrinsics,
                 image_shape=np.array(image_shape))
        mask = eligible_mask(data, "projected")
        digest = hashlib.sha256(f"{self.seed}:{scene}".encode()).digest()
        priority = np.random.default_rng(int.from_bytes(digest[:8], "little")).random(len(opacity))
        self.sampler.add(priority[mask], log_s_max=np.log10(data["s_px"][mask]),
                         opacity=data["opacity"][mask], scene_index=np.full(int(mask.sum()), index),
                         gaussian_index=np.flatnonzero(mask))
        self.n_projected += int(mask.sum())
        self.n_world += int((opacity >= ACTIVE).sum())
        self.seen += 1

    def finish(self):
        if self.seen != len(self.scenes):
            raise ValueError("Incomplete LBF collection")
        arrays = self.sampler.arrays()
        arrays["log_s_px"] = arrays.pop("log_s_max")
        (self.directory / "lbf_samples").mkdir(exist_ok=True)
        save_npz(self.directory / "lbf_samples" / f"{self.name}.npz", **arrays)
        marker = {"schema": SCHEMA, "manifest_hash": manifest_hash(self.scenes),
                  "n_scenes": self.seen, "N_projected": self.n_projected, "N_world": self.n_world,
                  "sample_size": self.sampler.capacity}
        (self.directory / "lbf_cache" / self.name / "complete.json").write_text(
            json.dumps(marker, indent=2), encoding="utf-8")


def validate_cache(directory, name, scenes):
    marker = json.loads((Path(directory) / "lbf_cache" / name / "complete.json").read_text(encoding="utf-8"))
    if (marker["schema"] != SCHEMA or marker["manifest_hash"] != manifest_hash(scenes)
            or marker["n_scenes"] != len(scenes)):
        raise ValueError(f"LBF cache protocol/scene mismatch for {name}")
    for index in range(len(scenes)):
        if not cache_path(directory, name, index).is_file():
            raise FileNotFoundError(cache_path(directory, name, index))
    with np.load(Path(directory) / "lbf_samples" / f"{name}.npz", allow_pickle=False) as data:
        count = min(marker["sample_size"], marker["N_projected"])
        for key in ("log_s_px", "opacity", "scene_index", "gaussian_index"):
            if data[key].shape != (count,) or not np.isfinite(data[key]).all():
                raise ValueError(f"Invalid LBF plot cache for {name}: {key}")
    return marker


def exact_percentiles(directory, scenes, space, count):
    """Exact linear quantiles of ALL baseline values; disk-backed partition."""
    if count < 1:
        raise ValueError(f"No eligible baseline Gaussians in {space} space")
    path = Path(directory) / "results" / f"percentile_work_{space}.npy"
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(count,))
    offset = 0
    try:
        for index in range(len(scenes)):
            with np.load(cache_path(directory, "original", index), allow_pickle=False) as data:
                selected = size_values(data, space)[eligible_mask(data, space)]
                if not np.isfinite(selected).all() or np.any(selected <= 0):
                    raise ValueError("Invalid baseline percentile population")
                if offset + len(selected) > count:
                    raise ValueError("Baseline cache count mismatch")
                values[offset:offset + len(selected)] = selected
                offset += len(selected)
        if offset != count:
            raise ValueError("Baseline cache count mismatch")
        positions = (count - 1) * np.array(PERCENTILES) / 100
        lower, upper = np.floor(positions).astype(int), np.ceil(positions).astype(int)
        values.partition(np.unique(np.concatenate((lower, upper))))
        quantiles = values[lower] + (positions - lower) * (values[upper] - values[lower])
        return {f"P{p}": float(v) for p, v in zip(PERCENTILES, quantiles)}
    finally:
        del values
        path.unlink(missing_ok=True)


def freeze_thresholds(directory, scenes):
    directory = Path(directory)
    marker = validate_cache(directory, "original", scenes)
    result_dir = directory / "results"
    result_dir.mkdir(exist_ok=True)
    path = result_dir / "lbf_thresholds.json"
    if path.exists():
        thresholds = json.loads(path.read_text(encoding="utf-8"))
        if thresholds["baseline_cache"] != marker or thresholds["schema"] != SCHEMA:
            raise ValueError("Frozen LBF thresholds do not match baseline cache")
        return thresholds
    thresholds = {"schema": SCHEMA, "baseline_cache": marker, "source": "original only; full population",
                  "alpha_main": 0.25, "active_cutoff": ACTIVE,
                  "reference_camera": "origin context view; exclude nonpositive origin depth",
                  "focal": "horizontal pixel focal implied by repo get_fov",
                  "area": "pi*sqrt(det(J R Sigma R^T J^T)); unfiltered 1-sigma ellipse",
                  "projected": exact_percentiles(directory, scenes, "projected", marker["N_projected"]),
                  "world": exact_percentiles(directory, scenes, "world", marker["N_world"])}
    thresholds["projected"].update({"4px": 4.0, "8px": 8.0, "16px": 16.0})
    path.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    return thresholds


def scene_statistics(data, space, threshold, alpha, compute_diagnostics=True):
    mask = eligible_mask(data, space)
    sizes, opacity = size_values(data, space)[mask], data["opacity"][mask]
    if not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("Invalid selected sizes in LBF cache")
    large = sizes > threshold
    lbf = large & (opacity < alpha)
    result = {"N_analysis": len(sizes), "N_large": int(large.sum()), "N_lbf": int(lbf.sum()),
              "median_size": float(np.median(sizes)) if len(sizes) and compute_diagnostics else np.nan,
              "spearman_rho": np.nan, "area_total": np.nan, "area_lbf": np.nan}
    if compute_diagnostics and len(sizes) > 1 and np.ptp(sizes) > 0 and np.ptp(opacity) > 0:
        result["spearman_rho"] = float(spearmanr(np.log10(sizes), opacity)[0])
    if space == "projected":
        area = data["area_px"][mask]
        result.update(area_total=float(area.sum()), area_lbf=float(area[lbf].sum()))
    return result


def transition_counts(original, ours, threshold, alpha=0.25):
    if original["opacity"].shape != ours["opacity"].shape:
        raise ValueError("Origin-slot layouts do not match")
    baseline = (eligible_mask(original, "projected") & (original["s_px"] > threshold)
                & (original["opacity"] < alpha))
    active = ours["opacity"] >= ACTIVE
    valid = ours["valid_ref"]
    if not np.isfinite(ours["s_px"][active & valid]).all():
        raise ValueError("Invalid projected values in slot transitions")
    small = ours["s_px"] <= threshold
    opaque = ours["opacity"] >= alpha
    masks = (baseline & ~active, baseline & active & ~valid,
             baseline & active & valid & small & ~opaque,
             baseline & active & valid & ~small & opaque,
             baseline & active & valid & small & opaque,
             baseline & active & valid & ~small & ~opaque)
    result = {key: int(mask.sum()) for key, mask in zip(FATES, masks)}
    result["N_original_lbf"] = int(baseline.sum())
    if sum(result[key] for key in FATES) != result["N_original_lbf"]:
        raise ValueError("Invalid/missing projected values in slot transitions")
    return result
