"""Recover context cameras from cached point/pixel correspondences.

Dataset chunks supply only original context-frame intrinsics and clipping-scale
metadata. No context images are decoded and no encoder or target-camera fitting
is run. PnP fits cameras to original support maps in the cached canonical frame.
"""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch


def crop_intrinsics(intrinsics, original_shape, output_shape):
    """Match this repo's rescale_and_crop/center_crop, including rounded sizes."""
    h, w = original_shape
    oh, ow = output_shape
    scale = max(oh / h, ow / w)
    result = intrinsics.clone()
    result[..., 0, 0] *= round(w * scale) / ow
    result[..., 1, 1] *= round(h * scale) / oh
    return result


@lru_cache(maxsize=1)
def chunk_camera_metadata(path):
    try:
        chunk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError as error:
        if 'mmap' not in str(error).lower():
            raise
        chunk = torch.load(path, map_location='cpu', weights_only=True)
    result = {item['key']: item['cameras'].clone() for item in chunk}
    del chunk
    return result


class ContextCalibration:
    def __init__(self, manifest, dataset_roots=None):
        self.cfg = manifest['identity']['dataset']['re10k']
        self.index = {}
        roots = dataset_roots or self.cfg['roots']
        for root in roots:
            base = Path(root).expanduser().resolve() / 'test'
            entries = json.loads((base / 'index.json').read_text(encoding='utf-8'))
            for scene, relative in entries.items():
                if scene in self.index:
                    raise ValueError(f'Duplicate RE10K scene: {scene}')
                self.index[scene] = base / relative

    def intrinsics_and_bounds(self, payload):
        scene = payload['scene']
        rows = chunk_camera_metadata(str(self.index[scene]))[scene][payload['context_indices']].float()
        intrinsic = torch.eye(3).repeat(len(rows), 1, 1)
        intrinsic[:, 0, 0], intrinsic[:, 1, 1] = rows[:, 0], rows[:, 1]
        intrinsic[:, 0, 2], intrinsic[:, 1, 2] = rows[:, 2], rows[:, 3]
        shape = payload['support']['image_shape'][1:]
        intrinsic = crop_intrinsics(intrinsic, self.cfg['original_image_shape'], shape)
        scale = 1.
        if self.cfg.get('make_baseline_1', True):
            w2c = torch.eye(4).repeat(len(rows), 1, 1)
            w2c[:, :3] = rows[:, 6:].reshape(-1, 3, 4)
            centers = w2c.inverse()[:, :3, 3]
            scale = float((centers[0] - centers[-1]).norm())
            if scale <= 0:
                raise ValueError(f'Invalid context baseline for {scene}')
        return intrinsic, torch.full((len(rows),), .1 / scale), torch.full((len(rows),), 100. / scale)


def fit_context_cameras(payload, intrinsics, near, far, seed=111123, max_points=8192):
    import cv2
    cv2.setNumThreads(1)
    cv2.setRNGSeed(int(seed) % (2**31 - 1))
    views, height, width = payload['support']['image_shape']
    points = payload['support']['points'].cpu().double().numpy().copy()
    points *= float(payload['support']['scene_scale'])
    points = points.reshape(views, height * width, 3)
    y, x = np.mgrid[:height, :width]
    pixels = np.stack((x, y), -1).reshape(-1, 2).astype(np.float64)
    poses, reports = [], []
    for view in range(views):
        xyz = points[view]
        valid = np.flatnonzero(np.isfinite(xyz).all(-1) & (np.linalg.norm(xyz, axis=-1) > 1e-9))
        if len(valid) > max_points:
            valid = valid[np.linspace(0, len(valid) - 1, max_points).astype(np.int64)]
        if len(valid) < 6:
            raise ValueError('Too few finite support/pixel correspondences for PnP')
        k = intrinsics[view].double().numpy().copy()
        k[0] *= width
        k[1] *= height
        xyz, uv = np.ascontiguousarray(xyz[valid]), np.ascontiguousarray(pixels[valid])
        success, rotation, translation, inliers = cv2.solvePnPRansac(
            xyz, uv, k, None, iterationsCount=100, reprojectionError=5., flags=cv2.SOLVEPNP_SQPNP,
        )
        if not success or inliers is None or len(inliers) < 6:
            raise RuntimeError(f'Context PnP failed for {payload["scene"]}, view {view}')
        ids = inliers[:, 0]
        if hasattr(cv2, 'solvePnPRefineLM'):
            rotation, translation = cv2.solvePnPRefineLM(xyz[ids], uv[ids], k, None, rotation, translation)
        projection, _ = cv2.projectPoints(xyz[ids], rotation, translation, k, None)
        error = np.linalg.norm(projection.reshape(-1, 2) - uv[ids], axis=-1)
        w2c = np.eye(4)
        w2c[:3, :3] = cv2.Rodrigues(rotation)[0]
        w2c[:3, 3] = translation[:, 0]
        if not np.isfinite(w2c).all() or np.median(error) > 5:
            raise RuntimeError('Invalid context-camera reprojection')
        poses.append(np.linalg.inv(w2c))
        reports.append({'sampled_points': len(xyz), 'inliers': len(ids),
                        'median_reprojection_px': float(np.median(error))})
    return {'extrinsics': torch.tensor(np.stack(poses), dtype=torch.float32),
            'intrinsics': intrinsics.float(), 'near': near, 'far': far}, reports
