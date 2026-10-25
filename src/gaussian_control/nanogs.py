"""In-memory adapter for the pinned, unmodified official CPU NanoGS kernels."""

import numpy as np
import torch

from ..model.types import Gaussians
from ..third_party.nanogs.simplification import (
    edge_costs, greedy_pairs_from_edges, knn_indices, knn_undirected_edges,
)
from ..third_party.nanogs.utils.merge import merge_pairs
from ..third_party.nanogs.utils.params import CostParams
from ..third_party.nanogs.utils.splat_utils import rotmat_to_quat_batch, sigma_from_scale_quat_batch


UPSTREAM_COMMIT = 'cc64a752b5def8d17f9992216bfbb9e533f8364f'


def to_nanogs(gaussians):
    if gaussians.means.shape[0] != 1:
        raise ValueError('NanoGS adapter expects one scene')
    def array(tensor):
        return tensor.detach().float().cpu().numpy().copy()
    mu, op = array(gaussians.means[0]), array(gaussians.opacities[0])
    covariance = array(gaussians.covariances[0])
    values, rotation = np.linalg.eigh(.5 * (covariance + covariance.transpose(0, 2, 1)))
    order = np.argsort(values, axis=1)[:, ::-1]
    values = np.take_along_axis(values, order, axis=1)
    rotation = np.take_along_axis(rotation, order[:, None, :], axis=2)
    rotation[np.linalg.det(rotation) < 0, :, 2] *= -1
    scales = np.sqrt(np.maximum(values, 1e-18)).astype(np.float32)
    quaternion = rotmat_to_quat_batch(rotation.astype(np.float32))
    sh = array(gaussians.harmonics[0]).reshape(len(mu), -1)
    return mu, scales, quaternion, op, sh


@torch.no_grad()
def merge(gaussians, count, k=16):
    """No opacity pre-prune; exact integer budget, native cost/operator otherwise.

    Conversion, host/device transfers, graph construction and all merge passes
    are inside this function and therefore inside the reported wall time.
    """
    device = gaussians.means.device
    mu, sc, quat, op, sh = to_nanogs(gaussians)
    original = len(mu)
    if not 1 <= count <= original:
        raise ValueError('Invalid merge target')
    cost = CostParams(lam_geo=1.0, lam_sh=1.0)  # native MC=1, seed=0
    cap = max(1, int(.5 * original))
    passes = 0
    while len(mu) > count:
        neighbors = knn_indices(mu, k=min(k, len(mu) - 1))
        edges = knn_undirected_edges(neighbors)
        costs = edge_costs(edges, mu, sc, quat, op, sh, cost)
        pairs = greedy_pairs_from_edges(edges, costs, len(mu), min(len(mu) - count, cap))
        if not len(pairs):
            raise RuntimeError('NanoGS has no valid merge; refusing an incorrect primitive budget')
        mu, sc, quat, op, sh = merge_pairs(mu, sc, quat, op, sh, pairs)
        passes += 1
    def tensor(array):
        return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=torch.float32)[None]
    result = Gaussians(
        means=tensor(mu), covariances=tensor(sigma_from_scale_quat_batch(sc, quat)),
        harmonics=tensor(sh.reshape(count, *gaussians.harmonics.shape[2:])),
        opacities=tensor(np.clip(op, 0, 1)),
    )
    return result, {'merge_passes': passes, 'backend': 'official_numpy_scipy_cpu'}
