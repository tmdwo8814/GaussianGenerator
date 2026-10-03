"""Memory optimization against the previous full-batch appearance flow."""

import copy
import unittest
from unittest.mock import patch

import torch

from test_appearance_capacity import cfg, activate, neighborhood, selection
from test_moment_gaussian_decoder import Decoder, attributes, moment


def batch_reference(model, points, features, rgb, image_shape, diagnostics=None):
    """Pre-optimization flow: pool/read all scenes, stack features, then SH.

    Uses the separately tested reader/math primitives, but deliberately does
    not call the new scene checkpoint or streaming forward path.
    """
    moments, selected_features, contexts, reports = [], [], [], []
    image = neighborhood.make_image_neighborhood(image_shape, model.cfg.appearance_2d_radii, points.device)
    with torch.autocast(device_type=points.device.type, enabled=False):
        points, features = points.float(), features.float()
        scales = points.detach().norm(dim=-1).median(dim=-1).values.clamp_min(model.cfg.scene_epsilon)
        for index, (p, f, scale) in enumerate(zip(points, features, scales)):
            neighbors = moment.build_knn(p, model.cfg.num_neighbors, model.cfg.knn_workers,
                                         model.cfg.knn_backend, check_finite=False,
                                         query_backend=model.cfg.knn_query_backend)
            p = p / scale
            q, scores = model.predict_allocation(p, f, neighbors, return_appearance=True)
            *values, geometry = model.aggregate_moments(p, f, neighbors, q, return_weights=True)
            selected = selection.appearance_weights(geometry, scores, neighbors)
            selected_features.append(moment.pool_features(f, selected, neighbors, model.cfg.chunk_size))
            context, report = model.appearance_reader(
                p, f, rgb[index].float(), values[1], values[3], neighbors, image,
                chunk_size=model.cfg.chunk_size, checkpoint_chunks=model.cfg.checkpoint_chunks,
                collect_statistics=diagnostics is not None,
            )
            contexts.append(context)
            moments.append(values)
            if report is not None:
                report.update(selection.appearance_weight_statistics(geometry, selected, neighbors))
                reports.append(report)
        result = model.build_gaussians(
            *[torch.stack(values) for values in zip(*moments)], scales,
            appearance_context=torch.stack(contexts), appearance_pooled=torch.stack(selected_features),
            diagnostics=diagnostics,
        )
        if diagnostics is not None:
            diagnostics.update({key: torch.stack([report[key] for report in reports]).mean()
                                for key in reports[0]})
        return result


def saved_storage_bytes(model, inputs, forward):
    """Unique storage saved by autograd; excludes pre-existing inputs/weights.

    This is a CPU-compatible retention check, NOT CUDA peak allocated memory.
    Views sharing storage count once; ordinary Python lists/temporary tensors
    and the backbone/renderer's memory are not measured by these hooks.
    """
    def key(tensor):
        return (str(tensor.device), tensor.untyped_storage().data_ptr())

    excluded = {key(value) for value in (*inputs, *model.parameters(), *model.buffers())}
    saved = {}

    def pack(tensor):
        identity = key(tensor)
        if tensor.numel() and identity not in excluded:
            saved[identity] = tensor.untyped_storage().nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = forward()
    # Keep the graph alive until all forward saves have been counted.
    return sum(saved.values()), output


class CapacityMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(397)
        torch.set_num_threads(1)

    def test_streamed_values_gradients_and_metrics_match_previous_batch_flow(self):
        for heads in (2, 4):
            with self.subTest(heads=heads):
                model = Decoder(cfg(appearance_heads=heads))
                activate(model)
                # Nonconstant coverage verifies per-scene geometry heads too.
                torch.nn.init.normal_(model.scale_head.weight, std=.1)
                reference = copy.deepcopy(model)
                reference.load_state_dict(model.state_dict(), strict=True)
                inputs = [torch.randn(3, 12, d, requires_grad=True) for d in (3, 5, 3)]
                other = [x.detach().clone().requires_grad_() for x in inputs]
                report, old_report = {}, {}
                actual = model(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3), diagnostics=report)
                expected = batch_reference(reference, *other, (2, 2, 3), diagnostics=old_report)
                self.assertEqual(set(report), set(old_report))
                for key in report:
                    self.assertFalse(report[key].requires_grad)
                    torch.testing.assert_close(report[key], old_report[key], rtol=8e-5, atol=8e-6)
                signals = [torch.randn_like(x) for x in attributes(actual)]
                for a, b in zip(attributes(actual), attributes(expected)):
                    torch.testing.assert_close(a, b, rtol=4e-5, atol=4e-6)
                sum((x * s).sum() for x, s in zip(attributes(actual), signals)).backward()
                sum((x * s).sum() for x, s in zip(attributes(expected), signals)).backward()
                for a, b in [*zip(inputs, other), *zip(model.parameters(), reference.parameters())]:
                    self.assertIsNotNone(a.grad)
                    self.assertTrue(torch.isfinite(a.grad).all())
                    torch.testing.assert_close(a.grad, b.grad, rtol=8e-4, atol=2e-5)

    def test_checkpoint_switch_and_inference_preserve_function_without_new_knn(self):
        model = Decoder(cfg(appearance_heads=2))
        activate(model)
        reference = copy.deepcopy(model)
        reference.cfg.checkpoint_appearance = False
        inputs = [torch.randn(2, 12, d, requires_grad=True) for d in (3, 5, 3)]
        other = [x.detach().clone().requires_grad_() for x in inputs]
        with patch.object(moment, 'build_knn', wraps=moment.build_knn) as search:
            actual = model(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3))
            sum(x.square().mean() for x in attributes(actual)).backward()
            self.assertEqual(search.call_count, 2)  # Includes backward recomputation.
        expected = reference(other[0], other[1], rgb=other[2], image_shape=(2, 2, 3))
        sum(x.square().mean() for x in attributes(expected)).backward()
        for a, b in [*zip(inputs, other), *zip(model.parameters(), reference.parameters())]:
            torch.testing.assert_close(a.grad, b.grad, rtol=8e-4, atol=8e-6)
        model.eval()
        with torch.no_grad():
            inference = model(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3))
        for a, b, c in zip(attributes(actual), attributes(expected), attributes(inference)):
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(a, c)

    def test_actual_widths_save_less_autograd_storage(self):
        model = Decoder(cfg(feature_dim=256, hidden_dim=64, appearance_heads=2,
                            appearance_dim=32, appearance_mlp_dim=128, num_neighbors=16,
                            chunk_size=64))
        activate(model)
        inputs = [torch.randn(2, 512, d, requires_grad=True) for d in (3, 256, 3)]
        new_bytes, actual = saved_storage_bytes(model, inputs, lambda: model(
            inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 16, 16),
        ))
        old_bytes, expected = saved_storage_bytes(model, inputs, lambda: batch_reference(
            model, *inputs, (2, 16, 16),
        ))
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-6)
        self.assertLess(new_bytes, old_bytes * .85,
                        f'Expected meaningful retention reduction: new={new_bytes}, old={old_bytes}')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for device comparison')
    def test_cuda_legacy_and_streamed_values_and_gradients(self):
        model = Decoder(cfg(appearance_heads=2)).cuda()
        activate(model)
        reference = copy.deepcopy(model)
        inputs = [torch.randn(2, 12, d, device='cuda', requires_grad=True) for d in (3, 5, 3)]
        other = [x.detach().clone().requires_grad_() for x in inputs]
        actual = model(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3))
        expected = batch_reference(reference, *other, (2, 2, 3))
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=8e-4, atol=8e-5)
        sum(x.square().mean() for x in attributes(actual)).backward()
        sum(x.square().mean() for x in attributes(expected)).backward()
        for a, b in [*zip(inputs, other), *zip(model.parameters(), reference.parameters())]:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-3, atol=2e-4)


if __name__ == '__main__':
    unittest.main()
