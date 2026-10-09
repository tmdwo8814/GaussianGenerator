"""Slot selection, budget/gradient isolation, and real rasterizer compaction."""

import ast
import copy
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from einops import rearrange, repeat

from test_moment_gaussian_decoder import ROOT, Cfg, Decoder, attributes, moment

control = importlib.import_module('_moment_test.encoder.common.slot_control')
Gaussians = moment.Gaussians


def controller(**kwargs):
    return control.SlotController(
        5, control.SlotControlCfg(enabled=True, warmup_steps=10, ramp_steps=10, **kwargs),
        chunk_size=2, epsilon=1e-6, checkpoint_chunks=True, compile_kernels=False,
    )


def graph():
    return torch.tensor([[0, 1, 2], [1, 0, 2], [2, 0, 1]])


def method(path, cls_name, name, namespace):
    cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
               if isinstance(n, ast.ClassDef) and n.name == cls_name)
    func = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    # Renderer uses runtime jaxtyping annotations, unrelated to this CPU test.
    func.returns = None
    for arg in (*func.args.args, *func.args.kwonlyargs):
        arg.annotation = None
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class SlotControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_redistribution_conserves_supports_and_matches_restricted_softmax(self):
        neighbors = graph()
        logits = torch.randn(3, 3, requires_grad=True)
        budgets = torch.tensor([[.8], [.3], [1.2]], requires_grad=True)
        q = logits.softmax(-1) * budgets
        gate = torch.tensor([1., 0., 1.], requires_grad=True)
        allocated = control.redistribute_mass(q, neighbors, gate)
        expected = budgets * logits.masked_fill(gate.detach()[neighbors] == 0, -torch.inf).softmax(-1)
        torch.testing.assert_close(allocated, expected)
        torch.testing.assert_close(allocated.sum(-1), budgets[:, 0])
        self.assertEqual(allocated[neighbors == 1].abs().sum().item(), 0)
        # Closed slot 1 remains a SOURCE of its complete budget.
        self.assertGreater(allocated[1].sum().item(), 0)
        torch.testing.assert_close(control.redistribute_mass(q, neighbors, torch.ones(3)), q,
                                   rtol=0, atol=0)
        (allocated * torch.randn_like(allocated)).sum().backward()
        for gradient in (logits.grad, budgets.grad, gate.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_coverage_repair_is_global_and_accounts_for_zero_mass_edges(self):
        neighbors = graph()
        p = torch.tensor([.1, .4, .2])
        opened = control.repair_coverage(p > .5, p, neighbors, torch.ones(3, 3))
        self.assertEqual(opened.tolist(), [False, True, False])
        self.assertTrue(opened[neighbors].any(-1).all())
        # A selected zero-probability edge cannot absorb the whole budget.
        q = torch.eye(3)  # one positive edge per row, indices vary with the graph
        opened = control.repair_coverage(torch.tensor([False, True, False]), p, neighbors, q)
        self.assertTrue((opened[neighbors] & (q > 0)).any(-1).all())
        out = control.redistribute_mass(q, neighbors, opened.float())
        torch.testing.assert_close(out.sum(-1), q.sum(-1))
        out = control.redistribute_mass(torch.zeros(3, 3), neighbors, opened.float())
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.sum().item(), 0)

    def test_budget_gradient_trains_controller_not_original_features_or_mass(self):
        model = controller()
        with torch.no_grad():
            model.gate[-1].weight.normal_(std=.02)
        points = torch.randn(3, 3, requires_grad=True)
        features = torch.randn(3, 5, requires_grad=True)
        q = torch.rand(3, 3, requires_grad=True)
        _, mask, state = model(points, features, graph(), q, global_step=20)
        state['scene_count'] = torch.tensor(1.)
        loss, logs = model.budget_loss(state, 20)
        loss.backward()
        self.assertIsNone(points.grad)
        self.assertIsNone(features.grad)
        self.assertIsNone(q.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertGreater(model.projection.weight.grad.abs().sum().item(), 0)
        self.assertEqual(logs['gaussians_after'].item(), mask.sum().item())
        self.assertAlmostEqual(logs['active_fraction'].item() + logs['removed_fraction'].item(), 1)

    def test_warmup_and_ramp_preserve_identity_then_enable_the_budget(self):
        model = controller()
        points, features, q = torch.randn(3, 3), torch.randn(3, 5), torch.rand(3, 3)
        for step, coefficient in ((0, 0), (10, 0), (15, .005), (20, .01)):
            output, mask, state = model(points, features, graph(), q, global_step=step)
            torch.testing.assert_close(output, q, rtol=0, atol=0)
            self.assertTrue(mask.all())
            state['scene_count'] = torch.tensor(1.)
            loss, logs = model.budget_loss(state, step)
            self.assertAlmostEqual(logs['budget_weight'], coefficient)
            self.assertAlmostEqual(loss.item(), coefficient * .25**2)
        self.assertEqual(model.progress(None), 1.)
        self.assertEqual(model.progress(100000), 1.)
        # Warmup still connects every gate parameter, with zero gradients, for DDP.
        model.zero_grad(set_to_none=True)
        _, _, state = model(points, features, graph(), q, global_step=0)
        state['scene_count'] = torch.tensor(1.)
        model.budget_loss(state, 0)[0].backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.abs().sum().item(), 0)

    def test_count_penalty_stops_at_budget_and_counts_repaired_slots(self):
        model = controller()
        active = torch.tensor(7., requires_grad=True)
        loss, logs = model.budget_loss(dict(active_count=active, candidate_count=torch.tensor(10.),
                                          scene_count=torch.tensor(1.)), 20)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(active.grad.item(), 0)
        p = torch.tensor([.1, .4, .2], requires_grad=True)
        with patch.object(model, '_probabilities', return_value=p):
            q, opened, state = model(torch.randn(3, 3), torch.randn(3, 5), graph(), torch.ones(3, 3),
                                     global_step=20)
        self.assertEqual(state['active_count'].item(), 1)
        self.assertEqual(state['raw_active_fraction'].item(), 0)
        self.assertAlmostEqual(state['fallback_fraction'].item(), 1/3)
        self.assertEqual(q[~opened[graph()]].abs().sum().item(), 0)
        torch.testing.assert_close(q.sum(-1), torch.full((3,), 3.))

    def test_real_controller_checkpoint_backward_and_inference_use_same_slots(self):
        config = Cfg(feature_dim=5, hidden_dim=8, num_neighbors=3, chunk_size=4,
                     appearance_2d=True, separate_appearance=True, appearance_capacity=True,
                     appearance_heads=2, appearance_dim=3, appearance_mlp_dim=7,
                     slot_control=control.SlotControlCfg(enabled=True, warmup_steps=10, ramp_steps=10))
        checked = Decoder(config)
        with torch.no_grad():
            checked.slot_controller.gate[-1].weight.normal_(std=.5)
            checked.slot_controller.gate[-1].bias.zero_()
            checked.appearance_sh_head[-1].weight.normal_(std=.05)
        reference = copy.deepcopy(checked)
        reference.cfg.checkpoint_appearance = False
        reference.cfg.checkpoint_chunks = False
        reference.slot_controller.checkpoint_chunks = False
        source = [torch.randn(2, 12, d) for d in (3, 5, 3)]
        outputs, gradients = [], []
        for decoder in (checked, reference):
            inputs = [x.clone().requires_grad_() for x in source]
            state = {}
            output = decoder(inputs[0], inputs[1], image_shape=(2, 2, 3), rgb=inputs[2],
                             global_step=20, control_dump=state)
            active_loss = sum(x.square().sum() for i in range(2)
                              for x in attributes(output.compact_scene(i)))
            (active_loss + decoder.slot_controller.budget_loss(state, 20)[0]).backward()
            outputs.append(output)
            gradients.append([x.grad for x in inputs] + [p.grad for p in decoder.parameters()])
            for gradient in gradients[-1]:
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
            self.assertEqual(state['active_count'].item(), output.active_mask.sum().item())
        for a, b in zip(attributes(outputs[0]), attributes(outputs[1])):
            torch.testing.assert_close(a, b)
        for a, b in zip(gradients[0], gradients[1]):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
        checked.eval()
        with torch.no_grad():
            inference = checked(source[0], source[1], image_shape=(2, 2, 3), rgb=source[2], global_step=0)
        torch.testing.assert_close(inference.active_mask, outputs[0].active_mask)
        for a, b in zip(attributes(inference), attributes(outputs[0])):
            torch.testing.assert_close(a, b)

    def test_global_budget_value_and_ddp_average_gradient(self):
        # Unequal local candidate counts; hinge must see global 25/30, not
        # independent 6/10 and 19/20 ratios. Simulate the count all-reduce.
        for local_count, candidates in ((6., 10.), (19., 20.)):
            count = torch.tensor(local_count, requires_grad=True)
            with patch.object(control.dist, 'is_initialized', return_value=True), \
                 patch.object(control.dist, 'get_world_size', return_value=2), \
                 patch.object(control.dist, 'all_reduce', side_effect=lambda x: x.copy_(torch.tensor([25., 30.]))):
                ratio = control.global_active_fraction(count, torch.tensor(candidates))
            loss = (ratio - .75).clamp_min(0).square()
            loss.backward()
            self.assertAlmostEqual(ratio.item(), 25/30)
            # DDP averages the two rank gradients after backward.
            self.assertAlmostEqual(count.grad.item() / 2, 2 * (25/30 - .75) / 30)

    def test_decoder_all_open_matches_final_and_closed_supports_keep_gradients(self):
        config = Cfg(feature_dim=5, hidden_dim=8, num_neighbors=3, chunk_size=4,
                     appearance_2d=True, separate_appearance=True, appearance_capacity=True,
                     appearance_heads=2, appearance_dim=3, appearance_mlp_dim=7)
        base = Decoder(config)
        controlled_config = copy.deepcopy(config)
        controlled_config.slot_control = control.SlotControlCfg(enabled=True, warmup_steps=10, ramp_steps=10)
        controlled = Decoder(controlled_config)
        controlled.load_state_dict(base.state_dict(), strict=False)
        for step in (0, 20):
            points, features, rgb = torch.randn(1, 12, 3), torch.randn(1, 12, 5), torch.rand(1, 12, 3)
            a = base(points, features, image_shape=(2, 2, 3), rgb=rgb)
            state = {}
            b = controlled(points, features, image_shape=(2, 2, 3), rgb=rgb, global_step=step, control_dump=state)
            for x, y in zip(attributes(a), attributes(b)):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            self.assertEqual(state['active_count'].item(), 12)

        # Fully connected candidate sets make coverage deterministic in a tiny
        # scene. Only slot 1 closes, but its point/feature still supply geometry/SH.
        points = torch.tensor([[[0., 0., 1.], [.2, .1, 1.], [.4, .3, 1.]]], requires_grad=True)
        features = torch.randn(1, 3, 5, requires_grad=True)
        probability = torch.tensor([.9, .1, .9], requires_grad=True)
        state = {}
        with patch.object(controlled.slot_controller, '_probabilities', return_value=probability):
            output = controlled(points, features, image_shape=(1, 1, 3), rgb=torch.rand(1, 3, 3),
                                global_step=20, control_dump=state)
            self.assertEqual(output.active_mask.tolist(), [[True, False, True]])
            compact = output.compact_scene(0)
            self.assertEqual(compact.means.shape[1], 2)
            self.assertEqual(output.opacities[0, 1].item(), 0)
            sum(x.square().sum() for x in attributes(compact)).backward()
        self.assertGreater(points.grad[0, 1].abs().sum().item(), 0)
        self.assertGreater(features.grad[0, 1].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(probability.grad).all())
        self.assertGreater(probability.grad.abs().sum().item(), 0)

    def test_renderer_receives_variable_compacted_counts_and_preserves_pose_gradients(self):
        calls = []

        def render(extrinsics, intrinsics, near, far, shape, bg, means, covariances, sh, opacities, **kwargs):
            calls.append(means.shape)
            color = ((means.sum(-1) + covariances.sum((-2, -1)) + sh.sum((-2, -1))) * opacities).sum(-1)
            color = color + kwargs['cam_rot_delta'].sum(-1) + kwargs['cam_trans_delta'].sum(-1)
            return color[:, None, None, None].expand(-1, 3, *shape), color[:, None, None].expand(-1, *shape)

        namespace = dict(torch=torch, rearrange=rearrange, repeat=repeat, render_cuda=render,
                         DecoderOutput=lambda color, depth: SimpleNamespace(color=color, depth=depth))
        forward = method(ROOT / 'src/model/decoder/decoder_splatting_cuda.py', 'DecoderSplattingCUDA', 'forward', namespace)
        renderer = type('Renderer', (), dict(forward=forward))()
        renderer.background_color, renderer.make_scale_invariant = torch.zeros(3), True
        data = [torch.randn(*s, requires_grad=True) for s in ((2, 3, 3), (2, 3, 3, 3), (2, 3, 3, 4), (2, 3))]
        mask = torch.tensor([[True, False, True], [False, True, False]])
        gaussian = Gaussians(*data, active_mask=mask)
        rotation, translation = [torch.randn(2, 2, 3, requires_grad=True) for _ in range(2)]
        poses, intrinsics = torch.eye(4).expand(2, 2, 4, 4), torch.eye(3).expand(2, 2, 3, 3)
        bounds = torch.ones(2, 2)
        result = renderer.forward(gaussian, poses, intrinsics, bounds, bounds, (2, 3),
                                  cam_rot_delta=rotation, cam_trans_delta=translation)
        self.assertEqual(calls, [torch.Size([2, 2, 3]), torch.Size([2, 1, 3])])
        dense = Gaussians(data[0], data[1], data[2], data[3] * mask)
        reference = renderer.forward(dense, poses, intrinsics, bounds, bounds, (2, 3),
                                    cam_rot_delta=rotation, cam_trans_delta=translation)
        torch.testing.assert_close(result.color, reference.color)
        torch.testing.assert_close(result.depth, reference.depth)
        result.color.sum().backward()
        for x in data:
            self.assertEqual(x.grad[~mask].abs().sum().item(), 0)
            self.assertGreater(x.grad[mask].abs().sum().item(), 0)
        self.assertGreater(rotation.grad.abs().sum().item(), 0)
        self.assertGreater(translation.grad.abs().sum().item(), 0)

    def test_training_adds_budget_loss_and_logs_true_retained_and_removed_ratios(self):
        model = controller()
        logs = {}

        class Encoder:
            gaussian_decoder = SimpleNamespace(cfg=SimpleNamespace(appearance_2d=False), slot_controller=model)

            def __call__(self, context, step, visualization_dump=None, control_dump=None):
                control_dump.update(active_count=torch.tensor(9., requires_grad=True),
                                    candidate_count=torch.tensor(10.), scene_count=torch.tensor(1.))

        namespace = dict(torch=torch, rearrange=rearrange,
                         compute_psnr=lambda target, image: image.flatten(1).mean(1))
        train = method(ROOT / 'src/model/model_wrapper.py', 'ModelWrapper', 'training_step', namespace)
        batch = {'context': {'image': torch.zeros(1, 2, 3, 2, 2)},
                 'target': {'image': torch.zeros(1, 1, 3, 2, 2), 'extrinsics': None,
                            'intrinsics': None, 'near': None, 'far': None}}
        wrapper = SimpleNamespace(
            data_shim=lambda x: x, encoder=Encoder(), distiller=None,
            decoder=SimpleNamespace(forward=lambda *a, **kw: SimpleNamespace(color=batch['target']['image'])),
            train_cfg=SimpleNamespace(depth_mode=None, print_log_every_n_steps=100),
            losses=[SimpleNamespace(name='mse', forward=lambda *a: torch.tensor(2.))],
            global_rank=1, global_step=20, step_tracker=None,
            log=lambda k, v, **kw: logs.update({k: v}), log_dict=lambda values, **kw: logs.update(values),
        )
        loss = train(wrapper, batch, 0)
        self.assertAlmostEqual(loss.item(), 2 + .01 * (.9 - .75)**2, places=6)
        self.assertAlmostEqual(logs['slot/active_fraction'].item(), .9)
        self.assertAlmostEqual(logs['slot/removed_fraction'].item(), .1)
        self.assertEqual(logs['slot/gaussians_before'].item(), 10)
        self.assertEqual(logs['slot/gaussians_after'].item(), 9)
        self.assertIn('loss/slot_budget', logs)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for controller device parity')
    def test_cuda_controller_matches_cpu_outputs_and_gradients(self):
        cpu = controller()
        gpu = copy.deepcopy(cpu).cuda()
        inputs = [torch.randn(3, 3), torch.randn(3, 5), graph(), torch.rand(3, 3)]
        states = []
        for model, args in ((cpu, inputs), (gpu, [x.cuda() for x in inputs])):
            allocated, mask, state = model(*args, global_step=20)
            state['scene_count'] = allocated.new_tensor(1)
            model.budget_loss(state, 20)[0].backward()
            states.append((allocated, mask))
        for a, b in zip(states[0], states[1]):
            torch.testing.assert_close(a, b.cpu())
        for a, b in zip(cpu.parameters(), gpu.parameters()):
            torch.testing.assert_close(a.grad, b.grad.cpu(), atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
