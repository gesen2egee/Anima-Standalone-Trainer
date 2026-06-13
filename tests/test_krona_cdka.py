import math
import unittest
import torch

from networks.cdka import CdkaInfModule, CdkaModule
from networks.krona import KronaInfModule, KronaModule


def _fill_weights(module):
    with torch.no_grad():
        module.lokr_w1.copy_(torch.randn_like(module.lokr_w1))
        module.lokr_w2.copy_(torch.randn_like(module.lokr_w2))


def _exported_delta(module):
    state = module.state_dict()
    return torch.kron(state["lokr_w1"], state["lokr_w2"])


class TestKronaCdka(unittest.TestCase):

    def test_krona_default_uses_diffusekrona_lokr_factor_layout(self):
        module = KronaModule("test", torch.nn.Linear(2048, 2048, bias=False))
        self.assertEqual(module.lokr_w1.shape, (32, 512))
        self.assertEqual(module.lokr_w2.shape, (64, 4))
        self.assertEqual(module.scale, 1.0)

    def test_cdka_default_uses_cdka_single_component_layout(self):
        module = CdkaModule("test", torch.nn.Linear(2048, 2048, bias=False))
        self.assertEqual(module.lokr_w1.shape, (1024, 8))
        self.assertEqual(module.lokr_w2.shape, (2, 256))
        self.assertEqual(module.scale, 16.0 / math.sqrt(8))



    def test_cdka_factor_in_takes_priority_over_factor_in(self):
        module = CdkaModule(
            "test",
            torch.nn.Linear(2048, 2048, bias=False),
            factor_in=4,
            cdka_factor_in=8,
        )
        self.assertEqual(module.lokr_w1.shape, (1024, 8))
        self.assertEqual(module.lokr_w2.shape, (2, 256))

    def test_cdka_alpha_none_string_disables_scale(self):
        module = CdkaModule(
            "test",
            torch.nn.Linear(2048, 2048, bias=False),
            cdka_alpha="None",
        )

        self.assertEqual(module.scale, 1.0)

    def test_krona_can_switch_to_cdka_layout_with_cdka_factor_in(self):
        module = KronaModule(
            "test",
            torch.nn.Linear(2048, 2048, bias=False),
            factor_out=2,
            factor_in=4,
            cdka_factor_in=8,
        )
        self.assertEqual(module.lokr_w1.shape, (1024, 8))
        self.assertEqual(module.lokr_w2.shape, (2, 256))

    def test_cdka_mica_both_initializes_minor_direction_without_freezing(self):
        module = CdkaModule(
            "test",
            torch.nn.Linear(16, 12, bias=False),
            factor_out=3,
            cdka_factor_in=4,
            w2_init="mica_both",
        )

        self.assertFalse(module.is_mica)
        self.assertTrue(module.lokr_w1.requires_grad)
        self.assertTrue(module.lokr_w2.requires_grad)
        self.assertTrue(torch.allclose(module.lokr_w1, torch.zeros_like(module.lokr_w1)))
        self.assertGreater(module.lokr_w2.norm().item(), 0.0)
        self.assertTrue(torch.allclose(module.get_diff_weight(), torch.zeros_like(module.get_diff_weight())))

    def test_krona_mica_both_initializes_minor_direction_without_freezing(self):
        module = KronaModule(
            "test",
            torch.nn.Linear(16, 12, bias=False),
            factor_out=3,
            factor_in=4,
            w2_init="mica_both",
        )

        self.assertFalse(module.is_mica)
        self.assertTrue(module.lokr_w1.requires_grad)
        self.assertTrue(module.lokr_w2.requires_grad)
        self.assertTrue(torch.allclose(module.lokr_w1, torch.zeros_like(module.lokr_w1)))
        self.assertGreater(module.lokr_w2.norm().item(), 0.0)
        self.assertTrue(torch.allclose(module.get_diff_weight(), torch.zeros_like(module.get_diff_weight())))

    def test_cdka_alpha_enables_paper_scale_and_exports_lossless_full_lokr(self):
        module = CdkaModule(
            "test",
            torch.nn.Linear(16, 12, bias=False),
            factor_out=3,
            cdka_factor_in=4,
            cdka_alpha=16,
        )
        _fill_weights(module)

        expected_scale = 16 / math.sqrt(4)
        expected_delta = module.get_diff_weight().detach()
        exported_delta = _exported_delta(module)

        self.assertEqual(module.scale, expected_scale)
        self.assertTrue(torch.allclose(exported_delta, expected_delta, atol=1e-6, rtol=1e-6))

    def test_krona_cdka_alpha_exports_lossless_full_lokr(self):
        module = KronaModule(
            "test",
            torch.nn.Linear(16, 12, bias=False),
            factor_out=3,
            factor_in=4,
            cdka_alpha=16,
        )
        _fill_weights(module)

        expected_delta = module.get_diff_weight().detach()
        exported_delta = _exported_delta(module)

        self.assertTrue(torch.allclose(exported_delta, expected_delta, atol=1e-6, rtol=1e-6))

    def test_cdka_state_dict_loading_correctness_under_scaling(self):
        # 1. Create source module with scaling
        src_module = CdkaModule(
            "test",
            torch.nn.Linear(16, 12, bias=False),
            factor_out=3,
            cdka_factor_in=4,
            cdka_alpha=16,
        )
        _fill_weights(src_module)
        original_w1 = src_module.lokr_w1.clone().detach()
        original_w2 = src_module.lokr_w2.clone().detach()
        expected_delta = src_module.get_diff_weight().clone().detach()

        # Export state_dict (this folds the scale into lokr_w1)
        sd = src_module.state_dict()
        
        # 2. Create destination module with same scaling configuration
        dst_module = CdkaModule(
            "test",
            torch.nn.Linear(16, 12, bias=False),
            factor_out=3,
            cdka_factor_in=4,
            cdka_alpha=16,
        )
        
        # Load state_dict (should automatically divide lokr_w1 by scale)
        dst_module.load_state_dict(sd)
        
        # 3. Assert correctness
        # The internal parameters should match the original unfolded values
        self.assertTrue(torch.allclose(dst_module.lokr_w1, original_w1, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(dst_module.lokr_w2, original_w2, atol=1e-6, rtol=1e-6))
        # The generated delta weights should be exactly correct (no repeated scaling)
        loaded_delta = dst_module.get_diff_weight().detach()
        self.assertTrue(torch.allclose(loaded_delta, expected_delta, atol=1e-6, rtol=1e-6))

    def test_krona_state_dict_loading_correctness_under_scaling(self):
        # 1. Create source module with scaling
        src_module = KronaModule(
            "test",
            torch.nn.Linear(16, 12, bias=False),
            factor_out=3,
            factor_in=4,
            cdka_alpha=16,
        )
        _fill_weights(src_module)
        original_w1 = src_module.lokr_w1.clone().detach()
        original_w2 = src_module.lokr_w2.clone().detach()
        expected_delta = src_module.get_diff_weight().clone().detach()

        # Export state_dict (this folds the scale into lokr_w1)
        sd = src_module.state_dict()
        
        # 2. Create destination module with same scaling configuration
        dst_module = KronaModule(
            "test",
            torch.nn.Linear(16, 12, bias=False),
            factor_out=3,
            factor_in=4,
            cdka_alpha=16,
        )
        
        # Load state_dict (should automatically divide lokr_w1 by scale)
        dst_module.load_state_dict(sd)
        
        # 3. Assert correctness
        self.assertTrue(torch.allclose(dst_module.lokr_w1, original_w1, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(dst_module.lokr_w2, original_w2, atol=1e-6, rtol=1e-6))
        loaded_delta = dst_module.get_diff_weight().detach()
        self.assertTrue(torch.allclose(loaded_delta, expected_delta, atol=1e-6, rtol=1e-6))

    def test_cdka_inference_merge_does_not_apply_folded_scale_twice(self):
        src_module = CdkaModule(
            "test",
            torch.nn.Linear(16, 12, bias=False),
            factor_out=3,
            cdka_factor_in=4,
            cdka_alpha=16,
        )
        _fill_weights(src_module)
        expected_delta = src_module.get_diff_weight().detach()
        sd = src_module.state_dict()

        base = torch.nn.Linear(16, 12, bias=False)
        with torch.no_grad():
            base.weight.zero_()
        inf_module = CdkaInfModule(
            "test",
            base,
            factor_out=3,
            cdka_factor_in=4,
            cdka_alpha=16,
        )

        inf_module.merge_to(sd, dtype=torch.float32, device=torch.device("cpu"))

        self.assertTrue(torch.allclose(base.weight, expected_delta, atol=1e-6, rtol=1e-6))

    def test_krona_inference_merge_does_not_apply_folded_scale_twice(self):
        src_module = KronaModule(
            "test",
            torch.nn.Linear(16, 12, bias=False),
            factor_out=3,
            factor_in=4,
            cdka_alpha=16,
        )
        _fill_weights(src_module)
        expected_delta = src_module.get_diff_weight().detach()
        sd = src_module.state_dict()

        base = torch.nn.Linear(16, 12, bias=False)
        with torch.no_grad():
            base.weight.zero_()
        inf_module = KronaInfModule(
            "test",
            base,
            factor_out=3,
            factor_in=4,
            cdka_alpha=16,
        )

        inf_module.merge_to(sd, dtype=torch.float32, device=torch.device("cpu"))

        self.assertTrue(torch.allclose(base.weight, expected_delta, atol=1e-6, rtol=1e-6))

    def test_cdka_dora_initialization_and_forward(self):
        linear = torch.nn.Linear(16, 12, bias=False)
        module = CdkaModule(
            "test",
            linear,
            factor_out=3,
            cdka_factor_in=4,
            weight_decompose=True,
            wd_on_out=True,
        )
        self.assertTrue(module.wd)
        self.assertEqual(module.dora_scale.shape, (12, 1))
        self.assertTrue(hasattr(module, "org_weight"))
        self.assertTrue(torch.allclose(module.org_weight, linear.weight))

        _fill_weights(module)
        sd = module.state_dict()
        self.assertIn("dora_scale", sd)

        x = torch.randn(2, 16)
        module.apply_to()
        out = module(x)
        self.assertEqual(out.shape, (2, 12))

    def test_krona_dora_initialization_and_forward(self):
        linear = torch.nn.Linear(16, 12, bias=False)
        module = KronaModule(
            "test",
            linear,
            factor_out=3,
            factor_in=4,
            weight_decompose=True,
            wd_on_out=True,
        )
        self.assertTrue(module.wd)
        self.assertEqual(module.dora_scale.shape, (12, 1))
        self.assertTrue(hasattr(module, "org_weight"))
        self.assertTrue(torch.allclose(module.org_weight, linear.weight))

        _fill_weights(module)
        sd = module.state_dict()
        self.assertIn("dora_scale", sd)

        x = torch.randn(2, 16)
        module.apply_to()
        out = module(x)
        self.assertEqual(out.shape, (2, 12))


    def test_cdka_dora_inference_merge(self):
        linear = torch.nn.Linear(16, 12, bias=False)
        src_module = CdkaModule(
            "test",
            linear,
            factor_out=3,
            cdka_factor_in=4,
            weight_decompose=True,
            wd_on_out=True,
        )
        _fill_weights(src_module)
        
        # Calculate expected Dora merged weight manually
        diff_weight = src_module.get_diff_weight().detach()
        expected_merged = src_module.apply_weight_decompose(linear.weight + diff_weight, multiplier=1.0).detach()
        sd = src_module.state_dict()

        base = torch.nn.Linear(16, 12, bias=False)
        # copy weights to base
        with torch.no_grad():
            base.weight.copy_(linear.weight)
        
        inf_module = CdkaInfModule(
            "test",
            base,
            factor_out=3,
            cdka_factor_in=4,
            weight_decompose=True,
            wd_on_out=True,
        )
        inf_module.merge_to(sd, dtype=torch.float32, device=torch.device("cpu"))
        
        self.assertTrue(torch.allclose(base.weight, expected_merged, atol=1e-6, rtol=1e-6))

    def test_cdka_allora_backward_scaling(self):
        linear = torch.nn.Linear(16, 12, bias=False)
        module = CdkaModule(
            "test",
            linear,
            factor_out=3,
            cdka_factor_in=4,
            allora=True,
            allora_eta=2.0,
        )
        _fill_weights(module)
        
        self.assertTrue(module.allora)
        self.assertEqual(module.allora_eta, 2.0)
        
        x = torch.randn(2, 16, requires_grad=True)
        module.apply_to()
        
        diff_weight = module.get_diff_weight()
        diff_weight_static = diff_weight.detach()
        norms = torch.norm(diff_weight_static.reshape(diff_weight_static.shape[0], -1), dim=1)
        norms = norms.reshape(diff_weight_static.shape[0], *[1] * (diff_weight_static.dim() - 1))
        rsq_scale = 1.0 / (2.0 ** 2)
        acc_val = 1.0 / torch.sqrt(norms + rsq_scale)
        
        out = module(x)
        loss = out.sum()
        loss.backward()
        
        grad_w1_allora = module.lokr_w1.grad.clone()
        grad_w2_allora = module.lokr_w2.grad.clone()
        grad_x_allora = x.grad.clone()
        
        # Now create reference run without ALLoRA
        linear_ref = torch.nn.Linear(16, 12, bias=False)
        with torch.no_grad():
            linear_ref.weight.copy_(linear.weight)
        module_ref = CdkaModule(
            "test",
            linear_ref,
            factor_out=3,
            cdka_factor_in=4,
            allora=False,
        )
        with torch.no_grad():
            module_ref.lokr_w1.copy_(module.lokr_w1)
            module_ref.lokr_w2.copy_(module.lokr_w2)
            
        x_ref = x.clone().detach().requires_grad_(True)
        module_ref.apply_to()
        
        out_ref = module_ref(x_ref)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        
        grad_w1_ref = module_ref.lokr_w1.grad.clone()
        grad_w2_ref = module_ref.lokr_w2.grad.clone()
        grad_x_ref = x_ref.grad.clone()
        
        # 1. Inputs gradients should be IDENTICAL
        self.assertTrue(torch.allclose(grad_x_allora, grad_x_ref, atol=1e-6, rtol=1e-6))
        
        # 2. Parameters gradients should be DIFFERENT (due to scaling)
        self.assertFalse(torch.allclose(grad_w1_allora, grad_w1_ref, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(grad_w2_allora, grad_w2_ref, atol=1e-6, rtol=1e-6))
        
        # 3. If we apply the manual hook to the reference run, the gradients of w1 and w2 should match ALLoRA's
        linear_manual = torch.nn.Linear(16, 12, bias=False)
        with torch.no_grad():
            linear_manual.weight.copy_(linear.weight)
        module_manual = CdkaModule(
            "test",
            linear_manual,
            factor_out=3,
            cdka_factor_in=4,
            allora=False,
        )
        with torch.no_grad():
            module_manual.lokr_w1.copy_(module.lokr_w1)
            module_manual.lokr_w2.copy_(module.lokr_w2)
            
        x_manual = x.clone().detach().requires_grad_(True)
        module_manual.apply_to()
        
        def manual_hook(grad):
            return grad * acc_val.to(grad.device).to(grad.dtype)
            
        orig_get_diff_weight = module_manual.get_diff_weight
        def hook_get_diff_weight():
            dw = orig_get_diff_weight()
            dw.register_hook(manual_hook)
            return dw
        module_manual.get_diff_weight = hook_get_diff_weight
        
        out_manual = module_manual(x_manual)
        loss_manual = out_manual.sum()
        loss_manual.backward()
        
        self.assertTrue(torch.allclose(module_manual.lokr_w1.grad, grad_w1_allora, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(module_manual.lokr_w2.grad, grad_w2_allora, atol=1e-5, rtol=1e-5))

    def test_krona_allora_backward_scaling(self):
        linear = torch.nn.Linear(16, 12, bias=False)
        module = KronaModule(
            "test",
            linear,
            factor_out=3,
            factor_in=4,
            allora=True,
            allora_eta=2.0,
        )
        _fill_weights(module)
        
        self.assertTrue(module.allora)
        self.assertEqual(module.allora_eta, 2.0)
        
        x = torch.randn(2, 16, requires_grad=True)
        module.apply_to()
        
        diff_weight = module.get_diff_weight()
        diff_weight_static = diff_weight.detach()
        norms = torch.norm(diff_weight_static.reshape(diff_weight_static.shape[0], -1), dim=1)
        norms = norms.reshape(diff_weight_static.shape[0], *[1] * (diff_weight_static.dim() - 1))
        rsq_scale = 1.0 / (2.0 ** 2)
        acc_val = 1.0 / torch.sqrt(norms + rsq_scale)
        
        out = module(x)
        loss = out.sum()
        loss.backward()
        
        grad_w1_allora = module.lokr_w1.grad.clone()
        grad_w2_allora = module.lokr_w2.grad.clone()
        grad_x_allora = x.grad.clone()
        
        # Now create reference run without ALLoRA
        linear_ref = torch.nn.Linear(16, 12, bias=False)
        with torch.no_grad():
            linear_ref.weight.copy_(linear.weight)
        module_ref = KronaModule(
            "test",
            linear_ref,
            factor_out=3,
            factor_in=4,
            allora=False,
        )
        with torch.no_grad():
            module_ref.lokr_w1.copy_(module.lokr_w1)
            module_ref.lokr_w2.copy_(module.lokr_w2)
            
        x_ref = x.clone().detach().requires_grad_(True)
        module_ref.apply_to()
        
        out_ref = module_ref(x_ref)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        
        grad_w1_ref = module_ref.lokr_w1.grad.clone()
        grad_w2_ref = module_ref.lokr_w2.grad.clone()
        grad_x_ref = x_ref.grad.clone()
        
        # 1. Inputs gradients should be IDENTICAL
        self.assertTrue(torch.allclose(grad_x_allora, grad_x_ref, atol=1e-6, rtol=1e-6))
        
        # 2. Parameters gradients should be DIFFERENT (due to scaling)
        self.assertFalse(torch.allclose(grad_w1_allora, grad_w1_ref, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(grad_w2_allora, grad_w2_ref, atol=1e-6, rtol=1e-6))
        
        # 3. If we apply the manual hook to the reference run, the gradients of w1 and w2 should match ALLoRA's
        linear_manual = torch.nn.Linear(16, 12, bias=False)
        with torch.no_grad():
            linear_manual.weight.copy_(linear.weight)
        module_manual = KronaModule(
            "test",
            linear_manual,
            factor_out=3,
            factor_in=4,
            allora=False,
        )
        with torch.no_grad():
            module_manual.lokr_w1.copy_(module.lokr_w1)
            module_manual.lokr_w2.copy_(module.lokr_w2)
            
        x_manual = x.clone().detach().requires_grad_(True)
        module_manual.apply_to()
        
        def manual_hook(grad):
            return grad * acc_val.to(grad.device).to(grad.dtype)
            
        orig_get_diff_weight = module_manual.get_diff_weight
        def hook_get_diff_weight():
            dw = orig_get_diff_weight()
            dw.register_hook(manual_hook)
            return dw
        module_manual.get_diff_weight = hook_get_diff_weight
        
        out_manual = module_manual(x_manual)
        loss_manual = out_manual.sum()
        loss_manual.backward()
        
        self.assertTrue(torch.allclose(module_manual.lokr_w1.grad, grad_w1_allora, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(module_manual.lokr_w2.grad, grad_w2_allora, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
