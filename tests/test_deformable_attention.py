"""Regression coverage for installations without GroundingDINO's C++ extension."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "weights/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py"
)
spec = importlib.util.spec_from_file_location("deformable_attention", SOURCE)
attention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention)


class AttentionFallbackTest(unittest.TestCase):
    def run_fallback(self, device, reference_dims):
        torch.manual_seed(7)
        cpu_model = attention.MultiScaleDeformableAttention(
            embed_dim=8, num_heads=2, num_levels=2, num_points=2, batch_first=True
        )
        model = copy.deepcopy(cpu_model).to(device)
        query = torch.randn(1, 3, 8)
        value = torch.randn(1, 5, 8)
        refs = torch.rand(1, 3, 2, reference_dims)
        shapes = torch.tensor([[2, 2], [1, 1]])
        starts = torch.tensor([0, 4])
        with patch.object(attention, "_C", None), patch.object(
            attention.MultiScaleDeformableAttnFunction, "apply",
            side_effect=AssertionError("Unavailable extension was called"),
        ):
            expected = cpu_model(query, value=value, reference_points=refs,
                                 spatial_shapes=shapes, level_start_index=starts)
            query = query.to(device).requires_grad_()
            value = value.to(device).requires_grad_()
            result = model(query, value=value, reference_points=refs.to(device),
                           spatial_shapes=shapes.to(device),
                           level_start_index=starts.to(device))
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-5)
            result.square().sum().backward()
            self.assertTrue(torch.isfinite(query.grad).all())
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertEqual(result.device.type, device)

    def test_cpu_without_extension(self):
        for dims in (2, 4):
            with self.subTest(reference_dims=dims):
                self.run_fallback("cpu", dims)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_cuda_without_extension(self):
        for dims in (2, 4):
            with self.subTest(reference_dims=dims):
                self.run_fallback("cuda", dims)


if __name__ == "__main__":
    unittest.main()
