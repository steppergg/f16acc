import unittest

import torch

import f16acc
from f16acc import patch

CUDA = torch.cuda.is_available()


def rel(x, ref):
    return ((x.double() - ref).abs().max() / ref.abs().max()).item()


@unittest.skipUnless(CUDA, "needs a CUDA GPU")
class Kernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    def test_odd_sizes_match_fp64(self):
        a = torch.randn(1000, 777, device="cuda").half()
        b = torch.randn(777, 1234, device="cuda").half()
        self.assertLess(rel(f16acc.mm(a, b, torch.float32), a.double() @ b.double()), 2e-3)

    def test_close_to_fp32_accumulate_and_better_than_plain_fp16(self):
        torch.manual_seed(0)
        a = torch.randn(4096, 4096, device="cuda").half()
        b = torch.randn(4096, 4096, device="cuda").half()
        ref = a.double() @ b.double()
        m = torch.backends.cuda.matmul
        fp32 = rel(torch.mm(a, b), ref)
        m.allow_fp16_accumulation = True
        try:
            plain = rel(torch.mm(a, b), ref)
        finally:
            m.allow_fp16_accumulation = False
        ours = rel(f16acc.mm(a, b), ref)
        self.assertLess(ours, 3 * fp32)      # measured 4.8e-4 vs 3.6e-4
        self.assertLess(ours * 5, plain)     # measured 4.8e-4 vs 8.8e-3

    def test_transposed_weight_view(self):
        x = torch.randn(256, 384, device="cuda").half()
        w = torch.randn(640, 384, device="cuda").half()
        self.assertLess(rel(f16acc.mm(x, w.t()), x.double() @ w.double().t()), 2e-3)


@unittest.skipUnless(CUDA, "needs a CUDA GPU")
class Patch(unittest.TestCase):
    def setUp(self):
        for k in patch.calls:
            patch.calls[k] = 0
        self.lin = torch.nn.Linear(384, 640).cuda().half()
        f16acc.enable()

    def tearDown(self):
        f16acc.disable()

    def test_linear_routed_with_bias_and_batch_dims(self):
        x = torch.randn(2, 150, 384, device="cuda").half()
        with torch.no_grad():
            y = self.lin(x)
        ref = x.double() @ self.lin.weight.double().t() + self.lin.bias.double()
        self.assertEqual(y.shape, (2, 150, 640))
        self.assertLess(rel(y, ref), 2e-3)
        self.assertEqual(patch.calls["chunked"], 1)

    def test_small_and_grad_inputs_fall_back(self):
        with torch.no_grad():
            self.lin(torch.randn(8, 384, device="cuda").half())
        self.lin(torch.randn(256, 384, device="cuda").half())  # grad enabled
        self.assertEqual(patch.calls, {"chunked": 0, "original": 2})

    def test_fp16_chunk_overflow_is_redone_with_fp32_accumulate(self):
        # each 32-long chunk sums to +-96000 (> 65504, inf in FP16) but the
        # total is 100: a tile left unrepaired would be inf or NaN
        lin = torch.nn.Linear(64, 128, bias=False).cuda().half()
        with torch.no_grad():
            lin.weight[:, :32] = 30.0
            lin.weight[:, 32:] = -30.0
            lin.weight[:, 63] = -29.0
            y = lin(torch.full((256, 64), 100.0, device="cuda").half())
        self.assertEqual(patch.calls["chunked"], 1)
        self.assertTrue(torch.all(y == 100.0))

    def test_no_overflow_leaves_flags_clear(self):
        x = torch.randn(300, 384, device="cuda").half()
        w = torch.randn(384, 500, device="cuda").half()
        self.assertTrue(torch.isfinite(f16acc.mm(x, w)).all())

    def test_disable_restores_original(self):
        f16acc.disable()
        self.assertIs(torch.nn.functional.linear, patch._orig)


if __name__ == "__main__":
    unittest.main()
