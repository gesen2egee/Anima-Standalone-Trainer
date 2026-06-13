import torch

from library import anima_models
from library.custom_train_functions import apply_masked_loss


def test_apply_masked_loss_accepts_5d_anima_loss_with_alpha_mask():
    loss = torch.ones(2, 3, 1, 4, 4)
    alpha = torch.zeros(2, 4, 4)
    alpha[:, 1:3, 1:3] = 1.0

    masked = apply_masked_loss(loss, {"alpha_masks": alpha})

    assert masked.shape == loss.shape
    assert torch.equal(masked[:, :, :, 0, 0], torch.zeros_like(masked[:, :, :, 0, 0]))
    assert torch.equal(masked[:, :, :, 1, 1], torch.ones_like(masked[:, :, :, 1, 1]))


def test_adaln_fp32_helper_returns_fp32_for_stock_linear():
    modulation = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        modulation[0].weight.copy_(torch.eye(2))

    out = anima_models._run_adaln_modulation_fp32(modulation, torch.ones(1, 1, 2, dtype=torch.float16))

    assert out.dtype == torch.float32
    assert torch.equal(out, torch.ones_like(out))


def test_adaln_fp32_helper_preserves_patched_linear_forward():
    class PatchedLinear(torch.nn.Linear):
        def forward(self, input):
            return super().forward(input) + 1

    modulation = torch.nn.Sequential(PatchedLinear(2, 2, bias=False))
    with torch.no_grad():
        modulation[0].weight.zero_()

    out = anima_models._run_adaln_modulation_fp32(modulation, torch.ones(1, 1, 2, dtype=torch.float16))

    assert out.dtype == torch.float32
    assert torch.equal(out, torch.ones_like(out))
