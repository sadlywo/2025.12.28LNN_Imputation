import torch

from torch import nn

from validation_v2.modern.sssd_worker import (
    diffusion_loss,
    sample_sssd_batched,
    sssd_parameters,
)


def test_sssd_parameters_match_preregistered_architecture():
    params = sssd_parameters(residual_width=32)
    assert params["T"] == 200
    assert params["num_res_layers"] == 36
    assert params["s4_lmax"] == 30
    assert params["s4_d_state"] == 64
    assert params["res_channels"] == params["skip_channels"] == 32


def test_diffusion_loss_scores_only_artificially_missing_values():
    target = torch.tensor([[[1.0, 2.0, 3.0]]])
    mask = torch.tensor([[[1.0, 0.0, 1.0]]])
    predicted_noise = torch.tensor([[[100.0, 0.0, 100.0]]])
    true_noise = torch.zeros_like(predicted_noise)
    assert diffusion_loss(predicted_noise, true_noise, mask).item() == 0.0


def test_batched_sampling_preserves_observations_and_sample_count():
    class ZeroDenoiser(nn.Module):
        def forward(self, inputs):
            return torch.zeros_like(inputs[0])

    observed = torch.arange(24, dtype=torch.float32).reshape(4, 1, 6)
    mask = torch.ones_like(observed)
    mask[:, :, 2:4] = 0
    schedule = {
        "T": 1,
        "Alpha": torch.tensor([0.9]),
        "Alpha_bar": torch.tensor([0.9]),
        "Sigma": torch.tensor([0.0]),
    }
    samples = sample_sssd_batched(
        ZeroDenoiser(), observed, mask, schedule, n_samples=6, batch_size=2
    )
    assert samples.shape == (4, 6, 1, 6)
    expected = (observed[:, None] * mask[:, None]).expand_as(samples)
    torch.testing.assert_close(
        samples * mask[:, None], expected
    )
