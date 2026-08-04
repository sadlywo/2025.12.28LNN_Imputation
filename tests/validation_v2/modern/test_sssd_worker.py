import torch

from validation_v2.modern.sssd_worker import diffusion_loss, sssd_parameters


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
