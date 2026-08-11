"""Autograd-safe SO(3) operations.

The rotation convention is always ``R_body_to_world``.  Rotation vectors are
axis-angle vectors in radians.  Every operation stays in PyTorch so the loss
path works on CPU/CUDA and retains gradients.
"""

from __future__ import annotations

import torch


def skew(vector: torch.Tensor) -> torch.Tensor:
    """Return the skew-symmetric matrix ``[vector]_x``.

    Args:
        vector: Floating tensor with shape ``[..., 3]``.
    """

    if not isinstance(vector, torch.Tensor) or vector.shape[-1:] != (3,):
        raise ValueError("vector must be a torch tensor with final dimension 3")
    if not vector.is_floating_point():
        raise TypeError("vector must have a floating dtype")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ),
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def _taylor_a(theta2: torch.Tensor) -> torch.Tensor:
    return 1.0 - theta2 / 6.0 + theta2.square() / 120.0


def _taylor_b(theta2: torch.Tensor) -> torch.Tensor:
    return 0.5 - theta2 / 24.0 + theta2.square() / 720.0


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    """Map rotation vectors ``[...,3]`` to rotation matrices ``[...,3,3]``."""

    if not isinstance(phi, torch.Tensor) or phi.shape[-1:] != (3,):
        raise ValueError("phi must be a torch tensor with final dimension 3")
    if not phi.is_floating_point():
        raise TypeError("phi must have a floating dtype")
    theta2 = phi.square().sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2)
    small = theta2 < (1e-4 if phi.dtype == torch.float32 else 1e-8)
    safe_theta = theta.clamp_min(torch.finfo(phi.dtype).eps)
    safe_theta2 = theta2.clamp_min(torch.finfo(phi.dtype).eps)
    a = torch.where(small, _taylor_a(theta2), torch.sin(theta) / safe_theta)
    b = torch.where(
        small,
        _taylor_b(theta2),
        (1.0 - torch.cos(theta)) / safe_theta2,
    )
    matrix = skew(phi)
    identity = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(
        *phi.shape[:-1], 3, 3
    )
    return identity + a[..., None] * matrix + b[..., None] * (matrix @ matrix)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """Map rotation matrices ``[...,3,3]`` to rotation vectors ``[...,3]``.

    The small-angle branch uses the first terms of ``theta/sin(theta)``.  The
    regular branch is stable for the short-window residuals used by this
    project; exact pi rotations are deliberately rejected by clamping because
    their logarithm has a non-unique axis.
    """

    if not isinstance(rotation, torch.Tensor) or rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must be a torch tensor with shape [...,3,3]")
    if not rotation.is_floating_point():
        raise TypeError("rotation must have a floating dtype")
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0)
    theta = torch.acos(cosine)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    ) * 0.5
    theta2 = theta.square()
    small = theta2 < (1e-4 if rotation.dtype == torch.float32 else 1e-8)
    sine = torch.sin(theta)
    regular_scale = theta / sine.clamp_min(torch.finfo(rotation.dtype).eps)
    small_scale = 1.0 + theta2 / 6.0 + 7.0 * theta2.square() / 360.0
    return vee * torch.where(small, small_scale, regular_scale)[..., None]


def quat_to_rotmat(quaternion: torch.Tensor, *, order: str = "xyzw") -> torch.Tensor:
    """Convert unit quaternions to body-to-world rotation matrices.

    Args:
        quaternion: Tensor ``[...,4]``. It is safely normalized internally.
        order: Either ``"xyzw"`` (OxIOD/Vicon) or ``"wxyz"``.
    """

    if not isinstance(quaternion, torch.Tensor) or quaternion.shape[-1:] != (4,):
        raise ValueError("quaternion must be a torch tensor with final dimension 4")
    if not quaternion.is_floating_point():
        raise TypeError("quaternion must have a floating dtype")
    if order not in {"xyzw", "wxyz"}:
        raise ValueError("quaternion order must be 'xyzw' or 'wxyz'")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if torch.any(norm <= torch.finfo(quaternion.dtype).eps):
        raise ValueError("quaternion norm must be positive")
    unit = quaternion / norm
    if order == "xyzw":
        x, y, z, w = unit.unbind(dim=-1)
    else:
        w, x, y, z = unit.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


__all__ = ["quat_to_rotmat", "skew", "so3_exp", "so3_log"]
