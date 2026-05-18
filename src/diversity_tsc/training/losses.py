"""Feature-decorrelation penalty used by the co-training procedure.

The penalty encourages the feature maps produced by the model under training
to be orthogonal to those produced by each frozen reference model.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = ["feature_diversity_penalty", "pairwise_diversity_penalty"]


_DEFAULT_SCALE = 40.0


def pairwise_diversity_penalty(
    feat_student: Tensor,
    feat_reference: Tensor,
    scale: float = _DEFAULT_SCALE,
) -> Tensor:
    """Compute the off-diagonal Frobenius penalty between two feature maps.

    Parameters
    ----------
    feat_student:
        Student feature map of shape ``(B, C, T)``.
    feat_reference:
        Reference feature map of shape ``(B, C, T)``, treated as a constant
        (must be ``detach()``-ed by the caller or come from a frozen model).
    scale:
        Divisor applied to the final penalty. Mirrors the ``/40`` factor used
        in the published experiments.

    Returns
    -------
    penalty: scalar tensor.
    """
    if feat_student.shape != feat_reference.shape:
        raise ValueError(
            f"Shape mismatch: student {tuple(feat_student.shape)} "
            f"vs reference {tuple(feat_reference.shape)}"
        )

    dot = torch.matmul(feat_student, feat_reference.transpose(2, 1))
    num_features = feat_student.size(1)
    mask = 1 - torch.eye(num_features, device=dot.device).unsqueeze(0)
    off_diag = dot * mask

    fro = torch.norm(off_diag, p="fro", dim=(1, 2))
    nz = torch.count_nonzero(off_diag, dim=(1, 2)) + 1
    return torch.mean(fro / nz) / scale


def feature_diversity_penalty(
    feat_student: Tensor,
    feats_reference: Sequence[Tensor],
    scale: float = _DEFAULT_SCALE,
) -> Tensor:
    """Mean pairwise feature-diversity penalty over a list of references.

    Parameters
    ----------
    feat_student:
        Student feature map, ``(B, C, T)``.
    feats_reference:
        Iterable of reference feature maps, each ``(B, C, T)``. They are
        detached internally so callers do not need to remember to do it.
    scale:
        Per-pair divisor; see :func:`pairwise_diversity_penalty`.
    """
    if not feats_reference:
        return torch.zeros((), device=feat_student.device)

    penalties = [
        pairwise_diversity_penalty(feat_student, ref.detach(), scale=scale)
        for ref in feats_reference
    ]
    return torch.stack(penalties).mean()
