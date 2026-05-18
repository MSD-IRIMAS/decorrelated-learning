"""LITE: Lightweight Inception-based architecture for time series classification.

Adapted from:
    Ismail-Fawaz, A. et al. (2023). LITE: Light Inception with boosTing
    tEchniques for Time Series Classification.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Tuple

import torch
from torch import Tensor, nn

__all__ = ["LITE", "InceptionBlock", "HybridBlock", "FCNBlock"]


class HybridBlock(nn.Module):
    """Hybrid block with hand-crafted, non-trainable convolutional filters.

    Builds three families of fixed filters: increasing/decreasing detectors
    and peak detectors with multiple scales. Their weights are frozen.

    Parameters
    ----------
    input_channels:
        Number of channels in the input tensor.
    kernel_sizes:
        Kernel sizes used for the increase/decrease detector filters. Peak
        detector filters use the same kernel sizes (skipping the first).
    """

    def __init__(
        self,
        input_channels: int = 1,
        kernel_sizes: Sequence[int] = (2, 4, 8, 16, 32, 64),
    ) -> None:
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.layers = nn.ModuleList()

        # Increase detectors: alternating +/- where even indices are negative.
        for k in self.kernel_sizes:
            filt = torch.ones((input_channels, 1, k))
            idx = torch.arange(k)
            filt[:, :, idx % 2 == 0] *= -1
            self.layers.append(self._make_fixed_conv(input_channels, k, filt))

        # Decrease detectors: alternating +/- where odd indices are negative.
        for k in self.kernel_sizes:
            filt = torch.ones((input_channels, 1, k))
            idx = torch.arange(k)
            filt[:, :, idx % 2 > 0] *= -1
            self.layers.append(self._make_fixed_conv(input_channels, k, filt))

        # Peak detectors (skip smallest kernel as in the original LITE paper).
        for k in self.kernel_sizes[1:]:
            filt = self._build_peak_filter(input_channels, k)
            self.layers.append(
                self._make_fixed_conv(input_channels, k + k // 2, filt)
            )

        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _make_fixed_conv(in_channels: int, kernel_size: int, weight: Tensor) -> nn.Conv1d:
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding="same",
            bias=False,
        )
        with torch.no_grad():
            conv.weight = nn.Parameter(weight, requires_grad=False)
        return conv

    @staticmethod
    def _build_peak_filter(input_channels: int, kernel_size: int) -> Tensor:
        k = kernel_size
        filt = torch.zeros((k + k // 2, input_channels, 1))
        ramp = torch.linspace(0, 1, k // 4 + 1)[1:].reshape(-1, 1, 1)
        ramp_rev = ramp.flip(0)

        filt[0 : k // 4] = -ramp
        filt[k // 4 : k // 2] = -ramp_rev
        filt[k // 2 : 3 * k // 4] = 2 * ramp
        filt[3 * k // 4 : k] = 2 * ramp_rev
        filt[k : 5 * k // 4] = -ramp
        filt[5 * k // 4 :] = -ramp_rev
        return filt.permute(2, 1, 0)

    @property
    def out_channels(self) -> int:
        # Two families of len(kernel_sizes) + one family of len(kernel_sizes) - 1
        return 2 * len(self.kernel_sizes) + (len(self.kernel_sizes) - 1)

    def forward(self, x: Tensor) -> Tensor:
        outputs = [conv(x) for conv in self.layers]
        x = torch.cat(outputs, dim=1)
        return self.relu(x)


class InceptionBlock(nn.Module):
    """Multi-scale Inception block combining standard and hybrid convolutions."""

    def __init__(
        self,
        n_filters: int = 32,
        kernel_size: int = 40,
        dilation_rate: int = 1,
        stride: int = 1,
        use_hybrid_layer: bool = True,
        use_multiplexing: bool = True,
    ) -> None:
        super().__init__()
        self.use_hybrid_layer = use_hybrid_layer

        if use_multiplexing:
            n_convs = 3
            n_filters = 32
        else:
            n_convs = 1
            n_filters = n_filters * 3

        kernel_size_s = [kernel_size // (2**i) for i in range(n_convs)]

        self.inception_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=n_filters,
                    kernel_size=ks,
                    stride=stride,
                    padding="same",
                    dilation=dilation_rate,
                    bias=False,
                )
                for ks in kernel_size_s
            ]
        )

        if use_hybrid_layer:
            self.hybrid: nn.Module = HybridBlock(input_channels=1)
            out_channels = n_filters * n_convs + self.hybrid.out_channels
        else:
            self.hybrid = nn.Identity()
            out_channels = n_filters * n_convs

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._out_channels = out_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        outs = [layer(x) for layer in self.inception_layers]
        x = torch.cat(outs, dim=1)
        if self.use_hybrid_layer:
            x = torch.cat([x, self.hybrid(identity)], dim=1)
        x = self.bn(x)
        return self.relu(x)


class FCNBlock(nn.Module):
    """Depthwise-separable FCN block (depthwise + pointwise conv + BN + ReLU)."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        n_filters: int,
        dilation_rate: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            dilation=dilation_rate,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        depth_out = self.depthwise(x)
        x = self.pointwise(depth_out)
        x = self.bn(x)
        x = self.relu(x)
        return x, depth_out


class LITE(nn.Module):
    """LITE classifier for univariate time series.

    Parameters
    ----------
    length_TS:
        Length of the input time series.
    n_classes:
        Number of target classes.
    n_filters:
        Width of the FCN blocks.
    kernel_size:
        Initial kernel size of the inception block (the effective kernel size
        is ``kernel_size - 1`` as in the original implementation).
    use_custom_filters:
        Whether to attach the hand-crafted ``HybridBlock``.
    use_dilation:
        Reserved for API compatibility. Dilation values are fixed inside the
        FCN blocks (2 and 4) following the original paper.
    use_multiplexing:
        Whether to use three parallel convolutions of decreasing kernel size
        inside the inception block.
    """

    def __init__(
        self,
        length_TS: int,
        n_classes: int,
        n_filters: int = 32,
        kernel_size: int = 41,
        use_custom_filters: bool = True,
        use_dilation: bool = True,
        use_multiplexing: bool = True,
    ) -> None:
        super().__init__()
        self.length_TS = length_TS
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.use_custom_filters = use_custom_filters
        self.use_dilation = use_dilation
        self.use_multiplexing = use_multiplexing

        eff_kernel = kernel_size - 1
        self.inception = InceptionBlock(
            n_filters=n_filters,
            kernel_size=eff_kernel,
            dilation_rate=1,
            use_hybrid_layer=use_custom_filters,
            use_multiplexing=use_multiplexing,
        )

        eff_kernel //= 2
        self.fcn1 = FCNBlock(
            in_channels=self.inception.out_channels,
            kernel_size=eff_kernel,
            n_filters=n_filters,
            dilation_rate=2 if use_dilation else 1,
        )
        self.fcn2 = FCNBlock(
            in_channels=n_filters,
            kernel_size=eff_kernel // 2,
            n_filters=n_filters,
            dilation_rate=4 if use_dilation else 1,
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(n_filters, n_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass returning ``(logits, gap_features)``."""
        x = self.inception(x)
        x, _ = self.fcn1(x)
        x, _ = self.fcn2(x)
        x = self.gap(x).flatten(1)
        gap_features = x
        logits = self.classifier(x)
        return logits, gap_features

    def extract_features(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Forward pass returning logits and intermediate feature maps.

        The returned list contains the output of the inception block and the
        two FCN blocks, in order. The last element is the one used for the
        diversity penalty during co-training.
        """
        features: List[Tensor] = []
        x = self.inception(x)
        features.append(x)
        x, _ = self.fcn1(x)
        features.append(x)
        x, _ = self.fcn2(x)
        features.append(x)
        gap = self.gap(x).flatten(1)
        logits = self.classifier(gap)
        return logits, features
