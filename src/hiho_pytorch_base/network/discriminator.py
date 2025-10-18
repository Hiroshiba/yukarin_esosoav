"""Discriminatorネットワークモジュール（HiFi-GANベース）"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import AvgPool1d, Conv1d, Conv2d
from torch.nn.utils import spectral_norm, weight_norm

from hiho_pytorch_base.network.vocoder import get_padding

LRELU_SLOPE = 0.1


class DiscriminatorP(nn.Module):
    """Period-based Discriminator"""

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:  # noqa: D102
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.flatten(1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """複数のPeriod-based Discriminatorを組み合わせたもの"""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(  # noqa: D102
        self,
        y: Tensor,  # (B, 1, wL)
        y_hat: Tensor,  # (B, 1, wL)
    ) -> tuple[list[Tensor], list[Tensor], list[list[Tensor]], list[list[Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for discriminator in self.discriminators:
            y_d_r, fmap_r = discriminator(y)
            y_d_g, fmap_g = discriminator(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def forward_list(  # noqa: D102
        self,
        y_list: list[Tensor],  # [(wL,)]
        y_hat_list: list[Tensor],  # [(wL,)]
    ) -> tuple[list[Tensor], list[Tensor], list[list[Tensor]], list[list[Tensor]]]:
        all_y_d_rs = []
        all_y_d_gs = []
        all_fmap_rs = []
        all_fmap_gs = []

        for y, y_hat in zip(y_list, y_hat_list, strict=True):
            y_batch = y.unsqueeze(0).unsqueeze(0)
            y_hat_batch = y_hat.unsqueeze(0).unsqueeze(0)

            y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.forward(y_batch, y_hat_batch)

            all_y_d_rs.append(y_d_rs)
            all_y_d_gs.append(y_d_gs)
            all_fmap_rs.append(fmap_rs)
            all_fmap_gs.append(fmap_gs)

        num_discriminators = len(all_y_d_rs[0])
        y_d_rs_merged = []
        y_d_gs_merged = []
        fmap_rs_merged = []
        fmap_gs_merged = []

        for i in range(num_discriminators):
            y_d_rs_merged.append(torch.cat([batch[i] for batch in all_y_d_rs], dim=0))
            y_d_gs_merged.append(torch.cat([batch[i] for batch in all_y_d_gs], dim=0))

            num_layers = len(all_fmap_rs[0][i])
            fmap_r_layers = []
            fmap_g_layers = []
            for j in range(num_layers):
                fmap_r_layers.append(
                    torch.cat([batch[i][j] for batch in all_fmap_rs], dim=0)
                )
                fmap_g_layers.append(
                    torch.cat([batch[i][j] for batch in all_fmap_gs], dim=0)
                )
            fmap_rs_merged.append(fmap_r_layers)
            fmap_gs_merged.append(fmap_g_layers)

        return y_d_rs_merged, y_d_gs_merged, fmap_rs_merged, fmap_gs_merged


class DiscriminatorS(nn.Module):
    """Scale-based Discriminator"""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(  # noqa: D102
        self,
        x: Tensor,  # (B, 1, wL)
    ) -> tuple[Tensor, list[Tensor]]:
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.flatten(1, -1)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """複数のScale-based Discriminatorを組み合わせたもの"""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def forward(  # noqa: D102
        self,
        y: Tensor,  # (B, 1, wL)
        y_hat: Tensor,  # (B, 1, wL)
    ) -> tuple[list[Tensor], list[Tensor], list[list[Tensor]], list[list[Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, discriminator in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = discriminator(y)
            y_d_g, fmap_g = discriminator(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def forward_list(  # noqa: D102
        self,
        y_list: list[Tensor],  # [(wL,)]
        y_hat_list: list[Tensor],  # [(wL,)]
    ) -> tuple[list[Tensor], list[Tensor], list[list[Tensor]], list[list[Tensor]]]:
        all_y_d_rs = []
        all_y_d_gs = []
        all_fmap_rs = []
        all_fmap_gs = []

        for y, y_hat in zip(y_list, y_hat_list, strict=True):
            y_batch = y.unsqueeze(0).unsqueeze(0)
            y_hat_batch = y_hat.unsqueeze(0).unsqueeze(0)

            y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.forward(y_batch, y_hat_batch)

            all_y_d_rs.append(y_d_rs)
            all_y_d_gs.append(y_d_gs)
            all_fmap_rs.append(fmap_rs)
            all_fmap_gs.append(fmap_gs)

        num_discriminators = len(all_y_d_rs[0])
        y_d_rs_merged = []
        y_d_gs_merged = []
        fmap_rs_merged = []
        fmap_gs_merged = []

        for i in range(num_discriminators):
            y_d_rs_merged.append(torch.cat([batch[i] for batch in all_y_d_rs], dim=0))
            y_d_gs_merged.append(torch.cat([batch[i] for batch in all_y_d_gs], dim=0))

            num_layers = len(all_fmap_rs[0][i])
            fmap_r_layers = []
            fmap_g_layers = []
            for j in range(num_layers):
                fmap_r_layers.append(
                    torch.cat([batch[i][j] for batch in all_fmap_rs], dim=0)
                )
                fmap_g_layers.append(
                    torch.cat([batch[i][j] for batch in all_fmap_gs], dim=0)
                )
            fmap_rs_merged.append(fmap_r_layers)
            fmap_gs_merged.append(fmap_g_layers)

        return y_d_rs_merged, y_d_gs_merged, fmap_rs_merged, fmap_gs_merged
