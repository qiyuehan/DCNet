import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class DeformConv_experimental(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv_experimental, self).__init__()

        self.conv_channel_adjust = nn.Conv2d(in_channels=in_channels, out_channels=2 * kernel_size[0] * kernel_size[1],
                                             kernel_size=(1, 1))

        self.offset_net = nn.Conv2d(in_channels=2 * kernel_size[0] * kernel_size[1],
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    groups=2 * kernel_size[0] * kernel_size[1],
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        x_chan = self.conv_channel_adjust(x)
        offsets = self.offset_net(x_chan)
        out = self.deform_conv(x, offsets)
        return out


# LKA Deformable
class deformable_LKA_experimental(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv_experimental(dim, kernel_size=(5,5), padding=2, groups=dim)
        self.conv_spatial = DeformConv_experimental(dim, kernel_size=(7,7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

# Original
class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)  # [1, 384, 14, 14]
        return out

# GDC
class GatedDeformableConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=1, dilation=1):
        super(GatedDeformableConv2d, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding, dilation=dilation)
        self.gate_conv = nn.Conv2d(in_channels, 4, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation)
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation)

    def forward(self, x):
        # Calculate offsets
        offsets = self.offset_conv(x)

        # Calculate gating values
        gates = torch.sigmoid(self.gate_conv(x))

        # Create base grid
        N, C, H, W = x.shape
        base_grid = self._create_base_grid(N, H, W, offsets.device)

        # Reshape offsets to match the base grid
        offsets = offsets.permute(0, 2, 3, 1)
        offsets = offsets.view(N, H, W, 2 * -1 // (H * W), 2)
        offsets = offsets.sum(dim=3)

        # Apply deformable convolution
        grid = base_grid + offsets
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # Standard convolution
        x = self.conv(x)

        # Apply gating mechanism
        x = x * gates

        return x

    def _create_base_grid(self, N, H, W, device):
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
        grid = torch.stack((grid_x, grid_y), dim=-1).float()
        grid = grid.unsqueeze(0).expand(N, -1, -1, -1)
        grid = grid / (torch.tensor([W - 1, H - 1], device=device) / 2) - 1  # Normalize to [-1, 1]
        return grid

