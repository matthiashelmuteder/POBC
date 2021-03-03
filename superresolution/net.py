"""Defines the neural network and the loss function"""

import torch
from torch import nn, optim

def pixel_shuffle(input, upscale_factor):
# Autor linzwatt @ https://github.com/pytorch/pytorch/pull/6340/files
    input_size = list(input.size())
    dimensionality = len(input_size) - 2

    input_size[1] //= (upscale_factor ** dimensionality)
    output_size = [dim * upscale_factor for dim in input_size[2:]]

    input_view = input.contiguous().view(
        input_size[0], input_size[1],
        *(([upscale_factor] * dimensionality) + input_size[2:])
    )

    indicies = list(range(2, 2 + 2 * dimensionality))
    indicies = indicies[1::2] + indicies[0::2]

    shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()
    return shuffle_out.view(input_size[0], input_size[1], *output_size)

class PixelShuffle3D(nn.Module):

    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_shuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResidualDoubleConv(nn.Module):

    def __init__(self, channels):
        super().__init__()
        
        self.doubleConv = DoubleConv(channels, channels)

    def forward(self, x):
        return x + self.doubleConv(x)


class VSSR(nn.Module):
    """
       This is the standard way to define a network in PyTorch. The components
       (layers) of the network are defined in the __init__ function.
       Then, in the forward function it is defined how to apply these layers on the input step-by-step.
    """

    def __init__(self):
        super(VSSR, self).__init__()

        self.doubleConv1 = DoubleConv(1, 64)
        self.downConv1 = nn.Conv3d(64, 128,  (6, 6, 6), stride=2, padding=2) #15x15x15
        self.doubleConv2 = ResidualDoubleConv(128)

        self.downConv2 = nn.Conv3d(128, 256,  (5, 5, 5), stride=2, padding=0) #6x6x6
        self.doubleConv3 = ResidualDoubleConv(256)

        self.Transpose6x6 = nn.ConvTranspose3d(256, 128, 5, 2)
        self.doubleConv4 = DoubleConv(256, 128)

        self.conv1x1 = nn.Conv3d(64, 1, (1, 1, 1), stride=1)
        self.conv1 = nn.Conv3d(32, 1, (3, 3, 3), stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU( inplace=True )

        self.pixelShuffle = PixelShuffle3D(2)
        self.doubleConv5 = DoubleConv(16, 64)
        self.doubleConv6 = DoubleConv(128, 256, 128)

        self.conv3x3 = nn.Sequential(
                        nn.Conv3d(32, 32,  3, stride=1, padding=1),
                        nn.BatchNorm3d(32),
                        nn.ReLU(inplace=True) )
        self.conv5x5 = nn.Sequential(
                        nn.Conv3d(32, 32,  5, stride=1, padding=2),
                        nn.BatchNorm3d(32),
                        nn.ReLU(inplace=True) )

    def forward(self, x):
        """
        This function defines how to use the components of the network to operate on an input batch.
        """
        x30 = self.doubleConv1(x)

        x = self.downConv1(x30)
        x = self.relu(x)
        x15 = self.doubleConv2(x)

        x = self.downConv2(x15)
        x = self.relu(x)
        x = self.doubleConv3(x)

        x = self.Transpose6x6(x)
        x = self.relu(x)
        x = torch.cat([x, x15], dim=1)
        x = self.doubleConv4(x)

        x = self.pixelShuffle(x)
        x = self.relu(x)
        x = self.doubleConv5(x)
        x = torch.cat([x, x30], dim=1)
        x = self.doubleConv6(x)

        x = self.pixelShuffle(x)
        x = self.relu(x)

        x_bigFilter = self.conv5x5(x)

        x = torch.cat([x, x_bigFilter], dim=1)

        x = self.conv1x1(x)

        # x = self.sigmoid(x)
        return x

def loss_fn(outputs, targets):
    """
    Computes the cross entropy loss given outputs and labels
    """
    loss = nn.BCEWithLogitsLoss()

    return loss(outputs, targets)