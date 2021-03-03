"""Defines the neural network and the loss function"""

import torch
from torch import nn, optim, cat

# Codesnippets taken from github milesial / Pytorch-UNet, and changed to 3D functionallity by Matthias Eder
# https://github.com/milesial/Pytorch-UNet

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


class VolAutoEncoder(nn.Module):
    """
       This is the standard way to define a network in PyTorch. The components
       (layers) of the network are defined in the __init__ function.
       Then, in the forward function it is defined how to apply these layers on the input step-by-step.
    """

    def __init__(self):
        super(VolAutoEncoder, self).__init__()

        self.doubleConv1 = DoubleConv(1, 64)
        self.downConv1 = nn.Conv3d(64, 128,  (8, 8, 8), stride=3, padding=1) #9x9x9
        self.doubleConv2 = DoubleConv(128, 128)

        self.Transpose9x9 = nn.ConvTranspose3d(128, 64, 6, 3)
        self.doubleConv3 = DoubleConv(128, 64)

        self.conv1x1 = nn.Conv3d(64, 1, (1, 1, 1), stride=1)
        self.dropout = nn.Dropout(p=0.5)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU( inplace=True )

    def forward(self, x):
        """
        This function defines how to use the components of the network to operate on an input batch.
        """
        x = self.doubleConv1(x)
        x30 = self.dropout(x)

        x = self.downConv1(x)
        x = self.relu(x)
        x = self.doubleConv2(x)

        x = self.Transpose9x9(x)
        x = self.relu(x)
        x = torch.cat([x, x30], dim=1)
        x = self.doubleConv3(x)
        x = self.conv1x1(x)

        x = self.sigmoid(x)
        return x

def loss_fn(outputs, targets):
    """
    Computes the cross entropy loss given outputs and labels
    """
    loss = nn.BCELoss()

    return loss(outputs, targets)