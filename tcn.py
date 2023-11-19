import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    """A temporal block that performs 1D convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        """Creates a temporal block.

        The temporal block is a simple 1D convolution, followed by a ReLU non-linearity and a dropout.

        :arg
            in_channels (int): the number of input channels to the 1D convolution.
            out_channels (int): the number of output channels of the 1D convolution.
            kernel_size (int): the filter size of the 1D convolution.
            dilation (int): the spacing between values in the kernel.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation,
                               padding=(kernel_size - 1) * dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x


class TCN(nn.Module):
    """A Temporal Convolutional Network (TCN)."""

    def __init__(self, num_inputs: int, num_channels: int, kernel_size: int, num_blocks: int):
        """Creates a TCN.

        :arg
            num_inputs (int): the number of input features.
            num_channels (int): the number of channels in each temporal convolution.
            kernel_size (int): the size of the kernel used in the temporal convolution operation.
            num_blocks (int): the number of temporal blocks in the network.
        """
        super().__init__()

        layers = []

        for i in range(num_blocks):
            # the dilation of each subsequent layer is doubled to increase the receptive field
            dilation = 2 ** i

            # each temporal block has the same number of inputs, channels, and kernel size, but a different dilation
            block = TemporalBlock(num_inputs, num_channels, kernel_size, dilation)
            layers.append(block)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    # Example usage:
    batch_size = 32

    # the number of frames in our data
    num_frames = 25

    # the number of joints
    num_joints = 32

    # the number of spatial dimensions
    spatial_dimensions = 3

    # Assuming input with shape (batch_size, num_frames, num_joints, spatial_dimensions)
    input_data = torch.randn((batch_size, num_frames, num_joints, spatial_dimensions))

    # Create TCN model
    # this is our sequence length
    input_size = num_joints * spatial_dimensions  # Fix here
    num_channels = 64
    kernel_size = 3
    num_blocks = 1

    model = TCN(input_size, num_channels, kernel_size, num_blocks)

    # Forward pass
    output = model(input_data.view(batch_size, num_frames, -1))  # Reshape input_data
