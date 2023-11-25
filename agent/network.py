import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Deep Q Network (DQN) model.

        Parameters:
            input_size (int): Size of the input state.
            output_size (int): Size of the output action space.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * input_size * input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def _get_conv_output_size(self, input_size: int) -> int:
        """
        Calculate the output size after passing through convolutional layers.

        Parameters:
            input_size (int): Size of the input state.

        Returns:
            int: Output size after passing through convolutional layers.
        """
        # Create a dummy tensor to calculate the size
        dummy_input = torch.randn(1, 1, input_size, input_size)
        x = self.conv1(dummy_input)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.view(-1, 1, self.input_size, self.input_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
