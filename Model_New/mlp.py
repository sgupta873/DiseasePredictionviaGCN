import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        Multi-Layer Perceptron (MLP) with dynamic depth.

        Args:
            num_layers (int): Total layers excluding input.
            input_dim (int): Input feature size.
            hidden_dim (int): Size of hidden layers.
            output_dim (int): Final output dimension.
        """
        super(MLP, self).__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.is_linear = num_layers == 1

        if self.is_linear:
            self.layers = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through MLP.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Processed output.
        """
        return self.layers(x) if not self.is_linear else self.layers(x)
