import torch
import torch.nn as nn
import torch.nn.functional as F

import math


# Define your neural networks in this class.
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.

class ValueNetwork(nn.Module):
    def __init__(self, input_dim=68, num_actions=4, hidden_dim=64):
        super(ValueNetwork, self).__init__()

        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_values = nn.Linear(hidden_dim, num_actions)


    def forward(self, inputs):
        h1 = F.elu(self.layer_1(inputs))
        h2 = F.elu(self.layer_2(h1))

        return self.q_values(h2)
