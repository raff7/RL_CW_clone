import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Define your neural networks in this class. 
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.

class ValueNetwork(nn.Module):
	def __init__(self,input_dim = 15,hidden_dim = 30,nActions=4):
		super(ValueNetwork, self).__init__()

		self.l1 = nn.Linear(input_dim,hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, nActions)

	def forward(self, inputs) :
		h = F.leaky_relu(self.l1(inputs))
		h = F.leaky_relu(self.l2(h))
		value = self.l3(h)
		return value
