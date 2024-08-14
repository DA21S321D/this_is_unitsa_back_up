"""
    This file contains a neural network module for us to
    define our actor and critic networks in PPO.
"""
    
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from EAttention.eattention import EAttention
class FeedForwardNN(nn.Module):
    """
        A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """
    def __init__(self, in_dim, out_dim):
        """
            Initialize the network and set up the layers.

            Parameters:
                in_dim - input dimensions as an int
                out_dim - output dimensions as an int

            Return:
                None
        """
        super(FeedForwardNN, self).__init__()

        self.feature_extractor = EAttention(in_dim)  # 这里需要in_dim = observation_space

        self.layer1 = nn.Linear(64, 64) #特征提取出来是（1，320）向量
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        """
            Runs a forward pass on the neural network.

            Parameters:
                obs - observation to pass as input

            Return:
                output - the output of our forward pass
        """

        # Convert observation to tensor if it's a numpy array

        features = self.feature_extractor(obs)
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float)
        features1 = features

        activation1 = F.relu(self.layer1(features)) #layer1的形状是420*64
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output