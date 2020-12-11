import torch
import torch.nn as nn

class Adversary(nn.Module):
    def __init__(self,feature_dims):
        super().__init__()

        feature_dim = feature_dims[1]

        self.mlp = nn.Sequential(nn.Linear(2*feature_dim,feature_dim),nn.ReLU(),nn.Linear(feature_dim,feature_dim),nn.ReLU(),nn.Linear(feature_dim,1))



    def forward(self,combined_objs):
        return self.mlp(combined_objs)