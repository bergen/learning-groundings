import os
import torch

import torch.nn as nn
import torchvision
import types



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, args):
        super(SimCLR, self).__init__()

        self.args = args

        self.encoder = self.get_resnet(args.resnet)

        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.encoder.fc = Identity()  # remove fully-connected layer after pooling layer

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, args.projection_dim, bias=False),
        )
        

    def get_resnet(self, name):
        resnets = {
            "resnet18": torchvision.models.resnet18(),
            "resnet50": torchvision.models.resnet50(),
        }
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]


    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        #x = self.encoder.layer4(x)
        
        return x


def load_pretrained_simclr():
    args = types.SimpleNamespace()
    args.projection_dim = 64
    args.resnet_type = '18'
    args.resnet = 'resnet' + args.resnet_type
    model = SimCLR(args)

    model_fp = os.path.join(
        os.path.dirname(__file__), "cached_models/simclr_resnet_{}.tar".format(args.resnet_type)
    )
    model.load_state_dict(torch.load(model_fp))




    return model


