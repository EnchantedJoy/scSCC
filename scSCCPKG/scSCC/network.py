import torch.nn as nn
import torch
from torch.nn.functional import normalize
import torch.nn.functional as F


def buildNetwork(layers, dropoutRate, alpha, type, activation="relu"):
    ''' build network by layers

    Args: 
        layers: list, 
            dimension of each layer.
        type: decalre name of the layer
        activation: string, 
            type of activation function, default as "relu".

    Returns: 
        nn.Sequential(*net): built network
    '''
    net = []
    for i in range(1, len(layers)):
        if i == 1 and type == "encoder":
            net.append(nn.Dropout(p=dropoutRate))
            net.append(GaussianNoise(alpha=alpha))
        net.append(nn.Linear(layers[i - 1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":  ## define your activation here
            net.append(nn.Sigmoid())
        elif activation == "elu":
            net.append(nn.ELU())
        elif activation == "gelu":
            net.append(nn.GELU())
    return nn.Sequential(*net)


class GaussianNoise(nn.Module):
    def __init__(self, alpha=0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x + self.alpha * torch.randn_like(x)


## =========== scSCC network =======
class scSCCNet(nn.Module):
    def __init__(self,
                 input_dim,
                 z_dim,
                 headDim,
                 n_classes,
                 alpha,
                 activation,
                 dropoutRate=0.9,
                 swavTemperature=0.1,
                 enc_dim=[],
                 dec_dim=[]):
        super(scSCCNet, self).__init__()

        # network parameters
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.headDim = headDim
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.Temperature = swavTemperature

        self.n_classes = n_classes

        self.enc_network = buildNetwork([self.input_dim] + self.enc_dim,
                                        dropoutRate,
                                        alpha,
                                        type="encoder",
                                        activation=activation)
        self.fc_z = nn.Linear(enc_dim[-1], z_dim)
        self.projectionHead = nn.Sequential(
            nn.Linear(z_dim, self.headDim), nn.ReLU(),
            nn.Linear(self.headDim, self.headDim))

        ## declare prototypes, ref SwAV
        self.prototypes = nn.Linear(headDim, n_classes, bias=False)

        # self._set_params()

    def encoder(self, x):
        h = self.enc_network(x)
        return normalize(self.fc_z(h), dim=1)

    def forward(self, x):
        z = self.encoder(x)  ## normalized z
        ## projection
        inst = normalize(self.projectionHead(z), dim=1)  ## h

        score = self.prototypes(inst)  # score

        return inst, score

    def get_cluster(self, x):
        _, p = self.forward(x)
        return torch.argmax(F.log_softmax(p / self.Temperature, dim=1), dim=1)

    def _set_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
        return
