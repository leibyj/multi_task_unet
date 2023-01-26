import torch
from torch import nn
import torch.nn.functional as F

class ATTN_module(nn.Module):
  def __init__(self, dims = [512, 256], dropout = True):
    """
    Basic 3 layer attention network
    Forward function will return attention tensor (Px1) and original input
    """
    super(ATTN_module, self).__init__()

    self.net = [nn.ReLU(), nn.Linear(dims[0], dims[1]), nn.ReLU()]
    if dropout:
            self.net.append(nn.Dropout(0.25))

    self.net.append(nn.Linear(dims[1], 1, bias=False))

    self.net = nn.Sequential(*self.net)

  def forward(self, x):
    return self.net(x), x


class ATTN_module_gated(nn.Module):
  def __init__(self, dims = [512, 256], dropout = True):
    """
    Gated attention network 
    Forward function will return attention tensor (Px1) and original input
    """
    super(ATTN_module_gated, self).__init__()

    # Tanh arm
    self.net_t = [nn.ReLU(), nn.Linear(dims[0], dims[1]), nn.Tanh()]
    # Signmoid arm
    self.net_s = [nn.ReLU(), nn.Linear(dims[0], dims[1]), nn.Sigmoid()]

    if dropout:
            self.net_t.append(nn.Dropout(0.25))
            self.net_s.append(nn.Dropout(0.25))

    self.net_t = nn.Sequential(*self.net_t)
    self.net_s = nn.Sequential(*self.net_s)

    self.net_combined = nn.Linear(dims[1], 1, bias=False)

  def forward(self, x):
    t = self.net_t(x)
    s = self.net_s(x)
    A = torch.mul(t,s)
    return self.net_combined(A), x


class MIL_model(nn.Module):
    def __init__(self, dims = [1120, 512, 256], dropout = True, return_features=True):
        super(MIL_model, self).__init__()

        self.return_feats = return_features

        fc = [nn.Linear(dims[0], dims[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(dims[1], dims[1])]) #, nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.append(ATTN_module_gated([dims[1], dims[2]], dropout = dropout))

        self.attn_net = nn.Sequential(*fc)

        self.classifier = nn.Sequential(
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        A, x = self.attn_net(x) # A.shape == Px1
        A = F.softmax(A, dim=0)
        # A * feature map
        x = torch.mul(A, x) # x.shape == Px512
        # collapse 
        x = x.sum(dim=0) # x.shape == 512
        if self.return_feats:
            feats = self.classifier[0:2](x)
            return self.classifier[2:](feats), A, feats

        out = self.classifier(x)
        return out, A