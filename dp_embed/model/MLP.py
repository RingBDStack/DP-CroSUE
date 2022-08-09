import torch
from torch.nn import Linear
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, indim, hiddim, outdim):
        super(MLP,self).__init__()
        self.lin1 = Linear(indim, hiddim)
        self.lin2 = Linear(hiddim, outdim)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.sigmoid(self.lin2(x))
        return x

