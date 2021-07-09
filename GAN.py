import torch
from torch import nn
from torch.autograd import Variable


class Discriminator(nn.Module):
    def __init__(self, dim=1000, nz=1000):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.nz = nz
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.nz),
            #nn.BatchNorm1d(self.nz),
            nn.ReLU(),
            nn.Linear(self.nz, self.nz),
            #nn.BatchNorm1d(self.nz),
            nn.ReLU(),
            nn.Linear(self.nz, self.nz),
            #nn.BatchNorm1d(self.nz),
            nn.ReLU(),
            nn.Linear(self.nz, 1),
        )

    def forward(self, x):
        #x = x.view(-1, self.nz)
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, nz, hidden):
        super(Generator, self).__init__()
        self.nz = nz
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(self.nz, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(), 
            nn.Linear(self.hidden, self.nz),
        )
        self.scale = nn.Sequential(
            nn.Linear(self.nz, self.nz),
            #nn.BatchNorm1d(self.nz),
            nn.ReLU(),
            nn.Linear(self.nz, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x), self.scale(x)
