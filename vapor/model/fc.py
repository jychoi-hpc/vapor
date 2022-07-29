import torch.nn as nn

class ResidualBlockLinear(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlockLinear, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.PReLU(),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class FC(nn.Module):
    def __init__(self, in_channels, out_channels, nbatch, nh, nw):
        super(FC, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nbatch = nbatch
        self.nh = nh
        self.nw = nw
        
        n0 = self.in_channels * self.nh * self.nw
        n1 = self.out_channels * self.nh * self.nw
        
        self.conv = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(n0, n0),
            ResidualBlockLinear(n0),
            nn.ReLU(),
            nn.Linear(n0, n1),
            # nn.PReLU(),
            # nn.Linear(n1, n1),
            ResidualBlockLinear(n1),
            nn.ReLU(),
            nn.Unflatten(1, (self.out_channels, self.nh, self.nw))
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x