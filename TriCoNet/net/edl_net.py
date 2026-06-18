import torch
from torch import nn
import torch.nn.functional as F


class EDLNet(nn.Module):
    def __init__(self, n_feats=32):
        super().__init__()

        self.edge_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.edge_gate_enhance = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.Sigmoid()
        )

        self.conv_local = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True)
        )

        self.conv_context = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True)
        )

        self.edge_gate = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, bias=False),
            nn.Sigmoid()
        )

        self.out_edge = nn.Conv2d(n_feats, 1, 1)

    def forward(self, x):
        low_freq = self.edge_pool(x)
        edge = x - low_freq
        w_edge = self.edge_gate_enhance(edge)
        x = x + edge * w_edge  

        feat = self.conv_local(x) + self.conv_context(x)

        w = self.edge_gate(feat)
        feat = x + feat * w

        edge_map = torch.tanh(self.out_edge(feat))

        return feat, edge_map

if __name__ == "__main__":
    x = torch.randn(2, 32, 144, 144)
    net = EDLNet()
    f, e = net(x)
    print(f.shape, e.shape)
