import torch
import torch.nn as nn
import torch.nn.functional as F


class CCINet(nn.Module):
    def __init__(self, icha=32):
        super().__init__()
        self.icha = icha

        def block(dilation):
            return nn.Sequential(
                nn.Conv2d(icha, icha, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(icha),
                nn.ReLU(inplace=True)
            )

        self.enc1 = block(1)
        self.enc2 = block(2)
        self.enc3 = block(3)

        self.consis_conv = nn.Sequential(
            nn.Conv2d(icha, icha, 3, padding=1, bias=False),
            nn.BatchNorm2d(icha),
            nn.ReLU(inplace=True)
        )

        self.diff_conv = nn.Sequential(
            nn.Conv2d(icha, icha, 3, padding=1, bias=False),
            nn.BatchNorm2d(icha),
            nn.ReLU(inplace=True)
        )

        self.consis_gate = nn.Sequential(
            nn.Conv2d(icha, 1, 1),
            nn.Sigmoid()
        )

        self.out_conv = nn.Conv2d(icha, 1, 1)

    def forward(self, fea_1, fea_2):
        f1 = self.enc3(self.enc2(self.enc1(fea_1)))
        f2 = self.enc3(self.enc2(self.enc1(fea_2)))

        consis = self.consis_conv(0.5 * (f1 + f2))
        diff = self.diff_conv(torch.abs(f1 - f2))

        gate = self.consis_gate(consis)   # [B,1,H,W]

        fuse = gate * consis + (1 - gate) * diff

        fuse_l = torch.tanh(self.out_conv(fuse))

        return fuse, fuse_l

if __name__ == '__main__':
    img_ct = torch.randn(1, 32, 144, 144)
    img_mrict = torch.randn(1, 32, 144, 144)

    cmk = CCINet()
    fuse_fea, fuse_l = cmk(img_mrict, img_ct)
    print(fuse_fea.shape)  # [1,32,144,144]
    print(fuse_l.shape)    # [1,1,144,144]
