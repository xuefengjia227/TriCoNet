import torch
import torch.nn as nn
import torch.nn.functional as F


class SSGNet(nn.Module):
    def __init__(self, func_ch=1, ref_ch=1, embed_dim=32):
        super().__init__()

        self.spa_enc = nn.Sequential(
            nn.Conv2d(func_ch + 1, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, embed_dim)
        )

        self.ref_tok = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Conv2d(ref_ch, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # [B,16,1,1]
        )

        self.func_stat = nn.Sequential(
            nn.Conv2d(func_ch * 2, embed_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 1)
        )

        self.func_dist_tok = nn.Sequential(
            nn.Conv2d(func_ch, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # [B,E,1,1]
        )

        self.tok_fuse = nn.Sequential(
            nn.Conv2d(embed_dim + 16, embed_dim, 1),
            nn.ReLU(inplace=True)
        )

        self.film = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim * 2, 1)  # gamma,beta
        )

        self.sem_map_head = nn.Conv2d(embed_dim, 1, 1)

    @staticmethod
    def _blur_ref(x_ref, k=9):
        return F.avg_pool2d(x_ref, k, stride=1, padding=k // 2)

    def forward(self, x_func, x_ref):
        ref_blur = self._blur_ref(x_ref, k=9)
        diff = torch.abs(x_func - ref_blur)
        spa_in = torch.cat([x_func, diff], dim=1)  # [B,2,H,W]
        feat = self.spa_enc(spa_in)  # [B,E,H,W]

        r_tok = self.ref_tok(x_ref)  # [B,16,1,1]
        mean = x_func.mean(dim=[2, 3], keepdim=True)
        std = x_func.std(dim=[2, 3], keepdim=True) + 1e-6
        stat = torch.cat([mean, std], dim=1)  # [B,2,1,1]
        stat_e = self.func_stat(stat)  # [B,E,1,1]
        dist_e = self.func_dist_tok(x_func)  # [B,E,1,1]

        tok = self.tok_fuse(torch.cat([stat_e + dist_e, r_tok], dim=1))  # [B,E,1,1]

        gb = self.film(tok)  # [B,2E,1,1]
        gamma, beta = torch.chunk(gb, 2, dim=1)
        mod_feat = feat * (1 + torch.tanh(gamma)) + beta  # [B,E,H,W]

        sem_map = torch.tanh(self.sem_map_head(feat))

        return mod_feat, sem_map

if __name__ == "__main__":
    x_func = torch.randn(2, 1, 144, 144)
    x_ref  = torch.randn(2, 1, 144, 144)

    net = SSGNet(func_ch=1, ref_ch=1, embed_dim=32)
    semantic, sem_map = net(x_func, x_ref)

    print(semantic.shape, sem_map.shape)
