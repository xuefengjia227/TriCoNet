import math
import torch.nn as nn
from ssim import SSIM
import torch
import torch.nn.functional as F


class EDLLoss(nn.Module):
    def __init__(self):
        super(EDLLoss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_vis, generate_img):
        vis_grad = self.sobelconv(image_vis)
        generate_img_grad = self.sobelconv(generate_img)
        loss_grad = F.l1_loss(generate_img_grad, vis_grad)

        return loss_grad


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class SSGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8
        self.k = 7
        self.pad = self.k // 2

    def forward(self, ssd_map):
        ssd_map = (ssd_map + 1.0) / 2.0
        ssd_map = ssd_map.clamp(0.0, 1.0)

        mu_ssd = F.avg_pool2d(ssd_map, self.k, stride=1, padding=self.pad)
        var_ssd = F.avg_pool2d(ssd_map**2, self.k, stride=1, padding=self.pad) - mu_ssd**2

        loss_l1 = F.l1_loss(ssd_map, mu_ssd) / (mu_ssd.mean().detach() + self.eps)
        loss_var = F.l1_loss(var_ssd, var_ssd.mean(dim=[2,3], keepdim=True)) / (var_ssd.mean().detach() + self.eps)

        return loss_l1 + loss_var


class CCILoss(nn.Module):
    def __init__(self):
        super().__init__()
        ksize, sigma = 11, 2.0
        ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / kernel.sum()
        self.register_buffer("g", kernel.view(1, 1, ksize, ksize))
        self.pad = ksize // 2

    def blur(self, x):
        return F.conv2d(x, self.g, padding=self.pad)

    def forward(self, img1, img2, f_img):
        img1 = img1.clamp(0, 1)
        img2 = img2.clamp(0, 1)
        f_img = (f_img + 1) / 2
        f_img = f_img.clamp(0, 1)

        s1 = torch.abs(img1 - self.blur(img1))
        s2 = torch.abs(img2 - self.blur(img2))

        consis = torch.exp(-torch.abs(s1 - s2))
        consis = consis.detach()

        loss = consis * (
            torch.abs(f_img - img1) +
            torch.abs(f_img - img2)
        )

        return loss.mean()


class FusionLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ssim_loss = SSIM()
        self.sobel = Sobelxy()

    @staticmethod
    def charbonnier(x, eps=1e-3):
        return torch.mean(torch.sqrt(x * x + eps * eps))

    def forward(self, img1, img2, fused):
        fused = fused.clamp(0.0, 1.0)

        ssim1 = (1 - self.ssim_loss(img1, fused)) / 2
        ssim2 = (1 - self.ssim_loss(img2, fused)) / 2
        loss_ssim = self.alpha * (ssim1 + ssim2)

        g1 = self.sobel(img1).abs()
        g2 = self.sobel(img2).abs()
        gf = self.sobel(fused).abs()

        target = torch.sqrt(g1 * g1 + g2 * g2 + 1e-6)
        loss_edge = self.beta * self.charbonnier(gf - target)

        return loss_ssim + loss_edge
