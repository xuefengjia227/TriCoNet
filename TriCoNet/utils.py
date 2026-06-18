import string
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from glob import glob
import os
from PIL import Image, ImageFile
import torch
import time
import cv2


ImageFile.LOAD_TRUNCATED_IMAGES = True
_tensor = transforms.ToTensor()
_pil_gray = transforms.ToPILImage()
device = 'cuda'

def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x

def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def load_img(img_path):
    img = Image.open(img_path)
    img = img.convert('L')
    return _tensor(img).unsqueeze(0)

class Strategy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y1, y2):
        return (y1 + y2) / 2

def read_image(path):
    I = np.array(Image.open(path))
    return I

def fusion(x1, x2, model):
    with torch.no_grad():
        start = time.time()
        out_f = model(x1, x2)[0]
        out_cpu = out_f.detach().cpu()
        vmin = out_cpu.min().item()
        vmax = out_cpu.max().item()

        if vmin < 0.0 or vmax > 1.0:
            out_cpu = (out_cpu + 1.0) / 2.0

        out_cpu = out_cpu.clamp(0.0, 1.0)
        time_used = time.time() - start
        print(f"fusion time: {time_used:.4f}s, raw range before clamp: {vmin:.4f}-{vmax:.4f}, after map: {out_cpu.min().item():.4f}-{out_cpu.max().item():.4f}")
        return out_cpu

class Test:
    def __init__(self):
        pass

    def load_imgs(self, img1_path, img2_path, device):
        img1 = load_img(img1_path).to(device)
        img2 = load_img(img2_path).to(device)
        return img1, img2

    def save_imgs(self, save_path, save_name, img_fusion):
        mkdir(save_path)
        save_path = os.path.join(save_path, save_name)
        img_fusion.save(save_path)

class test_gray(Test):
    def __init__(self):
        super().__init__()
        self.img_type = 'gray'

    def get_fusion(self, img1_path, img2_path, model,
                   save_path='none', save_name='none'):
        img1, img2 = self.load_imgs(img1_path, img2_path, device)
        img_fusion = fusion(x1=img1, x2=img2, model=model)

        img_fusion_tensor = img_fusion[0].squeeze(0)
        img_fusion_pil = _pil_gray(img_fusion_tensor)

        self.save_imgs(save_path, save_name, img_fusion_pil)
        return img_fusion_pil

def test(test_path, model, save_path='./test_result/'):
    img_list = glob(test_path + '*')
    img_num = len(img_list) / 2
    suffix = img_list[0].split('.')[-1]
    img_name_list = list(
        set([img_list[i].split('/')[-1].split('.')[0].strip(string.digits) for i in range(len(img_list))]))

    fusion_phase = test_gray()

    for i in range(int(img_num)):
        img1_path = test_path + img_name_list[0] + str(i) + '.' + suffix
        img2_path = test_path + img_name_list[1] + str(i) + '.' + suffix
        save_name = 'fusion_' + str(i) + '.' + suffix
        fusion_phase.get_fusion(img1_path, img2_path, model,
                                save_path=save_path, save_name=save_name)

def ycrcb_to_rgb_tensor(y, cr, cb):
    y = (y.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    cr = (cr.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    cb = (cb.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    ycrcb = cv2.merge([y, cr, cb])
    rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(rgb)

class test_rgb(Test):
    def __init__(self):
        super().__init__()
        self.img_type = 'rgb'

    def get_fusion(self, mri_path, pet_rgb_path, model,
                   save_path='none', save_name='none'):
        img1 = load_img(mri_path).to(device)

        pet_rgb = Image.open(pet_rgb_path).convert('RGB')
        pet_np = cv2.cvtColor(np.array(pet_rgb), cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(pet_np)

        pet_y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float() / 255.0  # [1,1,H,W]
        cr = torch.from_numpy(cr).unsqueeze(0).float() / 255.0
        cb = torch.from_numpy(cb).unsqueeze(0).float() / 255.0

        pet_y = pet_y.to(device)

        img_fusion = fusion(img1, pet_y, model)
        y_tensor = img_fusion[0].squeeze(0)

        fusion_rgb = ycrcb_to_rgb_tensor(y_tensor, cr, cb)
        self.save_imgs(save_path, save_name, fusion_rgb)
        return fusion_rgb
    
def test_gray_dataset(test_root, model, save_path='./test_result/',
                      modality1='MRI', modality2='CT'):

    m1_dir = os.path.join(test_root, modality1)
    m2_dir = os.path.join(test_root, modality2)

    files = sorted(os.listdir(m1_dir))

    fusion_phase = test_gray()

    for name in files:
        img1_path = os.path.join(m1_dir, name)
        img2_path = os.path.join(m2_dir, name)
        fusion_phase.get_fusion(img1_path, img2_path, model,
                                save_path=save_path, save_name=name)

def test_rgb_dataset(test_root, model, save_path='./test_result/',
                     modality1='MRI', modality2='PET'):

    mri_dir = os.path.join(test_root, modality1)
    pet_dir = os.path.join(test_root, modality2)
    files = sorted(os.listdir(mri_dir))

    fusion_phase = test_rgb()

    for name in files:
        mri_path = os.path.join(mri_dir, name)
        pet_path = os.path.join(pet_dir, name)
        fusion_phase.get_fusion(mri_path, pet_path, model,
                                save_path=save_path, save_name=name)
