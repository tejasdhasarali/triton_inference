# import segmentation_models_pytorch as smp
import torch
import os
import torch.nn as nn
from torchvision import models
from torchvision.models import 
import torch.nn.functional as F

if __name__ == "__main__":
    # model = smp.Unet('resnet34', classes=2, activation='softmax')
    model = torch.load("linknet.pt")
    state_dict = torch.load("LINK34_ATTN_BOS1__10mar2020_nemo512_final.pkl")
    model.load_state_dict(state_dict)
    traced = torch.jit.trace(model, torch.rand(1, 3, 512, 512))
    print(model)
    save_path = "/home/tejas/Compgeom/triton/scripts/salnet/Linknet/1/model.pt"
    torch.jit.save(traced, save_path)
    # pt.save(model, save_path)
    # print("Model saved in", save_path)