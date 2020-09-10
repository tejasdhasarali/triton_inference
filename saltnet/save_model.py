# import segmentation_models_pytorch as smp
import torch
import os
import torch.nn as nn
from torchvision import models  
import torch.nn.functional as F
from linknet34_attn import LinkNet34_attn
from collections import OrderedDict

model = LinkNet34_attn()
state_dict = torch.load("LINK34_ATTN_BOS1_final.pkl", map_location=lambda storage, loc: storage)['model_state']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
#print(new_state_dict.keys())
model.load_state_dict(new_state_dict)
traced = torch.jit.trace(model, torch.rand(1,3,512,512))
save_path = "/homedirs/tejahasa/TGS/triton_inference/saltnet/model.pt"
torch.jit.save(traced, save_path)
# print(model)
