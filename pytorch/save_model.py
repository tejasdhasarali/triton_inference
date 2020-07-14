import segmentation_models_pytorch as smp
import torch
import os

if __name__ == "__main__":
    model = smp.Unet('resnet34', classes=2, activation='softmax')
    traced = torch.jit.trace(model, torch.rand(1, 3, 256, 256))
    print(model)
    save_path = os.getcwd()+"/unet_pt/1/model.pt"
    torch.jit.save(traced, save_path)
    # pt.save(model, save_path)
    print("Model saved in", save_path)