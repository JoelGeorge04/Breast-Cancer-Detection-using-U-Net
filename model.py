import torch
import torch.nn as nn

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU()
    )

def build_unet():
    model = nn.Sequential()
    # this is only a partial unet but simple enough
    model.down1 = double_conv(3, 64)
    model.pool1 = nn.MaxPool2d(2)

    model.down2 = double_conv(64, 128)
    model.pool2 = nn.MaxPool2d(2)

    model.bridge = double_conv(128, 256)

    model.up1_trans = nn.ConvTranspose2d(256, 128, 2, stride=2)
    model.up1 = double_conv(256, 128)

    model.up2_trans = nn.ConvTranspose2d(128, 64, 2, stride=2)
    model.up2 = double_conv(128, 64)

    model.final = nn.Conv2d(64, 1, 1)

    return model
