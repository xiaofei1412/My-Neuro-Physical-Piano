"""
Pedal Prediction Network (PedalNet).
A U-Net based architecture acting as the 'Perception Module' in the 
Neuro-Physical Piano Synthesis system. It extracts continuous pedal 
control signals from input Mel-spectrograms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Standard Double Convolution Block: [Conv2d -> BatchNorm2d -> ReLU] x 2
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class PedalNet(nn.Module):
    """
    U-Net Architecture for Pedal Onset/Depth Regression.
    
    Inputs:
        x: Tensor of shape (Batch, 1, Freq, Time) representing Mel-spectrograms.
    Outputs:
        out: Tensor of shape (Batch, 1, Time) representing the continuous pedal 
             control curve bounded in [0, 1].
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(PedalNet, self).__init__()
        
        # --- Encoder (Downsampling) ---
        self.enc1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ConvBlock(64, 128)
        self.pool4 = nn.MaxPool2d(2)
        
        # --- Bottleneck ---
        self.bottleneck = ConvBlock(128, 256)
        
        # --- Decoder (Upsampling) ---
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256, 128) 
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        
        # --- Head ---
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder with skip connections and robust size matching
        d4 = self.up4(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:])
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:])
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:])
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:])
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output Logits
        out = self.final_conv(d1) 
        
        # Collapse Frequency Dimension (Max pooling along the frequency axis)
        out, _ = torch.max(out, dim=2) 
        
        # Sigmoid Activation for Regression bounding
        out = torch.sigmoid(out)
        return out