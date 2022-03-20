import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(out_c),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(out_c),
                                  nn.ReLU(inplace=True)
                                 )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, features=[64, 128, 256, 512]) -> None:
        super(UNet, self).__init__()
        self.ups   = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.maxpool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down convolution part of UNet 
        for feature in features:
            self.downs.append(DoubleConv(in_c, feature))
            in_c = feature
            
        # Up convolution part of UNet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(features*2, features))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_c, kernel_size=1)
     
        
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.maxpool_2x2(x)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i]
            skip_connection = skip_connections[i//2]
            if x.size != skip_connection.size:
                F.resize(x, size=skip_connection.shape[2:])
                
            skip_join = torch.concat([skip_connection,x], dim=1)    
            x = self.ups[i+1](skip_join)
        
        return self.final_conv(x)
