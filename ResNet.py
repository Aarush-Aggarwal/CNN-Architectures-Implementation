import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_c, inter_c, I_downsample=None, stride=1) -> None:
        super(Block, self).__init__()
        self.expansion = 4
        self.Conv1 = nn.Conv2d(in_c, inter_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.BN1   = nn.BatchNorm2d(inter_c) 
        self.Conv2 = nn.Conv2d(inter_c, inter_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN2   = nn.BatchNorm2d(inter_c)
        self.Conv3 = nn.Conv2d(inter_c, inter_c * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.BN3   = nn.BatchNorm2d(inter_c*self.expansion)
        self.ReLU  = nn.ReLU()
        self.I_downsample = I_downsample
        self.stride = stride
        
    def forward(self, x) :
        I = x
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.ReLU(x)
        x = self.Conv2(x)
        x = self.BN2(x)
        x = self.ReLU(x)
        x = self.Conv3(x)
        x = self.BN3(x)
        
        if self.I_downsample is not None:
            I = self.I_downsample(I)
        
        x += I
        x = self.ReLU(x)
        
        return x
    

class ResNet(nn.Module):
    def __init__(self, Block, layers_repeat, image_channels, num_classes) -> None:
        super(ResNet, self).__init__()
        self.in_c  = 64
        self.Conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.BN1   = nn.BatchNorm2d(64)
        self.ReLU  = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Layers
        self.layer1 = self.make_layer(Block, 64,  layers_repeat[0], stride=1)
        self.layer2 = self.make_layer(Block, 128, layers_repeat[1], stride=2)
        self.layer3 = self.make_layer(Block, 256, layers_repeat[2], stride=2)
        self.layer4 = self.make_layer(Block, 512, layers_repeat[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.ReLU(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def make_layer(self, Block, inter_c, num_residual_blocks, stride):
        I_downsample = None
        layers = []
        # Either if the input space is halved, or num_channels changes,
        # the Identity (skip connection) needs to be adapted so that it will be able to be added to the layer that is ahead
        if stride != 1 or self.in_c != inter_c*4:
            I_downsample = nn.Sequential(nn.Conv2d(self.in_c, inter_c*4, kernel_size=1, stride=stride, bias=False), 
                                         nn.BatchNorm2d(inter_c*4) 
                                        )
            
        layers.append(Block(self.in_c, inter_c, I_downsample, stride))
        
        # The expansion size is always 4 for ResNet 50,101,152
        self.in_c = inter_c * 4
        
        # For example, in first ResNet layer: 256 will be mapped to 64 as intermediate layer, then finally back to 256. 
        # Hence no I_downsample is needed, as stride = 1, and also num_channels is not changed
        for _ in range(num_residual_blocks-1):
            layers.append(Block(self.in_c, inter_c))
        
        return nn.Sequential(*layers)

    
def ResNet50(img_channels=3, num_classes=100):
    return ResNet(Block, [3, 4, 6, 3], img_channels, num_classes)

def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)

def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)
