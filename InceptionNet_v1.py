import torch
import torch.nn as nn

class InceptionNet(nn.Module):
    def __init__(self, num_classes, aux_logits=True) -> None:
        super(InceptionNet, self).__init__()
        
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits
        
        # Conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0) ###########
        
        self.conv1 = Conv_block(3, 64, (7,7), 2, 3)
        self.maxpool1 = Conv_block(64, 64, (3,3), 2, 1)
        self.conv2 = Conv_block(64, 192, (3,3), 1, 1)
        self.maxpool2 = Conv_block(192, 192, (3,3), 2, 1)
        
        # Inception_block(in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool)
        
        self.incept3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.incept3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.incept4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.incept4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.incept4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.incept4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.incept4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.incept5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.incept5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool  = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout  = nn.Dropout(p=0.4)
        self.fc1      = nn.Linear(1024, num_classes)

    def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.maxpool2(x)

            x = self.inception3a(x)
            x = self.inception3b(x)
            x = self.maxpool3(x)

            x = self.inception4a(x)

            # Auxiliary Softmax classifier 1
            if self.aux_logits and self.training:
                aux1 = self.aux1(x)

            x = self.inception4b(x)
            x = self.inception4c(x)
            x = self.inception4d(x)

            # Auxiliary Softmax classifier 2
            if self.aux_logits and self.training:
                aux2 = self.aux2(x)

            x = self.inception4e(x)
            x = self.maxpool4(x)
            x = self.inception5a(x)
            x = self.inception5b(x)
            x = self.avgpool(x)
            x = x.reshape(x.shape[0], -1)
            x = self.dropout(x)
            x = self.fc1(x)

            if self.aux_logits and self.training:
                return aux1, aux2, x
            else:
                return x


class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool) -> None:
        super(Inception_block, self).__init__()
        self.branch1 = Conv_block(in_channels, out_1x1, kernel_size=(1,1))
        self.branch2 = nn.Sequential(Conv_block(in_channels, red_3x3, kernel_size=(1,1)),
                                     Conv_block(red_3x3, out_3x3, kernel_size=(3,3), stride=1, padding=1)
                                    )
        self.branch3 = nn.Sequential(Conv_block(in_channels, red_5x5, kernel_size=(1,1)),
                                     Conv_block(red_5x5, out_5x5, kernel_size=(5,5), stride=1, padding=3)
                                    )
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=(3,3), padding=1),
                                     Conv_block(in_channels, out_1x1pool, kernel_size=(1,1))
                                    )
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = Conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0) -> None:
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
    
    
if __name__ == "__main__":
    # N = 3 (Mini batch size)
    x = torch.randn(3, 3, 224, 224)
    model = InceptionNet(num_classes=1000, aux_logits=True)
