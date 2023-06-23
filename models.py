import torch
import torch.nn as nn
import torch.nn.functional as F

# C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10

class Model0(nn.Module):
    def __init__(self):
        super(Model0, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1, bias=False), # 32 >> 32 || 1 >> 3
            nn.ReLU(),
            nn.BatchNorm2d(num_features=4),

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, bias=False), # 32 >> 32 || 3 >> 5
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8)
        ) # C1 C2

        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, padding=0, bias=False), # 32 >> 32 || 7 >> 7
            nn.ReLU()
        ) # c3
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32 >> 16 || 7 >> 8  || P1

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1, bias=False), # 16 >> 16 || 8 >> 12
            nn.ReLU(),
            nn.BatchNorm2d(num_features=4),

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, bias=False), # 16 >> 16 || 12 >> 16
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, bias=False), # 16 >> 16 || 16 >> 20
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16)
        ) # C3 C4 C5

        self.transition2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1, padding=0, bias=False), # 16 >> 16 || 20 >> 20
            nn.ReLU()
        ) # c6
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16 >> 8 || 20 >> 22 || P2

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1, bias=False), # 8 >> 8 || 22 >> 30
            nn.ReLU(),
            nn.BatchNorm2d(num_features=4),

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, bias=False), # 8 >> 8 || 30 >> 38
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, bias=False), # 8 >> 8 || 38 >> 46
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16)
        ) # C7 C8 C9

        self.gap = nn.AdaptiveAvgPool2d(output_size=1) # 8 >> 1 || GAP
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, padding=0, bias=0) # 1 >> 1 || c10

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.transition2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.gap(x)
        x = self.conv4(x)

        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)
    

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, bias=False), # 32 >> 32 || 1 >> 3
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, bias=False), # 32 >> 32 || 3 >> 5
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16)
        ) # C1 C2

        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, padding=0, bias=False), # 32 >> 32 || 7 >> 7
            nn.ReLU()
        ) # c3
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32 >> 16 || 7 >> 8  || P1

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, bias=False), # 16 >> 16 || 8 >> 12
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, bias=False), # 16 >> 16 || 12 >> 16
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False), # 16 >> 16 || 16 >> 20
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16)
        ) # C3 C4 C5

        self.transition2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, padding=0, bias=False), # 16 >> 16 || 20 >> 20
            nn.ReLU()
        ) # c6
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16 >> 8 || 20 >> 22 || P2

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, bias=False), # 8 >> 8 || 22 >> 30
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, bias=False), # 8 >> 8 || 30 >> 38
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False), # 8 >> 8 || 38 >> 46
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16)
        ) # C7 C8 C9

        self.gap = nn.AdaptiveAvgPool2d(output_size=1) # 8 >> 1 || GAP
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, padding=0, bias=0) # 1 >> 1 || c10

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.transition2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.gap(x)
        x = self.conv4(x)

        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)
    
class Model(nn.Module):
    def __init__(self, normalization="batchnorm", num_group=2):
        super(Model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False), # 32 >> 32 || 1 >> 3
            nn.ReLU(),
            self.norm(normalization=normalization, num_features=16, num_groups=num_group, num_channels=16),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False), # 32 >> 32 || 3 >> 5
            nn.ReLU(),
            self.norm(normalization=normalization, num_features=32, num_groups=num_group, num_channels=32)
        ) # C1 C2

        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=0, bias=False), # 32 >> 32 || 7 >> 7
            nn.ReLU()
        ) # c3
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32 >> 16 || 7 >> 8  || P1

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False), # 16 >> 16 || 8 >> 12
            nn.ReLU(),
            self.norm(normalization=normalization, num_features=16, num_groups=num_group, num_channels=16),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False), # 16 >> 16 || 12 >> 16
            nn.ReLU(),
            self.norm(normalization=normalization, num_features=16, num_groups=num_group, num_channels=16),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False), # 16 >> 16 || 16 >> 20
            nn.ReLU(),
            self.norm(normalization=normalization, num_features=32, num_groups=num_group, num_channels=32)
        ) # C3 C4 C5

        self.transition2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=0, bias=False), # 16 >> 16 || 20 >> 20
            nn.ReLU()
        ) # c6
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16 >> 8 || 20 >> 22 || P2

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False), # 8 >> 8 || 22 >> 30
            nn.ReLU(),
            self.norm(normalization=normalization, num_features=16, num_groups=num_group, num_channels=16),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False), # 8 >> 8 || 30 >> 38
            nn.ReLU(),
            self.norm(normalization=normalization, num_features=16, num_groups=num_group, num_channels=16),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False), # 8 >> 8 || 38 >> 46
            nn.ReLU(),
            self.norm(normalization=normalization, num_features=16, num_groups=num_group, num_channels=16)
        ) # C7 C8 C9

        self.gap = nn.AdaptiveAvgPool2d(output_size=1) # 8 >> 1 || GAP
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, padding=0, bias=0) # 1 >> 1 || c10
    
    def norm(self, normalization, **kwargs):
        if normalization == "batchnorm":
            return nn.BatchNorm2d(num_features=kwargs["num_features"])
        elif normalization == "groupnorm":
            return nn.GroupNorm(num_groups=kwargs["num_groups"], num_channels=kwargs["num_channels"])
        elif normalization == "layernorm":
            return nn.GroupNorm(num_groups=1, num_channels=kwargs["num_channels"])

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.pool1(x)

        x = x + self.transition2(self.conv2(x))
        x = self.pool2(x)

        x = x + self.conv3(x)
        x = self.gap(x)
        x = self.conv4(x)

        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)