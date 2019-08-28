from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

class SpatialTransformerNet(nn.Module):
    def __init__(self, input_channel=1, num_class=10):
        super(SpatialTransformerNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        print(theta)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x, grid

    def forward(self, x):
        # transform the input
        x, grid = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        # self.nll_loss = nn.NLLLoss(size_average=True)
    
    def forward(self, predict, target):
        return F.nll_loss(predict, target, size_average=False)



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class SpatialTransformerPretrained(nn.Module):
    def __init__(self, input_channel=1, num_class=10):
        super(SpatialTransformerPretrained, self).__init__()
        self.model_ft = models.resnet18(pretrained=True)
        set_parameter_requires_grad(self.model_ft, False)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, num_class)
       
        # Spatial transformer localization-network
        self.localization = models.densenet121(pretrained=True)
        set_parameter_requires_grad(self.localization, False)
        num_ftrs = self.localization.classifier.in_features
        self.localization.classifier = nn.Linear(num_ftrs, 512)
        # self.localization = nn.Sequential(
        #     nn.Conv2d(input_channel, 8, kernel_size=9),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 10, kernel_size=7),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(10, 10, kernel_size=5),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True)
        # )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2),
            nn.Sigmoid()
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 512)
        theta = self.fc_loc(xs)
        # grid generator
        mask = torch.cuda.FloatTensor(*theta.size()).fill_(1.0)
        mask[:, 1] = 0
        mask[:, 3] = 0
        mask[:, 2] = 0
        mask[:, 5] = 0
        theta = theta * (mask)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        # sampler
        x = F.grid_sample(x, grid, padding_mode="zeros")
       
       
        return x, grid

    def forward(self, x):
        # transform the input
        x, gird = self.stn(x)

        # Perform the usual forward pass
        x = self.model_ft(x)
        return F.log_softmax(x, dim=1)