import torch.nn as nn
import torch

class ParameterRegressor(nn.Module):
    def __init__(self, n_f, num_joints):
        super(ParameterRegressor, self).__init__()
        """Input is 3x256x256 image and int num_joints"""
        self.num_joints = num_joints
        self.main = nn.Sequential(
            nn.Conv2d(3+self.num_joints, n_f, 3, 1, 1),
            nn.BatchNorm2d(n_f),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f, n_f, 3, 1, 1, dilation=4),
            nn.BatchNorm2d(n_f),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f, n_f*2, 3, 2, 1),
            nn.BatchNorm2d(n_f*2),
            nn.LeakyReLU(inplace=True),
            # 128, 128
            nn.Conv2d(n_f*2, n_f*2, 3, 1, 1, dilation=8),
            nn.BatchNorm2d(n_f*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f*2, n_f*4, 3, 2, 1),
            nn.BatchNorm2d(n_f*4),
            nn.LeakyReLU(inplace=True),
            # 64, 64
            nn.Conv2d(n_f*4, n_f * 4, 3, 1, 1),
            nn.BatchNorm2d(n_f * 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f * 4, n_f * 8, 3, 2, 1),
            nn.BatchNorm2d(n_f * 8),
            nn.LeakyReLU(inplace=True),
            # 32, 32
            nn.Conv2d(n_f * 8, n_f * 8, 3, 1, 1),
            nn.BatchNorm2d(n_f * 8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f * 8, n_f*8, 3, 2, 1),
            nn.BatchNorm2d(n_f * 8),
            nn.LeakyReLU(inplace=True),
            # 16, 16
            nn.Conv2d(n_f * 8, n_f * 8, 3, 1, 1),
            nn.BatchNorm2d(n_f * 8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f * 8, n_f * 8, 3, 2, 1),
            nn.BatchNorm2d(n_f * 8),
            nn.LeakyReLU(inplace=True),
            # 8, 8
            nn.Conv2d(n_f * 8, n_f * 8, 3, 1, 1),
            nn.BatchNorm2d(n_f * 8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f * 8, n_f * 8, 3, 2, 1),
            nn.BatchNorm2d(n_f * 8),
            nn.LeakyReLU(inplace=True),

        )

        self.param_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_f * 8 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            # parameters for affine matrix
            nn.Linear(512, num_joints * 6)
        )

        self.depth_branch = nn.Sequential(
            nn.Conv2d(n_f * 8, n_f * 4, 3, 1, 1),
            nn.BatchNorm2d(n_f * 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f * 4, n_f * 4, 3, 1, 1),
            nn.BatchNorm2d(n_f * 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f * 4, self.num_joints, 4, 1, 0),
            # bottleneck, allow channel-wise multiplication by scalar
            # (1, 1)
        )

    def forward(self, input, template):
        cat = torch.cat([input, template], dim=1)
        features = self.main(cat)
        params = self.param_branch(features)
        depth = self.depth_branch(features)
        return params.view(-1, self.num_joints, 2, 3), depth