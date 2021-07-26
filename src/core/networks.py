import torch.nn as nn
import torch


class ParameterRegressor(nn.Module):
    def __init__(self, num_features, num_parts):
        super(ParameterRegressor, self).__init__()
        """
        convolutional encoder + linear layer at the end
        Args:
            num_features: list of ints containing number of features per layer
            num_parts: number of body parts for which we regress affine parameters
        Returns:
            torch.tensor (batch, num_parts, 2, 3), (2, 3) affine matrix for each body part
        """
        self.num_features = num_features
        self.num_parts = num_parts
        self.layers = self.define_network(num_features)

    def _add_conv_layer(self, in_ch, nf):
        return nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True)
        )

    def _add_down_layer(self, in_ch, nf):
        return nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, 2, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True)
        )

    def define_network(self, num_features):
        layers = [self._add_conv_layer(in_ch=3, nf=num_features[0])]
        for i in range(1, len(num_features)):
            layers.append(self._add_conv_layer(num_features[i-1], num_features[i-1]))
            layers.append(self._add_down_layer(num_features[i-1], num_features[i]))
        layers.append(nn.Sequential(
            nn.Conv2d(num_features[-1], 256, 1, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.LazyLinear(self.num_parts*6)
        ))

        return nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input).view(-1, self.num_parts, 2, 3)


class ImageTranslator(nn.Module):
    def __init__(self, num_features, num_parts):
        super(ImageTranslator, self).__init__()
        """
        convolutional enncoder, decoder
        """
        self.num_features = num_features
        self.num_parts = num_parts
        self.layers = self.define_network(self.num_features)

    def _add_conv_layer(self, in_ch, nf):
        return nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True)
        )

    def _add_down_layer(self, in_ch, nf):
        return nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, 2, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True)
        )

    def _add_up_layer(self, in_ch, nf):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True)
        )

    def define_network(self, num_features):
        # encoder
        layers = [self._add_conv_layer(in_ch=3+self.num_parts, nf=num_features[0])]
        for i in range(1, len(num_features)):
            layers.append(self._add_conv_layer(num_features[i - 1], num_features[i - 1]))
            layers.append(self._add_down_layer(num_features[i - 1], num_features[i]))

        # decoder mirrors the encoder
        for i in range(len(num_features)-1, 0, -1):
            layers.append(self._add_conv_layer(num_features[i], num_features[i]))
            layers.append(self._add_up_layer(num_features[i], num_features[i-1]))

        layers.append(nn.Conv2d(num_features[i-1], 3, 3, 1, 1))
        return nn.Sequential(*layers)

    def forward(self, input, template):
        return self.layers(torch.cat([input, template], dim=1))


class ParameterRegressor1(nn.Module):
    def __init__(self, n_f, num_joints):
        super(ParameterRegressor1, self).__init__()
        """Input is 3x256x256 image and int num_joints"""
        self.num_joints = num_joints
        self.main = nn.Sequential(
            nn.Conv2d(3, n_f, 3, 1, 1),
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
            # 4, 4
            nn.Flatten(),
            nn.Linear(n_f * 8 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            # parameters for affine matrix
            nn.Linear(512, num_joints * 6)

        )

    def forward(self, input):

        return self.main(input).view(-1, self.num_joints, 2, 3)














