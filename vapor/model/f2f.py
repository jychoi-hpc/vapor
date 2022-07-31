import torch
import torch.nn as nn

## Credit: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/models.py


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class F2F(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_hiddens,
        num_residual_layers,
        ks,
        upsampling=False,
    ):
        super().__init__()

        self.upsampling = upsampling
        k1, k2, k3 = ks

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, k1, 1, k1 // 2),
            nn.BatchNorm2d(num_hiddens, 0.8),
            nn.PReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_residual_layers):
            res_blocks.append(ResidualBlock(num_hiddens))
        self.residual_stack = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_hiddens, num_hiddens, k2, 1, k2 // 2),
            nn.BatchNorm2d(num_hiddens, 0.8),
            nn.PReLU(),
        )

        if self.upsampling:
            # Upsampling layers
            upsampling = []
            for out_features in range(2):
                upsampling += [
                    # nn.Upsample(scale_factor=2),
                    nn.Conv2d(num_hiddens, num_hiddens * 4, 3, 1, 1),
                    nn.BatchNorm2d(256),
                    nn.PixelShuffle(upscale_factor=2),
                    nn.PReLU(),
                ]
            self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_hiddens, num_hiddens // 2, k3, 1, k3 // 2),
            nn.PReLU(),
            nn.Conv2d(num_hiddens // 2, num_hiddens // 4, k3, 1, k3 // 2),
            nn.PReLU(),
            nn.Conv2d(num_hiddens // 4, out_channels, k3, 1, k3 // 2),
            nn.PReLU(),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.residual_stack(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        if self.upsampling:
            out = self.upsampling(out)
        out = self.conv3(out)
        return out
