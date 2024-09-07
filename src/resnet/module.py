import torch.nn as nn


def conv_block_1(
    in_dim: int,
    out_dim: int,
    activation: nn.Module = nn.ReLU(),
    stride: int = 1,
):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_dim),
        activation,
    )


def conv_block_3(
    in_dim: int,
    out_dim: int,
    activation: nn.Module = nn.ReLU(),
    stride: int = 1,
):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,
    )


class BottleNeck(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
        activation: nn.Module = nn.ReLU(),
        down: bool = False,
    ):
        super(BottleNeck, self).__init__()
        self.down = down

        if self.down:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, activation=activation, stride=2),
                conv_block_3(mid_dim, mid_dim, activation=activation, stride=1),
                conv_block_1(mid_dim, out_dim, activation=activation, stride=1),
            )
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2)
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, activation=activation, stride=1),
                conv_block_3(mid_dim, mid_dim, activation=activation, stride=1),
                conv_block_1(mid_dim, out_dim, activation=activation, stride=1),
            )

        self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.down:
            down = self.downsample(x)
            out = self.layer(x)
            out = out + down
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x

        return out


class ResNet(nn.Module):
    def __init__(self, base_dim: int, num_classes: int = 10):
        super(ResNet, self).__init__()
        self.activation = nn.ReLU()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(
                base_dim,
                base_dim,
                base_dim * 4,
                self.activation,
            ),
            BottleNeck(
                base_dim * 4,
                base_dim,
                base_dim * 4,
                self.activation,
            ),
            BottleNeck(
                base_dim * 4,
                base_dim,
                base_dim * 4,
                self.activation,
                down=True,
            ),
        )
        self.layer_3 = nn.Sequential(
            BottleNeck(
                base_dim * 4,
                base_dim * 2,
                base_dim * 8,
                self.activation,
            ),
            BottleNeck(
                base_dim * 8,
                base_dim * 2,
                base_dim * 8,
                self.activation,
            ),
            BottleNeck(
                base_dim * 8,
                base_dim * 2,
                base_dim * 8,
                self.activation,
            ),
            BottleNeck(
                base_dim * 8,
                base_dim * 2,
                base_dim * 8,
                self.activation,
                down=True,
            ),
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(
                base_dim * 8,
                base_dim * 4,
                base_dim * 16,
                self.activation,
            ),
            BottleNeck(
                base_dim * 16,
                base_dim * 4,
                base_dim * 16,
                self.activation,
            ),
            BottleNeck(
                base_dim * 16,
                base_dim * 4,
                base_dim * 16,
                self.activation,
            ),
            BottleNeck(
                base_dim * 16,
                base_dim * 4,
                base_dim * 16,
                self.activation,
            ),
            BottleNeck(
                base_dim * 16,
                base_dim * 4,
                base_dim * 16,
                self.activation,
            ),
            BottleNeck(
                base_dim * 16,
                base_dim * 4,
                base_dim * 16,
                self.activation,
                down=True,
            ),
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(
                base_dim * 16,
                base_dim * 8,
                base_dim * 32,
                self.activation,
            ),
            BottleNeck(
                base_dim * 32,
                base_dim * 8,
                base_dim * 32,
                self.activation,
            ),
            BottleNeck(
                base_dim * 32,
                base_dim * 8,
                base_dim * 32,
                self.activation,
            ),
        )
        self.avgpool = nn.AvgPool2d(1, 1)
        self.fc_layer = nn.Linear(base_dim * 32, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)

        return out
