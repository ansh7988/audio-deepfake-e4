import torch
import torch.nn as nn


class MultiScaleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        # Three different temporal scales
        c = out_channels // 3

        self.branch_small = nn.Sequential(
            nn.Conv1d(
                in_channels,
                c,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(c),
            nn.ReLU()
        )

        self.branch_medium = nn.Sequential(
            nn.Conv1d(
                in_channels,
                c,
                kernel_size=11,
                padding=5
            ),
            nn.BatchNorm1d(c),
            nn.ReLU()
        )

        self.branch_large = nn.Sequential(
            nn.Conv1d(
                in_channels,
                c,
                kernel_size=21,
                padding=10
            ),
            nn.BatchNorm1d(c),
            nn.ReLU()
        )

        # Make sure output is exactly out_channels
        self.projection = nn.Sequential(
            nn.Conv1d(
                c * 3,
                out_channels,
                kernel_size=1
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )


    def forward(self, x):

        small = self.branch_small(x)

        medium = self.branch_medium(x)

        large = self.branch_large(x)

        x = torch.cat(
            [small, medium, large],
            dim=1
        )

        return self.projection(x)


class E4MultiScaleCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # =================================
        # Initial feature extraction
        # =================================

        self.stem = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=80,
                stride=4
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(4)
        )


        # =================================
        # Multi-scale blocks
        # =================================

        self.block1 = MultiScaleBlock(
            32,
            64
        )

        self.pool1 = nn.MaxPool1d(2)


        self.block2 = MultiScaleBlock(
            64,
            128
        )

        self.pool2 = nn.MaxPool1d(2)


        self.block3 = MultiScaleBlock(
            128,
            256
        )

        self.pool3 = nn.MaxPool1d(2)


        # =================================
        # Global feature extraction
        # =================================

        self.global_pool = nn.AdaptiveAvgPool1d(1)


        # =================================
        # Classifier
        # =================================

        self.classifier = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.35),

            nn.Linear(
                128,
                1
            )
        )


    def forward(self, x):

        x = self.stem(x)

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        x = self.global_pool(x)

        x = x.squeeze(-1)

        return self.classifier(x).squeeze(1)