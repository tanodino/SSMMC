"""wrn.py -- Wide ResNet as used by FixMatch (Sohn et al., 2020) / FreeMatch / SoftMatch for CIFAR:
WRN-28-2 = 3 groups of 4 pre-activation BasicBlocks, widths 32/64/128, BN momentum 0.001, LeakyReLU 0.1,
no dropout, global average pooling, linear classifier; kaiming-normal conv init, xavier linear init, zero bias.
Follows the reference implementations (google-research/fixmatch, TorchSSL / USB wrn.py).
Agreed 2026-11-21: FixMatch-family competitors with their original backbone, one WRN-28-2 per modality,
score fusion (average of the two softmax outputs), from scratch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.001
LEAKY = 0.1


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, activate_before_residual=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=BN_MOMENTUM)
        self.relu1 = nn.LeakyReLU(LEAKY, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)
        self.relu2 = nn.LeakyReLU(LEAKY, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=True)
        self.equal = in_planes == out_planes
        self.shortcut = None if self.equal else nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=True)
        self.activate_before_residual = activate_before_residual

    def forward(self, x):   # exactly as TorchSSL / USB wrn.py
        if not self.equal and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
            out = None
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal else x)))
        out = self.conv2(out)
        return (x if self.equal else self.shortcut(x)) + out


class NetworkBlock(nn.Module):
    def __init__(self, n, in_planes, out_planes, stride, activate_before_residual=False):
        super().__init__()
        self.layer = nn.Sequential(*[BasicBlock(in_planes if i == 0 else out_planes, out_planes,
                                                stride if i == 0 else 1, activate_before_residual)
                                     for i in range(n)])

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, in_chans=3, depth=28, widen_factor=2):
        super().__init__()
        ch = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        n = (depth - 4) // 6
        self.conv1 = nn.Conv2d(in_chans, ch[0], 3, 1, 1, bias=True)
        self.block1 = NetworkBlock(n, ch[0], ch[1], 1, activate_before_residual=True)
        self.block2 = NetworkBlock(n, ch[1], ch[2], 2)
        self.block3 = NetworkBlock(n, ch[2], ch[3], 2)
        self.bn1 = nn.BatchNorm2d(ch[3], momentum=BN_MOMENTUM)
        self.relu = nn.LeakyReLU(LEAKY, inplace=True)
        self.fc = nn.Linear(ch[3], num_classes)
        self.num_features = ch[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.block3(self.block2(self.block1(self.conv1(x))))
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)          # logits


class WRNScoreFusion(nn.Module):
    """Two WRN-28-2 streams (one per modality); output = mean of the two softmax distributions (as our ScoreFusion)."""

    def __init__(self, num_classes, in_chans_m1, in_chans_m2, depth=28, widen_factor=2):
        super().__init__()
        self.modality_1_encoder = WideResNet(num_classes, in_chans_m1, depth, widen_factor)
        self.modality_2_encoder = WideResNet(num_classes, in_chans_m2, depth, widen_factor)

    def forward(self, x1, x2):
        return (F.softmax(self.modality_1_encoder(x1), -1) + F.softmax(self.modality_2_encoder(x2), -1)) / 2

    def predict(self, x1, x2):
        return self.forward(x1, x2)
