import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, in_features=768):
        super().__init__()
        self.project = nn.Linear(in_features, 7 * 7 * 256)
        chans = [256, 128, 64, 32, 16]
        layers = []
        for a, b in zip(chans, chans[1:]):
            layers += [nn.ConvTranspose2d(a, b, 4, 2, 1), nn.BatchNorm2d(b), nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid()]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(self.project(x).view(-1, 256, 7, 7))


class WaferSwin(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=0)
        dim = self.backbone.num_features
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.3), nn.Linear(dim, num_classes))
        self.decoder = Decoder(dim)

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats), self.decoder(feats)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


def mixup(x, y, alpha):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam
