from torch import nn
from torchvision import models

class AudioNet(nn.Module):

    def __init__(self, num_classes=50, pretrained=True):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights='IMAGENET1K_V1' if pretrained else None)
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

