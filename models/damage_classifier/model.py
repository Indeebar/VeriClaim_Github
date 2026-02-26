import torch
import torch.nn as nn
import timm


CLASSES = ['minor', 'moderate', 'severe']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


class DamageClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, num_classes=3):
        model = cls(num_classes=num_classes, pretrained=False)
        model.load_state_dict(
            torch.load(path, map_location='cpu')
        )
        model.eval()
        return model