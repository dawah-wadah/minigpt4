import torch.nn as nn
from torchvision.models import resnet50


class ImageEncoder(nn.Module):
    """
    Image Encoder module based on ResNet50.
    """

    def __init__(self, pretrained=True):
        super(ImageEncoder, self).__init__()
        self.resnet = resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.num_features = self.resnet.fc.in_features

    def forward(self, images):
        features = self.features(images)
        features = features.view(features.size(0), -1)
        return features
