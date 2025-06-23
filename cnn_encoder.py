import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super(CNNEncoder, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]  # Remove final FC layer
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)  # shape: [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # → [B, 2048]
        features = self.fc(features)                   # → [B, embed_size]
        return self.bn(features)                       # → [B, embed_size]
