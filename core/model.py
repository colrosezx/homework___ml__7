from torchvision.models import resnet18
import torch.nn as nn

def prepare_model(num_classes=10):
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model