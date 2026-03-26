import torch.nn as nn

SUPPORTED_MODELS = {
    "resnet50": "ResNet-50",
    "efficientnet_b2": "EfficientNet-B2",
    "mobilenet_v3_large": "MobileNetV3-Large",
}


def build_model(model_name, num_classes, pretrained=True):
    """Load a torchvision model and replace its classification head."""
    import torchvision.models as models

    weights_arg = "DEFAULT" if pretrained else None

    if model_name == "resnet50":
        model = models.resnet50(weights=weights_arg)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights_arg)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model
