import torchvision.models as tvm
from torch import nn

WEIGHTS = {
    "resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
    "resnet34": tvm.ResNet34_Weights.IMAGENET1K_V1,
    "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2,
    "resnet101": tvm.ResNet101_Weights.IMAGENET1K_V2,
    "resnet152": tvm.ResNet152_Weights.IMAGENET1K_V2,
}

FEATURE_COUNT = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
}

MODELS = {
    "resnet18": tvm.resnet18,
    "resnet34": tvm.resnet34,
    "resnet50": tvm.resnet50,
    "resnet101": tvm.resnet101,
    "resnet152": tvm.resnet152,
}


def make_model(name="resnet18"):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODELS.keys())}")

    # ImageNet Weights
    backbone = MODELS[name](weights=WEIGHTS[name])

    # Patch First Layer
    w = backbone.conv1.weight.data.mean(dim=1, keepdim=True)
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    backbone.conv1.weight.data = w

    # Remove Classifier head
    backbone.fc = nn.Identity() # type: ignore

    # Add Regression head
    head = nn.Sequential(
        nn.Linear(FEATURE_COUNT[name], 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 1),
    )

    return nn.Sequential(backbone, head)
