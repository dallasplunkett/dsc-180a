from torch import nn
import torchvision.models as tvm

def make_model(out_dim=1, pretrained=True, freeze=False):
    backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    w = backbone.conv1.weight.data.mean(dim=1, keepdim=True)
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    backbone.conv1.weight.data = w
    backbone.fc = nn.Identity()
    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
    head = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, out_dim),
    )
    return nn.Sequential(backbone, head)
