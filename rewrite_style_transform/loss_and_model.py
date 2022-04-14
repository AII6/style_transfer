import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

DEVICE = "cuda"

preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize(512),
    T.CenterCrop(512),
    T.ToTensor(),
    # T.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
])

# 保存图片前的处理
show = T.ToPILImage()


# 图片预处理
def imageProcess(path):
    image = (preprocess(plt.imread(path))).to(DEVICE)
    image = image.unsqueeze(0).to(DEVICE).requires_grad_(False)
    return image


def Grim_Maxtrix(input):
    s = input.shape
    a, b, c, d = s[0], s[1], s[2], s[3]  # a为图片数，b为通道数，c和d分别为宽高
    F = input.resize(b, c * d)  # 先算F
    G = torch.mm(F, F.T)  # 再算外积(叉乘)
    return torch.div(G, 2 * b * c * d)


class ContentLoss(nn.Module):
    def __init__(self, content):
        super(ContentLoss, self).__init__()
        self.content = content.detach()

    def forward(self, x):
        self.loss = F.mse_loss(self.content, x)
        return x


class StyleLoss(nn.Module):
    def __init__(self, style):
        super(StyleLoss, self).__init__()
        self.style = Grim_Maxtrix(style).detach()

    def forward(self, x):
        x_ = Grim_Maxtrix(x)
        self.loss = F.mse_loss(self.style, x_)
        return x


class Normolization(nn.Module):
    def __init__(self, mean, std):
        super(Normolization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, input):
        nor = (input - self.mean) / self.std
        return nor


def model_generate(style_image, content_image):
    model = models.vgg19(pretrained=True).features.eval().requires_grad_(False).to(DEVICE)
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)

    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    content_layers = ['conv4_2']
    # Loop through vgg layers
    style_losses = []
    content_losses = []
    nor = Normolization(cnn_normalization_mean, cnn_normalization_std).to(DEVICE)
    self_model = nn.Sequential().to(DEVICE)
    self_model.add_module("normalize", nor)

    i = 0
    block = 0
    for layer in model.children():
        name = ""
        if isinstance(layer, nn.Conv2d):
            if (list(layer.parameters())[0].shape[0] != list(layer.parameters())[0].shape[1]) or i == 5:
                block += 1
                i = 0
            i += 1
            name = 'conv{}_{}'.format(block, i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu{}_{}'.format(block, i)
            layer = nn.ReLU(inplace=False).to(DEVICE)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool{}_{}'.format(block, i)
            layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False).to(DEVICE)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn{}_{}'.format(block, i)

        # Add layer to our model
        self_model.add_module(name, layer)
        # Insert style loss layer
        if name in style_layers:
            target_feature = self_model(style_image).detach()
            style_loss = StyleLoss(target_feature).to(DEVICE)
            self_model.add_module('style_loss_{}'.format(block), style_loss)
            style_losses.append(style_loss)
        # Insert content loss layer
        if name in content_layers:
            target = self_model(content_image).detach()
            content_loss = ContentLoss(target).to(DEVICE)
            self_model.add_module('content_loss_{}'.format(block), content_loss)
            content_losses.append(content_loss)
    idx = 0
    for i in range(len(self_model) - 1, 0, -1):
        if isinstance(self_model[i], StyleLoss):
            idx = i + 1
            break
    self_model = self_model[:idx]
    return self_model, content_losses, style_losses
