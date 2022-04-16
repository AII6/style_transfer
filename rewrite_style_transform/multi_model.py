from loss_and_model import *


class Multi_S_Model(nn.Module):
    def __init__(self, style_image1, style_image2, content_image):
        super(Multi_S_Model, self).__init__()
        self.self_model = None
        self.generate_model(style_image1, style_image2, content_image)

    def generate_model(self, style_image1, style_image2, content_image):
        model = models.vgg19(pretrained=True).features.eval().requires_grad_(False).to(DEVICE)
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)

        style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        content_layers = ['conv4_2']
        # Loop through vgg layers
        self.style_losses1 = []
        self.style_losses2 = []
        self.content_losses = []
        nor = Normolization(cnn_normalization_mean, cnn_normalization_std).to(DEVICE)
        self.self_model = nn.Sequential().to(DEVICE)
        self.self_model.add_module("normalize", nor)

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
            self.self_model.add_module(name, layer)
            # Insert style loss layer
            if name in style_layers:
                target_feature1 = self.self_model(style_image1).detach()
                style_loss1 = StyleLoss(target_feature1).to(DEVICE)
                self.self_model.add_module('style_loss1_{}'.format(block), style_loss1)
                self.style_losses1.append(style_loss1)

                target_feature2 = self.self_model(style_image2).detach()
                style_loss2 = StyleLoss(target_feature2).to(DEVICE)
                self.self_model.add_module('style_loss2_{}'.format(block), style_loss2)
                self.style_losses2.append(style_loss2)
            # Insert content loss layer
            if name in content_layers:
                target = self.self_model(content_image).detach()
                content_loss = ContentLoss(target).to(DEVICE)
                self.self_model.add_module('content_loss_{}'.format(block), content_loss)
                self.content_losses.append(content_loss)
        idx = 0
        for i in range(len(self.self_model) - 1, 0, -1):
            if isinstance(self.self_model[i], StyleLoss):
                idx = i + 1
                break
        self.self_model = self.self_model[:idx]
