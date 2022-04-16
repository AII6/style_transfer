import torch

from loss_and_model import *
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1000000

style_image = imageProcess("style_img/picasso_selfport.jpg")
content_image = imageProcess("content_img/tower-3.jpg")
# target_image = content_image.clone()
target_image = torch.randn([1, 3, 512, 512]).data.clamp_(0, 1).to(DEVICE)

self_model, content_losses, style_losses = model_generate(style_image, content_image)
opt = torch.optim.LBFGS([target_image.requires_grad_(True)])
e = 0
while e < 500:
    def closure():
        global e
        target_image.data.clamp_(0, 1)  # 消除杂点
        L_content = 0
        L_style = 0
        opt.zero_grad()
        self_model(target_image)

        for i in content_losses:
            L_content += i.loss

        for i in range(5):
            if i < 3:
                L_style += style_losses[i].loss*0.1
            else:
                L_style += style_losses[i].loss*3

        L = CONTENT_WEIGHT * L_content + STYLE_WEIGHT * L_style
        L.backward()

        e += 1
        if e % 10 == 0:
            print(e)
        return L

    opt.step(closure)
    target_image.data.clamp_(0, 1)

output = show(target_image.cpu().squeeze(0))
output.save("output/3.jpg")

