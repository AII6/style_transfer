from loss_and_model import *
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 100000

style_image = imageProcess("style_img/starry-night.jpg")
content_image = imageProcess("content_img/tower.jpg")
target_image = content_image.clone()

self_model, content_losses, style_losses = model_generate(style_image, content_image)
opt = torch.optim.LBFGS([target_image.requires_grad_(True)])
e = 0
while e < 300:
    def closure():
        global e
        target_image.data.clamp_(0, 1)  # 消除杂点
        L_content = 0
        L_style = 0
        opt.zero_grad()
        self_model(target_image)

        for i in content_losses:
            L_content += i.loss

        for i in style_losses:
            L_style += i.loss * 0.2

        L = CONTENT_WEIGHT * L_content + STYLE_WEIGHT * L_style
        L.backward()

        e += 1
        if e % 10 == 0:
            print(e)
        return L

    opt.step(closure)
    target_image.data.clamp_(0, 1)

output = show(target_image.cpu().squeeze(0))
output.save("output/1.jpg")
