from multi_model import *

CONTENT_WEIGHT = 1
STYLE_WEIGHT1 = 1000000
STYLE_WEIGHT2 = 1000000

style_image1 = imageProcess("style_img/picasso_selfport.jpg")
style_image2 = imageProcess("style_img/frida_kahlo.jpg")
content_image = imageProcess("content_img/tower-3.jpg")
# target_image = content_image.clone()
target_image = torch.randn([1, 3, 512, 512]).data.clamp_(0, 1).to(DEVICE)

multiModel = Multi_S_Model(style_image1, style_image2, content_image)
opt = torch.optim.LBFGS([target_image.requires_grad_(True)])
e = 0
while e < 350:
    def closure():
        global e
        target_image.data.clamp_(0, 1)  # 消除杂点
        L_content = 0
        L_style1 = 0
        L_style2 = 0
        opt.zero_grad()
        multiModel.self_model(target_image)

        for i in multiModel.content_losses:
            L_content += i.loss

        for i in range(5):
            if i < 3:
                L_style1 += multiModel.style_losses1[i].loss * 0.2
            else:
                L_style1 += multiModel.style_losses1[i].loss * 2

        for i in range(5):
            if i < 3:
                L_style2 += multiModel.style_losses2[i].loss * 0.2
            else:
                L_style2 += multiModel.style_losses2[i].loss * 2

        L = CONTENT_WEIGHT * L_content + STYLE_WEIGHT1 * L_style1 + STYLE_WEIGHT2 * L_style2
        L.backward()

        e += 1
        if e % 10 == 0:
            print(e)
        return L

    opt.step(closure)
    target_image.data.clamp_(0, 1)

output = show(target_image.cpu().squeeze(0))
output.save("output/2.jpg")