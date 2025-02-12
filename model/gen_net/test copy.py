from pix2Topix import pix2pixG_256
import torch
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def test(img_path):
    if img_path.endswith('.png'):
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]
    else:
        img = Image.open(img_path)

    transforms = transform.Compose([
        transform.ToTensor(),
        transform.Resize((256, 256)),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transforms(img.copy())
    img = img[None].to('cuda')  # [1,3,128,128]

    # 实例化网络
    G = pix2pixG_256().to('cuda')
    # 加载预训练权重
    ckpt = torch.load('weights/pix2pix_256.pth')
    G.load_state_dict(ckpt['G_model'], strict=False)

    G.eval()
    out = G(img)[0]
    out = out.permute(1,2,0)
    out = (0.5 * (out + 1)).cpu().detach().numpy()
    plt.figure()
    plt.imshow(out)
    plt.show()


if __name__ == '__main__':
    test('data/test_cnv/sys/CNV_196_9_1_d.jpg')
