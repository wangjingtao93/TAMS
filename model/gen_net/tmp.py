import cv2
from PIL import Image
import numpy as np
import cv2
import numpy as np

# 假设mask是一个二值图像
mask = np.random.choice([0, 255], size=(512, 512))

# 使用最近邻插值方法进行尺寸调整
resized_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

# 检查调整尺寸后的图像是否还是二值图
def is_binary_image(image):
    return np.all(np.logical_or(image == 0, image == 255))

print("Is the resized image binary? ", is_binary_image(resized_mask))
