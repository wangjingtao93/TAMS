
import cv2
import albumentations as A
import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    object_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation'

    csv_root = os.path.join(object_path, 'data/rips_fundus_rp/real_data/all_data.csv')
    dataframe = pd.read_csv(csv_root)
        
    real_imglist = dataframe['Image_path']
    for img in real_imglist:
        img_name = img.split('/')[-1].replace('.jpg', '')
        mask_path = img.replace('/x/', '/y/').replace('.jpg', '.png')

        src_img = cv2.imread(img)
        src_img = cv2.resize(src_img,(256, 256), interpolation=cv2.INTER_NEAREST)
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask,(256, 256), interpolation=cv2.INTER_NEAREST)



        lesion = np.where(mask, src_img, 0)

        dst_path = f'{img_name}_aug.png'
        transform = A.Compose([
            # A.ColorJitter(always_apply=True),
            A.HueSaturationValue(always_apply=True)
                            ])
        img_aug = transform(image=lesion)['image']
        # cv2.imwrite(dst_path, np.concatenate((img_aug, lesion)))
        cv2.imwrite(dst_path, img_aug)

        cv2.imwrite(f'{img_name}.jpg', lesion)