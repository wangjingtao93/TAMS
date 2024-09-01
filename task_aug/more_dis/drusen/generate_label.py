from glob import glob
import os
import cv2
import random
import numpy as np

def generate_kuang_all():
    mask_file = glob(os.path.join(output_y_fake_dir, "**", "*.png"), recursive=True)
    img_file = [elm.replace(output_y_fake_dir, output_x_fake_dir).replace("_label", "")[:-4] + ".*" for elm in
                mask_file]
    # mask_file_dir = "/192.168.128.161/temp/WJC/wjt_fake_dursen/result7/y"
    # img_file_dir = "/192.168.128.161/temp/WJC/wjt_fake_dursen/result7/x"
    # mask_file = glob(os.path.join(mask_file_dir, "**", "*.png"), recursive=True)
    # img_file = [elm.replace(mask_file_dir, img_file_dir).replace("_label", "")[:-4] + ".*" for elm in
    #             mask_file]
    for i in range(len(img_file)):
        img_file_i = glob(img_file[i])
        assert len(img_file_i) == 1, "multi img_file_i : %s" % img_file
        img_file_i = img_file_i[0]
        assert img_file_i.endswith("jpg") or img_file_i.endswith("png"), "img_file_i type error : %s" % img_file_i
        img_file[i] = img_file_i

    check_kuang_path = "./data/result/check_kuang.txt"
    # if os.path.isfile(check_kuang_path):  # 先检查文件是否存在
    #     os.remove(check_kuang_path)
    with open(check_kuang_path, "w", encoding="utf-8", ) as f:
        for i in range(len(mask_file)):
            image_name = img_file[i].split("\\")[1]
            y_raw = cv2.imread(mask_file[i], 0)
            # imageErZhi = np.array(255 * (y_raw == 6) + 255 * (y_raw == 176), dtype=np.uint8)
            imageErZhi = np.array(255 * (y_raw == 6), dtype=np.uint8)

            cnts, hierarchy = cv2.findContours(imageErZhi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                # 找到边界坐标
                x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
                idx_kuang_fir = [x - random.randint(3, 5), y - random.randint(3, 5)]
                idx_kuang_sec = [x + w + random.randint(3, 5), y + h + random.randint(3, 5)]
                info = image_name + " " + str(idx_kuang_fir[0]) + " " + str(idx_kuang_fir[1]) + " " + str(
                    idx_kuang_sec[0]) + " " + str(idx_kuang_sec[1]) + "\n"
                f.writelines(info)

if __name__ == "__main__":

    check_kuang_path = "./data/result/check_kuang.txt"

    output_x_fake_dir = "./data/result/succ_res/x"
    output_y_fake_dir = "./data/result/succ_res/y"

    generate_kuang_all()