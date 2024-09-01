import csv
import glob
import os

def write_csv_fs1000():
    img_dir = '/data1/wangjingtao/workplace/python/data/seg/meta/fss1000/meta_set'
    relative_path = '/data1/wangjingtao/workplace/python/data/seg/meta/'
    
    class_name_ls = os.listdir('/data1/wangjingtao/workplace/python/data/seg/meta/fss1000/meta_set')
    class_name_ls = set(class_name_ls)

    class_id_d = {}
    for i, v in enumerate(class_name_ls):
        class_id_d[v] = i



    with open('tmp.csv', 'w', newline='') as datacsvfile:
        fields = ['ID', 'Class_ID','Class', 'Image_path', 'Label_path']
        datawrite = csv.writer(datacsvfile, delimiter=',')
        datawrite.writerow(fields)
        
        img_ls = glob.glob(os.path.join(img_dir, "**", "*.jpg"),
                        recursive=True)
    
        
        # img_ls = img_ls


        for i, img in enumerate(img_ls):
            img_path = img.replace(relative_path, '')
            mask_path = img_path.replace('.jpg', '.png')
            if not os.path.isfile(os.path.join(relative_path, img_path)):
                raise ValueError(f'{img_path} file not exist')
            if not os.path.isfile(os.path.join(relative_path, mask_path)):
                raise ValueError(f'{mask_path} file not exist')

            value_ls = img_path.split('/')
            class_name =  value_ls[2]
            class_id = class_id_d[class_name]
            
            
            
            datawrite.writerow([i, class_id, class_name, img_path, mask_path])


write_csv_fs1000()