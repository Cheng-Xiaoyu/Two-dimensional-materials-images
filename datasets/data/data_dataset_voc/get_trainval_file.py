import os
import shutil

count=0
last_num=1
target_path=r'E:\jupyter notebook\DeepLabV3Plus-Pytorch-master\datasets\data\data_dataset_voc\JPEGImages/'
for root,dirs,files in os.walk(target_path):
    for file in files:
        
        # print(os.path.join(root,file))
        source_name=os.path.join(root,file)
        
        ext=os.path.splitext(file)[-1].lower()
        file_name=os.path.splitext(file)[0].lower()
        mirror=file_name.split("-", -1)[-1]
        num=file_name.split("-", -1)[0]
        #
        if ext=='.png':
            print(file_name)

 