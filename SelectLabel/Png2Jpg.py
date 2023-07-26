import os
import shutil
from PIL import Image



count=1057
last_num=1
source_path=r'./161-1056mask/'
target_path=r'./1057-2080mask/'
for root,dirs,files in os.walk(source_path):
    for file in files:
        
        #print(os.path.join(root,file))
        source_name=os.path.join(root,file)
        
        ext=os.path.splitext(file)[-1].lower()
        file_name=os.path.splitext(file)[0].lower()
        mirror=file_name.split("-", -1)[-1]
        num=file_name.split("-", -1)[0]
        #
        if ext=='.png':
            #im = Image.open(source_name)
            #im = im.convert('RGB')
            
            str_count=str(count)
            new_name=str_count.zfill(8)
            target_name=target_path+'img'+new_name+'.png'
            ##################################################
            #im.save(target_name, quality=95)
            count=count+1
            print(target_name)
            #shutil.copy(source_name, target_name)


