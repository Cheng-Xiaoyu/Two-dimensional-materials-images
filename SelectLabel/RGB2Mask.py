import os
import shutil
import numpy as np
import os.path as osp
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import labelme
import imgviz

count=673
last_num=1
source_path=r'./SelectLab/'
target_path=r'./673-800mask/'
for root,dirs,files in os.walk(source_path):
    for file in files:
        
        #print(os.path.join(root,file))
        source_name=os.path.join(root,file)
        
        ext=os.path.splitext(file)[-1].lower()
        filename=os.path.splitext(file)[0].lower()
        #
        if ext=='.png':
            img =Image.open(source_name)
            img_array=np.asarray(img)
            img_r=img_array[:,:,0]
            img_r=img_r.copy()
            img_r[img_r>0]=1
            img_g=img_array[:,:,1]
            img_g=img_g.copy()
            img_g[img_g>0]=1
            img_b=img_array[:,:,2]
            img_b=img_b.copy()
            img_b[img_b>0]=1

            img_class=img_r+img_g*2+img_b*3
            
            if osp.splitext(filename)[1] != ".png":
                filename += ".png"
            if img_class.min() >= -1 and img_class.max() < 255:
                lbl_pil = Image.fromarray(img_class.astype(np.uint8), mode="P")
                colormap = imgviz.label_colormap()
                lbl_pil.putpalette(colormap.flatten())
                
                str_count=str(count)
                new_name=str_count.zfill(8)
                target_name=target_path+'img'+new_name+'.png'
                ##################################################
                lbl_pil.save(target_name)
                count=count+1
                print(target_name+'...'+str(last_num))
                last_num+=1
            else:
                raise ValueError(
                    "[%s] Cannot save the pixel-wise class label as PNG. "
                    "Please consider using the .npy format." % filename
                )
            
            #shutil.copy(source_name, target_name)

