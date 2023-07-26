
import os
import shutil

count=129
last_num=1
target_path=r'E:\jupyter notebook\WSe2Picture\WSE2-50-nolabel-512x512\VitualPix1000\Vit256label/'
for root,dirs,files in os.walk("./"):
    for file in files:
        
        #print(os.path.join(root,file))
        source_name=os.path.join(root,file)
        ext=os.path.splitext(file)[-1].lower()
        file_name=os.path.splitext(file)[0].lower()
        mirror=file_name.split("-", -1)[-1]
        num=file_name.split("-", -1)[0]
        #
        if ext=='.png':
            rename='img'+str(count).zfill(8)
            print(file_name+'-->'+rename)
            target_name=(target_path+rename+'.png')
            count+=1
            shutil.copy(source_name, target_name)

                
            

            

            
            
            
            



'''
a = 1
i=str(a)
other_url = i.zfill(3)
print(other_url)
source_path = os.path.abspath(r'./1-20.bmp')
target_path = os.path.abspath(r'./rename/00020.bmp')

shutil.copy(source_path, target_path)
print('copy dir finished!')
'''

