import numpy as np
import os.path as osp
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import labelme
import imgviz
filename='seed1003.png'
img =Image.open('./test_results/seed1001.png')

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
# Assume label ranses [-1, 254] for int32,
# and [0, 255] for uint8 as VOC.
if img_class.min() >= -1 and img_class.max() < 255:
    lbl_pil = Image.fromarray(img_class.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(filename)
else:
    raise ValueError(
        "[%s] Cannot save the pixel-wise class label as PNG. "
        "Please consider using the .npy format." % filename
    )


'''
mask_path=r'E:\jupyter notebook\DeepLabV3Plus-Pytorch-master\test_results\img00000038.png'
img_path=r'E:\jupyter notebook\DeepLabV3Plus-Pytorch-master\test_target\img00000038.png'

mask=Image.open(mask_path).convert('RGB')
img =Image.open(img_path)
plt.axis('off')
plt.imshow(img)
plt.imshow(mask, alpha=0.6)
ax = plt.gca()
ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
plt.savefig('results/%d_overlay.png' % 38, bbox_inches='tight', pad_inches=0)
plt.show()

'''