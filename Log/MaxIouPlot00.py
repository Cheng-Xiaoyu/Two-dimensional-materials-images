import matplotlib.pyplot as plt
import matplotlib
import numpy as np

filepath = './MaxIouPlot_v.txt'
cl_aver_list = []
cl_max_list = []
with open(filepath, "r", encoding="utf-8") as txt:
    lines = txt.readlines()

# plt.ylim((0.8, 1))
classiou_1 = []
classiou_2 = []
classiou_3 = []
max_indes_list = []
plt.ylim((0.875, 0.91))
plt.xlim((200, 1100))
ticks = np.arange(256, 1152, 128)
plt.xticks(ticks)
plt.xlabel('Data set capacity', fontsize='10')
plt.ylabel('IoU', fontsize='10')
Width = 30
for li in lines:
    print(li)
    classiou = []

    for member in li.split('\t'):
        onetime = member.split('-->')
        Imgnum = float(onetime[0])
        classiou.append(float(member.split('-->')[1].split('   ')[-1].split('\n')[0]))


    print(classiou)

    cl_aver = sum(classiou) / len(classiou)
    cl_aver_list.append(cl_aver)
    cl_max_list.append(max(classiou))
    max_indes_list.append(Imgnum + (classiou.index(max(classiou)) - 1) * Width)


    if lines.index(li)==0:
        plt.bar(Imgnum - Width, classiou[0], width=Width,color='#1f77b4',label='first')
        plt.bar(Imgnum, classiou[1], width=Width,color='#ff7f0e',label='second')
        plt.bar(Imgnum + Width, classiou[2], width=Width,color='#2ca02c',label='third')
    else:
        plt.bar(Imgnum - Width, classiou[0], width=Width, color='#1f77b4')
        plt.bar(Imgnum, classiou[1], width=Width, color='#ff7f0e')
        plt.bar(Imgnum + Width, classiou[2], width=Width, color='#2ca02c')

    plt.plot(Imgnum + (classiou.index(max(classiou)) - 1) * Width, max(classiou), marker='o', markersize='7',
             linewidth='10',color='#d62728',mec='#d62728',mfc='#d62728')
    plt.text(Imgnum + (classiou.index(max(classiou)) - 1) * Width,max(classiou)+0.001,str(max(classiou)))
    

plt.plot(max_indes_list,cl_max_list,color='grey',linestyle='--',alpha=0.3)


plt.legend()

plt.savefig('./maxiou_00.png',dpi=600,bbox_inches='tight')
plt.show()

'''
    plt.subplot(133)
    plt.plot(Imgnum,max(classiou),marker='o',label=str(Imgnum))
    plt.plot(cl_max_list,color='grey',alpha=0.3,linestyle='--')
'''
