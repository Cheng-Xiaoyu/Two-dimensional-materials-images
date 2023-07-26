import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def plot_maxiou(filepath):
    #savepath='./maxiou_sup_real.png'
    #filepath = './MaxIouPlot.txt'
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
    Imgnum_list= []
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
        Imgnum_list.append(Imgnum)
        max_indes_list.append(Imgnum + (classiou.index(max(classiou)) - 1) * Width)

        plt.text(Imgnum,max(classiou)+0.001,str(max(classiou)))
        plt.text(Imgnum,cl_aver - 0.002, str(round(cl_aver,4)))


    plt.plot(Imgnum_list,cl_max_list,marker='o', markersize='7',
                 linewidth='3',label='max IoU',color='#d62728')

    plt.plot(Imgnum_list,cl_aver_list,marker='o', markersize='7',
                 linewidth='3',label='average Iou',color='#2ca02c')



    #plt.savefig(savepath,dpi=600,bbox_inches='tight')
    #plt.show()

'''
    plt.subplot(133)
    plt.plot(Imgnum,max(classiou),marker='o',label=str(Imgnum))
    plt.plot(cl_max_list,color='grey',alpha=0.3,linestyle='--')
'''
