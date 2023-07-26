from GetDeeplabLogData import Get_deeplab_log_data
import matplotlib.pyplot as plt
from numpy import nan
import math

plt.grid()
plt.rc('font',family='Times New Roman', size=13)
plt.xlabel('Epoch')
plt.ylabel('Class IoU')

def Plot_fewIoU(filepath,legend):
    LogData = Get_deeplab_log_data(filepath)
    Dataset = LogData[14]
    Class_IoU_1 = LogData[11]
    cur_itrs_score = LogData[5]
    TrainNum=int(Dataset['Trainset'])
    batchsize=4
    #epoch=[round(i*batchsize/TrainNum) for i in cur_itrs_score]
    maxindex=Class_IoU_1.index(max(Class_IoU_1))
    #plt.plot(cur_itrs_score[0:maxindex+1], Class_IoU_1[0:maxindex+1], label=legend,linewidth='3')
    plt.plot(cur_itrs_score[maxindex], round(max(Class_IoU_1),4),linestyle='',marker='o',markersize=round(math.log(TrainNum)*5),label=legend)
    plt.text(x=cur_itrs_score[maxindex], y=max(Class_IoU_1), s=round(max(Class_IoU_1),4))
    
Plot_fewIoU('101-16.txt','16')
Plot_fewIoU('101-32.txt','32')
Plot_fewIoU('101-128-1.txt','128')
Plot_fewIoU('101-384.txt','384')
Plot_fewIoU('101-512.txt','512')
plt.legend()
plt.show()
#plt.savefig('.\img.svg',dpi=600,bbox_inches='tight', pad_inches=0)