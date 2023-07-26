from GetDeeplabLogData import Get_deeplab_log_data
import matplotlib.pyplot as plt
from numpy import nan

Pixpath='./Pix/'
LogData = Get_deeplab_log_data('./101-128-1.txt')
cur_itrs_loss = LogData[0]
np_loss = LogData[1]
cur_epochs = LogData[2]
cur_itrs_epochs = LogData[3]
interval_loss = LogData[4]
cur_itrs_score = LogData[5]
Overall_Acc = LogData[6]
Mean_Acc = LogData[7]
FreqW_Acc = LogData[8]
Mean_IoU = LogData[9]
Class_IoU_0 = LogData[10]
Class_IoU_1 = LogData[11]
Class_IoU_2 = LogData[12]
data_table = LogData[13]
Dataset = LogData[14]

TrainNum=int(Dataset['Trainset'])
batchsize=4

epoch=[round(i*batchsize/TrainNum) for i in cur_itrs_score]
#print(epoch)

maxindex=Class_IoU_1.index(max(Class_IoU_1))

plt.grid()
plt.rc('font',family='Times New Roman', size=13)

#plt.title('Class IoU')
plt.plot(epoch[0:maxindex+1], Class_IoU_0[0:maxindex+1], label='Background',linewidth='3')
plt.plot(epoch[0:maxindex+1], Class_IoU_1[0:maxindex+1], label='Few Layers',linewidth='3')
plt.plot(epoch[0:maxindex+1], Class_IoU_2[0:maxindex+1], label='Bulk',linewidth='3')
plt.legend()

plt.plot(epoch[maxindex], round(max(Class_IoU_1),4), marker='o',linewidth='3')
plt.text(x=epoch[maxindex]-20, y=max(Class_IoU_1)+0.01, s=round(max(Class_IoU_1),4))
plt.xlabel('Epoch')
plt.ylabel('Class IoU')
plt.savefig('.\img.svg',dpi=600,bbox_inches='tight', pad_inches=0)


print(data_table['model'])
print(Dataset['Trainset']+'-->',round(max(Mean_IoU),4),' ',round(max(Class_IoU_1),4))
print('Maxitrs',cur_itrs_score[maxindex])

plt.show()

'''
plt.grid();
plt.title('Class IoU')
plt.plot(cur_itrs_score, Class_IoU_0, label='Background')
plt.plot(cur_itrs_score, Class_IoU_1, label='Bock')

plt.plot(cur_itrs_score, Class_IoU_2, label='Few Layers')
plt.legend()

plt.plot(cur_itrs_score[Class_IoU_1.index(max(Class_IoU_1))], round(max(Class_IoU_1),4), marker='o')
plt.text(x=cur_itrs_score[Class_IoU_1.index(max(Class_IoU_1))], y=max(Class_IoU_1), s=max(Class_IoU_1))
#plt.show()
'''
