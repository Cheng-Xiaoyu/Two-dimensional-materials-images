from GetDeeplabLogData import Get_deeplab_log_data
import matplotlib.pyplot as plt
from numpy import nan

Pixpath='./Pix/'
LogData = Get_deeplab_log_data('./101-v256-3.txt')

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

plt.figure(figsize=(16, 9))

plt.subplot(231)
plt.plot(cur_itrs_loss, np_loss)
plt.grid();
plt.title('Loss')

plt.subplot(232)
plt.plot(cur_itrs_epochs, interval_loss)
plt.grid();
plt.title('interval_loss')

plt.subplot(233)
plt.title('Acc')
plt.grid();
plt.plot(cur_itrs_score, Overall_Acc, label='Overall_Acc')
plt.plot(cur_itrs_score, Mean_Acc, label='Mean_Acc')
plt.plot(cur_itrs_score, FreqW_Acc, label='FreqW_Acc')
plt.legend()

plt.subplot(234)
plt.plot(cur_itrs_score, Mean_IoU)
plt.plot(cur_itrs_score[Mean_IoU.index(max(Mean_IoU))], max(Mean_IoU), marker='.')
plt.text(x=cur_itrs_score[Mean_IoU.index(max(Mean_IoU))], y=max(Mean_IoU), s=max(Mean_IoU))
plt.grid();
plt.title('Mean_IoU')

plt.subplot(235)
plt.grid();
plt.title('Class_IoU')
plt.plot(cur_itrs_score[50:-1], Class_IoU_0[50:-1], label='Class_IoU_0')
plt.plot(cur_itrs_score[50:-1], Class_IoU_1[50:-1], label='Class_IoU_1')
plt.plot(cur_itrs_score[50:-1], Class_IoU_2[50:-1], label='Class_IoU_2')
plt.legend()

plt.plot(cur_itrs_score[Class_IoU_1.index(max(Class_IoU_1))], max(Class_IoU_1), marker='.')
plt.text(x=cur_itrs_score[Class_IoU_1.index(max(Class_IoU_1))], y=max(Class_IoU_1), s=max(Class_IoU_1))

#print(Dataset)
print(data_table['model'])
print(Dataset['Trainset']+'-->',round(max(Mean_IoU),4),' ',round(max(Class_IoU_1),4))
#plt.savefig(Pixpath+'pix.png',dpi=600)
#plt.show()

