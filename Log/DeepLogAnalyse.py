import matplotlib.pyplot as plt
from numpy import nan
def convert_all_json_in_text_to_dict(text):
    dicts, stack = [], []
    for i in range(len(text)):#
        if text[i] == '{':
            stack.append(i)
        elif text[i] == '}':
            begin = stack.pop()
            if not stack:
                #str_member=text[begin:i+1].replace("'","\"")
                #print(str_member)
                dicts.append(eval(text[begin:i+1]))
    return dicts

cur_itrs=[]
with open("./resnet50_128_real01.txt","r",encoding="utf-8") as txt:
    txtdata=txt.read()

cur_itrs_loss,np_loss,cur_epochs,cur_itrs_epochs,interval_loss=[list() for x in range(5)]
cur_itrs_score,Overall_Acc,Mean_Acc,FreqW_Acc,Mean_IoU=[list() for x in range(5)]
Class_IoU_0,Class_IoU_1,Class_IoU_2=[list() for x in range(3)]
dictsdata = convert_all_json_in_text_to_dict(txtdata)
for i in dictsdata:

    if 'data_root' in i:
        # print(i)
        pass
    if 'Loss' in i:
        cur_itrs_loss.append(i['Loss']['cur_itrs'])
        np_loss.append(i['Loss']['np_loss'])

    if 'interval_loss' in i:
        cur_epochs.append(i['cur_epochs'])
        cur_itrs_epochs.append(i['cur_itrs'])
        interval_loss.append(i['interval_loss'])

    if 'val_score' in i:
        cur_itrs_score.append(i['cur_itrs'])

        Overall_Acc.append(i['val_score']['Overall Acc'])
        Mean_Acc.append(i['val_score']['Mean Acc'])
        FreqW_Acc.append(i['val_score']['FreqW Acc'])
        Mean_IoU.append(i['val_score']['Mean IoU'])
        Class_IoU_0.append(i['val_score']['Class IoU'][0])
        Class_IoU_1.append(i['val_score']['Class IoU'][1])
        Class_IoU_2.append(i['val_score']['Class IoU'][2])

plt.figure(figsize=(16, 9))

plt.subplot(231)
plt.plot(cur_itrs_loss,np_loss)
plt.grid();plt.title('Loss')

plt.subplot(232)
plt.plot(cur_itrs_epochs,interval_loss)
plt.grid();plt.title('interval_loss')

plt.subplot(233)
plt.title('Acc')
plt.grid();
plt.plot(cur_itrs_score,Overall_Acc,label='Overall_Acc')
plt.plot(cur_itrs_score,Mean_Acc,label='Mean_Acc')
plt.plot(cur_itrs_score,FreqW_Acc,label='FreqW_Acc')
plt.legend()

plt.subplot(234)
plt.plot(cur_itrs_score,Mean_IoU)
plt.plot(cur_itrs_score[Mean_IoU.index(max(Mean_IoU))],max(Mean_IoU),marker='.')
plt.text(x=cur_itrs_score[Mean_IoU.index(max(Mean_IoU))],y=max(Mean_IoU),s=max(Mean_IoU))
plt.grid();plt.title('Mean_IoU')

plt.subplot(235)
plt.grid();plt.title('Class_IoU')
plt.plot(cur_itrs_score,Class_IoU_0,label='Class_IoU_0')
plt.plot(cur_itrs_score,Class_IoU_1,label='Class_IoU_1')
print(max(Class_IoU_1))
plt.plot(cur_itrs_score,Class_IoU_2,label='Class_IoU_2')
plt.legend()

plt.plot(cur_itrs_score[Class_IoU_1.index(max(Class_IoU_1))],max(Class_IoU_1),marker='.')
plt.text(x=cur_itrs_score[Class_IoU_1.index(max(Class_IoU_1))],y=max(Class_IoU_1),s=max(Class_IoU_1))


'''
plt.subplot(236)
plt.xlim(0, 5);plt.ylim(0, 5)

plt.text(x=0,y=4,s='MaxClassIoU_thinlayer',fontsize=10)
plt.text(x=0,y=3,s=max(Class_IoU_1),fontsize=10)
'''
print(max(Mean_IoU))

plt.show()
