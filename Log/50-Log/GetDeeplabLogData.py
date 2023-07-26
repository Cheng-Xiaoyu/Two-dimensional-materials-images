from numpy import nan
import re
import matplotlib.pyplot as plt
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

def Get_deeplab_log_data(filepath):
    cur_itrs=[]
    with open(filepath,"r",encoding="utf-8") as txt:
        txtdata=txt.read()
    with open(filepath,"r",encoding="utf-8") as txt:
        lines=txt.readlines()
        for i in lines:
            if 'Dataset' in i:
                dataset_str=re.split(':|,',i.replace(" ","").replace('\n', ''))
                Dataset={'Dataset':dataset_str[1],'Trainset':dataset_str[3],'Valset':dataset_str[5]}

    dictsdata = convert_all_json_in_text_to_dict(txtdata)
    cur_itrs_loss,np_loss,cur_epochs,cur_itrs_epochs,interval_loss=[list() for x in range(5)]
    cur_itrs_score,Overall_Acc,Mean_Acc,FreqW_Acc,Mean_IoU=[list() for x in range(5)]
    Class_IoU_0,Class_IoU_1,Class_IoU_2=[list() for x in range(3)]
    
    
    for i in dictsdata:

        if 'data_root' in i:
            # print(i)
            data_table=i
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
            
    return cur_itrs_loss,np_loss,cur_epochs,cur_itrs_epochs,interval_loss, \
           cur_itrs_score,Overall_Acc,Mean_Acc,FreqW_Acc,Mean_IoU, \
           Class_IoU_0,Class_IoU_1,Class_IoU_2,data_table,Dataset
               
            
if __name__=='__main__':
    
    data=Get_deeplab_log_data('./mobile_64.txt')
    #plt.plot(data[13])
    #plt.show()
    print(data[13]['model'])
    print(data[14])
    '''
    Get_deeplab_log_data[0]=cur_itrs_loss
    Get_deeplab_log_data[1]=np_loss
    Get_deeplab_log_data[2]=cur_epochs
    Get_deeplab_log_data[3]=cur_itrs_epochs
    Get_deeplab_log_data[4]=interval_loss
    Get_deeplab_log_data[5]=cur_itrs_score
    Get_deeplab_log_data[6]=Overall_Acc
    Get_deeplab_log_data[7]=Mean_Acc
    Get_deeplab_log_data[8]=FreqW_Acc
    Get_deeplab_log_data[9]=Mean_IoU
    Get_deeplab_log_data[10]=Class_IoU_0
    Get_deeplab_log_data[11]=Class_IoU_1
    Get_deeplab_log_data[12]=Class_IoU_2
    Get_deeplab_log_data[13]=data_table
    Get_deeplab_log_data[14]=Dataset
    
    LogData = Get_deeplab_log_data('./mobile_64.txt')
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
    '''
    