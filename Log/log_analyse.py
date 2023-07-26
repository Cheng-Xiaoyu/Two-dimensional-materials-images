import matplotlib.pyplot as plt
import json
import time
import datetime
def convert_all_json_in_text_to_dict(text):
    dicts, stack = [], []
    for i in range(len(text)):
        if text[i] == '{':
            stack.append(i)
        elif text[i] == '}':
            begin = stack.pop()
            if not stack:
                dicts.append(json.loads(text[begin:i+1]))
    return dicts

with open("./log.txt","r",encoding="utf-8") as txt:
    txtdata=txt.read()


jsondata=convert_all_json_in_text_to_dict(txtdata)
FID=[]
datetimes=[]
for i in range(len(jsondata)):
    fid=jsondata[i]['results']['fid50k_full']
    timestamp=jsondata[i]['timestamp']
    # print(fid,timestamp)
    timedate_str=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(timestamp))
    timedate_date=datetime.datetime.strptime(timedate_str,'%Y-%m-%d %H:%M:%S')
    
    FID.append(fid)
    datetimes.append(timedate_date)
    
    
# print(jsondata)
datetime_delta=[(i-datetimes[0]) for i in datetimes]
datetime_steps=[round((i.days+i.seconds/86400),3) for i in datetime_delta]


plt.plot(datetime_steps[20:-1],FID[20:-1],color='orange',marker='o',linewidth='3')
plt.grid()
plt.title('FID-Step')
plt.xlabel('Days')
plt.ylabel('FID')
plt.show()

