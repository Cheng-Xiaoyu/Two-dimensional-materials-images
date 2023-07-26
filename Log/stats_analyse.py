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

with open("./stats.jsonl","r",encoding="utf-8") as txt:
    txtdata=txt.read()


jsondata=convert_all_json_in_text_to_dict(txtdata)

scores_fake=[]
signs_fake=[]
G_loss=[]
scores_real=[]
signs_real=[]
D_loss=[]
r1_penalty=[]
D_reg=[]
Progress_tick=[]
Progress_kimg=[]
Progress_augment=[]
Timing_total_hours=[]
Timing_total_days=[]
for i in jsondata:
    scores_fake.append(i['Loss/scores/fake']['mean'])
    signs_fake.append(i['Loss/signs/fake']['mean'])
    G_loss.append(i['Loss/G/loss']['mean'])
    scores_real.append(i['Loss/scores/real']['mean'])
    signs_real.append(i['Loss/signs/real']['mean'])
    D_loss.append(i['Loss/D/loss']['mean'])
    r1_penalty.append(i['Loss/r1_penalty']['mean'])
    D_reg.append(i['Loss/D/reg']['mean'])
    Progress_tick.append(i['Progress/tick']['mean'])
    Progress_kimg.append(i['Progress/kimg']['mean'])
    Progress_augment.append(i['Progress/augment']['mean'])
    Timing_total_hours.append(i['Timing/total_hours']['mean'])
    Timing_total_days.append(i['Timing/total_days']['mean'])


#plt.plot(Progress_tick,scores_fake,marker='',linewidth='1')
#plt.plot(Progress_tick,signs_fake,marker='',linewidth='1')
plt.plot(Progress_tick,G_loss,marker='',linewidth='2')
#plt.plot(Progress_tick,scores_real,marker='',linewidth='1')
#plt.plot(Progress_tick,signs_real,marker='',linewidth='1')
plt.plot(Progress_tick,D_loss,marker='',linewidth='2')

plt.grid()

'''
plt.title('FID-Step')
plt.xlabel('Days')
plt.ylabel('FID')
'''
plt.show()



'''

Loss/scores/fake=[]
Loss/signs/fake=[]
Loss/G/loss=[]
Loss/scores/real=[]
Loss/signs/real=[]
Loss/D/loss=[]
Loss/r1_penalty=[]
Loss/D/reg=[]
Progress/tick=[]
Progress/kimg=[]
Progress/augment=[]
Timing/total_hours=[]
Timing/total_days=[]


plt.plot(,color='orange',marker='o',linewidth='3')
plt.grid()
plt.title('FID-Step')
plt.xlabel('Days')
plt.ylabel('FID')
plt.show()
'''
