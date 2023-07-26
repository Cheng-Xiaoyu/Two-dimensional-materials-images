import matplotlib.pyplot as plt
import matplotlib
filepath='./MaxIouPlot.txt'
filepath_v='./MaxIouPlot_v.txt'



def plot_iou(path):
    with open(path,"r",encoding="utf-8") as txt:
        lines=txt.readlines()
        
            
    #plt.ylim((0.8, 1))
    cl_max_list=[]
    cl_aver_list=[]
    for li in lines:
        print(li)
        classiou=[]
        for member in li.split('\t'):
            onetime=member.split('-->')
            Imgnum=onetime[0]
            classiou.append(float(member.split('-->')[1].split('   ')[-1].split('\n')[0]))
        print(classiou)
        cl_aver=sum(classiou)/len(classiou)
        cl_aver_list.append(cl_aver)
        cl_max_list.append(max(classiou))
        #plt.subplot(121)
        #plt.plot(classiou,marker='o',label=str(Imgnum))
        #plt.subplot(122)
        plt.xlabel('Data set capacity',fontsize='10')
        plt.ylabel('Max IoU',fontsize='10')
        plt.plot(Imgnum,max(classiou),marker='o',markersize='7',label=str(Imgnum),linewidth='10')
        plt.plot(cl_max_list,color='grey',alpha=0.3,linestyle='--')

    plt.plot([0.8841 for i in range(7)],color='grey',alpha=0.3,linestyle='--')

plot_iou(filepath)
plot_iou(filepath_v)

plt.legend()
plt.savefig('./maxiou.png',dpi=600,bbox_inches='tight')
plt.show()

'''
    plt.subplot(133)
    plt.plot(Imgnum,max(classiou),marker='o',label=str(Imgnum))
    plt.plot(cl_max_list,color='grey',alpha=0.3,linestyle='--')
'''
    