import matplotlib.pyplot as plt
from MaxIouPlot_rAv import plot_maxiou
#plot_maxiou('./MaxIouPlot_v.txt')
plot_maxiou('./MaxIouPlot.txt')
plt.legend()
plt.savefig('./ok01.png',dpi=600,bbox_inches='tight')
plt.show()