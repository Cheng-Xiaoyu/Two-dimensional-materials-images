import matplotlib.pyplot as plt
Pix=[16,32,64,128,256]
Mean=[0.941066,0.941882,0.946337,0.949238,0.953951]
Class=[0.865307,0.867005,0.881484,0.885894,0.898214]

plt.plot(Pix,Mean)
plt.plot(Pix,Class)
plt.show()