# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# %% basic functions

plt.rcParams["figure.figsize"]=(8,6)
plt.rcParams["font.size"]=15
plt.rcParams["font.family"]='monospace'


fps_list = [1, 2, 5, 10, 15, 30]
rsl_list=[240,360,480,720]


data=np.load('data/feat-conf.npz', allow_pickle=True)
f1=data['f1']
f2=data['f2']
ps=data['ps']
data.close()

r = 80

fig, ax = plt.subplots(4, sharex=True)

plt.subplot(4,1,1)
plt.plot(f1)
plt.ylabel('max\nspeed',rotation=r,labelpad=10)

plt.subplot(4,1,2)
plt.plot(f2)
plt.ylabel('mean\ndistance',rotation=r,labelpad=10)


plt.subplot(4,1,3)
plt.plot(ps[:,0],'r')
plt.ylabel('resolution\n(height)',rotation=r,labelpad=10)
plt.yticks([0,1,2,3],rsl_list)


ax = plt.subplot(4,1,4)
plt.plot(5-ps[:,1],'g')
plt.ylabel('frmae-rate\n(FPS)',rotation=r,labelpad=10)
plt.xlabel('time (s)')
lgd = [f'{fps} ' if i%2==1 else f'{fps}' for i,fps in enumerate(fps_list)]
plt.yticks([0,1,2,3,4,5],lgd)

plt.tight_layout(h_pad=0.5)
