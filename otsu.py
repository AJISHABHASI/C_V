import numpy as np
import scipy.ndimage as ni
import matplotlib.pyplot as plt
import operator
import time
start_time = time.time()
#import cv2

Ima = np.round(ni.imread('cmyrgb1.jpg', flatten=True))
#if np.shape(Ima)[2]==3:
#    Ima = cv2.cvtColor(Ima, cv2.COLOR_BGR2GRAY)

fil, col = np.shape(Ima)

hist = np.zeros([256])

for i in range(255):
    hist[i] = np.sum(Ima == i)


pro = hist/(fil*col)

a = 0
c = 0
w0 = np.zeros(256)
w1 = np.zeros(256)
mu0 = np.zeros(256)
mu1 = np.zeros(256)
for i in range(256):
    a = pro[i]+a
    w0[i] = a
    c = (i*pro[i]) +c
    mu0[i] = c/(a+0.00001)
    b = 0
    d = 0
    for j in range(i+1,256):
        b = pro[j]+b
        d = (j*pro[j])+d
    w1[i]=b
    mu1[i] = d/(b+0.00001)



sig = w0*w1*((mu1-mu0)**2)
th = max(enumerate(sig),key=operator.itemgetter(1))[0]
print("--- %s seconds ---" % (time.time() - start_time))
Sal = Ima>(th)
Sal = ni.binary_closing(Sal)

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.plot(np.arange(256), hist)

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.imshow(Ima)

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
ax3.plot(np.arange(256), w0)
ax3.plot(np.arange(256), w1)
ax3.plot(np.arange(256), sig)

print th
fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)
ax4.imshow(Sal, cmap=plt.cm.gray)
plt.show()
