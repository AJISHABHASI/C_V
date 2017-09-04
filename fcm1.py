from __future__ import division
import numpy as np
import scipy.ndimage as ni
import matplotlib.pyplot as plt
import time
from skimage import color
#import dicom
#import os


#Instrucciones imagen
#Ima = ni.imread('jorge.jpg', flatten=True)

Ima = ni.imread('jorge.jpg')
Ima = color.rgb2hsv(Ima)
Ima = Ima[:,:,2]

# Instrucciones DIOM

#Ima = dicom.read_file()
im = np.reshape(Ima, np.product(np.shape(Ima)))
start_time = time.time()
L = len(im)
c = 3 #Numero de clases
mp = 2
eps = 0.00001
d = np.zeros([c, L])
th = 1000
v = np.random.rand(c)*255
for i in range(10):
    for i in range(c):
        d[i,:] = np.abs(im-v[i])

    d = d+eps
    m = np.zeros_like(d)
    for i in range(c):
        for j in range(c):
            m[i,:] = m[i,:] + (d[i,:]/(d[j,:]+eps))**2
        m[i,:] = m[i,:]**(-1)
    vp = v

    for i in range(c):
        a = np.sum(m[i,:]**2)
        b = np.sum((m[i,:]**2)*im)
        v[i] = b/a

    #th = np.subtract(v,vp)
    #print th

print("--- %s seconds ---" % (time.time() - start_time))
#print v

#Mostrar histograma
di = np.zeros([c,255])
mu = np.zeros([c,255])

for k in range(255):
    for i in range(c):
        di[i,k] = np.abs(k-v[i])

    for i in range(c):
        for j in range(c):
            mu[i,k] = mu[i,k] + (di[i,k]/(di[j,k]+eps))**2
        mu[i,k] = mu[i,k]**(-1)


ord = np.argsort(v)
#print v
print ord
plt.subplot(131)
plt.imshow(np.reshape(m[ord[0], :], np.shape(Ima)), cmap='gray', clim=(0.0, 1))

plt.subplot(132)
plt.imshow(np.reshape(m[ord[1], :], np.shape(Ima)), cmap='gray', clim=(0.0, 1))

plt.subplot(133)
plt.imshow(np.reshape(m[ord[2], :], np.shape(Ima)), cmap='gray', clim=(0.0, 1))
plt.show()




fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)

for i in range(c):
    ax1.plot(range(255),mu[i,:])

m = m/np.sum(m, axis=0)
lev = 255/(c-1)
ims = np.zeros_like(Ima)
for i in range(c):
    ims = np.reshape(m[ord[i], :], np.shape(Ima)) * lev*i + ims

#imf = ni.median_filter(ims, 3)
#imf = ni.
#ims = ni.gaussian_filter(imf, 1)
#alpha = 3
#imf1 = imf + alpha * (imf - ims)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.imshow(ims, cmap=plt.cm.gray )
'''mask = np.reshape(m[ord[c-1], :], np.shape(Ima))>0.7
mask = ni.binary_opening(mask, structure=np.ones((2,2)))
mask1 = ni.binary_erosion(mask, structure=np.ones((11,11)))
mask2 = mask-mask1 
mask2 = ni.binary_opening(mask2, structure=np.ones((5,5)))*250
'''
mask = np.reshape(m[ord[c-1], :], np.shape(Ima))
sx = ni.sobel(mask, axis=0, mode='constant')
sy = ni.sobel(mask, axis=1, mode='constant')
sob = np.hypot(sx, sy)*255
mask2 = (sob>370)*255
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
ax3.imshow(Ima, cmap=plt.cm.gray)
ax3.imshow(mask2, alpha=.3, cmap=plt.cm.gray)

plt.show()