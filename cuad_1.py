from __future__ import division
import numpy as np
import cv2
#import matplotlib.pyplot as plt


Ima = cv2.imread('cmyrgb1.jpg')


class CoordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self,event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                #cv2.circle(Ima,(x,y),3,(255,0,0),-1)
                self.points.append((x,y))


#instantiate class
coordinateStore1 = CoordinateStore()
coordinateStore2 = CoordinateStore()
coordinateStore3 = CoordinateStore()



cv2.namedWindow('image')
cv2.setMouseCallback('image',coordinateStore1.select_point)

while(1):
    cv2.imshow('image',Ima)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

loc = coordinateStore1.points
loc = np.array(loc)
loc = np.squeeze(loc)
x = loc[:,0]
y = loc[:,1]

rj = Ima[y,x]

rj_m = np.mean(rj, axis=0)
print rj_m

cv2.setMouseCallback('image',coordinateStore2.select_point)

while(1):
    cv2.imshow('image',Ima)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

loc2 = coordinateStore2.points
loc2 = np.array(loc2)
loc2 = np.squeeze(loc2)
x2 = loc2[:,0]
y2 = loc2[:,1]

vj = Ima[y2,x2]


vj_m = np.mean(vj, axis=0)
print vj_m

cv2.setMouseCallback('image',coordinateStore3.select_point)

while(1):
    cv2.imshow('image',Ima)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

loc3 = coordinateStore3.points
loc3 = np.array(loc3)
loc3 = np.squeeze(loc3)
x3 = loc3[:,0]
y3 = loc3[:,1]

aj = Ima[y3,x3]


aj_m = np.mean(aj, axis=0)
print aj_m

Ima = Ima.astype(float)

Im1 = np.zeros_like(Ima)

Im1[:,:,0] = np.sqrt(np.abs(Ima[:,:,0] - rj_m[0])**2 + np.abs(Ima[:,:,1] - rj_m[1])**2 + np.abs(Ima[:,:,2] - rj_m[2])**2)
Im1[:,:,1] = np.sqrt(np.abs(Ima[:,:,0] - vj_m[0])**2 + np.abs(Ima[:,:,1] - vj_m[1])**2 + np.abs(Ima[:,:,2] - vj_m[2])**2)
Im1[:,:,2] = np.sqrt(np.abs(Ima[:,:,0] - aj_m[0])**2 + np.abs(Ima[:,:,1] - aj_m[1])**2 + np.abs(Ima[:,:,2] - aj_m[2])**2)

Imf = np.argmin(Im1, axis=2)

Imf1 = np.zeros_like(Ima)
Imf1[Imf==2,0] = 1
Imf1[Imf==1,1] = 1
Imf1[Imf==0,2] = 1

Imf1 = np.uint8(Imf1 * 255)

while(1):
    cv2.imshow('img',Imf1)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
#plt.imshow(Imf1)
#plt.show()











