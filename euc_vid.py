from __future__ import division
import numpy as np
import cv2
#import matplotlib.pyplot as plt

eps = 0.0000001
def norm_1(Im):
    Im = Im.astype(float)
    Res = np.zeros_like(Im)
    b = Im[:,:,0]
    g = Im[:,:,1]
    r = Im[:,:,2]
    s = b+g+r
    Res[:, :, 0] = b/(s + eps) * 255
    Res[:, :, 1] = g/(s + eps) * 255
    Res[:, :, 2] = r/(s + eps) * 255
    return np.uint8(Res)

#p = np.array([78, 78, 100])
#pl = np.array([111, 87, 53])
#rj = np.array([53, 37, 162])
#rosa = np.array([206, 147, 227])
#amar = np.array([35, 152, 210])

# Vectores normalizados
#amar_n = np.array([11, 103, 140])
#nar_n = np.array([20, 53, 180])
#ros_n = np.array([74, 54, 125])
#ver_n = np.array([86, 112, 56])
#rojo_n = np.array([52, 41, 160])



col = np.array([[43, 57, 192], # rojo, verde, naranja, amarillo, rosa, azul
               [113, 204, 46],
               [18, 156, 243],
               [15, 196, 241],
               [153, 88, 181],
               [219, 152, 52]])

def eucl_dist(Im, fe):
    Im = Im.astype(float)
    b = Im[:, :, 0] - fe[0]
    g = Im[:, :, 1] - fe[1]
    r = Im[:, :, 2] - fe[2]
    s = np.clip(np.sqrt(b**2 + g**2 + r**2), 0, 255)
    s = np.uint8(s)
    #s = np.uint8((s<20) * 255)
    return s

cap = cv2.VideoCapture(1)
d = np.zeros((480, 640, 5))
#df = np.zeros((480, 640))

while(True):
    res = np.zeros((480, 640, 3))
    ret, frame = cap.read()
    #frame = cv2.medianBlur(frame, 7)
    nr = norm_1(frame)
    #nr = cv2.medianBlur(nr,7)

    d[:,:,0] = eucl_dist(nr, rojo_n)
    d[:,:,1] = eucl_dist(nr, ver_n)
    d[:,:,2] = eucl_dist(nr, nar_n)
    d[:,:,3] = eucl_dist(nr, amar_n)
    d[:,:,4] = eucl_dist(nr, ros_n)
    #df = np.uint8(np.argmin(d, axis=2) * 35)
    df = np.argmin(d, axis=2)

    #for i in range(480):
    #    for j in range(640):
    #        res[i, j, 0] = col[df[i, j], 2]
    #        res[i, j, 1] = col[df[i, j], 1]
    #        res[i, j, 2] = col[df[i, j], 0]

    res[df == 0, :] = col[0, :]
    res[df == 1, :] = col[1, :]
    res[df == 2, :] = col[2, :]
    res[df == 3, :] = col[3, :]
    res[df == 4, :] = col[4, :]

    #print res
    res = np.uint8(res)
    cv2.imshow('im', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#plt.imshow()
#plt.show()
cap.release()
cv2.destroyAllWindows()
