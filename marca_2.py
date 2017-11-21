#from __future__ import division
import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt

#Ima = cv2.imread('4k_1.jpg')

#gris = cv2.cvtColor(Ima, cv2.COLOR_BGR2GRAY)
gris = np.ones((512, 512)) * 255
marca = cv2.imread('opeth.jpg', 0)


#cv2.imshow('sdf', marca)
#cv2.waitKey()
#cv2.destroyAllWindows()

coeff = pywt.wavedec2(gris, 'db2', level=5)

'''cA, (cH, cV, cD) = coeff


cV[0:20,0:20] += 10


cv2.imshow('Im', Ima)

cv2.waitKey()
cv2.destroyAllWindows()

plt.subplot(221)
plt.imshow(cA)

plt.subplot(222)
plt.imshow(cH)

plt.subplot(223)
plt.imshow(cV)

plt.subplot(224)
plt.imshow(cD)

plt.show()

coeff2 = cA, (cH, cV, cD)

Ima2 = pywt.idwt2(coeff, 'db2')'''

cA = coeff[0]
cD1 = coeff[1]
cD2 = coeff[2]
cD3 = coeff[3]
cD4 = coeff[4]
cD5 = coeff[5]

cD41, cD42, cD43 = cD5




lymr = cD43
fil, col = np.shape(lymr)
marca = cv2.resize(marca, (col, fil))
marca =255-marca
marca = marca


#lymr[0:20,0:20] += 1
lymr = lymr + marca
cD43 = lymr

cD54 = (cD41, cD42, cD43)
coeff3 = (cA,cD1,cD2,cD3,cD4, cD54)

print np.shape(coeff[0])

plt.imshow(cD41)
plt.show()

Ima2= pywt.waverec2(coeff3, 'db2')

Ima2 = np.clip(Ima2, 0, 255)

plt.subplot(121)
plt.imshow(Ima2, cmap='gray')

plt.subplot(122)
plt.imshow(gris, cmap='gray')
plt.show()