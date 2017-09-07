from __future__ import division
import numpy as np
import cv2

Ima = cv2.imread('cmyrgb1.jpg', 0)

fil, col = np.shape(Ima)

def cd1(Im, th):
    f, c = np.shape(Im)
    c1 = Im[0: int(f/2), 0: int(c/2)]
    c2 = Im[int(f/2): f, 0: int(c/2)]
    c3 = Im[0: int(f/2), int(c/2): c]
    c4 = Im[int(f/2): f, int(c/2): c]
    s1 = np.std(c1)
    s2 = np.std(c2)
    s3 = np.std(c3)
    s4 = np.std(c4)
    #cv2.imshow('v1', c4)
    #print s1, s2, s3, s4
    res = np.zeros((f, c))

    if s1 > th and f > 4:
        res[0: int(f / 2), 0: int(c / 2)] = cd1(c1,th)
    #else s1 <= th:
    else:
        res[0: int(f/2), 0: int(c/2)] = np.mean(c1)

    if s2 > th and f > 4:
        res[int(f / 2): f, 0: int(c / 2)] = cd1(c2,th)
    #elif s2 <= th:
    else:
        res[int(f/2): f, 0: int(c/2)] = np.mean(c2)

    if s3 > th and f > 4:
        res[0: int(f / 2), int(c / 2): c] = cd1(c3,th)
    #elif s3 <= th:
    else:
        res[0: int(f/2), int(c/2): c] = np.mean(c3)

    if s4 > th and f > 4:
        res[int(f / 2): f, int(c / 2): c] = cd1(c4,th)
    #elif s4 <= th:
    else:
        res[int(f/2): f, int(c/2): c] = np.mean(c4)


    return res

# Obtener resultado

res = np.uint8(cd1(Ima,30))

cv2.namedWindow('win',cv2.WINDOW_NORMAL)
cv2.resizeWindow('win', fil*3, col*2)
cv2.imshow('win', res)


cv2.waitKey()
cv2.destroyAllWindows()

