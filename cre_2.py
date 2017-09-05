import cv2
import numpy as np
#import matplotlib.pyplot as plt

class CoordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self,event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                #cv2.circle(Ima,(x,y),3,(255,0,0),-1)
                self.points.append((x,y))


#instantiate class
coordinateStore1 = CoordinateStore()


# Create a black image, a window and bind the function to window
Ima = cv2.imread('abril2.jpg', 0)
Ima = cv2.resize(Ima,None,fx=0.75, fy=0.75, interpolation = cv2.INTER_CUBIC)
#Ima = cv2.imread('cameraman.tif', 0)
cv2.namedWindow('image')
cv2.setMouseCallback('image',coordinateStore1.select_point)

while(1):
    cv2.imshow('image',Ima)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
#cv2.destroyAllWindows()

trh = 20

print "Selected Coordinates: "

loc = coordinateStore1.points
x = loc[0][0]
y = loc[0][1]

#print x
#print y
fil, col = np.shape(Ima)

gri = Ima[y][x]
c = 1

print gri
#Ima = Ima.astype(float)
Ima2 = np.ones_like(Ima) * gri
Ima2 = cv2.absdiff(Ima2, Ima)

thr1,aux = cv2.threshold(Ima2,trh,255,cv2.THRESH_BINARY_INV)
#aux = np.abs(Ima-gri)<10
#print aux
#plt.imshow(aux)
#plt.show()
aux2 = np.uint8(np.zeros_like(aux))
aux3 = np.uint8(np.zeros_like(aux))
#aux2 = aux2.astype(int)
aux2[y,x] = 255
#kernel = np.ones((5,5),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
cv2.imshow('image', np.uint8(aux2*255))
cv2.moveWindow('image', 20,20)

#vid = np.zeros((fil, col, 3))
# Guardar video
out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (col, fil))

while c != 0:
    #aux3 = cv2.dilate(np.uint8(aux2), kernel, iterations=1)
    aux2 = cv2.dilate(np.uint8(aux2), kernel, iterations=1)
    aux2 = np.logical_and(aux2, aux)
    c = np.sum(aux2-aux3)
    aux3 = aux2
    cv2.imshow('image', np.uint8(aux2*255))
    aux2 = cv2.GaussianBlur(np.uint8(aux2*255),(5,5),0)
    vid = cv2.cvtColor(np.uint8(aux2), cv2.COLOR_GRAY2RGB)
    out.write(vid)
    #cv2.moveWindow('image', 20,20)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # optional key to end cycle
        break

out.release()
cv2.waitKey()
cv2.destroyAllWindows()

#plt.subplot(121)
#plt.imshow(aux2, cmap='gray')
#plt.subplot(122)
#plt.imshow(Ima, cmap='gray')
#plt.show()