# prob1.py
import numpy as np
import cv2
import matplotlib.pyplot as pyplot

# 1st
sam1 = cv2.imread("sample1.jpg")
print(sam1.shape)
h,w,c = sam1.shape
sam1_N = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        sam1_N[i][j] = sam1[i][j][0]

g1 = np.zeros([h,w], dtype=np.uint8)

for i in range(1,h-1):
    for j in range(1,w-1):
        g1[i][j] = np.power(np.power(1/4*(sam1_N[i-1][j+1]+2*sam1_N[i][j+1]+sam1_N[i+1][j+1]-sam1_N[i-1][j-1]-2*sam1_N[i][j-1]-sam1_N[i+1][j-1]),2)+np.power(1/4*(sam1_N[i-1][j-1]+2*sam1_N[i-1][j]+sam1_N[i-1][j+1]-sam1_N[i+1][j-1]-2*sam1_N[i+1][j]-sam1_N[i+1][j+1]),2),1/2)
# print(g1)
# pyplot.hist(g1)
# pyplot.show()
for i in range(h):
    for j in range(w):
        if g1[i][j] > 18 :
            g1[i][j] = 255
        else :
            g1[i][j] = 0
cv2.imwrite("result1.jpg",g1) 

# 2nd

g1 = np.zeros([h,w],dtype=np.uint8)
for i in range(1,h-1):
    for j in range(1,w-1):
        g1[i][j] = sam1_N[i][j]-1/4*(sam1_N[i-1][j]+sam1_N[i][j-1]+sam1_N[i][j+1]+sam1_N[i+1][j])
# pyplot.hist(g1)
# pyplot.show() 
g2 = np.zeros([h,w], dtype=np.int8)
for i in range(h):
    for j in range(w):
        if g1[i][j] >= 25 and g1[i][j] <=100 :
            g2[i][j] = 0
        elif g1[i][j] < 25:
            g2[i][j] = -1  
        else :
            g2[i][j] = 1
# pyplot.hist(g2)
# pyplot.show() 



res2 = np.zeros([h,w], dtype=np.uint8)
for i in range(1,h-1):
    for j in range(1,w-1):
        if g2[i][j] == 0:
            if np.abs(g2[i-1][j-1])+np.abs(g2[i+1][j+1]) > 0 and np.abs(g2[i-1][j])+np.abs(g2[i+1][j]) > 0  and np.abs(g2[i-1][j+1])+np.abs(g2[i+1][j-1]) > 0 and np.abs(g2[i][j-1])+np.abs(g2[i][j+1]) > 0 :
                res2[i][j] = 255
            else:
                res2[i][j] = 0    

        elif g2[i][j] == -1:
            res2[i][j] = 0
        else :
            res2[i][j] = 0                        
cv2.imwrite("result2.jpg",res2)

# canny

# noise reduction
for i in range(1,h-1):
    for j in range(1,w-1):
        sam1_N[i][j] = 1/16*(sam1_N[i-1][j-1]+2*sam1_N[i-1][j]+sam1_N[i-1][j+1]+2*sam1_N[i][j-1]+4*sam1_N[i][j]+2*sam1_N[i][j+1]+sam1_N[i+1][j-1]+2*sam1_N[i+1][j]+sam1_N[i+1][j+1])
# compute gradient mag$ori
g1 = np.zeros([h,w], dtype=np.uint8)
for i in range(1,h-1):
    for j in range(1,w-1):
        g1[i][j] = np.power(np.power(1/4*(sam1_N[i-1][j+1]+2*sam1_N[i][j+1]+sam1_N[i+1][j+1]-sam1_N[i-1][j-1]-2*sam1_N[i][j-1]-sam1_N[i+1][j-1]),2)+np.power(1/4*(sam1_N[i-1][j-1]+2*sam1_N[i-1][j]+sam1_N[i-1][j+1]-sam1_N[i+1][j-1]-2*sam1_N[i+1][j]-sam1_N[i+1][j+1]),2),1/2)
        

ori = np.zeros([h,w])
for i in range(1,h-1):
    for j in range(1,w-1):
        ori[i][j] = np.arctan((1/4*(sam1_N[i-1][j+1]+2*sam1_N[i][j+1]+sam1_N[i+1][j+1]-sam1_N[i-1][j-1]-2*sam1_N[i][j-1]-sam1_N[i+1][j-1]))/(1/4*(sam1_N[i-1][j-1]+2*sam1_N[i-1][j]+sam1_N[i-1][j+1]-sam1_N[i+1][j-1]-2*sam1_N[i+1][j]-sam1_N[i+1][j+1])))
# Non-maximal suppression
for i in range(1,h-1):
    for j in range(1,w-1):
        if np.abs(ori[i][j]*180/np.pi) < 22.5 or np.abs(ori[i][j]*180/np.pi) > 157.5:
            if g1[i][j]>g1[i-1][j] and g1[i][j]>g1[i+1][j]:
                g1[i][j] = g1[i][j]
            else:
                g1[i][j] = 0
        elif ori[i][j]*180/np.pi > 22.5 and ori[i][j]*180/np.pi < 67.5:
            if g1[i][j]>g1[i-1][j-1] and g1[i][j]>g1[i+1][j+1]:
                g1[i][j] = g1[i][j]
            else:
                g1[i][j] = 0 
        elif ori[i][j]*180/np.pi > -112.5 and ori[i][j]*180/np.pi < -157.5:
            if g1[i][j]>g1[i-1][j-1] and g1[i][j]>g1[i+1][j+1]:
                g1[i][j] = g1[i][j]
            else:
                g1[i][j] = 0         
        elif np.abs(ori[i][j]*180/np.pi) > 67.5 and np.abs(ori[i][j]*180/np.pi) < 112.5:
            if g1[i][j]>g1[i][j-1] and g1[i][j]>g1[i][j+1]:
                g1[i][j] = g1[i][j]
            else:
                g1[i][j] = 0 
        else :
            if g1[i][j]>g1[i+1][j-1] and g1[i][j]>g1[i-1][j+1]:
                g1[i][j] = g1[i][j]
            else:
                g1[i][j] = 0                           
# thresholding
for i in range(h):
    for j in range(w):
        if g1[i][j]>36:
            g1[i][j] = 255
        elif g1[i][j]<18:
            g1[i][j] = 0  
        else:
            g1[i][j] = 128                        
# cv2.imwrite("bb.jpg",g1)  
# connect component 
for i in range(1,h-1):
    for j in range(1,w-1):
        if g1[i][j] == 128:
            if np.abs(ori[i][j]*180/np.pi) < 22.5 or np.abs(ori[i][j]*180/np.pi) > 157.5:
                if g1[i][j-1] == 255 or g1[i][j+1] == 255 or g1[i][j-1] ==128 or g1[i][j+1] == 128:
                    g1[i][j] = 255
                else:
                    g1[i][j] = 0
            elif ori[i][j]*180/np.pi > 22.5 and ori[i][j]*180/np.pi < 67.5:
                if g1[i-1][j+1] == 255 or g1[i+1][j-1] == 255 or g1[i-1][j+1] == 128 or g1[i+1][j-1] == 128:
                    g1[i][j] = 255
                else:
                    g1[i][j] = 0 
            elif ori[i][j]*180/np.pi > -112.5 and ori[i][j]*180/np.pi < -157.5:
                if g1[i-1][j+1] == 255 or g1[i+1][j-1] == 255 or g1[i-1][j+1] == 128 or g1[i+1][j-1] == 128:
                    g1[i][j] = 255
                else:
                    g1[i][j] = 0         
            elif np.abs(ori[i][j]*180/np.pi) > 67.5 and np.abs(ori[i][j]*180/np.pi) < 112.5:
                if g1[i-1][j] == 255 or g1[i+1][j] == 255 or g1[i-1][j] == 128 or g1[i+1][j] == 128:
                    g1[i][j] = 255
                else:
                    g1[i][j] = 0 
            else :
                if g1[i-1][j-1] == 255 or g1[i+1][j+1] == 255 or g1[i-1][j-1] == 128 or g1[i+1][j+1] == 128:
                    g1[i][j] = 255
                else:
                    g1[i][j] = 0 
        # elif g1[i][j] == 255:
        #     g1[i][j] = 255
        # else :
        #     g1[i][j] = 0  
# pyplot.hist(g1)
# pyplot.show()                            
cv2.imwrite("result3.jpg",g1)  

# edge crispening
sam2 = cv2.imread("sample2.jpg")
print(sam2.shape)
h,w,c = sam2.shape
sam2_N = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        sam2_N[i][j] = sam2[i][j][0]

g1 = np.zeros([h,w], dtype=np.uint8)
g2 = np.zeros([h,w], dtype=np.uint8)      

for i in range(1,h-1):
    for j in range(1,w-1):
        g2[i][j] = 1/16*(sam2_N[i-1][j-1]+2*sam2_N[i-1][j]+sam2_N[i-1][j+1]+2*sam2_N[i][j-1]+4*sam2_N[i][j]+2*sam2_N[i][j+1]+sam2_N[i+1][j-1]+2*sam2_N[i+1][j]+sam2_N[i+1][j+1])

for i in range(h):
    for j in range(w):
        g1[i][j]=3*sam2_N[i][j]-2*g2[i][j]

cv2.imwrite("result4.jpg",g1)