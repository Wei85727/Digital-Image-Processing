import numpy as np
import cv2
import matplotlib.pyplot as pyplot

#sample4.jpg
sam = cv2.imread("sample4.jpg")
h,w,c = sam.shape
sam4 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        sam4[i][j] = sam[i][j][0]*0.299+sam[i][j][1]*0.587+sam[i][j][2]*0.114

for i in range(h):
    for j in range(w):
        if sam4[i][j] > 200:
            sam4[i][j] = 0
        else :
            sam4[i][j] = 255    

t_1 = sam4[38:98,10:50]
t_2 = sam4[38:98,50:82]
t_3 = sam4[38:98,82:112]
t_4 = sam4[38:98,125:155]
t_5 = sam4[38:98,155:185]
t_6 = sam4[38:98,185:215]
t_7 = sam4[38:98,215:245]
 
cv2.imwrite("sam4.jpg",sam4)
cv2.imwrite("t_1.jpg",t_1)
cv2.imwrite("t_2.jpg",t_2)
cv2.imwrite("t_3.jpg",t_3) 
cv2.imwrite("t_4.jpg",t_4)
cv2.imwrite("t_5.jpg",t_5)
cv2.imwrite("t_6.jpg",t_6)                   
cv2.imwrite("t_7.jpg",t_7)

