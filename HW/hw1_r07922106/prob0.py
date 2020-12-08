# prob0
import numpy as np
import cv2
import matplotlib.pyplot as pyplot

# 讀取raw檔存jpg
dt = np.ubyte
img = np.fromfile('sample1.raw',dtype=dt)
print(img.shape)
img = np.reshape(img,(400,600))
print(img[50][60])
cv2.imwrite("result1.jpg",img)

# 彩色轉灰階
img1 = cv2.imread("sample2.jpg")
# img1_G = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
h,w,c = img1.shape
print(h,w,c)
img1_G = np.zeros([h,w], dtype=np.uint8)
print(img1_G.shape)
for i in range(h):
    for j in range(w):
        img1_G[i][j] = img1[i][j][0]*0.299+img1[i][j][1]*0.587+img1[i][j][2]*0.114
cv2.imwrite("result2.jpg",img1_G)

# 逆時針旋轉90%
empty_img = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        empty_img[511-j,i] = img1_G[i,j]
cv2.imwrite("result3.jpg", empty_img)

# 對角線翻轉
for i in range(h):
    for j in range(w):
        empty_img[j,i] = img1_G[i,j]
cv2.imwrite("result4.jpg", empty_img)