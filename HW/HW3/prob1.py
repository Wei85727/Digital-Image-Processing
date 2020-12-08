# prob1
import numpy as np
import cv2
import matplotlib.pyplot as pyplot
import time

start = time.clock()
sam1 = cv2.imread("sample1.jpg")
h,w,c = sam1.shape
sam1_N = np.zeros([h+2,w+2], dtype=np.uint8)
# 延展圖片
for i in range(h):
    for j in range(w):
        sam1_N[i+1][0] = sam1[i][0][0]
        sam1_N[0][j+1] = sam1[0][j][0]
        sam1_N[i+1][1001] = sam1[i][999][0]
        sam1_N[706][j+1] = sam1[704][j][0]
        sam1_N[0][0] = sam1[0][0][0]
        sam1_N[706][0] = sam1[704][0][0]
        sam1_N[0][1001] = sam1[0][999][0]
        sam1_N[706][1001] = sam1[704][999][0]
        sam1_N[i+1][j+1] = sam1[i][j][0]      

M1 = np.zeros([h,w], dtype=np.uint8)        
for i in range(h):
    for j in range(w):
        M1[i][j] = 1/36*(sam1_N[i][j]+2*sam1_N[i][j+1]+sam1_N[i][j+2]+2*sam1_N[i+1][j]+4*sam1_N[i+1][j+1]+2*sam1_N[i+1][j+2]+sam1_N[i+2][j]+2*sam1_N[i+2][j+1]+sam1_N[i+2][j+2])

M2 = np.zeros([h,w], dtype=np.uint8)        
for i in range(h):
    for j in range(w):
        M2[i][j] = 1/12*(sam1_N[i][j]+0*sam1_N[i][j+1]+(-1)*sam1_N[i][j+2]+2*sam1_N[i+1][j]+0*sam1_N[i+1][j+1]+(-2)*sam1_N[i+1][j+2]+sam1_N[i+2][j]+0*sam1_N[i+2][j+1]+(-1)*sam1_N[i+2][j+2])

M3 = np.zeros([h,w], dtype=np.uint8)        
for i in range(h):
    for j in range(w):
        M3[i][j] = 1/12*((-1)*sam1_N[i][j]+2*sam1_N[i][j+1]+(-1)*sam1_N[i][j+2]+(-2)*sam1_N[i+1][j]+4*sam1_N[i+1][j+1]+(-2)*sam1_N[i+1][j+2]+(-1)*sam1_N[i+2][j]+2*sam1_N[i+2][j+1]+(-1)*sam1_N[i+2][j+2])

M4 = np.zeros([h,w], dtype=np.uint8)        
for i in range(h):
    for j in range(w):
        M4[i][j] = 1/12*((-1)*sam1_N[i][j]+(-2)*sam1_N[i][j+1]+(-1)*sam1_N[i][j+2]+0*sam1_N[i+1][j]+0*sam1_N[i+1][j+1]+0*sam1_N[i+1][j+2]+sam1_N[i+2][j]+2*sam1_N[i+2][j+1]+sam1_N[i+2][j+2])     

M5 = np.zeros([h,w], dtype=np.uint8)        
for i in range(h):
    for j in range(w):
        M5[i][j] = 1/4*(sam1_N[i][j]+0*sam1_N[i][j+1]+(-1)*sam1_N[i][j+2]+0*sam1_N[i+1][j]+0*sam1_N[i+1][j+1]+0*sam1_N[i+1][j+2]+(-1)*sam1_N[i+2][j]+0*sam1_N[i+2][j+1]+sam1_N[i+2][j+2])

M6 = np.zeros([h,w], dtype=np.uint8)        
for i in range(h):
    for j in range(w):
        M6[i][j] = 1/4*((-1)*sam1_N[i][j]+2*sam1_N[i][j+1]+(-1)*sam1_N[i][j+2]+0*sam1_N[i+1][j]+0*sam1_N[i+1][j+1]+0*sam1_N[i+1][j+2]+sam1_N[i+2][j]+(-2)*sam1_N[i+2][j+1]+sam1_N[i+2][j+2])

M7 = np.zeros([h,w], dtype=np.uint8)        
for i in range(h):
    for j in range(w):
        M7[i][j] = 1/12*((-1)*sam1_N[i][j]+(-2)*sam1_N[i][j+1]+(-1)*sam1_N[i][j+2]+2*sam1_N[i+1][j]+4*sam1_N[i+1][j+1]+2*sam1_N[i+1][j+2]+(-1)*sam1_N[i+2][j]+(-2)*sam1_N[i+2][j+1]+(-1)*sam1_N[i+2][j+2])    

M8 = np.zeros([h,w], dtype=np.uint8)        
for i in range(h):
    for j in range(w):
        M8[i][j] = 1/4*((-1)*sam1_N[i][j]+0*sam1_N[i][j+1]+1*sam1_N[i][j+2]+2*sam1_N[i+1][j]+0*sam1_N[i+1][j+1]+(-2)*sam1_N[i+1][j+2]+(-1)*sam1_N[i+2][j]+0*sam1_N[i+2][j+1]+1*sam1_N[i+2][j+2])

M9 = np.zeros([h,w], dtype=np.uint8)        
for i in range(h):
    for j in range(w):
        M9[i][j] = 1/4*(1*sam1_N[i][j]+(-2)*sam1_N[i][j+1]+1*sam1_N[i][j+2]+(-2)*sam1_N[i+1][j]+4*sam1_N[i+1][j+1]+(-2)*sam1_N[i+1][j+2]+1*sam1_N[i+2][j]+(-2)*sam1_N[i+2][j+1]+1*sam1_N[i+2][j+2])                       

# T1 = np.zeros([h,w], dtype=np.uint8)  
# for i in range(4,h-4):
#     for j in range(4,w-4):
#         T1[i][j] = np.power(M1[i-4][j-4],2)+np.power(M1[i-4][j-3],2)+np.power(M1[i-4][j-2],2)+np.power(M1[i-4][j-1],2)+np.power(M1[i-4][j],2)+np.power(M1[i-4][j+1],2)+np.power(M1[i-4][j+2],2)+np.power(M1[i-4][j+3],2)+np.power(M1[i-4][j+4],2)+np.power(M1[i-3][j-4],2)+np.power(M1[i-3][j-3],2)+np.power(M1[i-3][j-2],2)+np.power(M1[i-3][j-1],2)+np.power(M1[i-3][j],2)+np.power(M1[i-3][j+1],2)+np.power(M1[i-3][j+2],2)+np.power(M1[i-3][j+3],2)+np.power(M1[i-3][j+4],2)+np.power(M1[i-2][j-4],2)+np.power(M1[i-2][j-3],2)+np.power(M1[i-2][j-2],2)+np.power(M1[i-2][j-1],2)+np.power(M1[i-2][j],2)+np.power(M1[i-2][j+1],2)+np.power(M1[i-2][j+2],2)+np.power(M1[i-2][j+3],2)+np.power(M1[i-2][j+4],2)+np.power(M1[i-1][j-4],2)+np.power(M1[i-1][j-3],2)+np.power(M1[i-1][j-2],2)+np.power(M1[i-1][j-1],2)+np.power(M1[i-1][j],2)+np.power(M1[i-1][j+1],2)+np.power(M1[i-1][j+2],2)+np.power(M1[i-1][j+3],2)+np.power(M1[i-1][j+4],2)+np.power(M1[i][j-4],2)+np.power(M1[i][j-3],2)+np.power(M1[i][j-2],2)+np.power(M1[i][j-1],2)+np.power(M1[i][j],2)+np.power(M1[i][j+1],2)+np.power(M1[i][j+2],2)+np.power(M1[i][j+3],2)+np.power(M1[i][j+4],2)+np.power(M1[i+1][j-4],2)+np.power(M1[i+1][j-3],2)+np.power(M1[i+1][j-2],2)+np.power(M1[i+1][j-1],2)+np.power(M1[i+1][j],2)+np.power(M1[i+1][j+1],2)+np.power(M1[i+1][j+2],2)+np.power(M1[i+1][j+3],2)+np.power(M1[i+1][j+4],2)+np.power(M1[i+2][j-4],2)+np.power(M1[i+2][j-3],2)+np.power(M1[i+2][j-2],2)+np.power(M1[i+2][j-1],2)+np.power(M1[i+2][j],2)+np.power(M1[i+2][j+1],2)+np.power(M1[i+2][j+2],2)+np.power(M1[i+2][j+3],2)+np.power(M1[i+2][j+4],2)+np.power(M1[i+3][j-4],2)+np.power(M1[i+3][j-3],2)+np.power(M1[i+3][j-2],2)+np.power(M1[i+3][j-1],2)+np.power(M1[i+3][j],2)+np.power(M1[i+3][j+1],2)+np.power(M1[i+3][j+2],2)+np.power(M1[i+3][j+3],2)+np.power(M1[i+3][j+4],2)+np.power(M1[i+4][j-4],2)+np.power(M1[i+4][j-3],2)+np.power(M1[i+4][j-2],2)+np.power(M1[i+4][j-1],2)+np.power(M1[i+4][j],2)+np.power(M1[i+4][j+1],2)+np.power(M1[i+4][j+2],2)+np.power(M1[i+4][j+3],2)+np.power(M1[i+4][j+4],2)

T2 = np.zeros([h,w], dtype=np.uint8)  
for i in range(4,h-4):
    for j in range(4,w-4):
        T2[i][j] = np.power(M2[i-4][j-4],2)+np.power(M2[i-4][j-3],2)+np.power(M2[i-4][j-2],2)+np.power(M2[i-4][j-1],2)+np.power(M2[i-4][j],2)+np.power(M2[i-4][j+1],2)+np.power(M2[i-4][j+2],2)+np.power(M2[i-4][j+3],2)+np.power(M2[i-4][j+4],2)+np.power(M2[i-3][j-4],2)+np.power(M2[i-3][j-3],2)+np.power(M2[i-3][j-2],2)+np.power(M2[i-3][j-1],2)+np.power(M2[i-3][j],2)+np.power(M2[i-3][j+1],2)+np.power(M2[i-3][j+2],2)+np.power(M2[i-3][j+3],2)+np.power(M2[i-3][j+4],2)+np.power(M2[i-2][j-4],2)+np.power(M2[i-2][j-3],2)+np.power(M2[i-2][j-2],2)+np.power(M2[i-2][j-1],2)+np.power(M2[i-2][j],2)+np.power(M2[i-2][j+1],2)+np.power(M2[i-2][j+2],2)+np.power(M2[i-2][j+3],2)+np.power(M2[i-2][j+4],2)+np.power(M2[i-1][j-4],2)+np.power(M2[i-1][j-3],2)+np.power(M2[i-1][j-2],2)+np.power(M2[i-1][j-1],2)+np.power(M2[i-1][j],2)+np.power(M2[i-1][j+1],2)+np.power(M2[i-1][j+2],2)+np.power(M2[i-1][j+3],2)+np.power(M2[i-1][j+4],2)+np.power(M2[i][j-4],2)+np.power(M2[i][j-3],2)+np.power(M2[i][j-2],2)+np.power(M2[i][j-1],2)+np.power(M2[i][j],2)+np.power(M2[i][j+1],2)+np.power(M2[i][j+2],2)+np.power(M2[i][j+3],2)+np.power(M2[i][j+4],2)+np.power(M2[i+1][j-4],2)+np.power(M2[i+1][j-3],2)+np.power(M2[i+1][j-2],2)+np.power(M2[i+1][j-1],2)+np.power(M2[i+1][j],2)+np.power(M2[i+1][j+1],2)+np.power(M2[i+1][j+2],2)+np.power(M2[i+1][j+3],2)+np.power(M2[i+1][j+4],2)+np.power(M2[i+2][j-4],2)+np.power(M2[i+2][j-3],2)+np.power(M2[i+2][j-2],2)+np.power(M2[i+2][j-1],2)+np.power(M2[i+2][j],2)+np.power(M2[i+2][j+1],2)+np.power(M2[i+2][j+2],2)+np.power(M2[i+2][j+3],2)+np.power(M2[i+2][j+4],2)+np.power(M2[i+3][j-4],2)+np.power(M2[i+3][j-3],2)+np.power(M2[i+3][j-2],2)+np.power(M2[i+3][j-1],2)+np.power(M2[i+3][j],2)+np.power(M2[i+3][j+1],2)+np.power(M2[i+3][j+2],2)+np.power(M2[i+3][j+3],2)+np.power(M2[i+3][j+4],2)+np.power(M2[i+4][j-4],2)+np.power(M2[i+4][j-3],2)+np.power(M2[i+4][j-2],2)+np.power(M2[i+4][j-1],2)+np.power(M2[i+4][j],2)+np.power(M2[i+4][j+1],2)+np.power(M2[i+4][j+2],2)+np.power(M2[i+4][j+3],2)+np.power(M2[i+4][j+4],2)
       
T3 = np.zeros([h,w], dtype=np.uint8)  
for i in range(4,h-4):
    for j in range(4,w-4):
        T3[i][j] = np.power(M3[i-4][j-4],2)+np.power(M3[i-4][j-3],2)+np.power(M3[i-4][j-2],2)+np.power(M3[i-4][j-1],2)+np.power(M3[i-4][j],2)+np.power(M3[i-4][j+1],2)+np.power(M3[i-4][j+2],2)+np.power(M3[i-4][j+3],2)+np.power(M3[i-4][j+4],2)+np.power(M3[i-3][j-4],2)+np.power(M3[i-3][j-3],2)+np.power(M3[i-3][j-2],2)+np.power(M3[i-3][j-1],2)+np.power(M3[i-3][j],2)+np.power(M3[i-3][j+1],2)+np.power(M3[i-3][j+2],2)+np.power(M3[i-3][j+3],2)+np.power(M3[i-3][j+4],2)+np.power(M3[i-2][j-4],2)+np.power(M3[i-2][j-3],2)+np.power(M3[i-2][j-2],2)+np.power(M3[i-2][j-1],2)+np.power(M3[i-2][j],2)+np.power(M3[i-2][j+1],2)+np.power(M3[i-2][j+2],2)+np.power(M3[i-2][j+3],2)+np.power(M3[i-2][j+4],2)+np.power(M3[i-1][j-4],2)+np.power(M3[i-1][j-3],2)+np.power(M3[i-1][j-2],2)+np.power(M3[i-1][j-1],2)+np.power(M3[i-1][j],2)+np.power(M3[i-1][j+1],2)+np.power(M3[i-1][j+2],2)+np.power(M3[i-1][j+3],2)+np.power(M3[i-1][j+4],2)+np.power(M3[i][j-4],2)+np.power(M3[i][j-3],2)+np.power(M3[i][j-2],2)+np.power(M3[i][j-1],2)+np.power(M3[i][j],2)+np.power(M3[i][j+1],2)+np.power(M3[i][j+2],2)+np.power(M3[i][j+3],2)+np.power(M3[i][j+4],2)+np.power(M3[i+1][j-4],2)+np.power(M3[i+1][j-3],2)+np.power(M3[i+1][j-2],2)+np.power(M3[i+1][j-1],2)+np.power(M3[i+1][j],2)+np.power(M3[i+1][j+1],2)+np.power(M3[i+1][j+2],2)+np.power(M3[i+1][j+3],2)+np.power(M3[i+1][j+4],2)+np.power(M3[i+2][j-4],2)+np.power(M3[i+2][j-3],2)+np.power(M3[i+2][j-2],2)+np.power(M3[i+2][j-1],2)+np.power(M3[i+2][j],2)+np.power(M3[i+2][j+1],2)+np.power(M3[i+2][j+2],2)+np.power(M3[i+2][j+3],2)+np.power(M3[i+2][j+4],2)+np.power(M3[i+3][j-4],2)+np.power(M3[i+3][j-3],2)+np.power(M3[i+3][j-2],2)+np.power(M3[i+3][j-1],2)+np.power(M3[i+3][j],2)+np.power(M3[i+3][j+1],2)+np.power(M3[i+3][j+2],2)+np.power(M3[i+3][j+3],2)+np.power(M3[i+3][j+4],2)+np.power(M3[i+4][j-4],2)+np.power(M3[i+4][j-3],2)+np.power(M3[i+4][j-2],2)+np.power(M3[i+4][j-1],2)+np.power(M3[i+4][j],2)+np.power(M3[i+4][j+1],2)+np.power(M3[i+4][j+2],2)+np.power(M3[i+4][j+3],2)+np.power(M3[i+4][j+4],2)

T4 = np.zeros([h,w], dtype=np.uint8)  
for i in range(4,h-4):
    for j in range(4,w-4):
        T4[i][j] = np.power(M4[i-4][j-4],2)+np.power(M4[i-4][j-3],2)+np.power(M4[i-4][j-2],2)+np.power(M4[i-4][j-1],2)+np.power(M4[i-4][j],2)+np.power(M4[i-4][j+1],2)+np.power(M4[i-4][j+2],2)+np.power(M4[i-4][j+3],2)+np.power(M4[i-4][j+4],2)+np.power(M4[i-3][j-4],2)+np.power(M4[i-3][j-3],2)+np.power(M4[i-3][j-2],2)+np.power(M4[i-3][j-1],2)+np.power(M4[i-3][j],2)+np.power(M4[i-3][j+1],2)+np.power(M4[i-3][j+2],2)+np.power(M4[i-3][j+3],2)+np.power(M4[i-3][j+4],2)+np.power(M4[i-2][j-4],2)+np.power(M4[i-2][j-3],2)+np.power(M4[i-2][j-2],2)+np.power(M4[i-2][j-1],2)+np.power(M4[i-2][j],2)+np.power(M4[i-2][j+1],2)+np.power(M4[i-2][j+2],2)+np.power(M4[i-2][j+3],2)+np.power(M4[i-2][j+4],2)+np.power(M4[i-1][j-4],2)+np.power(M4[i-1][j-3],2)+np.power(M4[i-1][j-2],2)+np.power(M4[i-1][j-1],2)+np.power(M4[i-1][j],2)+np.power(M4[i-1][j+1],2)+np.power(M4[i-1][j+2],2)+np.power(M4[i-1][j+3],2)+np.power(M4[i-1][j+4],2)+np.power(M4[i][j-4],2)+np.power(M4[i][j-3],2)+np.power(M4[i][j-2],2)+np.power(M4[i][j-1],2)+np.power(M4[i][j],2)+np.power(M4[i][j+1],2)+np.power(M4[i][j+2],2)+np.power(M4[i][j+3],2)+np.power(M4[i][j+4],2)+np.power(M4[i+1][j-4],2)+np.power(M4[i+1][j-3],2)+np.power(M4[i+1][j-2],2)+np.power(M4[i+1][j-1],2)+np.power(M4[i+1][j],2)+np.power(M4[i+1][j+1],2)+np.power(M4[i+1][j+2],2)+np.power(M4[i+1][j+3],2)+np.power(M4[i+1][j+4],2)+np.power(M4[i+2][j-4],2)+np.power(M4[i+2][j-3],2)+np.power(M4[i+2][j-2],2)+np.power(M4[i+2][j-1],2)+np.power(M4[i+2][j],2)+np.power(M4[i+2][j+1],2)+np.power(M4[i+2][j+2],2)+np.power(M4[i+2][j+3],2)+np.power(M4[i+2][j+4],2)+np.power(M4[i+3][j-4],2)+np.power(M4[i+3][j-3],2)+np.power(M4[i+3][j-2],2)+np.power(M4[i+3][j-1],2)+np.power(M4[i+3][j],2)+np.power(M4[i+3][j+1],2)+np.power(M4[i+3][j+2],2)+np.power(M4[i+3][j+3],2)+np.power(M4[i+3][j+4],2)+np.power(M4[i+4][j-4],2)+np.power(M4[i+4][j-3],2)+np.power(M4[i+4][j-2],2)+np.power(M4[i+4][j-1],2)+np.power(M4[i+4][j],2)+np.power(M4[i+4][j+1],2)+np.power(M4[i+4][j+2],2)+np.power(M4[i+4][j+3],2)+np.power(M4[i+4][j+4],2)             

T5 = np.zeros([h,w], dtype=np.uint8)  
for i in range(4,h-4):
    for j in range(4,w-4):
        T5[i][j] = np.power(M5[i-4][j-4],2)+np.power(M5[i-4][j-3],2)+np.power(M5[i-4][j-2],2)+np.power(M5[i-4][j-1],2)+np.power(M5[i-4][j],2)+np.power(M5[i-4][j+1],2)+np.power(M5[i-4][j+2],2)+np.power(M5[i-4][j+3],2)+np.power(M5[i-4][j+4],2)+np.power(M5[i-3][j-4],2)+np.power(M5[i-3][j-3],2)+np.power(M5[i-3][j-2],2)+np.power(M5[i-3][j-1],2)+np.power(M5[i-3][j],2)+np.power(M5[i-3][j+1],2)+np.power(M5[i-3][j+2],2)+np.power(M5[i-3][j+3],2)+np.power(M5[i-3][j+4],2)+np.power(M5[i-2][j-4],2)+np.power(M5[i-2][j-3],2)+np.power(M5[i-2][j-2],2)+np.power(M5[i-2][j-1],2)+np.power(M5[i-2][j],2)+np.power(M5[i-2][j+1],2)+np.power(M5[i-2][j+2],2)+np.power(M5[i-2][j+3],2)+np.power(M5[i-2][j+4],2)+np.power(M5[i-1][j-4],2)+np.power(M5[i-1][j-3],2)+np.power(M5[i-1][j-2],2)+np.power(M5[i-1][j-1],2)+np.power(M5[i-1][j],2)+np.power(M5[i-1][j+1],2)+np.power(M5[i-1][j+2],2)+np.power(M5[i-1][j+3],2)+np.power(M5[i-1][j+4],2)+np.power(M5[i][j-4],2)+np.power(M5[i][j-3],2)+np.power(M5[i][j-2],2)+np.power(M5[i][j-1],2)+np.power(M5[i][j],2)+np.power(M5[i][j+1],2)+np.power(M5[i][j+2],2)+np.power(M5[i][j+3],2)+np.power(M5[i][j+4],2)+np.power(M5[i+1][j-4],2)+np.power(M5[i+1][j-3],2)+np.power(M5[i+1][j-2],2)+np.power(M5[i+1][j-1],2)+np.power(M5[i+1][j],2)+np.power(M5[i+1][j+1],2)+np.power(M5[i+1][j+2],2)+np.power(M5[i+1][j+3],2)+np.power(M5[i+1][j+4],2)+np.power(M5[i+2][j-4],2)+np.power(M5[i+2][j-3],2)+np.power(M5[i+2][j-2],2)+np.power(M5[i+2][j-1],2)+np.power(M5[i+2][j],2)+np.power(M5[i+2][j+1],2)+np.power(M5[i+2][j+2],2)+np.power(M5[i+2][j+3],2)+np.power(M5[i+2][j+4],2)+np.power(M5[i+3][j-4],2)+np.power(M5[i+3][j-3],2)+np.power(M5[i+3][j-2],2)+np.power(M5[i+3][j-1],2)+np.power(M5[i+3][j],2)+np.power(M5[i+3][j+1],2)+np.power(M5[i+3][j+2],2)+np.power(M5[i+3][j+3],2)+np.power(M5[i+3][j+4],2)+np.power(M5[i+4][j-4],2)+np.power(M5[i+4][j-3],2)+np.power(M5[i+4][j-2],2)+np.power(M5[i+4][j-1],2)+np.power(M5[i+4][j],2)+np.power(M5[i+4][j+1],2)+np.power(M5[i+4][j+2],2)+np.power(M5[i+4][j+3],2)+np.power(M5[i+4][j+4],2)

# T6 = np.zeros([h,w], dtype=np.uint8)  
# for i in range(4,h-4):
#     for j in range(4,w-4):
#         T6[i][j] = np.power(M6[i-4][j-4],2)+np.power(M6[i-4][j-3],2)+np.power(M6[i-4][j-2],2)+np.power(M6[i-4][j-1],2)+np.power(M6[i-4][j],2)+np.power(M6[i-4][j+1],2)+np.power(M6[i-4][j+2],2)+np.power(M6[i-4][j+3],2)+np.power(M6[i-4][j+4],2)+np.power(M6[i-3][j-4],2)+np.power(M6[i-3][j-3],2)+np.power(M6[i-3][j-2],2)+np.power(M6[i-3][j-1],2)+np.power(M6[i-3][j],2)+np.power(M6[i-3][j+1],2)+np.power(M6[i-3][j+2],2)+np.power(M6[i-3][j+3],2)+np.power(M6[i-3][j+4],2)+np.power(M6[i-2][j-4],2)+np.power(M6[i-2][j-3],2)+np.power(M6[i-2][j-2],2)+np.power(M6[i-2][j-1],2)+np.power(M6[i-2][j],2)+np.power(M6[i-2][j+1],2)+np.power(M6[i-2][j+2],2)+np.power(M6[i-2][j+3],2)+np.power(M6[i-2][j+4],2)+np.power(M6[i-1][j-4],2)+np.power(M6[i-1][j-3],2)+np.power(M6[i-1][j-2],2)+np.power(M6[i-1][j-1],2)+np.power(M6[i-1][j],2)+np.power(M6[i-1][j+1],2)+np.power(M6[i-1][j+2],2)+np.power(M6[i-1][j+3],2)+np.power(M6[i-1][j+4],2)+np.power(M6[i][j-4],2)+np.power(M6[i][j-3],2)+np.power(M6[i][j-2],2)+np.power(M6[i][j-1],2)+np.power(M6[i][j],2)+np.power(M6[i][j+1],2)+np.power(M6[i][j+2],2)+np.power(M6[i][j+3],2)+np.power(M6[i][j+4],2)+np.power(M6[i+1][j-4],2)+np.power(M6[i+1][j-3],2)+np.power(M6[i+1][j-2],2)+np.power(M6[i+1][j-1],2)+np.power(M6[i+1][j],2)+np.power(M6[i+1][j+1],2)+np.power(M6[i+1][j+2],2)+np.power(M6[i+1][j+3],2)+np.power(M6[i+1][j+4],2)+np.power(M6[i+2][j-4],2)+np.power(M6[i+2][j-3],2)+np.power(M6[i+2][j-2],2)+np.power(M6[i+2][j-1],2)+np.power(M6[i+2][j],2)+np.power(M6[i+2][j+1],2)+np.power(M6[i+2][j+2],2)+np.power(M6[i+2][j+3],2)+np.power(M6[i+2][j+4],2)+np.power(M6[i+3][j-4],2)+np.power(M6[i+3][j-3],2)+np.power(M6[i+3][j-2],2)+np.power(M6[i+3][j-1],2)+np.power(M6[i+3][j],2)+np.power(M6[i+3][j+1],2)+np.power(M6[i+3][j+2],2)+np.power(M6[i+3][j+3],2)+np.power(M6[i+3][j+4],2)+np.power(M6[i+4][j-4],2)+np.power(M6[i+4][j-3],2)+np.power(M6[i+4][j-2],2)+np.power(M6[i+4][j-1],2)+np.power(M6[i+4][j],2)+np.power(M6[i+4][j+1],2)+np.power(M6[i+4][j+2],2)+np.power(M6[i+4][j+3],2)+np.power(M6[i+4][j+4],2)

# T7 = np.zeros([h,w], dtype=np.uint8)  
# for i in range(4,h-4):
#     for j in range(4,w-4):
#         T7[i][j] = np.power(M7[i-4][j-4],2)+np.power(M7[i-4][j-3],2)+np.power(M7[i-4][j-2],2)+np.power(M7[i-4][j-1],2)+np.power(M7[i-4][j],2)+np.power(M7[i-4][j+1],2)+np.power(M7[i-4][j+2],2)+np.power(M7[i-4][j+3],2)+np.power(M7[i-4][j+4],2)+np.power(M7[i-3][j-4],2)+np.power(M7[i-3][j-3],2)+np.power(M7[i-3][j-2],2)+np.power(M7[i-3][j-1],2)+np.power(M7[i-3][j],2)+np.power(M7[i-3][j+1],2)+np.power(M7[i-3][j+2],2)+np.power(M7[i-3][j+3],2)+np.power(M7[i-3][j+4],2)+np.power(M7[i-2][j-4],2)+np.power(M7[i-2][j-3],2)+np.power(M7[i-2][j-2],2)+np.power(M7[i-2][j-1],2)+np.power(M7[i-2][j],2)+np.power(M7[i-2][j+1],2)+np.power(M7[i-2][j+2],2)+np.power(M7[i-2][j+3],2)+np.power(M7[i-2][j+4],2)+np.power(M7[i-1][j-4],2)+np.power(M7[i-1][j-3],2)+np.power(M7[i-1][j-2],2)+np.power(M7[i-1][j-1],2)+np.power(M7[i-1][j],2)+np.power(M7[i-1][j+1],2)+np.power(M7[i-1][j+2],2)+np.power(M7[i-1][j+3],2)+np.power(M7[i-1][j+4],2)+np.power(M7[i][j-4],2)+np.power(M7[i][j-3],2)+np.power(M7[i][j-2],2)+np.power(M7[i][j-1],2)+np.power(M7[i][j],2)+np.power(M7[i][j+1],2)+np.power(M7[i][j+2],2)+np.power(M7[i][j+3],2)+np.power(M7[i][j+4],2)+np.power(M7[i+1][j-4],2)+np.power(M7[i+1][j-3],2)+np.power(M7[i+1][j-2],2)+np.power(M7[i+1][j-1],2)+np.power(M7[i+1][j],2)+np.power(M7[i+1][j+1],2)+np.power(M7[i+1][j+2],2)+np.power(M7[i+1][j+3],2)+np.power(M7[i+1][j+4],2)+np.power(M7[i+2][j-4],2)+np.power(M7[i+2][j-3],2)+np.power(M7[i+2][j-2],2)+np.power(M7[i+2][j-1],2)+np.power(M7[i+2][j],2)+np.power(M7[i+2][j+1],2)+np.power(M7[i+2][j+2],2)+np.power(M7[i+2][j+3],2)+np.power(M7[i+2][j+4],2)+np.power(M7[i+3][j-4],2)+np.power(M7[i+3][j-3],2)+np.power(M7[i+3][j-2],2)+np.power(M7[i+3][j-1],2)+np.power(M7[i+3][j],2)+np.power(M7[i+3][j+1],2)+np.power(M7[i+3][j+2],2)+np.power(M7[i+3][j+3],2)+np.power(M7[i+3][j+4],2)+np.power(M7[i+4][j-4],2)+np.power(M7[i+4][j-3],2)+np.power(M7[i+4][j-2],2)+np.power(M7[i+4][j-1],2)+np.power(M7[i+4][j],2)+np.power(M7[i+4][j+1],2)+np.power(M7[i+4][j+2],2)+np.power(M7[i+4][j+3],2)+np.power(M7[i+4][j+4],2)

# T8 = np.zeros([h,w], dtype=np.uint8)  
# for i in range(4,h-4):
#     for j in range(4,w-4):
#         T8[i][j] = np.power(M8[i-4][j-4],2)+np.power(M8[i-4][j-3],2)+np.power(M8[i-4][j-2],2)+np.power(M8[i-4][j-1],2)+np.power(M8[i-4][j],2)+np.power(M8[i-4][j+1],2)+np.power(M8[i-4][j+2],2)+np.power(M8[i-4][j+3],2)+np.power(M8[i-4][j+4],2)+np.power(M8[i-3][j-4],2)+np.power(M8[i-3][j-3],2)+np.power(M8[i-3][j-2],2)+np.power(M8[i-3][j-1],2)+np.power(M8[i-3][j],2)+np.power(M8[i-3][j+1],2)+np.power(M8[i-3][j+2],2)+np.power(M8[i-3][j+3],2)+np.power(M8[i-3][j+4],2)+np.power(M8[i-2][j-4],2)+np.power(M8[i-2][j-3],2)+np.power(M8[i-2][j-2],2)+np.power(M8[i-2][j-1],2)+np.power(M8[i-2][j],2)+np.power(M8[i-2][j+1],2)+np.power(M8[i-2][j+2],2)+np.power(M8[i-2][j+3],2)+np.power(M8[i-2][j+4],2)+np.power(M8[i-1][j-4],2)+np.power(M8[i-1][j-3],2)+np.power(M8[i-1][j-2],2)+np.power(M8[i-1][j-1],2)+np.power(M8[i-1][j],2)+np.power(M8[i-1][j+1],2)+np.power(M8[i-1][j+2],2)+np.power(M8[i-1][j+3],2)+np.power(M8[i-1][j+4],2)+np.power(M8[i][j-4],2)+np.power(M8[i][j-3],2)+np.power(M8[i][j-2],2)+np.power(M8[i][j-1],2)+np.power(M8[i][j],2)+np.power(M8[i][j+1],2)+np.power(M8[i][j+2],2)+np.power(M8[i][j+3],2)+np.power(M8[i][j+4],2)+np.power(M8[i+1][j-4],2)+np.power(M8[i+1][j-3],2)+np.power(M8[i+1][j-2],2)+np.power(M8[i+1][j-1],2)+np.power(M8[i+1][j],2)+np.power(M8[i+1][j+1],2)+np.power(M8[i+1][j+2],2)+np.power(M8[i+1][j+3],2)+np.power(M8[i+1][j+4],2)+np.power(M8[i+2][j-4],2)+np.power(M8[i+2][j-3],2)+np.power(M8[i+2][j-2],2)+np.power(M8[i+2][j-1],2)+np.power(M8[i+2][j],2)+np.power(M8[i+2][j+1],2)+np.power(M8[i+2][j+2],2)+np.power(M8[i+2][j+3],2)+np.power(M8[i+2][j+4],2)+np.power(M8[i+3][j-4],2)+np.power(M8[i+3][j-3],2)+np.power(M8[i+3][j-2],2)+np.power(M8[i+3][j-1],2)+np.power(M8[i+3][j],2)+np.power(M8[i+3][j+1],2)+np.power(M8[i+3][j+2],2)+np.power(M8[i+3][j+3],2)+np.power(M8[i+3][j+4],2)+np.power(M8[i+4][j-4],2)+np.power(M8[i+4][j-3],2)+np.power(M8[i+4][j-2],2)+np.power(M8[i+4][j-1],2)+np.power(M8[i+4][j],2)+np.power(M8[i+4][j+1],2)+np.power(M8[i+4][j+2],2)+np.power(M8[i+4][j+3],2)+np.power(M8[i+4][j+4],2)        

# T9 = np.zeros([h,w], dtype=np.uint8)  
# for i in range(4,h-4):
#     for j in range(4,w-4):
#         T9[i][j] = np.power(M9[i-4][j-4],2)+np.power(M9[i-4][j-3],2)+np.power(M9[i-4][j-2],2)+np.power(M9[i-4][j-1],2)+np.power(M9[i-4][j],2)+np.power(M9[i-4][j+1],2)+np.power(M9[i-4][j+2],2)+np.power(M9[i-4][j+3],2)+np.power(M9[i-4][j+4],2)+np.power(M9[i-3][j-4],2)+np.power(M9[i-3][j-3],2)+np.power(M9[i-3][j-2],2)+np.power(M9[i-3][j-1],2)+np.power(M9[i-3][j],2)+np.power(M9[i-3][j+1],2)+np.power(M9[i-3][j+2],2)+np.power(M9[i-3][j+3],2)+np.power(M9[i-3][j+4],2)+np.power(M9[i-2][j-4],2)+np.power(M9[i-2][j-3],2)+np.power(M9[i-2][j-2],2)+np.power(M9[i-2][j-1],2)+np.power(M9[i-2][j],2)+np.power(M9[i-2][j+1],2)+np.power(M9[i-2][j+2],2)+np.power(M9[i-2][j+3],2)+np.power(M9[i-2][j+4],2)+np.power(M9[i-1][j-4],2)+np.power(M9[i-1][j-3],2)+np.power(M9[i-1][j-2],2)+np.power(M9[i-1][j-1],2)+np.power(M9[i-1][j],2)+np.power(M9[i-1][j+1],2)+np.power(M9[i-1][j+2],2)+np.power(M9[i-1][j+3],2)+np.power(M9[i-1][j+4],2)+np.power(M9[i][j-4],2)+np.power(M9[i][j-3],2)+np.power(M9[i][j-2],2)+np.power(M9[i][j-1],2)+np.power(M9[i][j],2)+np.power(M9[i][j+1],2)+np.power(M9[i][j+2],2)+np.power(M9[i][j+3],2)+np.power(M9[i][j+4],2)+np.power(M9[i+1][j-4],2)+np.power(M9[i+1][j-3],2)+np.power(M9[i+1][j-2],2)+np.power(M9[i+1][j-1],2)+np.power(M9[i+1][j],2)+np.power(M9[i+1][j+1],2)+np.power(M9[i+1][j+2],2)+np.power(M9[i+1][j+3],2)+np.power(M9[i+1][j+4],2)+np.power(M9[i+2][j-4],2)+np.power(M9[i+2][j-3],2)+np.power(M9[i+2][j-2],2)+np.power(M9[i+2][j-1],2)+np.power(M9[i+2][j],2)+np.power(M9[i+2][j+1],2)+np.power(M9[i+2][j+2],2)+np.power(M9[i+2][j+3],2)+np.power(M9[i+2][j+4],2)+np.power(M9[i+3][j-4],2)+np.power(M9[i+3][j-3],2)+np.power(M9[i+3][j-2],2)+np.power(M9[i+3][j-1],2)+np.power(M9[i+3][j],2)+np.power(M9[i+3][j+1],2)+np.power(M9[i+3][j+2],2)+np.power(M9[i+3][j+3],2)+np.power(M9[i+3][j+4],2)+np.power(M9[i+4][j-4],2)+np.power(M9[i+4][j-3],2)+np.power(M9[i+4][j-2],2)+np.power(M9[i+4][j-1],2)+np.power(M9[i+4][j],2)+np.power(M9[i+4][j+1],2)+np.power(M9[i+4][j+2],2)+np.power(M9[i+4][j+3],2)+np.power(M9[i+4][j+4],2)
a = 0
b = 0
c = 0
d1 = 0
d2 = 0
e1 = 0
e2 = 0
f1 = 0
f2 = 0

R1 = np.zeros([h,w], dtype=np.uint8) 
for i in range(h):
    for j in range(w):
        if np.power(T2[i][j]-T2[450][608],2)+np.power(T3[i][j]-T3[450][608],2)+np.power(T4[i][j]-T4[450][608],2)+np.power(T5[i][j]-T5[450][608],2) < np.power(T2[i][j]-T2[560][100],2)+np.power(T3[i][j]-T3[560][100],2)+np.power(T4[i][j]-T4[560][100],2)+np.power(T5[i][j]-T5[560][100],2) and np.power(T2[i][j]-T2[450][608],2)+np.power(T3[i][j]-T3[450][608],2)+np.power(T4[i][j]-T4[450][608],2)+np.power(T5[i][j]-T5[450][608],2) < np.power(T2[i][j]-T2[87][500],2)+np.power(T3[i][j]-T3[87][500],2)+np.power(T4[i][j]-T4[87][500],2)+np.power(T5[i][j]-T5[87][500],2) :
            R1[i][j] = 255
            a += 1
            d1 += i
            d2 += j
        elif np.power(T2[i][j]-T2[560][100],2)+np.power(T3[i][j]-T3[560][100],2)+np.power(T4[i][j]-T4[560][100],2)+np.power(T5[i][j]-T5[560][100],2) < np.power(T2[i][j]-T2[87][500],2)+np.power(T3[i][j]-T3[87][500],2)+np.power(T4[i][j]-T4[87][500],2)+np.power(T5[i][j]-T5[87][500],2) :
            R1[i][j] = 0
            b += 1
            e1 += i
            e2 += j
        else:
            R1[i][j] = 128 
            c += 1 
            f1 += i
            f2 += j  
cv2.imwrite("result1.jpg",R1)    

# z_i = round(d1/a)
# z_j = round(d2/a)
# g_i = round(e1/b)
# g_j = round(e2/b)
# s_i = round(f1/c)
# s_j = round(f2/c)
# print(z_i,z_j)
# print(g_i,g_j)
# print(s_i,s_j)
# a = 0
# b = 0
# c = 0
# d1 = 0
# d2 = 0
# e1 = 0
# e2 = 0
# f1 = 0
# f2 = 0

# R2 = np.zeros([h,w], dtype=np.uint8) 
# for i in range(h):
#     for j in range(w):
#         if np.power(T2[i][j]-T2[z_i][z_j],2)+np.power(T3[i][j]-T3[z_i][z_j],2)+np.power(T4[i][j]-T4[z_i][z_j],2)+np.power(T5[i][j]-T5[z_i][z_j],2) < np.power(T2[i][j]-T2[g_i][g_j],2)+np.power(T3[i][j]-T3[g_i][g_j],2)+np.power(T4[i][j]-T4[g_i][g_j],2)+np.power(T5[i][j]-T5[g_i][g_j],2) and np.power(T2[i][j]-T2[z_i][z_j],2)+np.power(T3[i][j]-T3[z_i][z_j],2)+np.power(T4[i][j]-T4[z_i][z_j],2)+np.power(T5[i][j]-T5[z_i][z_j],2) < np.power(T2[i][j]-T2[s_i][s_j],2)+np.power(T3[i][j]-T3[s_i][s_j],2)+np.power(T4[i][j]-T4[s_i][s_j],2)+np.power(T5[i][j]-T5[s_i][s_j],2) :
#             R2[i][j] = 255
#             a += 1
#             d1 += i
#             d2 += j
#         elif np.power(T2[i][j]-T2[g_i][g_j],2)+np.power(T3[i][j]-T3[g_i][g_j],2)+np.power(T4[i][j]-T4[g_i][g_j],2)+np.power(T5[i][j]-T5[g_i][g_j],2) < np.power(T2[i][j]-T2[s_i][s_j],2)+np.power(T3[i][j]-T3[s_i][s_j],2)+np.power(T4[i][j]-T4[s_i][s_j],2)+np.power(T5[i][j]-T5[s_i][s_j],2) :
#             R2[i][j] = 0
#             b += 1
#             e1 += i
#             e2 += j
#         else:
#             R2[i][j] = 128 
#             c += 1 
#             f1 += i
#             f2 += j 
# cv2.imwrite("R2.jpg",R2) 

# z_i = round(d1/a)
# z_j = round(d2/a)
# g_i = round(e1/b)
# g_j = round(e2/b)
# s_i = round(f1/c)
# s_j = round(f2/c)
# print(z_i,z_j)
# print(g_i,g_j)
# print(s_i,s_j)
# a = 0
# b = 0
# c = 0
# d1 = 0
# d2 = 0
# e1 = 0
# e2 = 0
# f1 = 0
# f2 = 0

# R3 = np.zeros([h,w], dtype=np.uint8) 
# for i in range(h):
#     for j in range(w):
#         if np.power(T2[i][j]-T2[z_i][z_j],2)+np.power(T3[i][j]-T3[z_i][z_j],2)+np.power(T4[i][j]-T4[z_i][z_j],2)+np.power(T5[i][j]-T5[z_i][z_j],2) < np.power(T2[i][j]-T2[g_i][g_j],2)+np.power(T3[i][j]-T3[g_i][g_j],2)+np.power(T4[i][j]-T4[g_i][g_j],2)+np.power(T5[i][j]-T5[g_i][g_j],2) and np.power(T2[i][j]-T2[z_i][z_j],2)+np.power(T3[i][j]-T3[z_i][z_j],2)+np.power(T4[i][j]-T4[z_i][z_j],2)+np.power(T5[i][j]-T5[z_i][z_j],2) < np.power(T2[i][j]-T2[s_i][s_j],2)+np.power(T3[i][j]-T3[s_i][s_j],2)+np.power(T4[i][j]-T4[s_i][s_j],2)+np.power(T5[i][j]-T5[s_i][s_j],2) :
#             R3[i][j] = 255
#             a += 1
#             d1 += i
#             d2 += j
#         elif np.power(T2[i][j]-T2[g_i][g_j],2)+np.power(T3[i][j]-T3[g_i][g_j],2)+np.power(T4[i][j]-T4[g_i][g_j],2)+np.power(T5[i][j]-T5[g_i][g_j],2) < np.power(T2[i][j]-T2[s_i][s_j],2)+np.power(T3[i][j]-T3[s_i][s_j],2)+np.power(T4[i][j]-T4[s_i][s_j],2)+np.power(T5[i][j]-T5[s_i][s_j],2) :
#             R3[i][j] = 0
#             b += 1
#             e1 += i
#             e2 += j
#         else:
#             R3[i][j] = 128 
#             c += 1 
#             f1 += i
#             f2 += j 

# z_i = round(d1/a)
# z_j = round(d2/a)
# g_i = round(e1/b)
# g_j = round(e2/b)
# s_i = round(f1/c)
# s_j = round(f2/c)
# print(z_i,z_j)
# print(g_i,g_j)
# print(s_i,s_j)
# cv2.imwrite("R3.jpg",R3) 
end = time.clock()
print(end-start)
