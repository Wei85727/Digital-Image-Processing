# prob2.py
import numpy as np
import cv2
import matplotlib.pyplot as pyplot

# a
sample3 = cv2.imread("sample3.jpg")
h ,w ,c = sample3.shape
sam3 = np.zeros([h,w],dtype=np.uint8)
for i in range(h):
    for j in range(w):
        sam3[i][j] = sample3[i][j][1]

res = np.zeros([h,w],dtype=np.uint8) 
for i in range(64,512):
    for j in range(96,576):
            res[np.int(np.round(1.8*i))][np.int(np.round(1.8*j))] = sam3[i][j]
for i in range(64,960):
    for j in range(96,1056):
        if res[i][j] == 0:
            res[i][j] = res[i-1][j]

for i in range(64,960):
    for j in range(96,1056):
        if res[i][j] == 0:
            res[i][j] = res[i][j-1]

res1 = np.zeros([h,w],dtype=np.uint8) 
for i in range(64,960):
    for j in range(96,1056):
        res1[i][j+64] = res[i][j] 
            
# pyplot.hist(sam3)
# pyplot.legend()
# pyplot.show()
cv2.imwrite("result5.jpg",res1)

# b
sam5 = cv2.imread("sample5.jpg")
h,w,c = sam5.shape
sam5_N = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        sam5_N[i][j] = sam5[i][j][0]


for i in range(32,h-32):
    for j in range(128): 
        if j <= 63:
            sam5_N[i][j] =  sam5_N[i+32-j][j]  
        else:
            sam5_N[i][j] = sam5_N[i-32+(j-64)][j]  

for i in range(32,h-32):
    for j in range(128,256): 
        if j <= 191:
            sam5_N[i][j] =  sam5_N[i+32-(j-128)][j]  
        else:
            sam5_N[i][j] = sam5_N[i-32+(j-192)][j]   

for i in range(32,h-32):
    for j in range(256,384): 
        if j <= 319:
            sam5_N[i][j] =  sam5_N[i+32-(j-256)][j]  
        else:
            sam5_N[i][j] = sam5_N[i-32+(j-320)][j] 

for i in range(32,h-32):
    for j in range(384,512): 
        if j <= 447:
            sam5_N[i][j] =  sam5_N[i+32-(j-384)][j]  
        else:
            sam5_N[i][j] = sam5_N[i-32+(j-448)][j]  

for i in range(32,h-32):
    for j in range(512,640): 
        if j <= 575:
            sam5_N[i][j] =  sam5_N[i+32-(j-512)][j]  
        else:
            sam5_N[i][j] = sam5_N[i-32+(j-576)][j]     

for i in range(32,h-32):
    for j in range(640,768): 
        if j <= 703:
            sam5_N[i][j] =  sam5_N[i+32-(j-640)][j]  
        else:
            sam5_N[i][j] = sam5_N[i-32+(j-704)][j]   

for i in range(32,h-32):
    for j in range(768,896): 
        if j <= 831:
            sam5_N[i][j] =  sam5_N[i+32-(j-768)][j]  
        else:
            sam5_N[i][j] = sam5_N[i-32+(j-832)][j]   

for i in range(32,h-32):
    for j in range(896,1024): 
        if j <= 959:
            sam5_N[i][j] =  sam5_N[i+32-(j-896)][j]  
        else:
            sam5_N[i][j] = sam5_N[i-32+(j-960)][j]   

for i in range(32,h-32):
    for j in range(1024,1152): 
        if j <= 1087:
            sam5_N[i][j] =  sam5_N[i+32-(j-1024)][j]  
        else:
            sam5_N[i][j] = sam5_N[i-32+(j-1088)][j]   

for i in range(32,h-32):
    for j in range(1152,1280): 
        if j <= 1215:
            sam5_N[i][j] =  sam5_N[i+32-(j-1152)][j]  
        else:
            sam5_N[i][j] = sam5_N[i-32+(j-1216)][j]   
cv2.imwrite("result6.jpg",sam5_N) 
