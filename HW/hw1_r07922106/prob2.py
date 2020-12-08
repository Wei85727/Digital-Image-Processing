# prob2
import numpy as np
import cv2
import matplotlib.pyplot as pyplot

#(a)
img = cv2.imread("sample4.jpg")
h,w,c = img.shape
img1 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        img1[i][j] = img[i][j][0]
mean = 0
var = 1
gauss = np.random.normal(mean,var,(h,w))
gauss = gauss.reshape(h,w)
noisy1 = img1 + 10*gauss
noisy2 = img1 + 20*gauss
cv2.imwrite("resultG1.jpg",noisy1)
cv2.imwrite("resultG2.jpg",noisy2)
      
#(b)
img = cv2.imread("sample4.jpg")
h,w,c = img.shape
img1 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        img1[i][j] = img[i][j][0]
sp = 0.5
amount = [0.025,0.05]
out = np.copy(img1)
for i in range(2):
    out = np.copy(img1)
    num_salt = np.ceil(amount[i] * img1.size * sp)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img1.shape]
    out[coords] = 255
    num_pepper = np.ceil(amount[i]* img1.size * sp)
    coords = [np.random.randint(0, i - 1, int(num_pepper))for i in img1.shape]
    out[coords] = 0
    if i == 0:
        cv2.imwrite("resultS1.jpg",out)
    else:
        cv2.imwrite("resultS2.jpg",out)

#(c)
for i in range(1,h-1):
    for j in range(1,w-1):
        img1[i][j] = 1/9*(noisy1[i-1][j-1]+noisy1[i-1][j]+noisy1[i-1][j+1]+noisy1[i][j-1]+noisy1[i][j]+noisy1[i][j+1]+noisy1[i+1][j-1]+noisy1[i+1][j]+noisy1[i+1][j+1])
mse = 0
for i in range(h):
    for j in range(w):
        mse += np.power((noisy1[i][j]-img1[i][j]),2)        
MSE1 = 1/240000*mse
PSNR1 = 10*np.log10(65025/MSE1)
print(PSNR1)
cv2.imwrite("resultR1.jpg",img1)      

for i in range(2,h-2):
    for j in range(2,w-2):
        img1[i][j] = 1/86*(noisy2[i-2][j-2]+2*noisy2[i-2][j-1]+4*noisy2[i-2][j]+2*noisy2[i-2][j+1]+noisy2[i-2][j+2]+2*noisy2[i-1][j-2]+4*noisy2[i-1][j-1]+6*noisy2[i-1][j]+4*noisy2[i-1][j+1]+2*noisy2[i-1][j+2]+4*noisy2[i][j-2]+6*noisy2[i][j-1]+10*noisy2[i][j]+6*noisy2[i][j+1]+4*noisy2[i][j+2]+2*noisy2[i+1][j-2]+4*noisy2[i+1][j-1]+6*noisy2[i+1][j]+4*noisy2[i+1][j+1]+2*noisy2[i+1][j+2]+noisy2[i+2][j-2]+2*noisy2[i+2][j-1]+4*noisy2[i+2][j]+2*noisy2[i+2][j+1]+noisy2[i+2][j+2])       
mse = 0
for i in range(h):
    for j in range(w):
        mse += np.power((noisy2[i][j]-img1[i][j]),2)        
MSE2 = 1/240000*mse
PSNR2 = 10*np.log10(65025/MSE2)
print(PSNR2)
cv2.imwrite("resultR2.jpg",img1)           

#(d)
for i in range(2):
    if i == 0:
        img = cv2.imread("resultS1.jpg")
        h, w, c = img.shape
        img0 = np.zeros([h+2,w+2], dtype=np.uint8)
        for i in range(1,h+1):
            for j in range(1,w+1):
                img0[i][j]=img[i-1][j-1][1]
                img0[401][j]=img[399][j-1][1]
                img0[0][j]=img[0][j-1][1]
                img0[i][601]=img[i-1][599][1]
                img0[i][0]=img[i-1][0][1]
        img0[0][0]=img[0][0][1] 
        img0[0][601]=img[0][599][1]
        img0[401][0]=img[399][0][1]
        img0[401][601]=img[399][599][1]   
        img1 = img0
        res1 = np.zeros([h,w], dtype=np.uint8)
        for i in range(1,h+1):
            for j in range(1,w+1):
                if (img1[i][j] <= 5 or img1[i][j] >= 250):
                    aa=sorted([img1[i-1][j-1],img1[i-1][j],img1[i-1][j+1],img1[i][j-1],img1[i][j],img1[i][j+1],img1[i+1][j-1],img1[i+1][j],img1[i+1][j+1]])
                    res1[i-1][j-1]=round(np.mean(aa[3:6]))
                else:
                    res1[i-1][j-1]=img1[i][j]
        cv2.imwrite("resultR3.jpg",res1)
    else :
        img = cv2.imread("resultS2.jpg")
        h, w, c = img.shape
        img0 = np.zeros([h+2,w+2], dtype=np.uint8)
        for i in range(1,h+1):
            for j in range(1,w+1):
                img0[i][j]=img[i-1][j-1][1]
                img0[401][j]=img[399][j-1][1]
                img0[0][j]=img[0][j-1][1]
                img0[i][601]=img[i-1][599][1]
                img0[i][0]=img[i-1][0][1]
        img0[0][0]=img[0][0][1] 
        img0[0][601]=img[0][599][1]
        img0[401][0]=img[399][0][1]
        img0[401][601]=img[399][599][1]   
        img1 = img0
        res2 = np.zeros([h,w], dtype=np.uint8)
        for i in range(1,h+1):
            for j in range(1,w+1):
                if (img1[i][j] <= 5 or img1[i][j] >= 250):
                    aa=sorted([img1[i-1][j-1],img1[i-1][j],img1[i-1][j+1],img1[i][j-1],img1[i][j],img1[i][j+1],img1[i+1][j-1],img1[i+1][j],img1[i+1][j+1]])
                    res2[i-1][j-1]=round(np.mean(aa[3:6]))
                else:
                    res2[i-1][j-1]=img1[i][j]
        cv2.imwrite("resultR4.jpg",res2)

# (e) PSNR
img = cv2.imread("resultS1.jpg")
img1 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        img1[i][j] = img[i][j][0]
mse = 0
for i in range(h):
    for j in range(w):
        mse += np.power((img1[i][j]-res1[i][j]),2)        
MSE3 = 1/240000*mse
PSNR3 = 10*np.log10(65025/MSE3)
print(PSNR3)

img = cv2.imread("resultS2.jpg")
img1 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        img1[i][j] = img[i][j][0]
mse = 0
for i in range(h):
    for j in range(w):
        mse += np.power((img1[i][j]-res2[i][j]),2)        
MSE4 = 1/240000*mse
PSNR4 = 10*np.log10(65025/MSE4)
print(PSNR4)    