# prob1
import numpy as np
import cv2
import matplotlib.pyplot as pyplot

# (a)
res1 = cv2.imread("result1.jpg")
sam3 = cv2.imread("sample3.jpg")
h,w,c = res1.shape
res1_N = np.zeros([h,w], dtype=np.uint8)
sam3_N = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        res1_N[i][j] = res1[i][j][0]
        sam3_N[i][j] = sam3[i][j][0]
# pyplot.hist(res1_N)
# pyplot.show()
# pyplot.hist(sam3_N)
# pyplot.show()  

# (b)
cdf = np.zeros(64)
for i in range(np.max(sam3_N)+1):
    cdf[i] = np.sum(sam3_N <= i)
print(cdf)    
res5 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        res5[i][j] = np.round((cdf[sam3_N[i][j]]-112)/239888*255)
print(res5) 
cv2.imwrite("result5.jpg",res5) 
# pyplot.hist(res5)
# pyplot.show()      

# res5 = cv2.equalizeHist(sam3_N)
# cv2.imwrite("result5.jpg",res5)

#(c)
sam3 = cv2.imread("sample3.jpg")
sam3_N = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        sam3_N[i][j] = sam3[i][j][0]       
# 先將圖片切隔成6塊 a1,a2,....,a6
a1 = sam3_N[0:200,0:200]
a2 = sam3_N[0:200,200:400]
a3 = sam3_N[0:200,400:600]
a4 = sam3_N[200:400,0:200]
a5 = sam3_N[200:400,200:400]
a6 = sam3_N[200:400,400:600]
# a1
cdf1 = np.zeros(np.max(a1)+1)
for i in range(np.max(a1)+1):
    cdf1[i] = np.sum(a1 <= i)    
t1 = np.zeros([200,200], dtype=np.uint8)
for i in range(200):
    for j in range(200):
        t1[i][j] = np.round((cdf1[a1[i][j]]-0)/40000*255) 
print(np.min(cdf1))    
# a2
cdf2 = np.zeros(np.max(a2)+1)
for i in range(np.max(a2)+1):
    cdf2[i] = np.sum(a2 <= i)    
t2 = np.zeros([200,200], dtype=np.uint8)
for i in range(200):
    for j in range(200):
        t2[i][j] = np.round((cdf2[a2[i][j]]-0)/40000*255)
print(np.min(cdf2))        
# a3
cdf3 = np.zeros(np.max(a3)+1)
for i in range(np.max(a3)+1):
    cdf3[i] = np.sum(a3 <= i)    
t3 = np.zeros([200,200], dtype=np.uint8)
for i in range(200):
    for j in range(200):
        t3[i][j] = np.round((cdf3[a3[i][j]]-0)/40000*255)
print(np.min(cdf3))        
# a4
cdf4 = np.zeros(np.max(a4)+1)
for i in range(np.max(a4)+1):
    cdf4[i] = np.sum(a1 <= i)    
t4 = np.zeros([200,200], dtype=np.uint8)
for i in range(200):
    for j in range(200):
        t4[i][j] = np.round((cdf4[a4[i][j]]-0)/40000*255)
print(np.min(cdf4))        
#a5
cdf5 = np.zeros(np.max(a5)+1)
for i in range(np.max(a5)+1):
    cdf5[i] = np.sum(a5 <= i)    
t5 = np.zeros([200,200], dtype=np.uint8)
for i in range(200):
    for j in range(200):
        t5[i][j] = np.round((cdf5[a5[i][j]]-53)/39947*255)
print(np.min(cdf5))        
#a6
cdf6 = np.zeros(np.max(a6)+1)
for i in range(np.max(a6)+1):
    cdf6[i] = np.sum(a6 <= i)    
t6 = np.zeros([200,200], dtype=np.uint8)
for i in range(200):
    for j in range(200):
        t6[i][j] = np.round((cdf6[a6[i][j]]-4)/39996*255)
print(np.min(cdf6))        


res6a = np.hstack((t1,t2,t3))
res6b = np.hstack((t4,t5,t6))
res6 = np.vstack((res6a,res6b))
cv2.imwrite("result6.jpg",res6)


#(e)
res7 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        res7[i][j] = 255*(np.log(sam3_N[i][j]/64+1))
print(res7) 
cv2.imwrite("result7.jpg",res7)  


res8 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        res8[i][j] = np.power(1.095,sam3_N[i][j])
print(res8) 
cv2.imwrite("result8.jpg",res8)

res9 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        res9[i][j] = np.power(sam3_N[i][j]/256,1/3)*256
print(res9) 
cv2.imwrite("result9.jpg",res9)  

