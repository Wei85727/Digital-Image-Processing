# prob2
import numpy as np
import cv2
import matplotlib.pyplot as pyplot
import time

sam = cv2.imread("TrainingSet.jpg")
h,w,c = sam.shape
sam1 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        sam1[i][j] = sam[i][j][0]
# for i in range(h):
#     for j in range(w):
#         if sam1[i][j] != 255:
#             sam1[i][j] = 0        
        
t1 = sam1[:180,:1200]
t_A = t1[:,:100]
t_B = t1[:,100:200]
t_C = t1[:,200:300]
t_D = t1[:,300:400]
t_E = t1[:,400:500]
t_F = t1[:,500:600]
t_G = t1[:,600:720]
t_H = t1[:,720:820]
t_I = t1[:,820:900]
t_J = t1[:,900:1000]
t_K = t1[:,1000:1100]
t_L = t1[:,1100:1200]

t2= sam1[180:360][:]
t_M = t2[:,:110]
t_N = t2[:,110:220]
t_O = t2[:,220:330]
t_P = t2[:,330:440]
t_Q = t2[:,440:550]
t_R = t2[:,550:660]
t_S = t2[:,660:770]
t_T = t2[:,770:880]
t_U = t2[:,880:990]
t_V = t2[:,990:1100]
t_W = t2[:,1100:1200]

t3 = sam1[360:][:]
t_X = t3[:,:100]
t_Y = t3[:,100:200]
t_Z = t3[:,200:300]
t_0 = t3[:,300:400]
t_1 = t3[:,400:500]
t_2 = t3[:,500:590]
t_3 = t3[:,590:680]
t_4 = t3[:,680:770]
t_5 = t3[:,770:860]
t_6 = t3[:,860:950]
t_7 = t3[:,950:1050]
t_8 = t3[:,1050:1150]
t_9 = t3[:,1150:1240]
cv2.imwrite("t_8.jpg",t_8)

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_A[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
A = t_A[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)] 

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_B[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
B = t_B[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_C[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
C = t_C[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_D[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
D = t_D[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_E[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
E = t_E[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(10,100):
        if t_F[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
F = t_F[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(120):
        if t_G[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
G = t_G[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_H[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
H = t_H[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(80):
        if t_I[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
I = t_I[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_J[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
J = t_J[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_K[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
K = t_K[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_L[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
L = t_L[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(110):
        if t_M[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
M = t_M[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(110):
        if t_N[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
N = t_N[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(110):
        if t_O[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
O = t_O[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(110):
        if t_P[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
P = t_P[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(110):
        if t_Q[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
Q = t_Q[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(110):
        if t_R[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
R = t_R[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(110):
        if t_S[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
S = t_S[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(110):
        if t_T[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
T = t_T[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(110):
        if t_U[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
U = t_U[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(110):
        if t_V[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
V = t_V[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_W[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
W = t_W[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_X[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
X = t_X[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_Y[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
Y = t_Y[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_Z[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
Z = t_Z[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_0[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
train_0 = t_0[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_1[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
train_1 = t_1[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(90):
        if t_2[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
train_2 = t_2[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(90):
        if t_3[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
train_3 = t_3[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(90):
        if t_4[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
train_4 = t_4[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(90):
        if t_5[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
train_5 = t_5[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(90):
        if t_6[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
train_6 = t_6[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_7[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
train_7 = t_7[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

list_A = []
list_B = []
for i in range(180):
    for j in range(100):
        if t_8[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
train_8 = t_8[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)] 

list_A = []
list_B = []
for i in range(180):
    for j in range(90):
        if t_9[i][j] == 0:
            list_A.append(i)
            list_B.append(j)
train_9 = t_9[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("char/A.jpg",A)
cv2.imwrite("char/B.jpg",B)
cv2.imwrite("char/C.jpg",C)
cv2.imwrite("char/D.jpg",D)
cv2.imwrite("char/E.jpg",E)
cv2.imwrite("char/F.jpg",F)
cv2.imwrite("char/G.jpg",G)
cv2.imwrite("char/H.jpg",H)
cv2.imwrite("char/I.jpg",I)
cv2.imwrite("char/J.jpg",J)
cv2.imwrite("char/K.jpg",K)
cv2.imwrite("char/L.jpg",L)
cv2.imwrite("char/M.jpg",M)
cv2.imwrite("char/N.jpg",N)
cv2.imwrite("char/O.jpg",O)
cv2.imwrite("char/P.jpg",P)
cv2.imwrite("char/Q.jpg",Q)
cv2.imwrite("char/R.jpg",R)
cv2.imwrite("char/S.jpg",S)
cv2.imwrite("char/T.jpg",T)
cv2.imwrite("char/U.jpg",U)
cv2.imwrite("char/V.jpg",V)
cv2.imwrite("char/W.jpg",W)
cv2.imwrite("char/X.jpg",X)
cv2.imwrite("char/Y.jpg",Y)
cv2.imwrite("char/Z.jpg",Z)
cv2.imwrite("char/0.jpg",train_0)
cv2.imwrite("char/1.jpg",train_1)
cv2.imwrite("char/2.jpg",train_2)
cv2.imwrite("char/3.jpg",train_3)
cv2.imwrite("char/4.jpg",train_4)
cv2.imwrite("char/5.jpg",train_5)
cv2.imwrite("char/6.jpg",train_6)
cv2.imwrite("char/7.jpg",train_7)
cv2.imwrite("char/8.jpg",train_8)
cv2.imwrite("char/9.jpg",train_9)




Q1 = 0
Q2 = 0
Q3 = 0
Q4 = 0 
Qn = 0
h,w = A.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i+1][j] == 255 and A[i][j+1] == 255 and A[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i+1][j] == 255 and A[i][j-1] == 255 and A[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i-1][j] == 255 and A[i][j-1] == 255 and A[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i-1][j] == 255 and A[i][j+1] == 255 and A[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i+1][j] == 255 and A[i][j+1] != 255 and A[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i+1][j] != 255 and A[i][j-1] == 255 and A[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i-1][j] == 255 and A[i][j-1] != 255 and A[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i-1][j] != 255 and A[i][j+1] == 255 and A[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i+1][j] == 255 and A[i][j+1] != 255 and A[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i+1][j] != 255 and A[i][j-1] == 255 and A[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i-1][j] == 255 and A[i][j-1] != 255 and A[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i-1][j] != 255 and A[i][j+1] == 255 and A[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i-1][j] != 255 and A[i][j+1] != 255 and A[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i+1][j] == 255 and A[i][j+1] == 255 and A[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if A[i][j] != 255 and A[i-1][j] == 255 and A[i][j+1] == 255 and A[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_A = Area/(55*59)            

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = B.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i+1][j] == 255 and B[i][j+1] == 255 and B[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i+1][j] == 255 and B[i][j-1] == 255 and B[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i-1][j] == 255 and B[i][j-1] == 255 and B[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i-1][j] == 255 and B[i][j+1] == 255 and B[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i+1][j] == 255 and B[i][j+1] != 255 and B[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i+1][j] != 255 and B[i][j-1] == 255 and B[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i-1][j] == 255 and B[i][j-1] != 255 and B[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i-1][j] != 255 and B[i][j+1] == 255 and B[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i+1][j] == 255 and B[i][j+1] != 255 and B[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i+1][j] != 255 and B[i][j-1] == 255 and B[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i-1][j] == 255 and B[i][j-1] != 255 and B[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i-1][j] != 255 and B[i][j+1] == 255 and B[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i-1][j] != 255 and B[i][j+1] != 255 and B[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i+1][j] == 255 and B[i][j+1] == 255 and B[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if B[i][j] != 255 and B[i-1][j] == 255 and B[i][j+1] == 255 and B[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_B = Area/(44*59)
                 
Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = C.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i+1][j] == 255 and C[i][j+1] == 255 and C[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i+1][j] == 255 and C[i][j-1] == 255 and C[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i-1][j] == 255 and C[i][j-1] == 255 and C[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i-1][j] == 255 and C[i][j+1] == 255 and C[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i+1][j] == 255 and C[i][j+1] != 255 and C[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i+1][j] != 255 and C[i][j-1] == 255 and C[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i-1][j] == 255 and C[i][j-1] != 255 and C[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i-1][j] != 255 and C[i][j+1] == 255 and C[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i+1][j] == 255 and C[i][j+1] != 255 and C[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i+1][j] != 255 and C[i][j-1] == 255 and C[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i-1][j] == 255 and C[i][j-1] != 255 and C[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i-1][j] != 255 and C[i][j+1] == 255 and C[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i-1][j] != 255 and C[i][j+1] != 255 and C[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i+1][j] == 255 and C[i][j+1] == 255 and C[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if C[i][j] != 255 and C[i-1][j] == 255 and C[i][j+1] == 255 and C[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_C = Area/(51*61)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = D.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i+1][j] == 255 and D[i][j+1] == 255 and D[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i+1][j] == 255 and D[i][j-1] == 255 and D[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i-1][j] == 255 and D[i][j-1] == 255 and D[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i-1][j] == 255 and D[i][j+1] == 255 and D[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i+1][j] == 255 and D[i][j+1] != 255 and D[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i+1][j] != 255 and D[i][j-1] == 255 and D[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i-1][j] == 255 and D[i][j-1] != 255 and D[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i-1][j] != 255 and D[i][j+1] == 255 and D[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i+1][j] == 255 and D[i][j+1] != 255 and D[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i+1][j] != 255 and D[i][j-1] == 255 and D[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i-1][j] == 255 and D[i][j-1] != 255 and D[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i-1][j] != 255 and D[i][j+1] == 255 and D[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i-1][j] != 255 and D[i][j+1] != 255 and D[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i+1][j] == 255 and D[i][j+1] == 255 and D[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if D[i][j] != 255 and D[i-1][j] == 255 and D[i][j+1] == 255 and D[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_D = Area/(48*59)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = E.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i+1][j] == 255 and E[i][j+1] == 255 and E[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i+1][j] == 255 and E[i][j-1] == 255 and E[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i-1][j] == 255 and E[i][j-1] == 255 and E[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i-1][j] == 255 and E[i][j+1] == 255 and E[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i+1][j] == 255 and E[i][j+1] != 255 and E[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i+1][j] != 255 and E[i][j-1] == 255 and E[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i-1][j] == 255 and E[i][j-1] != 255 and E[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i-1][j] != 255 and E[i][j+1] == 255 and E[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i+1][j] == 255 and E[i][j+1] != 255 and E[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i+1][j] != 255 and E[i][j-1] == 255 and E[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i-1][j] == 255 and E[i][j-1] != 255 and E[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i-1][j] != 255 and E[i][j+1] == 255 and E[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i-1][j] != 255 and E[i][j+1] != 255 and E[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i+1][j] == 255 and E[i][j+1] == 255 and E[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if E[i][j] != 255 and E[i-1][j] == 255 and E[i][j+1] == 255 and E[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_E = Area/(43*59)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = F.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i+1][j] == 255 and F[i][j+1] == 255 and F[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i+1][j] == 255 and F[i][j-1] == 255 and F[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i-1][j] == 255 and F[i][j-1] == 255 and F[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i-1][j] == 255 and F[i][j+1] == 255 and F[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i+1][j] == 255 and F[i][j+1] != 255 and F[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i+1][j] != 255 and F[i][j-1] == 255 and F[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i-1][j] == 255 and F[i][j-1] != 255 and F[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i-1][j] != 255 and F[i][j+1] == 255 and F[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i+1][j] == 255 and F[i][j+1] != 255 and F[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i+1][j] != 255 and F[i][j-1] == 255 and F[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i-1][j] == 255 and F[i][j-1] != 255 and F[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i-1][j] != 255 and F[i][j+1] == 255 and F[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i-1][j] != 255 and F[i][j+1] != 255 and F[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i+1][j] == 255 and F[i][j+1] == 255 and F[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if F[i][j] != 255 and F[i-1][j] == 255 and F[i][j+1] == 255 and F[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_F = Area/(40*59)


Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i+1][j] == 255 and G[i][j+1] == 255 and G[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i+1][j] == 255 and G[i][j-1] == 255 and G[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i-1][j] == 255 and G[i][j-1] == 255 and G[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i-1][j] == 255 and G[i][j+1] == 255 and G[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i+1][j] == 255 and G[i][j+1] != 255 and G[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i+1][j] != 255 and G[i][j-1] == 255 and G[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i-1][j] == 255 and G[i][j-1] != 255 and G[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i-1][j] != 255 and G[i][j+1] == 255 and G[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i+1][j] == 255 and G[i][j+1] != 255 and G[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i+1][j] != 255 and G[i][j-1] == 255 and G[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i-1][j] == 255 and G[i][j-1] != 255 and G[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i-1][j] != 255 and G[i][j+1] == 255 and G[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i-1][j] != 255 and G[i][j+1] != 255 and G[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i+1][j] == 255 and G[i][j+1] == 255 and G[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G[i][j] != 255 and G[i-1][j] == 255 and G[i][j+1] == 255 and G[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_G = Area/(55*61)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = H.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i+1][j] == 255 and H[i][j+1] == 255 and H[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i+1][j] == 255 and H[i][j-1] == 255 and H[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i-1][j] == 255 and H[i][j-1] == 255 and H[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i-1][j] == 255 and H[i][j+1] == 255 and H[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i+1][j] == 255 and H[i][j+1] != 255 and H[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i+1][j] != 255 and H[i][j-1] == 255 and H[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i-1][j] == 255 and H[i][j-1] != 255 and H[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i-1][j] != 255 and H[i][j+1] == 255 and H[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i+1][j] == 255 and H[i][j+1] != 255 and H[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i+1][j] != 255 and H[i][j-1] == 255 and H[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i-1][j] == 255 and H[i][j-1] != 255 and H[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i-1][j] != 255 and H[i][j+1] == 255 and H[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i-1][j] != 255 and H[i][j+1] != 255 and H[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i+1][j] == 255 and H[i][j+1] == 255 and H[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if H[i][j] != 255 and H[i-1][j] == 255 and H[i][j+1] == 255 and H[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_H = Area/(46*59)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = I.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i+1][j] == 255 and I[i][j+1] == 255 and I[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i+1][j] == 255 and I[i][j-1] == 255 and I[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i-1][j] == 255 and I[i][j-1] == 255 and I[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i-1][j] == 255 and I[i][j+1] == 255 and I[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i+1][j] == 255 and I[i][j+1] != 255 and I[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i+1][j] != 255 and I[i][j-1] == 255 and I[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i-1][j] == 255 and I[i][j-1] != 255 and I[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i-1][j] != 255 and I[i][j+1] == 255 and I[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i+1][j] == 255 and I[i][j+1] != 255 and I[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i+1][j] != 255 and I[i][j-1] == 255 and I[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i-1][j] == 255 and I[i][j-1] != 255 and I[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i-1][j] != 255 and I[i][j+1] == 255 and I[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i-1][j] != 255 and I[i][j+1] != 255 and I[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i+1][j] == 255 and I[i][j+1] == 255 and I[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if I[i][j] != 255 and I[i-1][j] == 255 and I[i][j+1] == 255 and I[i-1][j+1] != 255:
            Qn += 1 

Area = 0
for i in range(h):
    for j in range(w):
        if I[i][j] != 255:
            Area += 1

# Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_I = Area/(7*59)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = J.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i+1][j] == 255 and J[i][j+1] == 255 and J[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i+1][j] == 255 and J[i][j-1] == 255 and J[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i-1][j] == 255 and J[i][j-1] == 255 and J[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i-1][j] == 255 and J[i][j+1] == 255 and J[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i+1][j] == 255 and J[i][j+1] != 255 and J[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i+1][j] != 255 and J[i][j-1] == 255 and J[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i-1][j] == 255 and J[i][j-1] != 255 and J[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i-1][j] != 255 and J[i][j+1] == 255 and J[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i+1][j] == 255 and J[i][j+1] != 255 and J[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i+1][j] != 255 and J[i][j-1] == 255 and J[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i-1][j] == 255 and J[i][j-1] != 255 and J[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i-1][j] != 255 and J[i][j+1] == 255 and J[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i-1][j] != 255 and J[i][j+1] != 255 and J[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i+1][j] == 255 and J[i][j+1] == 255 and J[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if J[i][j] != 255 and J[i-1][j] == 255 and J[i][j+1] == 255 and J[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_J = Area/(31*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = K.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i+1][j] == 255 and K[i][j+1] == 255 and K[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i+1][j] == 255 and K[i][j-1] == 255 and K[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i-1][j] == 255 and K[i][j-1] == 255 and K[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i-1][j] == 255 and K[i][j+1] == 255 and K[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i+1][j] == 255 and K[i][j+1] != 255 and K[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i+1][j] != 255 and K[i][j-1] == 255 and K[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i-1][j] == 255 and K[i][j-1] != 255 and K[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i-1][j] != 255 and K[i][j+1] == 255 and K[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i+1][j] == 255 and K[i][j+1] != 255 and K[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i+1][j] != 255 and K[i][j-1] == 255 and K[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i-1][j] == 255 and K[i][j-1] != 255 and K[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i-1][j] != 255 and K[i][j+1] == 255 and K[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i-1][j] != 255 and K[i][j+1] != 255 and K[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i+1][j] == 255 and K[i][j+1] == 255 and K[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if K[i][j] != 255 and K[i-1][j] == 255 and K[i][j+1] == 255 and K[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_K = Area/(47*59)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = L.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i+1][j] == 255 and L[i][j+1] == 255 and L[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i+1][j] == 255 and L[i][j-1] == 255 and L[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i-1][j] == 255 and L[i][j-1] == 255 and L[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i-1][j] == 255 and L[i][j+1] == 255 and L[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i+1][j] == 255 and L[i][j+1] != 255 and L[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i+1][j] != 255 and L[i][j-1] == 255 and L[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i-1][j] == 255 and L[i][j-1] != 255 and L[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i-1][j] != 255 and L[i][j+1] == 255 and L[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i+1][j] == 255 and L[i][j+1] != 255 and L[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i+1][j] != 255 and L[i][j-1] == 255 and L[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i-1][j] == 255 and L[i][j-1] != 255 and L[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i-1][j] != 255 and L[i][j+1] == 255 and L[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i-1][j] != 255 and L[i][j+1] != 255 and L[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i+1][j] == 255 and L[i][j+1] == 255 and L[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if L[i][j] != 255 and L[i-1][j] == 255 and L[i][j+1] == 255 and L[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_L = Area/(36*59)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = M.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i+1][j] == 255 and M[i][j+1] == 255 and M[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i+1][j] == 255 and M[i][j-1] == 255 and M[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i-1][j] == 255 and M[i][j-1] == 255 and M[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i-1][j] == 255 and M[i][j+1] == 255 and M[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i+1][j] == 255 and M[i][j+1] != 255 and M[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i+1][j] != 255 and M[i][j-1] == 255 and M[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i-1][j] == 255 and M[i][j-1] != 255 and M[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i-1][j] != 255 and M[i][j+1] == 255 and M[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i+1][j] == 255 and M[i][j+1] != 255 and M[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i+1][j] != 255 and M[i][j-1] == 255 and M[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i-1][j] == 255 and M[i][j-1] != 255 and M[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i-1][j] != 255 and M[i][j+1] == 255 and M[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i-1][j] != 255 and M[i][j+1] != 255 and M[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i+1][j] == 255 and M[i][j+1] == 255 and M[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if M[i][j] != 255 and M[i-1][j] == 255 and M[i][j+1] == 255 and M[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_M = Area/(56*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = N.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i+1][j] == 255 and N[i][j+1] == 255 and N[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i+1][j] == 255 and N[i][j-1] == 255 and N[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i-1][j] == 255 and N[i][j-1] == 255 and N[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i-1][j] == 255 and N[i][j+1] == 255 and N[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i+1][j] == 255 and N[i][j+1] != 255 and N[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i+1][j] != 255 and N[i][j-1] == 255 and N[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i-1][j] == 255 and N[i][j-1] != 255 and N[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i-1][j] != 255 and N[i][j+1] == 255 and N[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i+1][j] == 255 and N[i][j+1] != 255 and N[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i+1][j] != 255 and N[i][j-1] == 255 and N[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i-1][j] == 255 and N[i][j-1] != 255 and N[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i-1][j] != 255 and N[i][j+1] == 255 and N[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i-1][j] != 255 and N[i][j+1] != 255 and N[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i+1][j] == 255 and N[i][j+1] == 255 and N[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if N[i][j] != 255 and N[i-1][j] == 255 and N[i][j+1] == 255 and N[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_N = Area/(47*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = O.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i+1][j] == 255 and O[i][j+1] == 255 and O[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i+1][j] == 255 and O[i][j-1] == 255 and O[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i-1][j] == 255 and O[i][j-1] == 255 and O[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i-1][j] == 255 and O[i][j+1] == 255 and O[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i+1][j] == 255 and O[i][j+1] != 255 and O[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i+1][j] != 255 and O[i][j-1] == 255 and O[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i-1][j] == 255 and O[i][j-1] != 255 and O[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i-1][j] != 255 and O[i][j+1] == 255 and O[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i+1][j] == 255 and O[i][j+1] != 255 and O[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i+1][j] != 255 and O[i][j-1] == 255 and O[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i-1][j] == 255 and O[i][j-1] != 255 and O[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i-1][j] != 255 and O[i][j+1] == 255 and O[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i-1][j] != 255 and O[i][j+1] != 255 and O[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i+1][j] == 255 and O[i][j+1] == 255 and O[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if O[i][j] != 255 and O[i-1][j] == 255 and O[i][j+1] == 255 and O[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_O = Area/(57*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = P.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i+1][j] == 255 and P[i][j+1] == 255 and P[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i+1][j] == 255 and P[i][j-1] == 255 and P[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i-1][j] == 255 and P[i][j-1] == 255 and P[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i-1][j] == 255 and P[i][j+1] == 255 and P[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i+1][j] == 255 and P[i][j+1] != 255 and P[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i+1][j] != 255 and P[i][j-1] == 255 and P[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i-1][j] == 255 and P[i][j-1] != 255 and P[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i-1][j] != 255 and P[i][j+1] == 255 and P[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i+1][j] == 255 and P[i][j+1] != 255 and P[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i+1][j] != 255 and P[i][j-1] == 255 and P[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i-1][j] == 255 and P[i][j-1] != 255 and P[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i-1][j] != 255 and P[i][j+1] == 255 and P[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i-1][j] != 255 and P[i][j+1] != 255 and P[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i+1][j] == 255 and P[i][j+1] == 255 and P[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if P[i][j] != 255 and P[i-1][j] == 255 and P[i][j+1] == 255 and P[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_P = Area/(44*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = Q.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i+1][j] == 255 and Q[i][j+1] == 255 and Q[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i+1][j] == 255 and Q[i][j-1] == 255 and Q[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i-1][j] == 255 and Q[i][j-1] == 255 and Q[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i-1][j] == 255 and Q[i][j+1] == 255 and Q[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i+1][j] == 255 and Q[i][j+1] != 255 and Q[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i+1][j] != 255 and Q[i][j-1] == 255 and Q[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i-1][j] == 255 and Q[i][j-1] != 255 and Q[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i-1][j] != 255 and Q[i][j+1] == 255 and Q[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i+1][j] == 255 and Q[i][j+1] != 255 and Q[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i+1][j] != 255 and Q[i][j-1] == 255 and Q[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i-1][j] == 255 and Q[i][j-1] != 255 and Q[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i-1][j] != 255 and Q[i][j+1] == 255 and Q[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i-1][j] != 255 and Q[i][j+1] != 255 and Q[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i+1][j] == 255 and Q[i][j+1] == 255 and Q[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if Q[i][j] != 255 and Q[i-1][j] == 255 and Q[i][j+1] == 255 and Q[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_Q = Area/(56*64)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = R.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i+1][j] == 255 and R[i][j+1] == 255 and R[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i+1][j] == 255 and R[i][j-1] == 255 and R[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i-1][j] == 255 and R[i][j-1] == 255 and R[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i-1][j] == 255 and R[i][j+1] == 255 and R[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i+1][j] == 255 and R[i][j+1] != 255 and R[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i+1][j] != 255 and R[i][j-1] == 255 and R[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i-1][j] == 255 and R[i][j-1] != 255 and R[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i-1][j] != 255 and R[i][j+1] == 255 and R[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i+1][j] == 255 and R[i][j+1] != 255 and R[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i+1][j] != 255 and R[i][j-1] == 255 and R[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i-1][j] == 255 and R[i][j-1] != 255 and R[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i-1][j] != 255 and R[i][j+1] == 255 and R[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i-1][j] != 255 and R[i][j+1] != 255 and R[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i+1][j] == 255 and R[i][j+1] == 255 and R[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if R[i][j] != 255 and R[i-1][j] == 255 and R[i][j+1] == 255 and R[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_R = Area/(50*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = S.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i+1][j] == 255 and S[i][j+1] == 255 and S[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i+1][j] == 255 and S[i][j-1] == 255 and S[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i-1][j] == 255 and S[i][j-1] == 255 and S[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i-1][j] == 255 and S[i][j+1] == 255 and S[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i+1][j] == 255 and S[i][j+1] != 255 and S[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i+1][j] != 255 and S[i][j-1] == 255 and S[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i-1][j] == 255 and S[i][j-1] != 255 and S[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i-1][j] != 255 and S[i][j+1] == 255 and S[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i+1][j] == 255 and S[i][j+1] != 255 and S[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i+1][j] != 255 and S[i][j-1] == 255 and S[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i-1][j] == 255 and S[i][j-1] != 255 and S[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i-1][j] != 255 and S[i][j+1] == 255 and S[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i-1][j] != 255 and S[i][j+1] != 255 and S[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i+1][j] == 255 and S[i][j+1] == 255 and S[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if S[i][j] != 255 and S[i-1][j] == 255 and S[i][j+1] == 255 and S[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_S = Area/(46*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = T.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i+1][j] == 255 and T[i][j+1] == 255 and T[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i+1][j] == 255 and T[i][j-1] == 255 and T[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i-1][j] == 255 and T[i][j-1] == 255 and T[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i-1][j] == 255 and T[i][j+1] == 255 and T[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i+1][j] == 255 and T[i][j+1] != 255 and T[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i+1][j] != 255 and T[i][j-1] == 255 and T[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i-1][j] == 255 and T[i][j-1] != 255 and T[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i-1][j] != 255 and T[i][j+1] == 255 and T[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i+1][j] == 255 and T[i][j+1] != 255 and T[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i+1][j] != 255 and T[i][j-1] == 255 and T[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i-1][j] == 255 and T[i][j-1] != 255 and T[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i-1][j] != 255 and T[i][j+1] == 255 and T[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i-1][j] != 255 and T[i][j+1] != 255 and T[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i+1][j] == 255 and T[i][j+1] == 255 and T[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if T[i][j] != 255 and T[i-1][j] == 255 and T[i][j+1] == 255 and T[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_T = Area/(47*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = U.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i+1][j] == 255 and U[i][j+1] == 255 and U[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i+1][j] == 255 and U[i][j-1] == 255 and U[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i-1][j] == 255 and U[i][j-1] == 255 and U[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i-1][j] == 255 and U[i][j+1] == 255 and U[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i+1][j] == 255 and U[i][j+1] != 255 and U[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i+1][j] != 255 and U[i][j-1] == 255 and U[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i-1][j] == 255 and U[i][j-1] != 255 and U[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i-1][j] != 255 and U[i][j+1] == 255 and U[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i+1][j] == 255 and U[i][j+1] != 255 and U[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i+1][j] != 255 and U[i][j-1] == 255 and U[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i-1][j] == 255 and U[i][j-1] != 255 and U[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i-1][j] != 255 and U[i][j+1] == 255 and U[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i-1][j] != 255 and U[i][j+1] != 255 and U[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i+1][j] == 255 and U[i][j+1] == 255 and U[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if U[i][j] != 255 and U[i-1][j] == 255 and U[i][j+1] == 255 and U[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_U = Area/(47*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = V.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i+1][j] == 255 and V[i][j+1] == 255 and V[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i+1][j] == 255 and V[i][j-1] == 255 and V[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i-1][j] == 255 and V[i][j-1] == 255 and V[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i-1][j] == 255 and V[i][j+1] == 255 and V[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i+1][j] == 255 and V[i][j+1] != 255 and V[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i+1][j] != 255 and V[i][j-1] == 255 and V[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i-1][j] == 255 and V[i][j-1] != 255 and V[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i-1][j] != 255 and V[i][j+1] == 255 and V[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i+1][j] == 255 and V[i][j+1] != 255 and V[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i+1][j] != 255 and V[i][j-1] == 255 and V[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i-1][j] == 255 and V[i][j-1] != 255 and V[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i-1][j] != 255 and V[i][j+1] == 255 and V[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i-1][j] != 255 and V[i][j+1] != 255 and V[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i+1][j] == 255 and V[i][j+1] == 255 and V[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if V[i][j] != 255 and V[i-1][j] == 255 and V[i][j+1] == 255 and V[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_V = Area/(55*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = W.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i+1][j] == 255 and W[i][j+1] == 255 and W[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i+1][j] == 255 and W[i][j-1] == 255 and W[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i-1][j] == 255 and W[i][j-1] == 255 and W[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i-1][j] == 255 and W[i][j+1] == 255 and W[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i+1][j] == 255 and W[i][j+1] != 255 and W[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i+1][j] != 255 and W[i][j-1] == 255 and W[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i-1][j] == 255 and W[i][j-1] != 255 and W[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i-1][j] != 255 and W[i][j+1] == 255 and W[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i+1][j] == 255 and W[i][j+1] != 255 and W[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i+1][j] != 255 and W[i][j-1] == 255 and W[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i-1][j] == 255 and W[i][j-1] != 255 and W[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i-1][j] != 255 and W[i][j+1] == 255 and W[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i-1][j] != 255 and W[i][j+1] != 255 and W[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i+1][j] == 255 and W[i][j+1] == 255 and W[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if W[i][j] != 255 and W[i-1][j] == 255 and W[i][j+1] == 255 and W[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_W = Area/(78*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = X.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i+1][j] == 255 and X[i][j+1] == 255 and X[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i+1][j] == 255 and X[i][j-1] == 255 and X[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i-1][j] == 255 and X[i][j-1] == 255 and X[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i-1][j] == 255 and X[i][j+1] == 255 and X[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i+1][j] == 255 and X[i][j+1] != 255 and X[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i+1][j] != 255 and X[i][j-1] == 255 and X[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i-1][j] == 255 and X[i][j-1] != 255 and X[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i-1][j] != 255 and X[i][j+1] == 255 and X[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i+1][j] == 255 and X[i][j+1] != 255 and X[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i+1][j] != 255 and X[i][j-1] == 255 and X[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i-1][j] == 255 and X[i][j-1] != 255 and X[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i-1][j] != 255 and X[i][j+1] == 255 and X[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i-1][j] != 255 and X[i][j+1] != 255 and X[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i+1][j] == 255 and X[i][j+1] == 255 and X[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if X[i][j] != 255 and X[i-1][j] == 255 and X[i][j+1] == 255 and X[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_X = Area/(51*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = Y.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i+1][j] == 255 and Y[i][j+1] == 255 and Y[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i+1][j] == 255 and Y[i][j-1] == 255 and Y[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i-1][j] == 255 and Y[i][j-1] == 255 and Y[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i-1][j] == 255 and Y[i][j+1] == 255 and Y[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i+1][j] == 255 and Y[i][j+1] != 255 and Y[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i+1][j] != 255 and Y[i][j-1] == 255 and Y[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i-1][j] == 255 and Y[i][j-1] != 255 and Y[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i-1][j] != 255 and Y[i][j+1] == 255 and Y[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i+1][j] == 255 and Y[i][j+1] != 255 and Y[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i+1][j] != 255 and Y[i][j-1] == 255 and Y[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i-1][j] == 255 and Y[i][j-1] != 255 and Y[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i-1][j] != 255 and Y[i][j+1] == 255 and Y[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i-1][j] != 255 and Y[i][j+1] != 255 and Y[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i+1][j] == 255 and Y[i][j+1] == 255 and Y[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if Y[i][j] != 255 and Y[i-1][j] == 255 and Y[i][j+1] == 255 and Y[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_Y = Area/(51*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = Z.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i+1][j] == 255 and Z[i][j+1] == 255 and Z[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i+1][j] == 255 and Z[i][j-1] == 255 and Z[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i-1][j] == 255 and Z[i][j-1] == 255 and Z[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i-1][j] == 255 and Z[i][j+1] == 255 and Z[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i+1][j] == 255 and Z[i][j+1] != 255 and Z[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i+1][j] != 255 and Z[i][j-1] == 255 and Z[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i-1][j] == 255 and Z[i][j-1] != 255 and Z[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i-1][j] != 255 and Z[i][j+1] == 255 and Z[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i+1][j] == 255 and Z[i][j+1] != 255 and Z[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i+1][j] != 255 and Z[i][j-1] == 255 and Z[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i-1][j] == 255 and Z[i][j-1] != 255 and Z[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i-1][j] != 255 and Z[i][j+1] == 255 and Z[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i-1][j] != 255 and Z[i][j+1] != 255 and Z[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i+1][j] == 255 and Z[i][j+1] == 255 and Z[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if Z[i][j] != 255 and Z[i-1][j] == 255 and Z[i][j+1] == 255 and Z[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_Z = Area/(46*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = train_8.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i+1][j] == 255 and train_8[i][j+1] == 255 and train_8[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i+1][j] == 255 and train_8[i][j-1] == 255 and train_8[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i-1][j] == 255 and train_8[i][j-1] == 255 and train_8[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i-1][j] == 255 and train_8[i][j+1] == 255 and train_8[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i+1][j] == 255 and train_8[i][j+1] != 255 and train_8[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i+1][j] != 255 and train_8[i][j-1] == 255 and train_8[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i-1][j] == 255 and train_8[i][j-1] != 255 and train_8[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i-1][j] != 255 and train_8[i][j+1] == 255 and train_8[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i+1][j] == 255 and train_8[i][j+1] != 255 and train_8[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i+1][j] != 255 and train_8[i][j-1] == 255 and train_8[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i-1][j] == 255 and train_8[i][j-1] != 255 and train_8[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i-1][j] != 255 and train_8[i][j+1] == 255 and train_8[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i-1][j] != 255 and train_8[i][j+1] != 255 and train_8[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i+1][j] == 255 and train_8[i][j+1] == 255 and train_8[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_8[i][j] != 255 and train_8[i-1][j] == 255 and train_8[i][j+1] == 255 and train_8[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_8 = Area/(38*80) 

print(Area_A,Area_P)


