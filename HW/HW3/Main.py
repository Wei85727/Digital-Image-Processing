import numpy as np
import cv2
import matplotlib.pyplot as pyplot

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
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
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
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
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
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
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
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_8 = Area/(38*80) 

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = train_0.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i+1][j] == 255 and train_0[i][j+1] == 255 and train_0[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i+1][j] == 255 and train_0[i][j-1] == 255 and train_0[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i-1][j] == 255 and train_0[i][j-1] == 255 and train_0[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i-1][j] == 255 and train_0[i][j+1] == 255 and train_0[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i+1][j] == 255 and train_0[i][j+1] != 255 and train_0[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i+1][j] != 255 and train_0[i][j-1] == 255 and train_0[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i-1][j] == 255 and train_0[i][j-1] != 255 and train_0[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i-1][j] != 255 and train_0[i][j+1] == 255 and train_0[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i+1][j] == 255 and train_0[i][j+1] != 255 and train_0[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i+1][j] != 255 and train_0[i][j-1] == 255 and train_0[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i-1][j] == 255 and train_0[i][j-1] != 255 and train_0[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i-1][j] != 255 and train_0[i][j+1] == 255 and train_0[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i-1][j] != 255 and train_0[i][j+1] != 255 and train_0[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i+1][j] == 255 and train_0[i][j+1] == 255 and train_0[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_0[i][j] != 255 and train_0[i-1][j] == 255 and train_0[i][j+1] == 255 and train_0[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_0 = Area/(38*60)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = train_5.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i+1][j] == 255 and train_5[i][j+1] == 255 and train_5[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i+1][j] == 255 and train_5[i][j-1] == 255 and train_5[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i-1][j] == 255 and train_5[i][j-1] == 255 and train_5[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i-1][j] == 255 and train_5[i][j+1] == 255 and train_5[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i+1][j] == 255 and train_5[i][j+1] != 255 and train_5[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i+1][j] != 255 and train_5[i][j-1] == 255 and train_5[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i-1][j] == 255 and train_5[i][j-1] != 255 and train_5[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i-1][j] != 255 and train_5[i][j+1] == 255 and train_5[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i+1][j] == 255 and train_5[i][j+1] != 255 and train_5[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i+1][j] != 255 and train_5[i][j-1] == 255 and train_5[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i-1][j] == 255 and train_5[i][j-1] != 255 and train_5[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i-1][j] != 255 and train_5[i][j+1] == 255 and train_5[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i-1][j] != 255 and train_5[i][j+1] != 255 and train_5[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i+1][j] == 255 and train_5[i][j+1] == 255 and train_5[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if train_5[i][j] != 255 and train_5[i-1][j] == 255 and train_5[i][j+1] == 255 and train_5[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ)
Area_5 = Area/(38*59)

# sample2.jpg
sam = cv2.imread("sample2.jpg")
h,w,c = sam.shape
sam2 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        sam2[i][j] = sam[i][j][0]

for i in range(h):
    for j in range(w):
        if sam2[i][j] > 80:
            sam2[i][j] = 255

t_1 = sam2[:,:60]
t_2 = sam2[:,60:120]
t_3 = sam2[:,120:180]
t_4 = sam2[:,195:255]
t_5 = sam2[:,255:315]
t_6 = sam2[:,315:375]
t_7 = sam2[:,375:435]

cv2.imwrite("sam2.jpg",sam2) 
cv2.imwrite("t_1.jpg",t_1)
cv2.imwrite("t_2.jpg",t_2)
cv2.imwrite("t_3.jpg",t_3) 
cv2.imwrite("t_4.jpg",t_4)
cv2.imwrite("t_5.jpg",t_5)
cv2.imwrite("t_6.jpg",t_6)                   
cv2.imwrite("t_7.jpg",t_7)

list_A = []
list_B = []
for i in range(178):
    for j in range(60):
        if t_1[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_1 = t_1[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("G_1.jpg",G_1)

list_A = []
list_B = []
for i in range(178):
    for j in range(60):
        if t_2[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_2 = t_2[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("G_2.jpg",G_2)

list_A = []
list_B = []
for i in range(178):
    for j in range(60):
        if t_3[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_3 = t_3[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("G_3.jpg",G_3)

list_A = []
list_B = []
for i in range(178):
    for j in range(60):
        if t_4[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_4 = t_4[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("G_4.jpg",G_4)

list_A = []
list_B = []
for i in range(178):
    for j in range(60):
        if t_5[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_5 = t_4[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("G_5.jpg",G_5)

list_A = []
list_B = []
for i in range(178):
    for j in range(60):
        if t_6[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_6 = t_4[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("G_6.jpg",G_6)

list_A = []
list_B = []
for i in range(178):
    for j in range(60):
        if t_7[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_7 = t_4[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("G_7.jpg",G_7)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_1.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j+1] == 255 and G_1[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j-1] == 255 and G_1[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j-1] == 255 and G_1[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j+1] == 255 and G_1[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j+1] != 255 and G_1[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] != 255 and G_1[i][j-1] == 255 and G_1[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j-1] != 255 and G_1[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] != 255 and G_1[i][j+1] == 255 and G_1[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j+1] != 255 and G_1[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] != 255 and G_1[i][j-1] == 255 and G_1[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j-1] != 255 and G_1[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] != 255 and G_1[i][j+1] == 255 and G_1[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] != 255 and G_1[i][j+1] != 255 and G_1[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j+1] == 255 and G_1[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j+1] == 255 and G_1[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)

G_1_vsym1 = G_1[:52,:]
G_1_vsym2 = G_1[52:,:]
Sym = np.sum(G_1_vsym1!=255)/np.sum(G_1_vsym2!=255)
G_1_fea = [Hole,Sym,Area/(h*w)]
G_1_sp1 = G_1[:,:23]
G_1_sp2 = G_1[:,23:]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_2.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j+1] == 255 and G_2[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j-1] == 255 and G_2[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j-1] == 255 and G_2[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j+1] == 255 and G_2[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j+1] != 255 and G_2[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] != 255 and G_2[i][j-1] == 255 and G_2[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j-1] != 255 and G_2[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] != 255 and G_2[i][j+1] == 255 and G_2[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j+1] != 255 and G_2[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] != 255 and G_2[i][j-1] == 255 and G_2[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j-1] != 255 and G_2[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] != 255 and G_2[i][j+1] == 255 and G_2[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] != 255 and G_2[i][j+1] != 255 and G_2[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j+1] == 255 and G_2[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j+1] == 255 and G_2[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)

G_2_vsym1 = G_2[:52,:]
G_2_vsym2 = G_2[52:,:]
Sym = np.sum(G_2_vsym1!=255)/np.sum(G_2_vsym2!=255)
G_2_fea = [Hole,Sym,Area/(h*w)]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0

h,w = G_3.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j+1] == 255 and G_3[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j-1] == 255 and G_3[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j-1] == 255 and G_3[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j+1] == 255 and G_3[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j+1] != 255 and G_3[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] != 255 and G_3[i][j-1] == 255 and G_3[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j-1] != 255 and G_3[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] != 255 and G_3[i][j+1] == 255 and G_3[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j+1] != 255 and G_3[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] != 255 and G_3[i][j-1] == 255 and G_3[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j-1] != 255 and G_3[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] != 255 and G_3[i][j+1] == 255 and G_3[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] != 255 and G_3[i][j+1] != 255 and G_3[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j+1] == 255 and G_3[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j+1] == 255 and G_3[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)

G_3_vsym1 = G_3[:52,:]
G_3_vsym2 = G_3[52:,:]
Sym = np.sum(G_3_vsym1!=255)/np.sum(G_3_vsym2!=255)
G_3_fea = [Hole,Sym,Area/(h*w)]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_4.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j+1] == 255 and G_4[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j-1] == 255 and G_4[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j-1] == 255 and G_4[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j+1] == 255 and G_4[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j+1] != 255 and G_4[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] != 255 and G_4[i][j-1] == 255 and G_4[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j-1] != 255 and G_4[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] != 255 and G_4[i][j+1] == 255 and G_4[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j+1] != 255 and G_4[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] != 255 and G_4[i][j-1] == 255 and G_4[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j-1] != 255 and G_4[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] != 255 and G_4[i][j+1] == 255 and G_4[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] != 255 and G_4[i][j+1] != 255 and G_4[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j+1] == 255 and G_4[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j+1] == 255 and G_4[i-1][j+1] != 255:
            Qn += 1 

G_4.shape = h,w
Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)

G_4_hsym1 = G_4[:52,:]
G_4_vsym2 = G_4[52:,:]
Sym = np.sum(G_4_hsym1!=255)/np.sum(G_4_vsym2!=255)
G_4_fea = [Hole,Sym,Area/(h*w)]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_5.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j+1] == 255 and G_5[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j-1] == 255 and G_5[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j-1] == 255 and G_5[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j+1] == 255 and G_5[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j+1] != 255 and G_5[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] != 255 and G_5[i][j-1] == 255 and G_5[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j-1] != 255 and G_5[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] != 255 and G_5[i][j+1] == 255 and G_5[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j+1] != 255 and G_5[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] != 255 and G_5[i][j-1] == 255 and G_5[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j-1] != 255 and G_5[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] != 255 and G_5[i][j+1] == 255 and G_5[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] != 255 and G_5[i][j+1] != 255 and G_5[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j+1] == 255 and G_5[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j+1] == 255 and G_5[i-1][j+1] != 255:
            Qn += 1 

G_5.shape = h,w
Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)


G_5_vsym1 = G_5[:52,:]
G_5_vsym2 = G_5[52:,:]
Sym = np.sum(G_5_vsym1!=255)/np.sum(G_5_vsym2!=255)
G_5_fea = [Hole,Sym,Area/(h*w)]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_6.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j+1] == 255 and G_6[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j-1] == 255 and G_6[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j-1] == 255 and G_6[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j+1] == 255 and G_6[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j+1] != 255 and G_6[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] != 255 and G_6[i][j-1] == 255 and G_6[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j-1] != 255 and G_6[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] != 255 and G_6[i][j+1] == 255 and G_6[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j+1] != 255 and G_6[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] != 255 and G_6[i][j-1] == 255 and G_6[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j-1] != 255 and G_6[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] != 255 and G_6[i][j+1] == 255 and G_6[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] != 255 and G_6[i][j+1] != 255 and G_6[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j+1] == 255 and G_6[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j+1] == 255 and G_6[i-1][j+1] != 255:
            Qn += 1 

G_6.shape = h,w
Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)


G_6_vsym1 = G_6[:52,:]
G_6_vsym2 = G_6[52:,:]
Sym = np.sum(G_6_vsym1!=255)/np.sum(G_6_vsym2!=255)
G_6_fea = [Hole,Sym,Area/(h*w)]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_7.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j+1] == 255 and G_7[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j-1] == 255 and G_7[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j-1] == 255 and G_7[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j+1] == 255 and G_7[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j+1] != 255 and G_7[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] != 255 and G_7[i][j-1] == 255 and G_7[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j-1] != 255 and G_7[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] != 255 and G_7[i][j+1] == 255 and G_7[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j+1] != 255 and G_7[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] != 255 and G_7[i][j-1] == 255 and G_7[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j-1] != 255 and G_7[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] != 255 and G_7[i][j+1] == 255 and G_7[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] != 255 and G_7[i][j+1] != 255 and G_7[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j+1] == 255 and G_7[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j+1] == 255 and G_7[i-1][j+1] != 255:
            Qn += 1 

G_7.shape = h,w
Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)


G_7_hsym1 = G_7[:52,:]
G_7_vsym2 = G_7[52:,:]
Sym = np.sum(G_7_hsym1!=255)/np.sum(G_7_vsym2!=255)
G_7_fea = [Hole,Sym,Area/(h*w)]

answer = []
h2_sym = [Area_B,Area_8]
h1_sym = [Area_D,Area_O,Area_0]
h1_nsym = [Area_A,Area_P,Area_Q,Area_R,Area_A,Area_A,Area_A]
h0_sym = [Area_C,Area_E,Area_H,Area_I,Area_K]
h0_nsym = [Area_F,Area_G,Area_L,Area_M,Area_J,Area_N,Area_S,Area_T,Area_W]

#G_1
area_list = []
if G_1_fea[0] == 2:
    if G_1_fea[1] > 0.9 and G_1_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_1_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(abs(G_1_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_1_fea[0] == 1:
    if G_1_fea[1] > 0.9 and G_1_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_1_fea[2]-h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(abs(G_1_fea[2]- h1_nsym[i]))
        if np.sum(G_1_sp1!=255)/np.sum(G_1_sp2!=255) > 0.9 and np.sum(G_1_sp1!=255)/np.sum(G_1_sp2!=255) < 1.1:
            answer.append("A")
        else:
            if area_list.index(min(area_list)) == 1:
                answer.append("P")
            elif area_list.index(min(area_list)) == 2:
                answer.append("Q")
            elif area_list.index(min(area_list)) == 3:
                answer.append("R")
            elif area_list.index(min(area_list)) == 4:
                answer.append("4")
            elif area_list.index(min(area_list)) == 5:
                answer.append("6")    
            else:
                answer.append("9")
else: 
    for i in range(14):
        area_list.append(abs(G_1_fea[2]- h0[i]))
    if area_list.index(min(area_list)) == 0:
            answer.append("C") 
    elif area_list.index(min(area_list)) == 1:
            answer.append("E")
    elif area_list.index(min(area_list)) == 2:
            answer.append("F")
    elif area_list.index(min(area_list)) == 3:
            answer.append("G")    
    elif area_list.index(min(area_list)) == 4:
            answer.append("H")
    elif area_list.index(min(area_list)) == 5:
            answer.append("I")     
    elif area_list.index(min(area_list)) == 6:
            answer.append("J")
    elif area_list.index(min(area_list)) == 7:
            answer.append("K")
    elif area_list.index(min(area_list)) == 8:
            answer.append("L")
    elif area_list.index(min(area_list)) == 9:
            answer.append("M")
    elif area_list.index(min(area_list)) == 10:
            answer.append("N")
    elif area_list.index(min(area_list)) == 11:
            answer.append("S")
    elif area_list.index(min(area_list)) == 12:
            answer.append("T") 
    else :
            answer.append("W")                       
                                                                                     


#G_2
area_list = []
if G_2_fea[0] == 2:
    if G_2_fea[1] > 0.9 and G_2_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_2_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(abs(G_2_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_2_fea[0] == 1:
    if G_2_fea[1] > 0.9 and G_2_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_2_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_2_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_2_vsym1!=255)/np.sum(G_2_vsym2!=255) > 0.9 and np.sum(G_2_vsym1!=255)/np.sum(G_2_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_2_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_2_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.appen("T")
        else :
            answer.appen("W")                      

#G_3
area_list = []
if G_3_fea[0] == 2:
    if G_3_fea[1] > 0.9 and G_3_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_3_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_3_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_3_fea[0] == 1:
    if G_3_fea[1] > 0.9 and G_3_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_3_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_3_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_3_vsym1!=255)/np.sum(G_3_vsym2!=255) > 0.9 and np.sum(G_3_vsym1!=255)/np.sum(G_3_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_3_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_3_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.appen("T")
        else :
            answer.appen("W")
#G_4
area_list = []
if G_4_fea[0] == 2:
    if G_4_fea[1] > 0.9 and G_4_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_4_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_4_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_4_fea[0] == 1:
    if G_4_fea[1] > 0.9 and G_4_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_4_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_4_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_4_vsym1!=255)/np.sum(G_4_vsym2!=255) > 0.9 and np.sum(G_4_vsym1!=255)/np.sum(G_4_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_4_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_4_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.appen("T")
        else :
            answer.appen("W")

#G_5
area_list = []
if G_5_fea[0] == 2:
    if G_5_fea[1] > 0.9 and G_5_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_5_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_5_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_5_fea[0] == 1:
    if G_5_fea[1] > 0.9 and G_5_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_5_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_5_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_5_vsym1!=255)/np.sum(G_5_vsym2!=255) > 0.9 and np.sum(G_5_vsym1!=255)/np.sum(G_5_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_5_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_5_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.appen("T")
        else :
            answer.appen("W")

#G_6
area_list = []
if G_6_fea[0] == 2:
    if G_6_fea[1] > 0.9 and G_6_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_6_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_6_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_6_fea[0] == 1:
    if G_6_fea[1] > 0.9 and G_6_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_6_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_6_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_6_vsym1!=255)/np.sum(G_6_vsym2!=255) > 0.9 and np.sum(G_6_vsym1!=255)/np.sum(G_6_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_6_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_6_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.appen("T")
        else :
            answer.appen("W")

#G_7
area_list = []
if G_7_fea[0] == 2:
    if G_7_fea[1] > 0.9 and G_7_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_7_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_7_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_7_fea[0] == 1:
    if G_7_fea[1] > 0.9 and G_7_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_7_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_7_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_7_vsym1!=255)/np.sum(G_7_vsym2!=255) > 0.9 and np.sum(G_7_vsym1!=255)/np.sum(G_7_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_7_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_7_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.appen("T")
        else :
            answer.appen("W")
print("license plate of sample2 :",answer[0]+answer[1]+answer[2]+answer[3]+answer[4]+answer[5]+answer[6])

# sample3.jpg
sam = cv2.imread("sample3.jpg")
h,w,c = sam.shape
sam3 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        sam3[i][j] = sam[i][j][0]*0.299+sam[i][j][1]*0.587+sam[i][j][2]*0.114

for i in range(h):
    for j in range(w):
        if sam3[i][j] > 80:
            sam3[i][j] = 255

t_1 = sam3[35:100,10:50]
t_2 = sam3[35:100,50:90]
t_3 = sam3[35:100,90:130]
t_4 = sam3[35:100,140:180]
t_5 = sam3[35:100,180:220]
t_6 = sam3[35:100,220:257]
t_7 = sam3[35:100,257:295]

list_A = []
list_B = []
for i in range(65):
    for j in range(40):
        if t_1[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_1 = t_1[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("G_1.jpg",G_1)

list_A = []
list_B = []
for i in range(65):
    for j in range(40):
        if t_2[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_2 = t_2[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("G_2.jpg",G_2)

list_A = []
list_B = []
for i in range(65):
    for j in range(40):
        if t_3[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_3 = t_3[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("G_3.jpg",G_3)

list_A = []
list_B = []
for i in range(65):
    for j in range(40):
        if t_4[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_4 = t_4[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("G_4.jpg",G_4)

list_A = []
list_B = []
for i in range(65):
    for j in range(40):
        if t_5[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_5 = t_5[min(list_A):max(list_A)+1,min(list_B):max(list_B)+1]

cv2.imwrite("G_5.jpg",G_5)

list_A = []
list_B = []
for i in range(65):
    for j in range(37):
        if t_6[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_6 = t_6[min(list_A):max(list_A)+2,min(list_B):max(list_B)+1]

cv2.imwrite("G_6.jpg",G_6)

list_A = []
list_B = []
for i in range(65):
    for j in range(38):
        if t_7[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_7 = t_7[min(list_A):max(list_A)+1,min(list_B):max(list_B)+1]

cv2.imwrite("G_7.jpg",G_7)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_1.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j+1] == 255 and G_1[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j-1] == 255 and G_1[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j-1] == 255 and G_1[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j+1] == 255 and G_1[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j+1] != 255 and G_1[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] != 255 and G_1[i][j-1] == 255 and G_1[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j-1] != 255 and G_1[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] != 255 and G_1[i][j+1] == 255 and G_1[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j+1] != 255 and G_1[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] != 255 and G_1[i][j-1] == 255 and G_1[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j-1] != 255 and G_1[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] != 255 and G_1[i][j+1] == 255 and G_1[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] != 255 and G_1[i][j+1] != 255 and G_1[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j+1] == 255 and G_1[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j+1] == 255 and G_1[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler-1)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)

G_1_vsym1 = G_1[:27,:]
G_1_vsym2 = G_1[27:,:]
Sym = np.sum(G_1_vsym1!=255)/np.sum(G_1_vsym2!=255)
G_1_fea = [Hole,Sym,Area/(h*w)]
G_1_sp1 = G_1[:,:23]
G_1_sp2 = G_1[:,23:]
cv2.imwrite("1.jpg",G_1_vsym1)
cv2.imwrite("2.jpg",G_1_vsym2)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_2.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j+1] == 255 and G_2[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j-1] == 255 and G_2[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j-1] == 255 and G_2[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j+1] == 255 and G_2[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j+1] != 255 and G_2[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] != 255 and G_2[i][j-1] == 255 and G_2[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j-1] != 255 and G_2[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] != 255 and G_2[i][j+1] == 255 and G_2[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j+1] != 255 and G_2[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] != 255 and G_2[i][j-1] == 255 and G_2[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j-1] != 255 and G_2[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] != 255 and G_2[i][j+1] == 255 and G_2[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] != 255 and G_2[i][j+1] != 255 and G_2[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j+1] == 255 and G_2[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j+1] == 255 and G_2[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler-1)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)

G_2_vsym1 = G_2[:26,:]
G_2_vsym2 = G_2[26:,:]
Sym = np.sum(G_2_vsym1!=255)/np.sum(G_2_vsym2!=255)
G_2_fea = [Hole,Sym,Area/(h*w)]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0

h,w = G_3.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j+1] == 255 and G_3[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j-1] == 255 and G_3[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j-1] == 255 and G_3[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j+1] == 255 and G_3[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j+1] != 255 and G_3[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] != 255 and G_3[i][j-1] == 255 and G_3[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j-1] != 255 and G_3[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] != 255 and G_3[i][j+1] == 255 and G_3[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j+1] != 255 and G_3[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] != 255 and G_3[i][j-1] == 255 and G_3[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j-1] != 255 and G_3[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] != 255 and G_3[i][j+1] == 255 and G_3[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] != 255 and G_3[i][j+1] != 255 and G_3[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j+1] == 255 and G_3[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j+1] == 255 and G_3[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)

G_3_vsym1 = G_3[:28,:]
G_3_vsym2 = G_3[28:,:]
Sym = np.sum(G_3_vsym1!=255)/np.sum(G_3_vsym2!=255)
G_3_fea = [Hole,Sym,Area/(h*w)]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_4.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j+1] == 255 and G_4[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j-1] == 255 and G_4[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j-1] == 255 and G_4[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j+1] == 255 and G_4[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j+1] != 255 and G_4[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] != 255 and G_4[i][j-1] == 255 and G_4[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j-1] != 255 and G_4[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] != 255 and G_4[i][j+1] == 255 and G_4[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j+1] != 255 and G_4[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] != 255 and G_4[i][j-1] == 255 and G_4[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j-1] != 255 and G_4[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] != 255 and G_4[i][j+1] == 255 and G_4[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] != 255 and G_4[i][j+1] != 255 and G_4[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j+1] == 255 and G_4[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j+1] == 255 and G_4[i-1][j+1] != 255:
            Qn += 1 

G_4.shape = h,w
Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)

G_4_hsym1 = G_4[:28,:]
G_4_vsym2 = G_4[28:,:]
Sym = np.sum(G_4_hsym1!=255)/np.sum(G_4_vsym2!=255)
G_4_fea = [Hole,Sym,Area/(h*w)]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_5.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j+1] == 255 and G_5[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j-1] == 255 and G_5[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j-1] == 255 and G_5[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j+1] == 255 and G_5[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j+1] != 255 and G_5[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] != 255 and G_5[i][j-1] == 255 and G_5[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j-1] != 255 and G_5[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] != 255 and G_5[i][j+1] == 255 and G_5[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j+1] != 255 and G_5[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] != 255 and G_5[i][j-1] == 255 and G_5[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j-1] != 255 and G_5[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] != 255 and G_5[i][j+1] == 255 and G_5[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] != 255 and G_5[i][j+1] != 255 and G_5[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j+1] == 255 and G_5[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j+1] == 255 and G_5[i-1][j+1] != 255:
            Qn += 1 

G_5.shape = h,w
Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler-1)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)


G_5_vsym1 = G_5[:28,:]
G_5_vsym2 = G_5[28:,:]
Sym = np.sum(G_5_vsym1!=255)/np.sum(G_5_vsym2!=255)
G_5_fea = [Hole,Sym,Area/(h*w)]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_6.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j+1] == 255 and G_6[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j-1] == 255 and G_6[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j-1] == 255 and G_6[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j+1] == 255 and G_6[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j+1] != 255 and G_6[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] != 255 and G_6[i][j-1] == 255 and G_6[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j-1] != 255 and G_6[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] != 255 and G_6[i][j+1] == 255 and G_6[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j+1] != 255 and G_6[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] != 255 and G_6[i][j-1] == 255 and G_6[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j-1] != 255 and G_6[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] != 255 and G_6[i][j+1] == 255 and G_6[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] != 255 and G_6[i][j+1] != 255 and G_6[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j+1] == 255 and G_6[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j+1] == 255 and G_6[i-1][j+1] != 255:
            Qn += 1 

G_6.shape = h,w
Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)


G_6_hsym1 = G_6[:28,:]
G_6_vsym2 = G_6[28:,:]
Sym = np.sum(G_6_hsym1!=255)/np.sum(G_6_vsym2!=255)
G_6_fea = [Hole,Sym,Area/(h*w)]
G_6_sp1 = G_6[:,:18]
G_6_sp2 = G_6[:,18:]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_7.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j+1] == 255 and G_7[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j-1] == 255 and G_7[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j-1] == 255 and G_7[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j+1] == 255 and G_7[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j+1] != 255 and G_7[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] != 255 and G_7[i][j-1] == 255 and G_7[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j-1] != 255 and G_7[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] != 255 and G_7[i][j+1] == 255 and G_7[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j+1] != 255 and G_7[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] != 255 and G_7[i][j-1] == 255 and G_7[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j-1] != 255 and G_7[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] != 255 and G_7[i][j+1] == 255 and G_7[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] != 255 and G_7[i][j+1] != 255 and G_7[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j+1] == 255 and G_7[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j+1] == 255 and G_7[i-1][j+1] != 255:
            Qn += 1 

G_7.shape = h,w
Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler-1)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)


G_7_vsym1 = G_7[:28,:]
G_7_vsym2 = G_7[28:,:]
Sym = np.sum(G_7_vsym1!=255)/np.sum(G_7_vsym2!=255)
G_7_fea = [Hole,Sym,Area/(h*w)]
G_7_sp1 = G_7[:,:18]
G_7_sp2 = G_7[:,18:]
answer = []
h2_sym = [Area_B,Area_8]
h1_sym = [Area_D,Area_O,Area_0]
h1_nsym = [Area_A,Area_P,Area_Q,Area_R,Area_A,Area_A,Area_A]
h0_sym = [Area_C,Area_E,Area_H,Area_I,Area_K]
h0_nsym = [Area_F,Area_G,Area_L,Area_M,Area_J,Area_N,Area_S,Area_T,Area_W,Area_5]

#G_1
area_list = []
if G_1_fea[0] == 2:
    if G_1_fea[1] > 0.9 and G_1_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_1_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(abs(G_1_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_1_fea[0] == 1:
    if G_1_fea[1] > 0.9 and G_1_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_1_fea[2]-h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(abs(G_1_fea[2]- h1_nsym[i]))
        if np.sum(G_1_sp1!=255)/np.sum(G_1_sp2!=255) > 0.9 and np.sum(G_1_sp1!=255)/np.sum(G_1_sp2!=255) < 1.1:
            answer.append("A")
        else:
            if area_list.index(min(area_list)) == 1:
                answer.append("P")
            elif area_list.index(min(area_list)) == 2:
                answer.append("Q")
            elif area_list.index(min(area_list)) == 3:
                answer.append("R")
            elif area_list.index(min(area_list)) == 4:
                answer.append("4")
            elif area_list.index(min(area_list)) == 5:
                answer.append("6")    
            else:
                answer.append("9")
else: 
    if np.sum(G_1_vsym1!=255)/np.sum(G_1_vsym2!=255) > 0.85 and np.sum(G_1_vsym1!=255)/np.sum(G_1_vsym2!=255) < 1.15:
        for i in range(5):
            area_list.append(abs(G_1_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("H")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("E")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_1_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.append("T")
        else :
            answer.append("W")                          
                                                                                     


#G_2
area_list = []
if G_2_fea[0] == 2:
    if G_2_fea[1] > 0.9 and G_2_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_2_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(abs(G_2_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_2_fea[0] == 1:
    if G_2_fea[1] > 0.9 and G_2_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_2_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_2_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_2_vsym1!=255)/np.sum(G_2_vsym2!=255) > 0.9 and np.sum(G_2_vsym1!=255)/np.sum(G_2_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_2_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_2_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.append("T")
        else :
            answer.append("W")                      

#G_3
area_list = []
if G_3_fea[0] == 2:
    if G_3_fea[1] > 0.9 and G_3_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_3_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_3_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_3_fea[0] == 1:
    if G_3_fea[1] > 0.9 and G_3_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_3_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_3_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_3_vsym1!=255)/np.sum(G_3_vsym2!=255) > 0.9 and np.sum(G_3_vsym1!=255)/np.sum(G_3_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_3_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_3_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.appen("T")
        else :
            answer.appen("W")
#G_4
area_list = []
if G_4_fea[0] == 2:
    if G_4_fea[1] > 0.9 and G_4_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_4_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_4_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_4_fea[0] == 1:
    if G_4_fea[1] > 0.9 and G_4_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_4_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("0")
        else:
            answer.append("O")       
    else: 
        for i in range(7):
           area_list.append(G_4_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_4_vsym1!=255)/np.sum(G_4_vsym2!=255) > 0.9 and np.sum(G_4_vsym1!=255)/np.sum(G_4_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_4_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_4_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.appen("T")
        else :
            answer.appen("W")

#G_5
area_list = []
if G_5_fea[0] == 2:
    if G_5_fea[1] > 0.9 and G_5_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_5_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_5_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_5_fea[0] == 1:
    if G_5_fea[1] > 0.9 and G_5_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_5_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_5_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_5_vsym1!=255)/np.sum(G_5_vsym2!=255) > 0.9 and np.sum(G_5_vsym1!=255)/np.sum(G_5_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_5_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(10):
            area_list.append(abs(G_5_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.appen("T")
        elif area_list.index(min(area_list)) == 8:
            answer.append("W")    
        else :
            answer.append("5")


#G_6
area_list = []
if G_6_fea[0] == 2:
    if G_6_fea[1] > 0.9 and G_6_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_6_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0 and abs(np.sum(G_6_sp1!=255)/np.sum(G_6_sp2!=255)-1) > 0.1:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_6_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0 and abs(np.sum(G_6_sp1!=255)/np.sum(G_6_sp2!=255)-1) > 0.1:
            answer.append("B")
        else:
            answer.append("8")
elif G_6_fea[0] == 1:
    if G_6_fea[1] > 0.9 and G_6_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_6_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_6_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_6_vsym1!=255)/np.sum(G_6_vsym2!=255) > 0.9 and np.sum(G_6_vsym1!=255)/np.sum(G_6_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_6_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_6_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.appen("T")
        else :
            answer.appen("W")

#G_7
area_list = []
if G_7_fea[0] == 2:
    if G_7_fea[1] > 0.9 and G_7_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_7_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0 and abs(np.sum(G_7_sp1!=255)/np.sum(G_7_sp2!=255)-1) > 0.1:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_7_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0 and abs(np.sum(G_7_sp1!=255)/np.sum(G_7_sp2!=255)-1) > 0.1:
            answer.append("B")
        else:
            answer.append("8")
elif G_7_fea[0] == 1:
    if G_7_fea[1] > 0.9 and G_7_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_7_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_7_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_7_vsym1!=255)/np.sum(G_7_vsym2!=255) > 0.9 and np.sum(G_7_vsym1!=255)/np.sum(G_7_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_7_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_7_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.append("T")
        else :
            answer.append("W")
print("license plate of sample3 :",answer[0]+answer[1]+answer[2]+answer[3]+answer[4]+answer[5]+answer[6])

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

list_A = []
list_B = []
for i in range(60):
    for j in range(40):
        if t_1[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_1 = t_1[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("G_1.jpg",G_1)

list_A = []
list_B = []
for i in range(60):
    for j in range(32):
        if t_2[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_2 = t_2[min(list_A)-1:max(list_A),min(list_B)-1:max(list_B)]

cv2.imwrite("G_2.jpg",G_2)

list_A = []
list_B = []
for i in range(60):
    for j in range(30):
        if t_3[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_3 = t_3[min(list_A):max(list_A)+1,min(list_B):max(list_B)+1]

cv2.imwrite("G_3.jpg",G_3)

list_A = []
list_B = []
for i in range(60):
    for j in range(30):
        if t_4[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_4 = t_4[min(list_A):max(list_A)+1,min(list_B):max(list_B)+1]

cv2.imwrite("G_4.jpg",G_4)

list_A = []
list_B = []
for i in range(60):
    for j in range(30):
        if t_5[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_5 = t_5[min(list_A):max(list_A)+1,min(list_B):max(list_B)+1]

cv2.imwrite("G_5.jpg",G_5)

list_A = []
list_B = []
for i in range(60):
    for j in range(30):
        if t_6[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_6 = t_6[min(list_A):max(list_A)+2,min(list_B):max(list_B)+1]

cv2.imwrite("G_6.jpg",G_6)

list_A = []
list_B = []
for i in range(60):
    for j in range(30):
        if t_7[i][j] != 255:
            list_A.append(i)
            list_B.append(j)
G_7 = t_7[min(list_A):max(list_A)+1,min(list_B):max(list_B)+1]

cv2.imwrite("G_7.jpg",G_7)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_1.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j+1] == 255 and G_1[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j-1] == 255 and G_1[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j-1] == 255 and G_1[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j+1] == 255 and G_1[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j+1] != 255 and G_1[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] != 255 and G_1[i][j-1] == 255 and G_1[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j-1] != 255 and G_1[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] != 255 and G_1[i][j+1] == 255 and G_1[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j+1] != 255 and G_1[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] != 255 and G_1[i][j-1] == 255 and G_1[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j-1] != 255 and G_1[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] != 255 and G_1[i][j+1] == 255 and G_1[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] != 255 and G_1[i][j+1] != 255 and G_1[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i+1][j] == 255 and G_1[i][j+1] == 255 and G_1[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_1[i][j] != 255 and G_1[i-1][j] == 255 and G_1[i][j+1] == 255 and G_1[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler-1)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)

G_1_vsym1 = G_1[:24,:]
G_1_vsym2 = G_1[24:,:]
Sym = np.sum(G_1_vsym1!=255)/np.sum(G_1_vsym2!=255)
G_1_fea = [Hole,Sym,Area/(h*w)]
G_1_sp1 = G_1[:,:23]
G_1_sp2 = G_1[:,23:]
cv2.imwrite("1.jpg",G_1_vsym1)
cv2.imwrite("2.jpg",G_1_vsym2)

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_2.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j+1] == 255 and G_2[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j-1] == 255 and G_2[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j-1] == 255 and G_2[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j+1] == 255 and G_2[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j+1] != 255 and G_2[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] != 255 and G_2[i][j-1] == 255 and G_2[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j-1] != 255 and G_2[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] != 255 and G_2[i][j+1] == 255 and G_2[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j+1] != 255 and G_2[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] != 255 and G_2[i][j-1] == 255 and G_2[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j-1] != 255 and G_2[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] != 255 and G_2[i][j+1] == 255 and G_2[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] != 255 and G_2[i][j+1] != 255 and G_2[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i+1][j] == 255 and G_2[i][j+1] == 255 and G_2[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_2[i][j] != 255 and G_2[i-1][j] == 255 and G_2[i][j+1] == 255 and G_2[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler-1)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)

G_2_vsym1 = G_2[:24,:]
G_2_vsym2 = G_2[24:,:]
Sym = np.sum(G_2_vsym1!=255)/np.sum(G_2_vsym2!=255)
G_2_fea = [Hole,Sym,Area/(h*w)]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0

h,w = G_3.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j+1] == 255 and G_3[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j-1] == 255 and G_3[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j-1] == 255 and G_3[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j+1] == 255 and G_3[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j+1] != 255 and G_3[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] != 255 and G_3[i][j-1] == 255 and G_3[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j-1] != 255 and G_3[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] != 255 and G_3[i][j+1] == 255 and G_3[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j+1] != 255 and G_3[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] != 255 and G_3[i][j-1] == 255 and G_3[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j-1] != 255 and G_3[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] != 255 and G_3[i][j+1] == 255 and G_3[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] != 255 and G_3[i][j+1] != 255 and G_3[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i+1][j] == 255 and G_3[i][j+1] == 255 and G_3[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_3[i][j] != 255 and G_3[i-1][j] == 255 and G_3[i][j+1] == 255 and G_3[i-1][j+1] != 255:
            Qn += 1 

Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler-1)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)

G_3_vsym1 = G_3[:24,:]
G_3_vsym2 = G_3[24:,:]
Sym = np.sum(G_3_vsym1!=255)/np.sum(G_3_vsym2!=255)
G_3_fea = [Hole,Sym,Area/(h*w)]
G_3_sp1 = G_3[:,:14]
G_3_sp2 =G_3[:,14:]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_4.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j+1] == 255 and G_4[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j-1] == 255 and G_4[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j-1] == 255 and G_4[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j+1] == 255 and G_4[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j+1] != 255 and G_4[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] != 255 and G_4[i][j-1] == 255 and G_4[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j-1] != 255 and G_4[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] != 255 and G_4[i][j+1] == 255 and G_4[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j+1] != 255 and G_4[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] != 255 and G_4[i][j-1] == 255 and G_4[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j-1] != 255 and G_4[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] != 255 and G_4[i][j+1] == 255 and G_4[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] != 255 and G_4[i][j+1] != 255 and G_4[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i+1][j] == 255 and G_4[i][j+1] == 255 and G_4[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_4[i][j] != 255 and G_4[i-1][j] == 255 and G_4[i][j+1] == 255 and G_4[i-1][j+1] != 255:
            Qn += 1 

G_4.shape = h,w
Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler-1)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)

G_4_vsym1 = G_4[:24,:]
G_4_vsym2 = G_4[24:,:]
Sym = np.sum(G_4_vsym1!=255)/np.sum(G_4_vsym2!=255)
G_4_fea = [Hole,Sym,Area/(h*w)]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_5.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j+1] == 255 and G_5[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j-1] == 255 and G_5[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j-1] == 255 and G_5[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j+1] == 255 and G_5[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j+1] != 255 and G_5[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] != 255 and G_5[i][j-1] == 255 and G_5[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j-1] != 255 and G_5[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] != 255 and G_5[i][j+1] == 255 and G_5[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j+1] != 255 and G_5[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] != 255 and G_5[i][j-1] == 255 and G_5[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j-1] != 255 and G_5[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] != 255 and G_5[i][j+1] == 255 and G_5[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] != 255 and G_5[i][j+1] != 255 and G_5[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i+1][j] == 255 and G_5[i][j+1] == 255 and G_5[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_5[i][j] != 255 and G_5[i-1][j] == 255 and G_5[i][j+1] == 255 and G_5[i-1][j+1] != 255:
            Qn += 1 

G_5.shape = h,w
Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler-1)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)


G_5_vsym1 = G_5[:28,:]
G_5_vsym2 = G_5[28:,:]
Sym = np.sum(G_5_vsym1!=255)/np.sum(G_5_vsym2!=255)
G_5_fea = [Hole,Sym,Area/(h*w)]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_6.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j+1] == 255 and G_6[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j-1] == 255 and G_6[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j-1] == 255 and G_6[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j+1] == 255 and G_6[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j+1] != 255 and G_6[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] != 255 and G_6[i][j-1] == 255 and G_6[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j-1] != 255 and G_6[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] != 255 and G_6[i][j+1] == 255 and G_6[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j+1] != 255 and G_6[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] != 255 and G_6[i][j-1] == 255 and G_6[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j-1] != 255 and G_6[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] != 255 and G_6[i][j+1] == 255 and G_6[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] != 255 and G_6[i][j+1] != 255 and G_6[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i+1][j] == 255 and G_6[i][j+1] == 255 and G_6[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_6[i][j] != 255 and G_6[i-1][j] == 255 and G_6[i][j+1] == 255 and G_6[i-1][j+1] != 255:
            Qn += 1 

G_6.shape = h,w
Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler-1)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)


G_6_hsym1 = G_6[:24,:]
G_6_vsym2 = G_6[24:,:]
Sym = np.sum(G_6_hsym1!=255)/np.sum(G_6_vsym2!=255)
G_6_fea = [Hole,Sym,Area/(h*w)]
G_6_sp1 = G_6[:,:13]
G_6_sp2 =G_6[:,13:]

Q1,Q2,Q3,Q4,Qn = 0,0,0,0,0
h,w = G_7.shape

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j+1] == 255 and G_7[i+1][j+1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j-1] == 255 and G_7[i+1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j-1] == 255 and G_7[i-1][j-1] == 255:
            Q1 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j+1] == 255 and G_7[i-1][j+1] == 255:
            Q1 += 1  

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j+1] != 255 and G_7[i+1][j+1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] != 255 and G_7[i][j-1] == 255 and G_7[i+1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j-1] != 255 and G_7[i-1][j-1] == 255:
            Q2 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] != 255 and G_7[i][j+1] == 255 and G_7[i-1][j+1] == 255:
            Q2 += 1

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j+1] != 255 and G_7[i+1][j+1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] != 255 and G_7[i][j-1] == 255 and G_7[i+1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j-1] != 255 and G_7[i-1][j-1] != 255:
            Q3 += 1
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] != 255 and G_7[i][j+1] == 255 and G_7[i-1][j+1] != 255:
            Q3 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] != 255 and G_7[i][j+1] != 255 and G_7[i-1][j+1] != 255:
            Q4 += 1 

for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i+1][j] == 255 and G_7[i][j+1] == 255 and G_7[i+1][j+1] != 255:
            Qn += 1 
for i in range(1,h-1):
    for j in range(1,w-1):
        if G_7[i][j] != 255 and G_7[i-1][j] == 255 and G_7[i][j+1] == 255 and G_7[i-1][j+1] != 255:
            Qn += 1 

G_7.shape = h,w
Area = 1/4*(Q1+2*Q2+(7/2)*Q3+4*Q4+3*Qn)
Para = Q2+(1/np.power(2,1/2)*(Q1+Q3+2*Qn))
Circ = 4*np.pi*Area/np.power(Para,2)
Euler = 1/4*(Q1-Q3+2*Qn)
Hole = np.floor(1-Euler-1)
# print(Q1,Q2,Q3,Q4,Qn,Area,Para,Circ,Hole)


G_7_vsym1 = G_7[:24,:]
G_7_vsym2 = G_7[24:,:]
Sym = np.sum(G_7_vsym1!=255)/np.sum(G_7_vsym2!=255)
G_7_fea = [Hole,Sym,Area/(h*w)]
G_7_sp1 = G_7[:,:13]
G_7_sp2 =G_7[:,13:]
answer = []
h2_sym = [Area_B,Area_8]
h1_sym = [Area_D,Area_O,Area_0]
h1_nsym = [Area_A,Area_P,Area_Q,Area_R,Area_A,Area_A,Area_A]
h0_sym = [Area_C,Area_E,Area_H,Area_I,Area_K]
h0_nsym = [Area_F,Area_G,Area_L,Area_M,Area_J,Area_N,Area_S,Area_T,Area_W,Area_5]

#G_1
area_list = []
if G_1_fea[0] == 2:
    if G_1_fea[1] > 0.9 and G_1_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_1_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(abs(G_1_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_1_fea[0] == 1:
    if G_1_fea[1] > 0.9 and G_1_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_1_fea[2]-h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(abs(G_1_fea[2]- h1_nsym[i]))
        if np.sum(G_1_sp1!=255)/np.sum(G_1_sp2!=255) > 0.9 and np.sum(G_1_sp1!=255)/np.sum(G_1_sp2!=255) < 1.1:
            answer.append("A")
        else:
            if area_list.index(min(area_list)) == 1:
                answer.append("P")
            elif area_list.index(min(area_list)) == 2:
                answer.append("Q")
            elif area_list.index(min(area_list)) == 3:
                answer.append("R")
            elif area_list.index(min(area_list)) == 4:
                answer.append("4")
            elif area_list.index(min(area_list)) == 5:
                answer.append("6")    
            else:
                answer.append("9")
else: 
    if np.sum(G_1_vsym1!=255)/np.sum(G_1_vsym2!=255) > 0.85 and np.sum(G_1_vsym1!=255)/np.sum(G_1_vsym2!=255) < 1.15:
        for i in range(5):
            area_list.append(abs(G_1_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("H")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("E")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_1_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.append("T")
        else :
            answer.append("W")                          
                                                                                     


#G_2
area_list = []
if G_2_fea[0] == 2:
    if G_2_fea[1] > 0.9 and G_2_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_2_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(abs(G_2_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_2_fea[0] == 1:
    if G_2_fea[1] > 0.9 and G_2_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_2_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_2_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_2_vsym1!=255)/np.sum(G_2_vsym2!=255) > 0.9 and np.sum(G_2_vsym1!=255)/np.sum(G_2_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_2_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_2_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.append("T")
        else :
            answer.append("W")                      

#G_3
area_list = []
if G_3_fea[0] == 2:
    if G_3_fea[1] > 0.9 and G_3_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_3_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_3_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_3_fea[0] == 1:
    if G_3_fea[1] > 0.9 and G_3_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_3_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_3_fea[2]- h1_nsym[i])
        if np.sum(G_3_sp1!=255)/np.sum(G_3_sp2!=255) > 0.85 and np.sum(G_3_sp1!=255)/np.sum(G_3_sp2!=255) < 1.15 :
            answer.append("A")
        else:
            if area_list.index(min(area_list)) == 1:
                answer.append("P")
            elif area_list.index(min(area_list)) == 2:
                answer.append("Q")
            elif area_list.index(min(area_list)) == 3:
                answer.append("R")
            elif area_list.index(min(area_list)) == 4:
                answer.append("4")
            elif area_list.index(min(area_list)) == 5:
                answer.append("6")    
            else:
                answer.append("9")
else:
    if np.sum(G_3_vsym1!=255)/np.sum(G_3_vsym2!=255) > 0.9 and np.sum(G_3_vsym1!=255)/np.sum(G_3_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_3_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_3_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.appen("T")
        else :
            answer.appen("W")
#G_4
area_list = []
if G_4_fea[0] == 2:
    if G_4_fea[1] > 0.9 and G_4_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_4_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_4_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_4_fea[0] == 1:
    if G_4_fea[1] > 0.9 and G_4_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_4_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("0")
        else:
            answer.append("O")       
    else: 
        for i in range(7):
           area_list.append(G_4_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_4_vsym1!=255)/np.sum(G_4_vsym2!=255) > 0.9 and np.sum(G_4_vsym1!=255)/np.sum(G_4_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_4_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(10):
            area_list.append(abs(G_4_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.append("T")
        elif area_list.index(min(area_list)) == 8:
            answer.append("W")    
        else :
            answer.append("5")


#G_5
area_list = []
if G_5_fea[0] == 2:
    if G_5_fea[1] > 0.9 and G_5_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_5_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_5_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("B")
        else:
            answer.append("8")
elif G_5_fea[0] == 1:
    if G_5_fea[1] > 0.9 and G_5_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_5_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_5_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_5_vsym1!=255)/np.sum(G_5_vsym2!=255) > 0.9 and np.sum(G_5_vsym1!=255)/np.sum(G_5_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_5_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(10):
            area_list.append(abs(G_5_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.appen("T")
        elif area_list.index(min(area_list)) == 8:
            answer.append("W")    
        else :
            answer.append("5")

#G_6
area_list = []
if G_6_fea[0] == 2:
    if G_6_fea[1] > 0.9 and G_6_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_6_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0 and abs(np.sum(G_6_sp1!=255)/np.sum(G_6_sp2!=255)-1) > 0.1:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_6_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0 and abs(np.sum(G_6_sp1!=255)/np.sum(G_6_sp2!=255)-1) > 0.1:
            answer.append("B")
        else:
            answer.append("8")
elif G_6_fea[0] == 1:
    if G_6_fea[1] > 0.9 and G_6_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_6_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_6_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_6_vsym1!=255)/np.sum(G_6_vsym2!=255) > 0.9 and np.sum(G_6_vsym1!=255)/np.sum(G_6_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_6_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_6_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.appen("T")
        else :
            answer.appen("W")

#G_7
area_list = []
if G_7_fea[0] == 2:
    if G_7_fea[1] > 0.9 and G_7_fea[1] < 1.1 :
        for i in range(2):
           area_list.append(abs(G_7_fea[2]- h2_sym[i]))
        if area_list.index(min(area_list)) == 0 and abs(np.sum(G_7_sp1!=255)/np.sum(G_7_sp2!=255)-1) > 0.1:
            answer.append("B")
        else:
            answer.append("8")   
    else:    
        for i in range(2):
           area_list.append(G_7_fea[2]- h2_sym[i])
        if area_list.index(min(area_list)) == 0 and abs(np.sum(G_7_sp1!=255)/np.sum(G_7_sp2!=255)-1) > 0.1:
            answer.append("B")
        else:
            answer.append("8")
elif G_7_fea[0] == 1:
    if G_7_fea[1] > 0.9 and G_7_fea[1] < 1.1 :
        for i in range(3):
           area_list.append(abs(G_7_fea[2]- h1_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("D")
        elif area_list.index(min(area_list)) == 1:
            answer.append("O")
        else:
            answer.append("0")       
    else: 
        for i in range(7):
           area_list.append(G_7_fea[2]- h1_nsym[i])
        if area_list.index(min(area_list)) == 0:
            answer.append("A")
        elif area_list.index(min(area_list)) == 1:
            answer.append("P")
        elif area_list.index(min(area_list)) == 2:
            answer.append("Q")
        elif area_list.index(min(area_list)) == 3:
            answer.append("R")
        elif area_list.index(min(area_list)) == 4:
            answer.append("4")
        elif area_list.index(min(area_list)) == 5:
            answer.append("6")    
        else:
            answer.append("9")
else:
    if np.sum(G_7_vsym1!=255)/np.sum(G_7_vsym2!=255) > 0.9 and np.sum(G_7_vsym1!=255)/np.sum(G_7_vsym2!=255) < 1.1:
        for i in range(5):
            area_list.append(abs(G_7_fea[2]- h0_sym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("C") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("E")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("H")
        elif area_list.index(min(area_list)) == 3:
            answer.append("I")
        else :
            answer.append("K")     
    else :
        for i in range(9):
            area_list.append(abs(G_7_fea[2]- h0_nsym[i]))
        if area_list.index(min(area_list)) == 0:
            answer.append("F") 
        elif area_list.index(min(area_list)) == 1:
            answer.append("G")   
        elif area_list.index(min(area_list)) == 2:
            answer.append("L")
        elif area_list.index(min(area_list)) == 3:
            answer.append("M")
        elif area_list.index(min(area_list)) == 4:
            answer.append("J") 
        elif area_list.index(min(area_list)) == 5:
            answer.append("N")
        elif area_list.index(min(area_list)) == 6:
            answer.append("S") 
        elif area_list.index(min(area_list)) == 7:
            answer.append("T")
        else :
            answer.append("W")
print("license plate of sample4 :",answer[0]+answer[1]+answer[2]+answer[3]+answer[4]+answer[5]+answer[6])