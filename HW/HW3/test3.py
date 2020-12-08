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