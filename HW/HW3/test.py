import numpy as np
import cv2
import matplotlib.pyplot as pyplot

DTW = np.zeros((u,v))

DTW[0][0] = 0
DTW[0][1] = 0
DTW[1][0] = 0
for i in range(1,u):
    for j in range(1,v):
        if DTW[i-1][j-1] + abs(Q[u]-T[v])/(Q[u]+T[v]) < DTW[i-1][j] + abs(Q[u]-T[v])/(Q[u]+T[v]) and DTW[i-1][j-1] + abs(Q[u]-T[v])/(Q[u]+T[v]) < DTW[i][j-1] + abs(Q[u]-T[v])/(Q[u]+T[v]):
            DTW[i][j] = DTW[i-1][j-1] + abs(Q[u]-T[v])/(Q[u]+T[v])
        elif  DTW[i-1][j] + abs(Q[u]-T[v])/(Q[u]+T[v]) <  DTW[i][j-1] + abs(Q[u]-T[v])/(Q[u]+T[v]):
            DTW[i][j] = DTW[i-1][j] + abs(Q[u]-T[v])/(Q[u]+T[v])  
        else
            DTW[i][j] = DTW[i][j-1] + abs(Q[u]-T[v])/(Q[u]+T[v])       

#DTW的對角項平方相加取更號 看值看哪個template做DTW為最小            