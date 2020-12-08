# bonus
import numpy as np
import cv2
import matplotlib.pyplot as pyplot

img = cv2.imread("sample5.jpg")
h,w,c = img.shape
sam5 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        sam5[i][j] = img[i][j][0]

# mean filter
for i in range(2,h-2):
    for j in range(2,w-2):
        sam5[i][j] = 1/86*(sam5[i-2][j-2]+2*sam5[i-2][j-1]+4*sam5[i-2][j]+2*sam5[i-2][j+1]+sam5[i-2][j+2]+2*sam5[i-1][j-2]+4*sam5[i-1][j-1]+6*sam5[i-1][j]+4*sam5[i-1][j+1]+2*sam5[i-1][j+2]+4*sam5[i][j-2]+6*sam5[i][j-1]+10*sam5[i][j]+6*sam5[i][j+1]+4*sam5[i][j+2]+2*sam5[i+1][j-2]+4*sam5[i+1][j-1]+6*sam5[i+1][j]+4*sam5[i+1][j+1]+2*sam5[i+1][j+2]+sam5[i+2][j-2]+2*sam5[i+2][j-1]+4*sam5[i+2][j]+2*sam5[i+2][j+1]+sam5[i+2][j+2])

# median filter
for i in range(1,h-1):
    for j in range(1,w-1):
        sam5[i][j] = 0.5*np.max([np.min([sam5[i-1][j],sam5[i][j-1],sam5[i][j]]),np.min([sam5[i][j-1],sam5[i][j],sam5[i][j+1]]),np.min([sam5[i][j],sam5[i][j+1],sam5[i+1][j]])]) + 0.5*np.min([np.max([sam5[i-1][j],sam5[i][j-1],sam5[i][j]]),np.max([sam5[i][j-1],sam5[i][j],sam5[i][j+1]]),np.max([sam5[i][j],sam5[i][j+1],sam5[i+1][j]])])
cv2.imwrite("bouns.jpg",sam5)  