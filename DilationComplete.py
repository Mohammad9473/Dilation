# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:26:04 2019

@author: Mohammadreza
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt



#image = Image.open('z.png').convert('L')
#image = cv2.imread('z.png')
image = plt.imread('Dilation.jpg', format=None)
image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
res, image = cv2.threshold(image,127,1,1)
#image = np.array([[1,0,1,0,1],[0,1,1,1,0],[0,0,1,0,1]])

#image = np.array([[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]])
#print(image)
dimensions = image.shape
height = image.shape[0]
width = image.shape[1]


#S = (1,3) 
#B = np.ones(S,dtype =int)
B=np.array([[0,1,0],[1,1,1],[0,1,0]])
#B[0,0] = 255
#B[0,1] = 255
#B[0,2] = 255
#print(B)
#B1 = np.size(B)
#b = np.linalg.inv(B)

D0 = np.zeros((height,width), dtype=int)
#print(D0)

a11=np.array(np.pad(image, ((1,1),(1,1)), 'constant'))

#print(a11)

#H = a11.shape[0]
#W = a11.shape[1]

#B = np.flip(B,(1))
#B = np.flip(B,(0))

#print(a11.shape)
for i in range(height):
    for j in range(width):
        test = np.logical_and(a11[i:(i+B.shape[0]),j:(j+B.shape[1])],B)
        
        if np.isin(True,test) == True:
                D0[i][j] = 1 
        
        
        
plt.imshow(image, cmap='gray')
plt.figure()        
plt.imshow(D0 ,cmap='gray')
plt.show()        
        
        