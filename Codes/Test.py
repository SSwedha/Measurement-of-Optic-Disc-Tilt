from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt
import cv2
import numpy as np

def removeBloodVessels(img):

    globalMin = 0 
    globalMax = 255   # it is the maximum gray level of the image

    img_flat = img.flatten()
    localMin = np.min(img_flat[np.nonzero(img_flat)])   # minimum non-zero intensity value of the taken image
    localMax = max(img_flat)   # maximum intensity value of the taken image

    m, n = img.shape[0:2]
    kernel1 = np.ones((25,25),np.uint8)
    kernel0 = np.zeros((25,25),np.uint8)
    for i in range(m):
        for j in range(n):
            img[i,j] = ((img[i,j]-localMin)* ((globalMax-globalMin))/((localMax-localMin))) + globalMin 
    closing1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
    closing0 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel0)

    bottomHat1 = closing1 - img
    bottomHat0 = closing0 - img
    
    cv2.imshow('bottomHat0',bottomHat0)
    cv2.waitKey(0)
    for i in range(m):
        for j in range(n):
            if (bottomHat0[i,j] < 20):
                bottomHat0[i,j] = 0
            else:
                bottomHat0[i,j] = 255
            if bottomHat1[i,j] < 60:
                bottomHat1[i,j] = 0
    cv2.imshow('bottomHat0 after for',bottomHat0)
    cv2.waitKey(0)                      

    alpha = 2
    beta = -125
    bottomHat1 = cv2.convertScaleAbs(bottomHat1, alpha=alpha, beta=beta)
    new_greyimage = np.zeros((m,n),np.uint8)
    for i in range(m):
        for j in range(n):
            if (bottomHat0[i,j] == 255):
                new_greyimage[i,j] = bottomHat1[i,j]
    cv2.imshow('grey',new_greyimage)
    cv2.waitKey(0)
    enhancedVessel1 = meijering(new_greyimage)
    enhancedVessel2 = frangi(new_greyimage)
    new_greyimage = cv2.addWeighted(enhancedVessel1,0.5,enhancedVessel2,0.5,0.0)
    cv2.imshow('fuse',new_greyimage)
    cv2.waitKey(0)
    
img = cv2.imread('worst_image.jpeg',0)
removeBloodVessels(img)