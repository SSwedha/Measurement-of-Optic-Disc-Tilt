import os
import cv2
import time
import math
import random
import argparse
import numpy as np
import numpy.matlib as npmatlib
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import statistics as stat
from skimage import io
from skimage import data
from skimage import color
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.segmentation import active_contour
from skimage.filters import meijering, sato, frangi, hessian
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

def nothing(x):
    pass

def write_image(path, img):
    # img = img*(2**16-1)
    # img = img.astype(np.uint16)
    img = img.astype(np.uint8)
    cv2.imwrite(path,img)
    # Convert the scale (values range) of the image
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    # Save file
    plt.savefig(path, bbox_inches='tight')#, img, format = 'png')

def threshold_tb(image, threshold = 110, th_type = 3):
    alpha = 100
    beta = 00
    s = 0
    threshold = 125

    """
    cv2.namedWindow('Contrast')
    # create trackbars for color change
    cv2.createTrackbar('alpha','Contrast', alpha, 255, nothing)
    cv2.createTrackbar('beta','Contrast', beta, 255, nothing)
    cv2.createTrackbar('Threshold','Contrast', threshold, 255, nothing)
    cv2.createTrackbar('Type','Contrast', th_type, 4, nothing)
    # create switch for ON/OFF functionality
    cv2.createTrackbar('Switch', 'Contrast', s, 1, nothing)
    # get current positions of four trackbars
    while(s == 0):
        alpha = cv2.getTrackbarPos('alpha','Contrast')
        beta = cv2.getTrackbarPos('beta','Contrast')
        threshold = cv2.getTrackbarPos('Threshold','Contrast')
        th_type = cv2.getTrackbarPos('Type','Contrast')
        s = cv2.getTrackbarPos('Switch','Contrast')

        _, dst = cv2.threshold(image, 70, 255, 3)
        i = cv2.convertScaleAbs(dst, alpha=alpha/100, beta=-1*beta)
        cv2.imshow('i',i)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
        #i = clahe.apply(i)

        _, t = cv2.threshold(i, threshold, 255, 0)
        cv2.imshow('Threshold', t)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    """
    i = cv2.convertScaleAbs(image, alpha=alpha/100, beta=-1*beta)
    _, dst = cv2.threshold(i, threshold, 255, th_type)
    
    #cv2.destroyAllWindows()
    
    return dst

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

if __name__=="__main__":
    
    # Image files location 
    location1 = 'Documents\GitHub\Optic_Disk\Images\_CS'
    # Loop through all the images

    for i in range(1, 29):

        image1 = location1 + str(i) + '.jpeg'    # Filename
        #image2 = location2 + str(i) + '.jpg'    # Filename
        img1 = cv2.imread(image1,0)              # Read image
        #img2 = cv2.imread(image2,0)             # Read image
        #img1 = cv2.medianBlur(img1,3)
        img1 = cv2.GaussianBlur(img1,(5,5),0)
        #img1 = cv2.blur(img1,(15,15))
        img = img1.copy()
        image = img.copy()

        thresh = threshold_tb(img1)

        height, width = img.shape[0:2]
        pivot = np.zeros((width,1))
        for b in range(width):
            for a in range(height):
                if thresh[a,b] > 200:
                    pivot[b] = a
                    break

        maximum = np.amax(pivot)
        for a in range(width):
            if pivot[a] == maximum:
                maxpos = a
                break

        partl = thresh.copy()
        
        #cv2.imshow('1-bef',partl)
        #cv2.imshow('r',partr)

        for b in range(width):
            fl = 0
            for a in range(height):
                if partl[a,b] > 200:
                    fl = fl + 1
                if partl[a,b] < 10 and fl < 20:
                    partl[a,b] = 255
                if fl > 20:
                    break
            fl = 0
            for a in range(height):
                if partl[a,b] < 10:
                    fl = fl + 1
                if partl[a,b] > 200 and fl < 20:
                    partl[a,b] = 0
                if fl > 20:
                    break
            fl = 0
            for a in range(height):
                if partl[a,b] > 200 and fl < 20:
                    fl = fl + 1
                    continue
                partl[a,b] = 0
        
        partr = partl.copy()

        for a in range(width):
            if a > 0.6*maxpos:
                partl[:, a] = 0
            if a < 1.4*maxpos:
                partr[:, a] = 0
        picture = partl + partr
        
        """
        path = 'Documents\GitHub\Optic_Disk\Images_Processed\_CS' + str(i) + '.jpeg'
        cv2.imwrite(path, picture)
        if not cv2.imwrite(path, picture):
            raise Exception("Could not write image")
        print('saved')
        """

        cv2.imshow('picture', picture)

        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(picture,kernel,iterations = 2)
        dilation = cv2.dilate(erosion,kernel,iterations = 3)
        opening = cv2.morphologyEx(picture, cv2.MORPH_OPEN, kernel)
        #closing = cv2.morphologyEx(picture, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('ero-dil', dilation)
        #cv2.imshow('ero', erosion)
        cv2.imshow('ope', opening)
        #cv2.imshow('clo', closing)
        cv2.waitKey(0) 

        """
        for b in range(width):
            fl = 0
            for a in range(height):
                if partr[a,b] > 200 and fl < 20:
                    fl = fl + 1
                    continue
                partr[a,b] = 0
        
        picture = partl + partr
        cv2.imshow('pic', picture)

        contour1,hierarchy1 = cv2.findContours(partl, 1, 2)
        contour2,hierarchy2 = cv2.findContours(partr, 1, 2)
        cnt1 = contour1[0]
        cnt2 = contour2[0]

        [vx1,vy1,x1,y1] = cv2.fitLine(cnt1, cv2.DIST_L2,0,0.01,0.01)
        print([vx1,vy1,x1,y1])
        lefty1 = int((-x1*vy1/vx1) + y1)
        righty1 = int(((width-x1)*vy1/vx1)+y1)
        print([lefty1,righty1])
        #cv2.imshow('l', partl)
        p = cv2.line(partl,(width-1,righty1),(0,lefty1),(0,255,0),2)
        cv2.imshow('p',p)
        q = cv2.drawContours(partl, contour1, -1, (0,255,0), 3)
        cv2.imshow('l', q)

        [vx2,vy2,x2,y2] = cv2.fitLine(cnt2, cv2.DIST_L2,0,0.01,0.01)
        print([vx2, vy2, 2, y2])
        lefty2 = int((-x2*vy2/vx2) + y2)
        righty2 = int(((width-x2)*vy2/vx2)+y2)
        print([lefty2,righty2])
        #cv2.imshow('r', partr)
        #cv2.line(picture,(width-1,righty2),(0,lefty2),(255,0,0),2) 
        
        cv2.waitKey(0)
        """

    cv2.destroyAllWindows
