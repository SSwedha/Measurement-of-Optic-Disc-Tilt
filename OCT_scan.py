import cv2
import time
import math
import random
import argparse
import numpy as np
import numpy.matlib as npmatlib
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps
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

def auto_canny(image, sigma = 0.7):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v)/2)
    #print(lower)
    #print(upper)
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

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

def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = variance(img)
    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def threshold_tb(image, threshold = 60, th_type = 3):
    alpha = 100
    beta = 0
    s = 0
    threshold = 125
    '''
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
        """
        #0: Binary
        #1: Binary Inverted
        #2: Threshold Truncated
        #3: Threshold to Zero
        #4: Threshold to Zero Inverted
        """
        i = cv2.convertScaleAbs(image, alpha=alpha/100, beta=-1*beta)
        _, dst = cv2.threshold(i, threshold, 255, th_type)
        cv2.imshow('Threshold', dst)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    '''
    i = cv2.convertScaleAbs(image, alpha=alpha/100, beta=-1*beta)
    _, dst = cv2.threshold(i, threshold, 255, th_type)
    #cv2.imshow('Threshold', dst)
    #cv2.waitKey(1)
    
    return dst

if __name__=="__main__":
    
    # Image files location 
    location1 = 'OCT_scan'
    location1 = 'Documents\GitHub\Optic_Disk\Images\_CS'
    #location2 = 'Documents\GitHub\Optic_Disk\Images_Processed\_OD'
    # Loop through all the images

    for i in range(1, 29):

        image1 = location1 + str(i) + '.jpeg'    # Filename
        #image2 = location2 + str(i) + '.jpg'    # Filename
        img1 = cv2.imread(image1,0)              # Read image
        #img2 = cv2.imread(image2,0)             # Read image
        #img1 = cv2.medianBlur(img1,15)
        #img1 = cv2.GaussianBlur(img1,(15,15),0)
        #img1 = cv2.blur(img1,(15,15))
        #cv2.imshow('image',img1)
        img = img1.copy()
        image = img.copy()
        plt.subplot(2,3,1),plt.imshow(image,cmap = 'gray')
        plt.title('OG_image'), plt.xticks([]), plt.yticks([])

        thresh = threshold_tb(img1, 125, 0)
        plt.subplot(2,3,5),plt.imshow(thresh,cmap = 'gray')
        plt.title('Th_image'), plt.xticks([]), plt.yticks([])

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
        partr = thresh.copy()

        for a in range(width):
            if a > 1*maxpos:
                partl[:, a] = 0
            if a < 1*maxpos:
                partr[:, a] = 0

        for b in range(int(0.25*width)):
            for a in range(int(0.75*height),height,1):
                partl[a,b] = 0

        #cv2.imshow('partl',partl)
        #cv2.waitKey(1)
        
        partl_top = np.zeros((height,width))
        partr_top = np.zeros((height,width))

        partl_bottom = np.zeros((height,width))
        partr_bottom = np.zeros((height,width))

        partl_top = partl.copy()
        partr_top = partr.copy()

        partl_bottom = partl.copy()
        partr_bottom = partr.copy()

        for b in range(width):
            fl = 0
            for a in range(height):
                if partl_top[a,b] > 200 and fl < 20:
                    fl = fl + 1
                    continue
                
                if fl < 20 and b < int(0.4*width):
                    partl_top[a,b] = 255

        for b in range(width):
            fl = 0
            for a in range(height):
                if partr_top[a,b] > 200 and fl < 20:
                    fl = fl + 1
                    continue
                
                if fl < 20 and b > int(0.65*width):
                    partr_top[a,b] = 255


        for b in range(width):
            fl = 0
            for a in range(height-1,0,-1): #thin line of layer
                if partl_bottom[a,b] > 200 and fl < 15:
                    fl = fl + 1
                    continue
                if fl>= 15:
                    partl_bottom[a,b] = 0
                if fl < 15 and b < int(0.4*width):
                    partl_bottom[a,b] = 255

        for b in range(width):
            fl = 0
            for a in range(height-1,0,-1): #thin line of layer
                if partr_bottom[a,b] > 200 and fl < 15:
                    fl = fl + 1
                    continue
                if fl>= 15:
                    partr_bottom[a,b] = 0
                if fl < 15 and b > int(0.65*width):
                    partr_bottom[a,b] = 255

        for b in range(width):
            for a in range(height):
                if partl_bottom[a,b] > 200 or partl_top[a,b] > 200:
                    partl[a,b] = 255
                if partr_bottom[a,b] > 200 or partr_top[a,b] > 200:
                    partr[a,b] = 255
                
             
        plt.subplot(2,3,2),plt.imshow(partl,cmap = 'gray')
        plt.title('Partl'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,3),plt.imshow(partr,cmap = 'gray')
        plt.title('Partr'), plt.xticks([]), plt.yticks([])      

        picture = partl + partr

        for b in range(int(0.4*width),int(0.7*width),1):
            for a in range(height):
                picture[a,b] = 255
        
        picture = np.invert(picture)
        
        plt.subplot(2,3,4),plt.imshow(picture,cmap = 'gray')
        plt.title('Combined_image'), plt.xticks([]), plt.yticks([])
        plt.show()

        '''
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
        '''
        #threshold_tb(image)

        #path = 'Documents\GitHub\Optic_Dsk\Images_Processed\_CS' + str(i) + '.jpeg'
        
        #image2 = crimmins(image)
        #cv2.imwrite(path, image2) 
        #print('Image ' + str(i) + ' - done')
        
        #image = img
        #equ = cv2.equalizeHist(image)
        #equ = image - image2
        
        """
        for i in range(height):
            for j in range(width):
                if (mask[i,j] != 0 and mask[i,j] != 255):
                    #final[i,j] = mask[i,j]
                    final[i,j] = 255 - final [i,j]
        
        plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
        plt.title('IMG'), plt.xticks([]), plt.yticks([])
        #plt.subplot(1,3,2),plt.imshow(equ,cmap = 'gray')
        #plt.title('EQU'), plt.xticks([]), plt.yticks([]) 
        plt.subplot(1,3,3),plt.imshow(final,cmap = 'gray')
        plt.title('Final'), plt.xticks([]), plt.yticks([])
        plt.show()
        """
        
        """
        crim = cl2
        unsharp = Image.fromarray(crim.astype('uint8'))
        unsharp_class = unsharp.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
        unsharp = np.array(unsharp_class)
        lee = lee_filter(img,10)
        conservative = conservative_smoothing_gray(unsharp,9)
        contrast = cv2.convertScaleAbs(conservative,alpha=1.8,beta=-120)
        
        #dst_col = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray)
        # Convert back to uint8
        noisy = np.uint8(np.clip(img,0,255))

        dst_bw = cv2.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)
        """
    
        cv2.waitKey(0)
    cv2.destroyAllWindows
