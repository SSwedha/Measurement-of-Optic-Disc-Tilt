import cv2
import time
import math
import random
import argparse
import numpy as np
import numpy.matlib as npmatlib
import matplotlib.pyplot as plt
import statistics as stat
from PIL import Image, ImageFilter
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

def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = variance(img)
    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def crimmins(data):
    new_image = data.copy()
    nrow = len(data)
    ncol = len(data[0])
    
    # Dark pixel adjustment
    
    # First Step
    # N-S
    for i in range(1, nrow):
        for j in range(ncol):
            if data[i-1,j] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol-1):
            if data[i,j+1] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow):
        for j in range(1, ncol):
            if data[i-1,j-1] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    #NE-SW
    for i in range(1, nrow):
        for j in range(ncol-1):
            if data[i-1,j+1] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, nrow-1):
        for j in range(ncol):
            if (data[i-1,j] > data[i,j]) and (data[i,j] <= data[i+1,j]):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol-1):
            if (data[i,j+1] > data[i,j]) and (data[i,j] <= data[i,j-1]):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i-1,j-1] > data[i,j]) and (data[i,j] <= data[i+1,j+1]):
                new_image[i,j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i-1,j+1] > data[i,j]) and (data[i,j] <= data[i+1,j-1]):
                new_image[i,j] += 1
    data = new_image
    #Third Step
    # N-S
    for i in range(1, nrow-1):
        for j in range(ncol):
            if (data[i+1,j] > data[i,j]) and (data[i,j] <= data[i-1,j]):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol-1):
            if (data[i,j-1] > data[i,j]) and (data[i,j] <= data[i,j+1]):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i+1,j+1] > data[i,j]) and (data[i,j] <= data[i-1,j-1]):
                new_image[i,j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i+1,j-1] > data[i,j]) and (data[i,j] <= data[i-1,j+1]):
                new_image[i,j] += 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow-1):
        for j in range(ncol):
            if (data[i+1,j] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1,ncol):
            if (data[i,j-1] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(nrow-1):
        for j in range(ncol-1):
            if (data[i+1,j+1] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    # NE-SW
    for i in range(nrow-1):
        for j in range(1,ncol):
            if (data[i+1,j-1] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    
    # Light pixel adjustment
    
    # First Step
    # N-S
    for i in range(1,nrow):
        for j in range(ncol):
            if (data[i-1,j] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol-1):
            if (data[i,j+1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1,nrow):
        for j in range(1,ncol):
            if (data[i-1,j-1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1,nrow):
        for j in range(ncol-1):
            if (data[i-1,j+1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1,nrow-1):
        for j in range(ncol):
            if (data[i-1,j] < data[i,j]) and (data[i,j] >= data[i+1,j]):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol-1):
            if (data[i,j+1] < data[i,j]) and (data[i,j] >= data[i,j-1]):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i-1,j-1] < data[i,j]) and (data[i,j] >= data[i+1,j+1]):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i-1,j+1] < data[i,j]) and (data[i,j] >= data[i+1,j-1]):
                new_image[i,j] -= 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1,nrow-1):
        for j in range(ncol):
            if (data[i+1,j] < data[i,j]) and (data[i,j] >= data[i-1,j]):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1,ncol-1):
            if (data[i,j-1] < data[i,j]) and (data[i,j] >= data[i,j+1]):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i+1,j+1] < data[i,j]) and (data[i,j] >= data[i-1,j-1]):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i+1,j-1] < data[i,j]) and (data[i,j] >= data[i-1,j+1]):
                new_image[i,j] -= 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow-1):
        for j in range(ncol):
            if (data[i+1,j] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1,ncol):
            if (data[i,j-1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(nrow-1):
        for j in range(ncol-1):
            if (data[i+1,j+1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(nrow-1):
        for j in range(1,ncol):
            if (data[i+1,j-1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    return new_image.copy()

def conservative_smoothing_gray(data, filter_size):
    temp = []
    indexer = filter_size // 2
    new_image = data.copy()
    nrow, ncol = data.shape[0:2]
    for i in range(nrow):
        for j in range(ncol):
            for k in range(i-indexer, i+indexer+1):
                for m in range(j-indexer, j+indexer+1):
                    if (k > -1) and (k < nrow):
                        if (m > -1) and (m < ncol):
                            temp.append(data[k,m])
            temp.remove(data[i,j])
            max_value = max(temp)
            min_value = min(temp)
            if data[i,j] > max_value:
                new_image[i,j] = max_value
            elif data[i,j] < min_value:
                new_image[i,j] = min_value
            temp =[]
    return new_image.copy()

"""
def window_function(img):
    min_avg_px = np.zeros(5)
    sum_px = 0
    avg_px = 0.0
    cntr = 0
    i_min = 0
    i_max = 0
    j_min = 0
    j_max = 0
    height,width = img.shape[0:2]
    for k in range(3):
        for l in range(3):
            i_min = math.floor(0.125*width)*k
            i_max = min((math.floor(0.5*width)+math.floor(0.125*width)*k),width)
            
            j_min = math.floor(0.1*height)*l
            j_max = min((math.floor(0.4*height)+math.floor(0.1*height)*l),height)

            sum_px = 0
            avg_px = 0.0
            cntr = 0
      
        for i in range(i_min,i_max):
            for j in range(j_min,j_max):
                sum_px = sum_px + img[j,i]
                cntr = cntr + 1

        if(k==0 and l==0):
            min_avg_px[0] = float(sum_px/cntr)
        avg_px = float(sum_px/cntr)
        if(avg_px<min_avg_px[0]):
            min_avg_px[0] = avg_px
            min_avg_px[1] = i_min
            min_avg_px[2] = i_max
            min_avg_px[3] = j_min
            min_avg_px[4] = j_max
        
    w_img = np.zeros((int(min_avg_px[4]-min_avg_px[3]),int(min_avg_px[2] - min_avg_px[1])))      
    for i in range(int(min_avg_px[1]),int(min_avg_px[2])):
        for j in range(int(min_avg_px[3]),int(min_avg_px[4])):
            w_img[(j-int(min_avg_px[3])),(i-int(min_avg_px[1]))] = img[j,i]
    
    cv2.imwrite('window6_open.jpeg',w_img)
    cv2.waitKey(0)
"""

def nothing(x):
    pass

def write_image(path, img):
    # img = img*(2**16-1)
    # img = img.astype(np.uint16)
    # img = img.astype(np.uint8)
    cv2.imwrite(path,img)
    # Convert the scale (values range) of the image
    #img = cv2.convertScaleAbs(img, alpha=(255.0))
    # Save file
    #plt.savefig(path, bbox_inches='tight')#, img, format = 'png')

def find_eyeside(img, i):
    #side = input('Enter the Optic-Disk Side (R or L): ')
    #side = 'r'
    kernel = kernel = np.ones((25,25),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    m, n = img.shape[0:2]
    min_intensity = 255
    for a in range(m):
        for b in range(n):
            if opening[a,b] < min_intensity:
                min_intensity = opening[a,b]
                x = a
                y = b
    if y > n/2:
        side = 'r'
    elif y <= n/2:
        side = 'l'  
    return side

def crop(img, i):
    # Returns dimensions of image
    height, width = img.shape[0:2]
    # Top and Bottom range
    startrow = int(height*.12)
    endrow = int(height*.88)
    startcol = int(width*.05)
    endcol = int(width*0.95)
    img = img[startrow:endrow, startcol:endcol]

    height, width = img.shape[0:2]
    # Finds the side of the Optic Disk
    side = find_eyeside(img, i)
    # Left and Right range 
    if side == 'r':
        startcol = int(width*.40)
        endcol = int(width-10)
    elif side == 'l':
        startcol = int(10)
        endcol = int(width*.60)
    image = img[0:height, startcol:endcol]
    height, width = img.shape[0:2]
    image = image[20:520, 20:520]
    return image

def segment(img, blur=0.01, alpha=0.1, beta=0.1, gamma=0.001):
    
    height, width = img.shape[0:2]
    # Initial contour
    s = np.linspace(0, 2*np.pi, 400)
    #r = int(height/2*np.ones(len(s)) + height*np.sin(s)/2)
    #c = int(width/2*np.ones(len(s)) + width*np.cos(s)/2)
    row = int(height/2)
    column = int(width/2)
    r = row + (row-10)*np.sin(s)
    c = column + (column-10)*np.cos(s)
    init = np.array([r, c]).T

    # Parameters for Active Contour
    blur = 0.01
    alpha = 0.1
    beta = 0.1
    gamma = 0.001
    w_line = -5
    w_edge = 0

    # Active contour
    snake = active_contour(gaussian(img, blur), init, alpha, beta, gamma, coordinates='rc')
    #snake = active_contour(gaussian(img, 1), init, alpha, beta, w_line, w_edge, gamma, coordinates='rc')
    # boundary_condition='fixed' blur = 1, alpha = 0.1, beta=1, w_line = -5, w_edge=0, gamma = 0.1
    # Display the image with contour
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()

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

def filterimage(img):
    #cv2.imshow('i',img)
    #croppedImage = img[startRow:endRow, startCol:endCol]
     
    kernel = np.ones((5,5),np.uint8)

    dilation = cv2.dilate(img,kernel,iterations = 1)
    blur = dilation
    #blur = cv2.GaussianBlur(dilation,(5,5),0)
    #blur = cv2.blur(dilation,(5,5))
    #plt.imshow(blur,cmap = 'gray')
    #plt.show()
    #dilation = cv2.dilate(blur,kernel,iterations = 1)
    #blur = cv2.GaussianBlur(dilation,(5,5),0)
    #blur = cv2.GaussianBlur(im,(5,5),0)

    erosion = cv2.erode(blur,kernel,iterations = 1)

    #blur = cv2.blur(erosion,(10,10),0)
    plt.imshow(blur,cmap = 'gray')
    plt.show()

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    dil = cv2.erode(gradient,kernel,iterations = 1) 

    plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,3,2),plt.imshow(erosion,cmap = 'gray')
    plt.title('Erosion'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,3,3),plt.imshow(dilation,cmap = 'gray')
    plt.title('Dilation'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,3,4),plt.imshow(opening,cmap = 'gray')
    plt.title('Opening'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,3,5),plt.imshow(dil,cmap = 'gray')
    plt.title('Dil'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,3,6),plt.imshow(gradient,cmap = 'gray')
    plt.title('Gradient'), plt.xticks([]), plt.yticks([])
    plt.show()

    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    sobel = cv2.addWeighted(np.absolute(sobelx), 0.5, np.absolute(sobely), 0.5, 0)
    #sobel = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)
    sobel = cv2.erode(sobel,kernel,iterations = 1)

    plt.subplot(2,2,1),plt.imshow(sobel,cmap = 'gray')
    plt.title('Sobel'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()

def removeBloodVessels(img,th_0,th_1):
    globalMin = 0 
    globalMax = 255   # it is the maximum gray level of the image

    img_flat = img.flatten()
    localMin = np.min(img_flat[np.nonzero(img_flat)])   # minimum non-zero intensity value of the taken image
    localMax = max(img_flat)   # maximum intensity value of the taken image

    m, n = img.shape[0:2]
    kernel1 = np.ones((25,25),np.uint8)
    kernel0 = np.zeros((25,25),np.uint8)
    copy_img = np.zeros((m,n))
    copy_img = img.copy()
    plt.subplot(2,3,4),plt.imshow(copy_img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    for i in range(m):
        for j in range(n):
            copy_img[i,j] = ((copy_img[i,j]-localMin)* ((globalMax-globalMin))/((localMax-localMin))) + globalMin 
    closing1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
    closing0 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel0)

    bottomHat1 = closing1 - copy_img
    bottomHat0 = closing0 - copy_img

    for i in range(m):
        for j in range(n):
            if bottomHat0[i,j] < th_0:
                bottomHat0[i,j] = 0
            else:
                bottomHat0[i,j] = 255
            if bottomHat1[i,j] < th_1:
                bottomHat1[i,j] = 0
    ''' 
    alpha = 2
    beta = -125
    bottomHat1 = cv2.convertScaleAbs(bottomHat1, alpha=alpha, beta=beta)
    new_greyimage = np.zeros((m,n),np.uint8)
    
    for i in range(m):
        for j in range(n):
            if (bottomHat0[i,j] == 255):
                new_greyimage[i,j] = bottomHat1[i,j]
    '''

    for i in range(m):
        for j in range(n):
            copy_img[i,j] = img[i,j] + bottomHat1[i,j]
            #if(copy_img[i,j]>255):
            #    copy_img[i,j] = 255
                
    plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')
    plt.title('crim'), plt.xticks([]), plt.yticks([])        
    plt.subplot(2,3,2),plt.imshow(bottomHat1,cmap = 'gray')
    plt.title('bottomHat1'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,3,3),plt.imshow(copy_img,cmap = 'gray')
    plt.title('crim+bottomHat1'), plt.xticks([]), plt.yticks([])
    plt.show()
    
"""
def removeBloodVessels(img, th0, th1, index):
    print('removingBloodVessels')
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
    cv2.waitKey(0)
    closing1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
    closing0 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel0)

    #cv2.imshow('closing0',closing0)
    #cv2.waitKey(0)
    #cv2.imshow('closing1',closing1)
    #cv2.waitKey(0)    
    
    bottomHat1 = closing1 - img
    bottomHat0 = closing0 - img
    
    #cv2.imshow('bottomHat0',bottomHat0)
    #cv2.waitKey(0)
    for i in range(m):
        for j in range(n):
            if (bottomHat0[i,j] < th0):
                bottomHat0[i,j] = 0
            else:
                bottomHat0[i,j] = 255
            if bottomHat1[i,j] < th1:
                bottomHat1[i,j] = 0
    #cv2.imshow('bottomHat0 after for',bottomHat0)
    #cv2.waitKey(0)
    #cv2.imshow('bottomHat1 after for',bottomHat1)
    #cv2.waitKey(0)                      

    alpha = 2
    beta = -125
    bottomHat1 = cv2.convertScaleAbs(bottomHat1, alpha=alpha, beta=beta)
    #cv2.imshow('bottomHat1 after contrast',bottomHat1)
    #cv2.waitKey(0)
    new_greyimage = np.zeros((m,n),np.uint8)
    for i in range(m):
        for j in range(n):
            if (bottomHat0[i,j] == 255):
                new_greyimage[i,j] = bottomHat1[i,j]
    cv2.imshow('grey',new_greyimage)
    v = 'Documents\GitHub\Optic_Disk\RemoveVessels\img_'+ str(index) + '-' + str(th0) + '_' + str(th1) + '.jpg'
    write_image(v,new_greyimage)
    
    cv2.waitKey(0)
    #enhancedVessel1 = meijering(new_greyimage)
    #enhancedVessel2 = frangi(new_greyimage)
    #new_greyimage = cv2.addWeighted(enhancedVessel1,0.5,enhancedVessel2,0.5,0.0)
    #cv2.imshow('fuse',new_greyimage)
    #cv2.waitKey(0)
"""

def trackbar(image):
    cv2.namedWindow('Contrast')

    # create trackbars for color change
    cv2.createTrackbar('alpha','Contrast', 0, 255, nothing)
    cv2.createTrackbar('beta','Contrast', 0, 255, nothing)
    cv2.createTrackbar('Threshold','Contrast', 0, 255, nothing)
    cv2.createTrackbar('Type','Contrast', 0, 4, nothing)
    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'Contrast',0,1,nothing)
        
    # get current positions of four trackbars
       
    while(1):
        alpha = cv2.getTrackbarPos('alpha','Contrast')
        beta = cv2.getTrackbarPos('beta','Contrast')
        threshold = cv2.getTrackbarPos('Threshold','Contrast')
        th_type = cv2.getTrackbarPos('Type','Contrast')
        s = cv2.getTrackbarPos(switch,'Contrast')
        new_image1 = cv2.convertScaleAbs(image, alpha=alpha/100.0, beta=-1*beta)
        cv2.imshow('image',new_image1)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        #0: Binary
        #1: Binary Inverted
        #2: Threshold Truncated
        #3: Threshold to Zero
        #4: Threshold to Zero Inverted
        _, dst = cv2.threshold(image, threshold, 255, th_type)
        cv2.imshow('Threshold', dst)


if __name__=="__main__":
    # Image files location 
    location = 'Documents\GitHub\Optic_Disk\Images\_OD'

    # Loop through all the images
    for i in range(1, 15):
        
        image = location + str(i) + '.jpeg'   # Filename
        #image = 'Documents\GitHub\Optic_Disk\BottomHat.jpg'
        img = cv2.imread(image,0)             # Read image
        img = crop(img, i)                    # Crop the image
        #img = cv2.imread('Documents\GitHub\Optic_Disk\OD.jpeg',0)
        image = img
        height, width = img.shape[0:2]
        
        image2 = crimmins(image)
        equ = cv2.equalizeHist(image2)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(image2)
        equ2 = crimmins(equ)
        cl2 = crimmins(cl1)
        
        """
        crim = cl2
        unsharp = Image.fromarray(crim.astype('uint8'))
        unsharp_class = unsharp.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
        unsharp = np.array(unsharp_class)
        lee = lee_filter(img,10)
        conservative = conservative_smoothing_gray(unsharp,9)
        contrast = cv2.convertScaleAbs(conservative,alpha=1.8,beta=-120)
        """
        
        removeBloodVessels(equ2,110,10)
        removeBloodVessels(cl2,110,10)

        plt.subplot(2,3,1),plt.imshow(image,cmap = 'gray')
        plt.title('IMG'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,2),plt.imshow(equ,cmap = 'gray')
        plt.title('EQU'), plt.xticks([]), plt.yticks([])   
        plt.subplot(2,3,3),plt.imshow(cl1,cmap = 'gray')
        plt.title('Cl1'), plt.xticks([]), plt.yticks([]) 
        plt.subplot(2,3,4),plt.imshow(image2,cmap = 'gray')
        plt.title('IMG'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,5),plt.imshow(equ2,cmap = 'gray')
        plt.title('EQU'), plt.xticks([]), plt.yticks([])   
        plt.subplot(2,3,6),plt.imshow(cl2,cmap = 'gray')
        plt.title('Cl1'), plt.xticks([]), plt.yticks([])    
        plt.show()
        
        #trackbar(image)

        """
        image = crimmins(image)
        #alpha = 1.8
        #beta = -125
        #new_image1 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        #new_image1 = cv2.GaussianBlur(new_image1,(5,5),8)
        alpha = 1.8
        beta = -125
        new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        new_image = cv2.GaussianBlur(new_image,(5,5),8)
        #alpha = 5
        #beta = -1000
        #new_image3 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        #new_image3 = cv2.GaussianBlur(new_image3,(5,5),8)
        #new_image = cv2.blur(new_image,(5,5))

        kernel = np.ones((5,5),np.uint8)

        dilation = cv2.dilate(image,kernel,iterations = 1)
        blur = dilation

        opening = cv2.morphologyEx(new_image, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(new_image, cv2.MORPH_CLOSE, kernel)
        gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
        dil = cv2.dilate(gradient,kernel,iterations = 1) 
        edge1 = auto_canny(opening)
        edge2 = auto_canny(closing)

        sobelx = cv2.Sobel(new_image,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(new_image,cv2.CV_64F,0,1,ksize=5)
        sobel = cv2.addWeighted(np.absolute(sobelx), 0.5, np.absolute(sobely), 0.5, 0)
        """
        image_filter = img
        """
        for a in range(height):
            for b in range(width):
                val = new_image2[a,b] - gradient[a,b]
                #val = new_image2[a,b] - sobel[a,b]
                if val < 0:
                    image_filter[a,b] = 0
                elif val > 255:
                    image_filter[a,b] = 255
                else:
                    image_filter[a,b] = val
        """
        """
        plt.subplot(2,3,1),plt.imshow(image_filter,cmap = 'gray')
        plt.title('Filter'), plt.xticks([]), plt.yticks([])
        #plt.subplot(2,3,2),plt.imshow(new_image1,cmap = 'gray')
        #plt.title('Contrast1'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,2),plt.imshow(new_image,cmap = 'gray')
        plt.title('Contrast'), plt.xticks([]), plt.yticks([])
        #plt.subplot(2,3,3),plt.imshow(new_image,cmap = 'gray')
        #plt.title('Invert'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,4),plt.imshow(edge1,cmap = 'gray')
        plt.title('Canny'), plt.xticks([]), plt.yticks([])
        #plt.subplot(3,3,6),plt.imshow(edge2,cmap = 'gray')
        #plt.title('Canny-Close'), plt.xticks([]), plt.yticks([])
        #plt.subplot(3,3,5),plt.imshow(opening,cmap = 'gray')
        #plt.title('Opening'), plt.xticks([]), plt.yticks([])
        #plt.subplot(3,3,6),plt.imshow(closing,cmap = 'gray')
        #plt.title('Closing'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,5),plt.imshow(gradient,cmap = 'gray')
        plt.title('Gradient'), plt.xticks([]), plt.yticks([])    
        plt.subplot(2,3,6),plt.imshow(sobel,cmap = 'gray')
        plt.title('Sobel'), plt.xticks([]), plt.yticks([])    
        plt.show()

        segment(image)
        segment(new_image)
        segment(image_filter)
        segment(gradient)
        """
        """
        img1 = new_image1
        img2 = new_image2
        img3 = new_image3
        img4 = gradient
        img5 = sobel
        #removeBloodVessels(img,110,10)
        #removeBloodVessels(img1,30,110)
        #for a in range(10,250,10):
            #for b in range(10,250,10):
                #removeBloodVessels(img2,a,b,i)
                #removeBloodVessels(img3,a,b,i)
                #removeBloodVessels(img4,a,b,i)
                #removeBloodVessels(img5,a,b,i)
        
        #v = 'img_'+str(i)
        #cv2.imshow(v,img)
        #cv2.imshow('new',new_image)
        #filterimage(img)
        
        segment(img1)
        segment(img2)
        segment(img4)
        """
        #segment(edge1)
        #segment(edge2)
        # cv2.imshow('canny',edge)
        # Wait for keystroke
        #FindBloodVesselPoint2(img)
        
        cv2.waitKey(0)
    
    cv2.destroyAllWindows
