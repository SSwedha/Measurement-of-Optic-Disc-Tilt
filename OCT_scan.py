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
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split 

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
    image = img[0:height, startcol:endcol]
    height, width = img.shape[0:2]
    image = image[20:520, 20:520]
    return image

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

def linear_regression(Xl,Xr):
    nl = Xl.shape[0]
    nr = Xr.shape[0]
  
    # mean of x and y vector 
    m_xl, m_yl = np.mean(Xl[:,0]), np.mean(Xl[:,1]) 
    m_xr, m_yr = np.mean(Xr[:,0]), np.mean(Xr[:,1]) 
  
    # calculating cross-deviation and deviation about Xl and Xr
    SS_xy_l = np.sum(Xl[:,1]*Xl[:,0]) - nl*m_yl*m_xl 
    SS_xx_l = np.sum(Xl[:,0]*Xl[:,0]) - nl*m_xl*m_xl 
    
    SS_xy_r = np.sum(Xr[:,1]*Xr[:,0]) - nr*m_yr*m_xr 
    SS_xx_r = np.sum(Xr[:,0]*Xr[:,0]) - nr*m_xr*m_xr 

    # calculating regression coefficients 
    bl = np.zeros((2))
    br = np.zeros((2))
    bl[1] = SS_xy_l / SS_xx_l
    bl[0] = m_yl - bl[1]*m_xl
    
    br[1] = SS_xy_r / SS_xx_r
    br[0] = m_yr - br[1]*m_xr

    return bl, br


if __name__=="__main__":
    
    # Image files location 
    location1 = 'OCT_scan'
    #location2 = 'Documents\GitHub\Optic_Disk\Images_Processed\_OD'
    # Loop through all the images

    for i in range(24, 29):

        image1 = location1 + str(i) + '.jpeg'    # Filename
        #image2 = location2 + str(i) + '.jpg'    # Filename
        img1 = cv2.imread(image1,0)              # Read image
        #img2 = cv2.imread(image2,0)             # Read image
        #img1 = cv2.medianBlur(img1,15)
        img1 = cv2.GaussianBlur(img1,(5,5),0)
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

        Xl = np.zeros((height*width,2))
        wpx_l = 0
        
        Xr = np.zeros((height*width,2))
        wpx_r = 0        
        

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
        partl_og = partl.copy()
        partr_og = partr.copy()
        
        kernel = np.ones((5,5),np.uint8)
        erosion_l = cv2.erode(partl,kernel,iterations = 1)
        dilation_l = cv2.dilate(erosion_l,kernel,iterations = 2)
        partl = dilation_l
        
        erosion_r = cv2.erode(partr,kernel,iterations = 1)
        dilation_r = cv2.dilate(erosion_r,kernel,iterations = 2)
        partr = dilation_r
        
        for b in range(width):
            for a in range(height):
                if partl[a,b] > 200:
                    Xl[wpx_l,0] = a  #column 0 has x coord of pixels which are white
                    Xl[wpx_l,1] = b  #column 1 has y coord of pixels which are white
                    wpx_l = wpx_l + 1 #counter to know the index of row
                    
                if partr[a,b] > 200:
                    Xr[wpx_r,0] = a
                    Xr[wpx_r,1] = b
                    wpx_r = wpx_r + 1

        print(wpx_l,wpx_r)

        Xl = np.delete(Xl, slice(wpx_l,height*width), 0) #to delete the unnecessary rows
        Xr = np.delete(Xr, slice(wpx_r,height*width), 0)
        
        print(Xl.shape[0:2])
        print(Xr.shape[0:2])

        Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl[:,1].reshape((-1,1)), Xl[:,0], test_size=0.2, 
                                                    random_state=1) 
        
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr[:,1].reshape((-1,1)), Xr[:,0], test_size=0.2, 
                                                    random_state=1) 

        # create linear regression object 
        reg_l = linear_model.LinearRegression() 
        
        # train the model using the training sets 
        reg_l.fit(Xl_train, yl_train) 

        ml = reg_l.coef_[0]
        cl = reg_l.intercept_
        
        yl_1 = cl + ml*1
        yl_2 = cl + ml*(width-1)

         # create linear regression object 
        reg_r = linear_model.LinearRegression() 
        
        # train the model using the training sets 
        reg_r.fit(Xr_train, yr_train) 

        mr = reg_r.coef_[0]
        cr = reg_r.intercept_
        
        yr_1 = cr + mr*1
        yr_2 = cr + mr*(width-1)

        picture = partl_og + partr_og
        picture_color = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        picture_l = cv2.line(picture_color, (1,int(yl_1)),(int(width-1),int(yl_2)), (0,255,0),3)
        picture_r = cv2.line(picture_l, (1,int(yr_1)),(int(width-1),int(yr_2)), (0,0,255),3)
        theta = math.degrees(math.atan(ml)-math.atan(mr))
        print('theta: ',theta)
        cv2.imshow('pic'+ str(i),picture_r)      

       
        '''        
        bl, br =linear_regression(Xl,Xr) 
        print(bl, br)

        yl_pred = np.zeros(2)
        yr_pred = np.zeros(2)

        yl_pred[0] = bl[0] + bl[1]*1
        int(width/2) = bl[0] + bl[1]*x

        yr_pred[0] = br[0] + bl[1]*int(width/2)
        yr_pred[1] = br[0] + bl[1]*(width-1)
        
        picture = partl + partr
        picture1 = cv2.line(picture,(1,int(yl_pred[0])),(int(width/2),int(yl_pred[1])),(0,255,0),3)
        picture2 = cv2.line(picture1,(int(width/2),int(yr_pred[0])),((width-1),int(yr_pred[1])),(0,255,0),3)
        
        cv2.imshow('picture with lines', picture2)

        print(partl.shape[0:2],partr.shape[0:2])
        '''
        '''
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(sobely,kernel,iterations = 2)
        dilation = cv2.dilate(erosion,kernel,iterations = 3)
        opening = cv2.morphologyEx(sobely, cv2.MORPH_OPEN, kernel)
        #closing = cv2.morphologyEx(picture, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('ero-dil', dilation)
        #cv2.imshow('ero', erosion)
        cv2.imshow('ope', opening)
        #cv2.imshow('clo', closing)
        cv2.waitKey(0) 
        '''

        

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
        
        '''
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
        '''
        
        ''''
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
        '''
    
        cv2.waitKey(0)
    cv2.destroyAllWindows
