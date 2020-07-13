from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from PIL import Image, ImageFilter

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
            if(copy_img[i,j]>255):
                copy_img[i,j] = 255

    plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')
    plt.title('crim'), plt.xticks([]), plt.yticks([])        
    plt.subplot(2,3,2),plt.imshow(bottomHat1,cmap = 'gray')
    plt.title('bottomHat1'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,3,3),plt.imshow(copy_img,cmap = 'gray')
    plt.title('crim+bottomHat1'), plt.xticks([]), plt.yticks([])
    plt.show()

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
def find_eyeside(img):
    #side = input('Enter the Optic-Disk Side (R or L): ')
    #side = 'r'
    kernel = kernel = np.ones((25,25),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #window_function(opening)
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

def crop(img):
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
  side = find_eyeside(img)
  # Left and Right range 
  if side == 'r':
    startcol = int(width*.40)
    endcol = int(width-10)
  elif side == 'l':
    startcol = int(10)
    endcol = int(width*.60)
  image = img[0:height, startcol:endcol]
  img_crop = np.zeros((500,500))
  for i in range(500):
    for j in range(500):
      img_crop[i,j] = image[i,j]
  return img_crop
        
img = cv2.imread('w3.jpeg',0)
img = crop(img)
crim = crimmins(img)
lee = lee_filter(crim,10)
unsharp = Image.fromarray(lee.astype('uint8'))
unsharp_class = unsharp.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
unsharp=np.array(unsharp_class)
conservative = conservative_smoothing_gray(unsharp,9)
contrast = cv2.convertScaleAbs(conservative,alpha=1.8,beta=-125)
# Convert back to uint8
#le = lee_filter()
removeBloodVessels(crim,110,10)
#PLOTS
plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(crim,cmap = 'gray')
plt.title('Crimmins'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(lee,cmap = 'gray')
plt.title('Lee'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(unsharp,cmap = 'gray')
plt.title('Unsharp'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(conservative,cmap = 'gray')
plt.title('Conservative'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(contrast,cmap = 'gray')
plt.title('Contrast'), plt.xticks([]), plt.yticks([])
plt.show()

#window_function(new_image1)
