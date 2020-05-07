import cv2
import time
import math
import random
import numpy as np
import numpy.matlib as npmatlib
import statistics as stat
import matplotlib.pyplot as plt
from skimage import io
from skimage import data
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.segmentation import active_contour

def write_image(path, img):
  # img = img*(2**16-1)
  # img = img.astype(np.uint16)
  # img = img.astype(np.uint8)

  # Convert the scale (values range) of the image
  img = cv2.convertScaleAbs(img, alpha=(255.0))
  # Save file
  plt.savefig(path, bbox_inches='tight')#, img, format = 'png')

def find_eyeside(img, i):
  #side = input('Enter the Optic-Disk Side (R or L): ')
  #side = 'r'
  if i%2:
    side = 'r'
  else:
    side = 'l'
  """
  if side == 'R':
    side = 'r'
  elif side == 'L':
    side = 'l'
  """
  return side

def crop(img, i):
  # Returns dimensions of image
  height, width = img.shape[0:2]
  
  # Top and Bottom range
  startrow = int(height*.10)
  endrow = int(height*.90)

  # Finds the side of the Optic Disk
  side = find_eyeside(img, i)

  # Left and Right range 
  if side == 'r':
    startcol = int(width*.40)
    endcol = int(width*.95)
  elif side == 'l':
    startcol = int(width*.05)
    endcol = int(width*.60)
  
  # Crop
  cropimg = img[startrow:endrow, startcol:endcol]
  return cropimg

def segment(img, blur=0.01, alpha=0.1, beta=0.1, gamma=0.001):
  height, width = img.shape[0:2]
  # Initial contour
  s = np.linspace(0, 2*np.pi, 400)
  # print(s)
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

def FindBloodVesselPoint2(M,N,BloodVessel,L,cx,cy,Eye_side):
  Edges = np.zeros(M,N) 
  #[x,y] = meshgrid(list(range(N)),list(range(M))) 
  x, y  = np.meshgrid(list(range(N)),list(range(M)))
  uperHalfL = round(L/2) 
  downHalfL = L - uperHalfL 
  flag = 0 
  x1 = cx 
  y1 = max(0,cy-uperHalfL) 
  y2 = y1+L 
  if Eye_side == 'R':
    a = 1 
  else:
    a = -1 
  max_pixl = np.zeros(1,0) 
  while(flag==0):
    Img_square = np.zeros(M,N) 
    x2 = x1+a*L 
    if x2>N or x2<0:
      flag = 1 
    else:
      xx1 = min(x1,x2) 
      xx2 = max(x1,x2) 
      for i in range(M):
        for j in range(N):
          if x[i][j]>=xx1 and x[i][j]<=xx2 and y[i][j]>=y1 and y[i][j]<=y2:
            #%Img_square(j,i) = 1 
            Img_square[i][j] = 1 
          #end
        #end
      #end
      Edges = Edges + edge(Img_square,'sobel') 
      point = np.zeros(M,N)
      for i in range(M):
        for j in range(N):
          point[i][j] = BloodVessel[i][j]*Img_square[i][j] 
      # max_pixl = [max_pixl,nnz(point)] 
      max_pixl = [max_pixl,np.count_nonzero(point)] 
      x1 = x2 
    #end
  #end
  I = max(max_pixl) 
  opt_square = np.zeros(M,N) 
  x1 = cx+a*L*(I-1) 
  x2 = x1+a*L 
  xx1 = min(x1,x2) 
  xx2 = max(x1,x2) 
  for i in range(M):
    for j in range(N):
      if x[i][j]>=xx1 and x[i][j]<=xx2 and y[i][j]>=y1 and y[i][j]<=y2:
        opt_square[i][j] = 1 
      #end
    #end
  #end
  return [Edges,I,opt_square,max_pixl]

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

def removeBloodVessels(img):
  m, n = img.shape[0:2]
  kernel1 = np.ones((25,25),np.uint8)
  kernel0 = np.zeros((25,25),np.uint8)
  closing1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
  closing0 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel0)

  bottomHat1 = closing1 - img
  bottomHat0 = closing0 - img

  for i in range(m):
    for j in range(n):
      if bottomHat1[i,j] < 60:
        bottomHat1 = 0
  
  alpha = 2
  beta = -125
  contrast_bottomHat1 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
  new_greyimage = np.zeros(m,n)
  for i in range(m):
    for j in range(n):
      new_greyimage[i,j] = bottomHat1[i,j]


if __name__=="__main__":

  # Image files location 
  location = 'Documents\GitHub\Optic_Disk\Images\_OD'
  
  # Loop through all the images
  for i in range(1, 15):
    image = location + str(i) + '.jpeg'   # Filename
    img = cv2.imread(image)               # Read image
    img = crop(img, i)                   # Crop the image
    image = img
    img = rgb2gray(img)                   # Convert to binary/greyscale
    alpha = 1.8
    beta = -125
    new_image1 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    new_image1 = cv2.GaussianBlur(new_image1,(5,5),8)
    alpha = 2
    beta = -125
    new_image2 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    new_image2 = cv2.GaussianBlur(new_image2,(5,5),8)
    alpha = 5
    beta = -1000
    new_image3 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    new_image3 = cv2.GaussianBlur(new_image3,(5,5),8)
    #new_image = cv2.blur(new_image,(5,5))

    kernel = np.ones((5,5),np.uint8)

    dilation = cv2.dilate(image,kernel,iterations = 1)
    blur = dilation

    opening = cv2.morphologyEx(new_image1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(new_image1, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
    dil = cv2.dilate(gradient,kernel,iterations = 1) 
    edge1 = auto_canny(opening)
    edge2 = auto_canny(closing)

    sobelx = cv2.Sobel(new_image1,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(new_image1,cv2.CV_64F,0,1,ksize=5)
    sobel = cv2.addWeighted(np.absolute(sobelx), 0.5, np.absolute(sobely), 0.5, 0)

    plt.subplot(3,3,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,2),plt.imshow(new_image1,cmap = 'gray')
    plt.title('Contrast1'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,3),plt.imshow(new_image2,cmap = 'gray')
    plt.title('Contrast2'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,4),plt.imshow(new_image3,cmap = 'gray')
    plt.title('Contrast3'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,5),plt.imshow(edge1,cmap = 'gray')
    plt.title('Canny-Open'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,6),plt.imshow(edge2,cmap = 'gray')
    plt.title('Canny-Close'), plt.xticks([]), plt.yticks([])
    #plt.subplot(3,3,5),plt.imshow(opening,cmap = 'gray')
    #plt.title('Opening'), plt.xticks([]), plt.yticks([])
    #plt.subplot(3,3,6),plt.imshow(closing,cmap = 'gray')
    #plt.title('Closing'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,7),plt.imshow(gradient,cmap = 'gray')
    plt.title('Gradient'), plt.xticks([]), plt.yticks([])    
    plt.subplot(3,3,8),plt.imshow(sobel,cmap = 'gray')
    plt.title('Sobel'), plt.xticks([]), plt.yticks([])    

    plt.show()
    img1 = rgb2gray(new_image1)
    img2 = rgb2gray(new_image2)
    img3 = rgb2gray(new_image3)
    img4 = rgb2gray(gradient)
    img5 = rgb2gray(sobel)

    #cv2.imshow('img',img)
    #cv2.imshow('new',new_image)
    #filterimage(img)
    segment(img1)
    segment(img2)
    segment(edge1)
    segment(edge2)
    segment(img4)
    # cv2.imshow('canny',edge)
    # Wait for keystroke
    #FindBloodVesselPoint2(img)
    
    cv2.waitKey(0)
  
  cv2.destroyAllWindows
