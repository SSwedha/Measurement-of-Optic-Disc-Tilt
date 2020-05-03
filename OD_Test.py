import cv2
import time
import math
import random
import numpy as np
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

def segment(img):
  height, width = img.shape[0:2]

  # Initial contour
  s = np.linspace(0, 2*np.pi, 400)
  r = int(0.5*height + 0.5*height*np.sin(s))
  c = int(0.5*width + 0.5*width*np.cos(s))
  init = np.array([r, c]).T

  # Parameters for Active Contour
  blur = 0.01
  alpha = 0.1
  beta = 0.1
  gamma = 0.001

  # Active contour
  snake = active_contour(gaussian(img, blur), init, alpha, beta, gamma, coordinates='rc')

  # Display the image with contour
  fig, ax = plt.subplots(figsize=(7, 7))
  ax.imshow(img, cmap=plt.cm.gray)
  ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
  ax.set_xticks([]), ax.set_yticks([])
  ax.axis([0, img.shape[1], img.shape[0], 0])
  plt.show()

def filter(img):
  #cv2.imshow('i',img)
  #croppedImage = img[startRow:endRow, startCol:endCol]
  """
  im = img.astype(np.uint8)
  edges = cv2.Canny(im, 20, 30)
  cv2.imshow('canny', edges)"""

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

if __name__=="__main__":

  # Image files location 
  location = 'Documents\GitHub\Optic_Disk\Images\_OD'
  
  # Loop through all the images
  for i in range(11,15):
    image = location + str(i) + '.jpeg'   # Filename
    img = cv2.imread(image)               # Read image
    img = rgb2gray(img)                   # Convert to binary/greyscale
    img = crop(img, i)                    # Crop the image
    cv2.imshow('img',img)
    filter(img)

    # Wait for keystroke
    cv2.waitKey(0)
  
  cv2.destroyAllWindows
