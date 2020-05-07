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

"""
Function to save images to a folder
  Input   : (string) location path, (numpy array) image
  Output  : -
"""
def write_image(path, img):
  # img = img*(2**16-1)
  # img = img.astype(np.uint16)
  # img = img.astype(np.uint8)

  # Convert the scale (values range) of the image
  img = cv2.convertScaleAbs(img, alpha=(255.0))
  # Save file
  plt.savefig(path, bbox_inches='tight')#, img, format = 'png')

"""
Find the side of the Optic Disk in the image
  Input   : (numpy array) image
  Output  : (character) side
"""
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

"""
Crop the image based on the location of the optic disk
  Input   : (numpy array) image
  Output  : (numpy image) cropped image
"""
def crop(img, i):
  # Returns dimensions of image
  height, width = img.shape[0:2]
  
  # Top and Bottom range
  startrow = int(height*.10)
  endrow = int(height*.90)

  startcol = int(width*.05)
  endcol = int(width*.95)

  # Finds the side of the Optic Disk
  side = find_eyeside(img, i)
  """
  # Left and Right range 
  if side == 'r':
    startcol = int(width*.40)
    endcol = int(width*.95)
  elif side == 'l':
    startcol = int(width*.05)
    endcol = int(width*.60)
  """
  # Crop
  cropimg = img[startrow:endrow, startcol:endcol]
  return cropimg

"""
Function for Optic Disk Segmentation
"""
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

"""
Function for prepossing the image
"""
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

def meshgrid(x,y,z):
  if (nargin == 0 or (nargin > 1 and nargout > nargin)):
    print('meshgrid:NotEnoughInputs') 
  #end

  if (nargin == 2 or (nargin == 1 and nargout < 3)): #% 2-D array case
    if nargin == 1:
      y = x 
    #end
    if isempty(x) or isempty(y):
      xx = np.zeros(0)
      yy = np.zeros(0) 
    else:
      xrow = np.transpose(np.transpose(x.flatten()))  #% Make sure x is a full row vector.
      ycol = np.transpose(y.flatten())                #% Make sure y is a full column vector.
      xx = npmatlib.repmat(xrow,size(ycol)) 
      yy = npmatlib.repmat(ycol,size(xrow)) 
    #end
  else:  #% 3-D array case
    if nargin == 1:
      y = x 
      z = x 
    #end
    if isempty(x) or isempty(y) or isempty(z):
      xx = np.zeros(0) 
      yy = np.zeros(0) 
      zz = np.zeros(0) 
    else:
      nx = numel(x) 
      ny = numel(y) 
      nz = numel(z) 
      xx = np.reshape(x,[1,nx,1])  #% Make sure x is a full row vector.
      yy = np.reshape(y,[ny,1,1])  #% Make sure y is a full column vector.
      zz = np.reshape(z,[1,1,nz])  #% Make sure z is a full page vector.
      xx = npmatlib.repmat(xx, ny, 1, nz) 
      yy = npmatlib.repmat(yy, 1, nx, nz) 
      zz = npmatlib.repmat(zz, ny, nx, 1) 
    #end
  #end
  return [xx, yy, zz]

def FindBloodVesselPoint2(M,N,BloodVessel,L,cx,cy,Eye_side):
  Edges = np.zeros(M,N) 
  [x,y] = meshgrid(list(range(N)),list(range(M))) 
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
      max_pixl = [max_pixl,nnz(point)] 
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

"""
Trapezoidal membership function generator.

  Input Parameters
  -----------------
    x : 1d array
      Independent variable.
    abcd : 1d array, length 4
      Four-element vector.  Ensure a <= b <= c <= d.

  Returns
  -------
    y : 1d array
      Trapezoidal membership function.
"""
def trapmf(x, abcd):
  assert len(abcd) == 4, 'abcd parameter must have exactly four elements.'
  a, b, c, d = np.r_[abcd]
  assert a <= b and b <= c and c <= d, 'abcd requires the four elements a <= b <= c <= d.'
  y = np.ones(len(x))

  idx = np.nonzero(x <= b)[0]
  y[idx] = trimf(x[idx], np.r_[a, b, b])

  idx = np.nonzero(x >= c)[0]
  y[idx] = trimf(x[idx], np.r_[c, c, d])

  idx = np.nonzero(x < a)[0]
  y[idx] = np.zeros(len(idx))

  idx = np.nonzero(x > d)[0]
  y[idx] = np.zeros(len(idx))

  return y

"""
Triangular membership function generator.

  Input Parameters
  ----------------
    x : 1d array
      Independent variable.
    abc : 1d array, length 3
      Three-element vector controlling shape of triangular function.
      Requires a <= b <= c.

  Returns
  -------
    y : 1d array
      Triangular membership function.
"""
def trimf(x, abc):
 
    assert len(abc) == 3, 'abc parameter must have exactly three elements.'
    a, b, c = np.r_[abc]     # Zero-indexing in Python
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = np.zeros(len(x))

    # Left side
    if a != b:
        idx = np.nonzero(np.logical_and(a < x, x < b))[0]
        y[idx] = (x[idx] - a) / float(b - a)

    # Right side
    if b != c:
        idx = np.nonzero(np.logical_and(b < x, x < c))[0]
        y[idx] = (c - x[idx]) / float(c - b)

    idx = np.nonzero(x == b)
    y[idx] = 1
    return y

def localization(I):
  
  trim_pxl = 100
  inputpixel=175
  #I_red=I(:,:,1)
  #if length(size(I))==3
      #I_bw = rgb2gray(I)
  #end
    
  M, N = I.shape[0:2]
  O = 3
    
  # global p
  # global L
  # global Th
  # global alpha
  # global max_val
  # global min_val
    
  L = 256
  H = cv2.calcHist([I], [0], None, [256], [0,256])
  # plt.hist(I.ravel(),256,[0,256])
  
  #print(H)
  plt.plot(H)
  plt.show()
  
  # figure,imhist(I)

  p = (1/ (M * N)) * H

  # figure(2) plot(h)

  max_val = float(max(I.flatten()))
  min_val = float(min(I.flatten())+1)

  #*********************************************************************
  # 1. Define problem hyperspace and plot in 2D
  #*********************************************************************
    
  Th = 2          # No of thresholds
  D = Th * 2      # no of dimensions
  range_min = min_val*np.ones((1,D), dtype = int) 
  range_max = max_val*np.ones((1,D), dtype = int)  # minimum & maximum range 
  print(range_max)
  alpha = 1.5 
    
  #*********************************************************************
  # 2. initialize the population
  #*********************************************************************
    
  NP = 10 * D    #% population size
  maxgen = 100  #% no of generations
  F = 0.5 
  CR = 0.9 
  max_runs = 2 
  globalbest1 = []  
  statistics_f = []  
  statistics_x = []  

  # tstart = tic 
  """
  for runn in range(max_runs):
    x=[]  
    for i in range(NP):
      for j in range(D):
        #x[i][j] = round(range_min[j] + ((range_max[j]-range_min[j])*(random.random()))  
        pass
      #x[i,:] = x[i,:].sort()
      #fitness_parent[i,1] = ultrafuzziness([0 0 x[i,:] 255 255],p,alpha)
    #v = zeros(np.size(x))  #from numpy import zeros
    #u = zeros(np.size(x)) 

    #*********************************************************************
    # 4. start iteration
    #*********************************************************************
        
    # tStart = tic 
        
    for gen in range(2, maxgen+1):
      # [o-2]
      #*********************************************************************
      # 3. find mutation population
      #*********************************************************************
      r = []
      for i in range(NP):
        #r[0] = ceil(random.random()*NP)  ###Pls check
        #r[1] = ceil(random.random()*NP) 
        #r[2] = ceil(random.random()*NP) 
        pass
        
        while r[1]==r[2] or r[2]== r[3] or min(r)==0 or max(r)>NP :
          #r[0] = ceil(random.random()*NP)  ###Pls check
          #r[1] = ceil(random.random()*NP) 
          #r[2] = ceil(random.random()*NP) 
          pass
                    
        v[i,:] = x[r[1],:] + F*(x[r[2],:] - x[r[3],:])  
        for j in range(D):
          if random.random() > CR:
            u[i,j] = x[i,j] 
          else:
            u[i,j] = v[i,j]
            
        u[i,:]= round(u[i,:])
        u[i,:]= u[i,:].sort()
            
      for i in range(NP):
        for jj in range(D):
          u[i,jj] = max(u[i,jj], range_min[jj]) 
          u[i,jj] = min(u[i,jj], range_max[jj]) 
        u[i,:]= u[i,:].sort() 
        fitness_child[i,1] = ultrafuzziness([0 0 u[i,:] 255 255],p,alpha) 

      for i in range(NP):
        if fitness_parent[i] < fitness_child[i]:
          fitness_parent[i] = fitness_child[i] 
          x[i,:] = u[i,:] 

      [globalbest,globalbest_index] = max(fitness_parent) 
      global_xbest = x[globalbest_index,:].sort() 
            
      # clc
      # runn
      # fprintf('Optimisation through Differential Evolution\n')
      # fprintf('Generation: %0.5g\nGlobalbest: %2.7g\n', gen, globalbest)
      # fprintf('Best particle position : %0.11g\n', global_xbest)
                
      globalbest1 = [globalbest1, globalbest] 
            
    # tElapsed = toc(tStart) 
    # tElapsed
    globalbest1 = [globalbest1, globalbest] 
    statistics_f = [statistics_f, globalbest] 
    statistics_x = [statistics_x; (global_xbest)] ###??? 
        
  # plot(1:NP:NP*50,globalbest1(1:50),'-bs','MarkerFaceColor','b')
  # hold on
  f_mean = stat.mean(statistics_f) #import statistics as stat
  f_stddev = np.std(statistics_f)
  best_fitness = max(statistics_f)
  worst_fitness = min(statistics_f)
  x_median = stat.median(statistics_x)
            
  ### select the threshold points
  # T = [min_val round(x_median) max_val]
  # Thres=T(2:2:2*Th)
  t1 = x_median(1:2:D) ###???
  t2 = x_median(2:2:D) ###???
  Thres = round((t1+t2)/2)
  X = grayslice(I,[Thres]) ### what is the source code? 
  out = np.zeros(X.shape, np.double)
  X1 = 255*cv2.normalize(X, out, 0, 1, cv2.NORM_MINMAX, dtype = cv2.CV_64F) ####X1 = uint8(255*mat2gray(X))
        
  # timestop=toc(tstart)
  # tstop1(im_num)=timestop/2
  # figure,imshow(X1)
        
  # Trimp the image first
  XX1 = Trimp2(X1,trim_pxl)
            
  # XX1 = X1
  # New Code cup is the largest spot
  X3 = XX1==255

  # Find all the connected components

  CC = bwconncomp(X3) ##### what is the source code
        
  # Number of pixels in each connected components
        
  numPixels = cellfun(@numel,CC.PixelIdxList) #### what is the source code?
          
  # Largest connected compnent
  [biggest,idx] = max(numPixels)
    
  # if(aaaa)
    ### Additional code to remove Fringe
    # numPixels_temp = sort(numPixels,'descend')
    # idx = find(numPixels==numPixels_temp(2))

  # Calculating the cetroid of the largest component

  S = regionprops(CC,'Centroid') #from skimage.measure import regionprops #% Calculate centroids for connected components in the image using regionprops.
  cntr = cat(1, S.Centroid) ###????? #%Concatenate structure array containing centroids into a single matrix.
  centroid_x = round(cntr[idx][1])
  centroid_y = round(cntr[idx][2])

  # figure, imshow(I)
  # hold on
  # plot(centroid_x,centroid_y, 'b*')
  # hold off
    
  # Calculating region

  newx_up = centroid_y-inputpixel
  newx_down = min(M,centroid_y+inputpixel)
  newy_left = centroid_x-inputpixel
  newy_right = min(N,centroid_x+inputpixel)
    
  # tstop2(im_num)=toc(tstart)
  # Extract image
      
  for i in range(newx_up,newx_down):
    for j in range(newy_left,newy_right):
      Xcol_seg[i - newx_up + 1][j - newy_left + 1,:] = I[i,j] ## print this in matlab
  """
  return 1 #Xcol_seg

"""
def trimp(BW, pixl):
  M, N = BW.shape[0:2]
  BW2 = BW

  for i in range(1, M + 1):
    for j in range(1, N + 1):
      if BW[i,j] > 0:
        BW[i,j] = 255
  
  BW1 = BW
  CC = bwconncomp(BW)
  numPixels_temp = numPixels = cellfun(@numel,CC.PixelIdxList)
  # Largest connected compnent
  numPixels_temp.sort(reversed = True)
  idx = find(numPixels==numPixels_temp(1))
  #figure, imshow(CC.PixelIdxList{idx})
  BW1(CC.PixelIdxList{idx}) = 0
  # figure, imshow(BW1)
  BW3 = BW-BW1
  # figure, imshow(BW3)
  CC = bwconncomp(BW3)
  # Number of pixels in each connected components
  numPixels = cellfun(@numel,CC.PixelIdxList)
  # Largest connected compnent
  [biggest,idx] = max(numPixels)
  S = regionprops(CC,'Centroid'); # Calculate centroids for connected components in the image using regionprops.
  cntr = cat(1, S.Centroid); #Concatenate structure array containing centroids into a single matrix.
  centroid_x=round(cntr(idx,1))
  centroid_y=round(cntr(idx,2))

  # hold on
  # plot(centroid_x,centroid_y, 'b*')
  # hold off
  x_left = []
  x_right = []
  y_left = []
  y_right = []
  count = 1

  for i in range(1, M + 1):
    for j in range(1, centroid_x + 1):
      if BW[i,j]==255:
        BW2[i,j:j+pixl] = 0
        break
  
  count = 1
  y_down = 0
  
  for i in range(1, M + 1):
    for j in range(N, centroid_x, -1):
      if BW[i,j]==255:
        BW2[i,j:-1:j-pixl] = 0
        y_down = i
        break
  
  for i in range(y_down, y_down-pixl+1, -1):
    BW2[i,:] = 0

  # figure();plot(x_left,y_left,'b*',x_right,y_right,'r*')
  # figure();imshow(BW2)

def ultrafuzziness(v,p,alpha):
  # if (len(v)==3):
    # u = trimf(1:256, v)
    # uL = u**(alpha)
    # uU = u**(1/alpha)
    # f = sum((uU-uL).*p')
  # else
    # f=ultrafuzziness(v(1:3))*ultrafuzziness(v(3:length(v)))

  if(len(v) == 4):
    v = v+1
    u = trapmf[1:256,v]
    uL=u.^(alpha)
    uU=u.^(1/alpha)
    y=(uU-uL).*p'
    x=sum(y)
    f = 0
    if x != 0:
      for i in range(v[1],v[4]):
        #for i=v(1):v(4):
        if y[i] != 0:
          f = f+(y[i]/x)*math.log(y[i]/x)
      f = -f
    else:
        f = ultrafuzziness(v[1:4],p,alpha) + ultrafuzziness(v[3:len(v)],p,alpha)
  return f
"""

def mainfunc():
  thLevel = 3 # starting with threshold number three
  Eye_side = 'R' #can be Left or Right
  im_num = 1 #image number

  max_pixl_yc = 10; # maximum distance between the disc and cetroid on Yaxis

  maxArea = 18000; # maximum cup area
  minArea = 2000; # minimum cup area

  Extra_side = 0.45 # percentage of extra radius

  Radius_up = 0.2 # percentage of extra radius
  Radius_down = 0.2 # percentage of extra radius
  Radius_right = 0.0 # percentage of extra radius
  Radius_left = 0.0 # percentage of extra radius

  Level2Set = 1 # if 1 then Level2Set is used to detect the disc (JPG) NONmydriatric retinal camera else (0) SmoothMuch will be used

  # Do not change this data
  error = 5003
  lower_area = 1000

  # Blood Vessel Extraction
  L = 45
  Extra_radius = [Radius_up,Radius_down,Radius_right,Radius_left]
  Enhanced = 1; # Enable enhance at the beginning
  Error = 0 # There is no error

  I = 'Documents\GitHub\Optic_Disk\_OD.jpeg'
  fname = I
  print('\nWorking on ' + I +'\n')
  I_temp = ['Localized',fname]
  inImg_temp1 = cv2.imread(I)
  b = inImg_temp1[:,:,0]
  g = inImg_temp1[:,:,1]
  r = inImg_temp1[:,:,2]

  #inImg_temp1 = rgb2gray(inImg_temp1)
  inImg_temp1 = cv2.cvtColor(inImg_temp1, cv2.COLOR_BGR2GRAY)
  cv2.imshow('Fig',inImg_temp1)

  inImg_temp2 = localization(inImg_temp1)

"""
Main function
"""
if __name__=="__main__":

  # Image files location 
  location = 'Documents\GitHub\Optic_Disk\Images\_OD'
  mainfunc()
  """
  # Loop through all the images
  for i in range(11,15):
    image = location + str(i) + '.jpeg'   # Filename
    img = cv2.imread(image)               # Read image
    img = rgb2gray(img)                   # Convert to binary/greyscale
    img = crop(img, i)                    # Crop the image
    cv2.imshow('img',img)
    #segment(img)

    # Wait for keystroke
    cv2.waitKey(0)
  """
  cv2.waitKey(0)
  cv2.destroyAllWindows
