import random
import numpy as np
import statistics as stat
import cv2

def localization(I):
    
    trim_pxl = 100
    inputpixel=175
    #I_red=I(:,:,1)
    #if length(size(I))==3
        #I_bw = rgb2gray(I)
    #end
    
    M, N = I.shape[0:2]
    
    # global p
    # global L
    # global Th
    # global alpha
    # global max_val
    # global min_val
    
    L = 256
    H = cv2.calcHist(I)
    
       # %figure,imhist(I)

    p = H / (M * N) 

       # %figure(2) plot(h)

    max_val = float(max(max(I)))
    min_val = float(min(min(I))+1)
    
    #*********************************************************************
    # 1. Define problem hyperspace and plot in 2D
    #*********************************************************************
    
    Th = 2          # No of thresholds
    D = Th * 2      # no of dimensions
    range_min = min_val*np.ones((1,D), dtype = int) 
    range_max = max_val*np.ones((1,D), dtype = int)  # minimum & maximum range 
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

    for runn in range(max_runs):
        x=[]  
        for i in range(NP):
            for j in range(D):
                x[i][j] = round(range_min[j] + ((range_max[j]-range_min[j])*(random.random()))  
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
            for i in range(NP):
                r[0] = ceil(random.random()*NP)  ###Pls check
                r[1] = ceil(random.random()*NP) 
                r[2] = ceil(random.random()*NP) 
                while r(1)==r(2) or r(2)== r(3) or min(r)==0 or max(r)>NP :
                    r[0] = ceil(random.random()*NP)  ###Pls check
                    r[1] = ceil(random.random()*NP) 
                    r[2] = ceil(random.random()*NP) 
                    
                v[i,:] = x[r[1],:] + F*(x[r[2],:] - x[r[3],:])  
                for j in range(D):
                    if random.random() > CR:
                        u[i][j] = x[i][j] 
                    else:
                        #u{i][j] = v[i][j]
                        pass 
                #u[i,:]= round(u[i,:])
                #u[i,:]= u[i,:].sort()
            
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
            #% Largest connected compnent

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
    
    return Xcol_seg
