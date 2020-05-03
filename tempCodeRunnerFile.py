
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
  