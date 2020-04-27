import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage import io
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2
import time

def write_image(path, img):
    # img = img*(2**16-1)
    # img = img.astype(np.uint16)
    # img = img.astype(np.uint8)
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    plt.savefig(path, bbox_inches='tight')
    #, img, format = 'png')

#img = io.imread('Desktop\Summer_Intern_20\_OD.jpeg')
img = io.imread('Documents\GitHub\Optic_Disk\_OD.jpeg')
plt.imshow(img)
plt.close()
#cv2.imshow('image',img)
img = rgb2gray(img)

s = np.linspace(0, 2*np.pi, 400)
r = 350 + 250*np.sin(s)
c = 550 + 500*np.cos(s)
init = np.array([r, c]).T

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
"""
plt.show()
plt.show(block=False)
time.sleep(1)
plt.close(1) 
v = 'Desktop\Summer_Intern_20\Output\Result' + '_' + str(0) + '_' + str(0) + '_' + str(0) + '_' + str(0)+ '.png'
"""
# plt.imsave(v, img)

blur = 0.01
alpha = 0.1
beta = 0.1
gamma = 0.001
"""
snake = active_contour(gaussian(img, blur), init, alpha, beta, gamma, coordinates='rc')

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

print(str(blur) + ', ' + str(alpha) + ', ' + str(beta) + ', ' + str(gamma))
#cv2.imwrite(img, 'result'+ str(blur) + ', ' + str(alpha) + ', ' + str(beta) + ', ' + str(gamma)+ '.jpg' )
v = 'Desktop\Summer_Intern_20\Output\Result' + '_' + str(blur) + '_' + str(alpha) + '_' + str(beta) + '_' + str(gamma) + '.png'
write_image(v, img)"""

val = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]# -0.0001, -0.0005, -0.001, -0.005, -0.01, -0.05, -0.1, -0.5, -1, -5, -10, -50, -100, -500, -1000]
flag = 0

for i in range(0, 16):
  blur = val[i]
  for j in range(0, len(val)):
    flag = flag + 1
    if flag < 3:
      continue
    alpha = val[j]
    for k in range(0, len(val)):
      beta = val[k]
      for l in range(0, len(val)):
        gamma = val[l]              
        snake = active_contour(gaussian(img, blur),
                               init, alpha, beta, gamma,
                               coordinates='rc')
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        print(str(blur) + ', ' + str(alpha) + ', ' + str(beta) + ', ' + str(gamma))
        #plt.show(block=False)
        #time.sleep(1)
        #plt.close(1) 
        v = 'Documents\GitHub\Optic_Disk\Output\Result' + '_' + str(blur) + '_' + str(alpha) + '_' + str(beta) + '_' + str(gamma) + '.png'
        write_image(v, img)
        plt.clf()
        plt.close('all')
