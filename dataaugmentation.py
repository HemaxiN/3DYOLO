'''Perform Data Augmentation'''

import numpy as np
import os
from tifffile import imread, imwrite
from da_utils import *

img_dir = r'/mnt/2TBData/hemaxi/YOLO3D/FINAL/dataset/images' #directory containing the .tif images (ZxXxYxC=64x416x416x3)
bbox_dir = r'/mnt/2TBData/hemaxi/YOLO3D/FINAL/dataset/bboxes' #directory containing the .npy bboxes (Nx6 array) (xmin, xmax, ymin, ymax, zmin, zmax)

save_dir_img = img_dir
save_dir_box = bbox_dir

maxpatches = 250 #number of augmented patches
npatches = len(os.listdir(img_dir)) #number of patches in "img_dir"


k=npatches
ii=0
while k<maxpatches+npatches:

    if ii==npatches:
        ii = 0

    img_aux = imread(os.path.join(img_dir, str(ii) + '.tif'))
    box_aux = np.load(os.path.join(bbox_dir, str(ii) + '.npy'))
    
    img_aux = img_aux.transpose(1,2,0,3)
    img_aux = img_aux/255.0
    
    #data augmentation
    if np.random.choice([0,1])==1:
        img_aux, box_aux = vertical_flip(img_aux, box_aux)
    if np.random.choice([0,1])==1:
        img_aux, box_aux = horizontal_flip(img_aux, box_aux)
    if np.random.choice([0,1])==1:
        img_aux, box_aux = intensity(img_aux, box_aux)
    if np.random.choice([0,1])==1:
        angle = np.random.choice(np.arange(0,360,90))
        img_aux, box_aux = rotation(img_aux, box_aux, angle)
      
        
    image_bbox = img_aux.copy()
    
    img_aux = img_aux.transpose(2,0,1,3)
    img_aux = img_aux*255.0
    img_aux = img_aux.astype('uint8')
    
    np.save(os.path.join(save_dir_box, str(k) + '.npy'), box_aux)
    imwrite(os.path.join(save_dir_img, str(k) + '.tif'), img_aux, photometric='rgb')
    k=k+1
    ii = ii + 1
    