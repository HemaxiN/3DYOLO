'''Functions for 3D Data Augmentation of Images and Bboxes'''

import cv2
import numpy as np
from scipy.ndimage.interpolation import rotate
import math

##auxiliary function to rotate the point
def rotate_around_point_lowperf(image, pointx, pointy, angle):
    """Rotate a point around a given point.
    
    I call this the "low performance" version since it's recalculating
    the same values more than once [cos(radians), sin(radians), x-ox, y-oy).
    It's more readable than the next function, though.
    """
    radians = (np.pi*angle)/(180)
    x, y = pointx, pointy
    ox, oy = image.shape[0]/2, image.shape[1]/2

    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy - math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

    return qx, qy


##rotation
def rotation(image, bbox, angle):
    
    rot_image = np.zeros(np.shape(image))
    for z in range(0, rot_image.shape[2]):
        rot_image[:,:,z,:] = rotate(image[:,:,z,:], angle, mode='constant', reshape=False)
        
    rot_bbox = []
    for b in bbox:
        rxmin, rymin = rotate_around_point_lowperf(image,b[0], b[2], angle)
        rxmax, rymax = rotate_around_point_lowperf(image,b[1], b[3], angle)
        xmin = min(rxmin, rxmax)
        xmax = max(rxmin, rxmax)
        ymin = min(rymin, rymax)
        ymax = max(rymin, rymax)
        rot_bbox.append([int(xmin),int(xmax),int(ymin),int(ymax), b[4], b[5]])
    return rot_image, rot_bbox


##vertical flip
def vertical_flip(image, bbox):
    
    flippedimage = np.zeros(np.shape(image))
    for z in range(0, flippedimage.shape[2]):
        flippedimage[:,:,z,:] = cv2.flip(image[:,:,z,:], 0)
    
    flippedbbox = []
    for b in bbox:
        xmin, ymax, zmin = b[0], (image.shape[1]-b[2]-1), b[4]
        xmax, ymin, zmax = b[1], (image.shape[1]-b[3]-1), b[5]
        flippedbbox.append([xmin, xmax, ymin, ymax, zmin, zmax])
    return flippedimage, flippedbbox


##horizontal flip
def horizontal_flip(image, bbox):
    
    flippedimage = np.zeros(np.shape(image))
    for z in range(0, flippedimage.shape[2]):
        flippedimage[:,:,z,:] = cv2.flip(image[:,:,z,:], 1)
        
        
    flippedbbox = []
    for b in bbox:
        xmax, ymin, zmin = image.shape[0]-b[0]-1, b[2], b[4]
        xmin, ymax, zmax = image.shape[0]-b[1]-1, b[3], b[5]
        flippedbbox.append([xmin, xmax, ymin, ymax, zmin, zmax])
    return flippedimage, flippedbbox

#intensity variations
def intensity(image, bbox, alpha=None):
    if alpha==None:
        alpha = np.random.choice(np.arange(0.75,1))
    
    image = image.astype('float64')
    image = image*alpha
    image = image.astype('float64')
    
    return image, bbox
