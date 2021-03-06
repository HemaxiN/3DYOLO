import numpy as np
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import copy
import cv2

class BoundBox:
    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    intersect_d = _interval_overlap([box1.zmin, box1.zmax], [box2.zmin, box2.zmax])  
    
    intersect = intersect_w * intersect_h * intersect_d

    w1, h1, d1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin, box1.zmax-box1.zmin
    w2, h2, d2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin, box2.zmax-box2.zmin
    
    union = w1*h1*d1 + w2*h2*d2 - intersect
    
    return float(intersect) / union

def draw_boxes(image, boxes, labels):
    image_d, image_h, image_w, _ = image.shape
    image_bbox = image.copy()
    #image_bbox = image_bbox.transpose(1,2,0,3)

    for box in boxes:
        xmin = int(box.xmin*image_w)
        ymin = int(box.ymin*image_h)
        xmax = int(box.xmax*image_w)
        ymax = int(box.ymax*image_h)
        zmin = int(box.zmin*image_d)
        zmax = int(box.zmax*image_d)

        if zmin>0 and xmin>0 and xmax<416 and ymin>0 and ymax<416 and zmax<64:

	        image_bbox[zmin:zmax,ymin,xmin,:] = 150.0*np.ones(np.shape(image_bbox[zmin:zmax,ymin,xmin,:]))
	        image_bbox[zmin:zmax,ymax,xmin,:] = 150.0*np.ones(np.shape(image_bbox[zmin:zmax,ymax,xmin,:]))
	        image_bbox[zmin:zmax,ymin,xmax,:] = 150.0*np.ones(np.shape(image_bbox[zmin:zmax,ymin,xmax,:]))
	        image_bbox[zmin:zmax,ymax,xmax,:] = 150.0*np.ones(np.shape(image_bbox[zmin:zmax,ymax,xmax,:]))

	        image_bbox[zmin,ymin:ymax,xmin,:] = 150.0*np.ones(np.shape(image_bbox[zmin,ymin:ymax,xmin,:]))
	        image_bbox[zmax,ymin:ymax,xmin,:] = 150.0*np.ones(np.shape(image_bbox[zmax,ymin:ymax,xmin,:]))
	        image_bbox[zmin,ymin:ymax,xmax,:] = 150.0*np.ones(np.shape(image_bbox[zmin,ymin:ymax,xmax,:]))
	        image_bbox[zmax,ymin:ymax,xmax,:] = 150.0*np.ones(np.shape(image_bbox[zmax,ymin:ymax,xmax,:]))

	        image_bbox[zmin,ymin,xmin:xmax,:] = 150.0*np.ones(np.shape(image_bbox[zmin,ymin,xmin:xmax,:]))
	        image_bbox[zmax,ymin,xmin:xmax,:] = 150.0*np.ones(np.shape(image_bbox[zmax,ymin,xmin:xmax,:]))
	        image_bbox[zmin,ymax,xmin:xmax,:] = 150.0*np.ones(np.shape(image_bbox[zmin,ymax,xmin:xmax,:]))
	        image_bbox[zmax,ymax,xmin:xmax,:] = 150.0*np.ones(np.shape(image_bbox[zmax,ymax,xmin:xmax,:]))
    
    return image_bbox             
        
def decode_netout(netout, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.3):
    grid_d, grid_h, grid_w, nb_box = netout.shape[:4]

    boxes = []
    
    # decode the output by the network
    netout[..., 6]  = _sigmoid(netout[..., 6])
    netout[..., 7:] = netout[..., 6][..., np.newaxis] * _softmax(netout[..., 7:])
    netout[..., 7:] *= netout[..., 7:] > obj_threshold

    for mmp in range(grid_d):
        for row in range(grid_h):
            for col in range(grid_w):

                for b in range(nb_box):
                    # from 4th element onwards are confidence and class classes
                    classes = netout[mmp,row,col,b,7:]
                
                    if np.sum(classes) > 0:
                        # first 4 elements are x, y, w, and h
                        x, y, z, w, h, d = netout[mmp,row,col,b,:6]

                        x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
                        y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
                        z = (mmp + _sigmoid(z)) / grid_d
                        w = anchors[3 * b + 0] * np.exp(w) / grid_w # unit: image width
                        h = anchors[3 * b + 1] * np.exp(h) / grid_h # unit: image height
                        d = anchors[3 * b + 2] * np.exp(d) / grid_d
                        confidence = netout[mmp,row,col,b,6]
                    
                        box = BoundBox(x-w/2, y-h/2, z-d/2, x+w/2, y+h/2, z+d/2, confidence, classes)
                    
                        boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes    

    
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap      
        
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3          

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    
    if np.min(x) < t:
        x = x/np.min(x)*t
        
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)
