import os
import cv2
import copy
import numpy as np
from keras.utils import Sequence
import xml.etree.ElementTree as ET
from utils3D import BoundBox, bbox_iou
from skimage import io
from keras.preprocessing.image import ImageDataGenerator
import tifffile


def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}

        tree = ET.parse(ann_dir + ann)
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'z' in elem.tag:
                img['z'] = 64
            if 'depth' in elem.tag:
                img['depth'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'zmin' in dim.tag:
                                obj['zmin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
                            if 'zmax' in dim.tag:
                                obj['zmax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels

class BatchGenerator(Sequence):
    def __init__(self, images, 
                       config, 
                       shuffle=True, 
                       jitter=True, 
                       norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        self.anchors = [BoundBox(0, 0, 0, config['ANCHORS'][3*i], config['ANCHORS'][3*i+1], config['ANCHORS'][3*i+2]) for i in range(int(len(config['ANCHORS'])//3))]

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))   

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)    

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['zmin'], obj['xmax'], obj['ymax'], obj['zmax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]
        return np.array(annots)

    def load_image(self, i):
        image = tifffile.imread(self.images[i]['filename'])
        #image = image.reshape((self.config['IMAGE_Z'],self.config['IMAGE_H'],self.config['IMAGE_W'],1))
        return image

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']
        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_Z'], self.config['IMAGE_H'], self.config['IMAGE_W'], 3))     # input images
        b_batch = np.zeros((r_bound - l_bound, 1,     1     ,  1  , 1 ,  self.config['TRUE_BOX_BUFFER'], 6))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_Z'], self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 6+1+len(self.config['LABELS'])))                # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)
            # construct output from object's x, y, w, h
            true_box_index = 0

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['zmax'] > obj['zmin'] and obj['name'] in self.config['LABELS']:
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])
                    center_z = .5*(obj['zmin'] + obj['zmax'])
                    center_z = center_z / (float(self.config['IMAGE_Z']) / self.config['GRID_Z'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))
                    grid_z = int(np.floor(center_z))
                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H'] and grid_z < self.config['GRID_Z']:
                        obj_indx  = self.config['LABELS'].index(obj['name'])
                        
                        center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                        center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
                        center_d = (obj['zmax'] - obj['zmin']) / (float(self.config['IMAGE_Z']) / self.config['GRID_Z']) # unit: grid cell
                        
                        box = [center_x, center_y, center_z, center_w, center_h, center_d]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou     = -1
                        
                        shifted_box = BoundBox(0, 
                                               0,
                                               0,
                                               center_w,                                                
                                               center_h,
                                               center_d)
                    
                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou    = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou     = iou
                                
                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_z, grid_y, grid_x, best_anchor, 0:6] = box
                        y_batch[instance_count, grid_z, grid_y, grid_x, best_anchor, 6  ] = 1.
                        y_batch[instance_count, grid_z, grid_y, grid_x, best_anchor, 7+obj_indx] = 1
#                        print (instance_count,grid_x,grid_y,grid_z,best_anchor)
                        
                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, 0,   true_box_index] = box
                        
                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                            
            # assign input image to x_batch
            if self.norm != None: 
                x_batch[instance_count] = img/255.
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if False:
                        cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']),  (255,0,0), 3)
                        cv2.putText(img[:,:,::-1], obj['name'], 
                                    (obj['xmin']+2, obj['ymin']+12), 
                                    0, 1.2e-3 * img.shape[0], 
                                    (0,255,0), 2)
                        
                x_batch[instance_count] = img/255.

            # increase instance counter in current batch
            instance_count += 1  
#        print (y_batch)
        #print(' new batch created', idx)
        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)




    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = tifffile.imread(image_name)

        if image is None: print('Cannot find ', image_name)

        h, w, d, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])

                
        return image, all_objs
