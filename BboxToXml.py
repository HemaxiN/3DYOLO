'''Create .xml annotations from bboxes'''

from xmls_yolo import beginXML, createXML
from lxml import etree
from tifffile import imread
import numpy as np
import os

img_dir = r'/mnt/2TBData/hemaxi/YOLO3D/FINAL/dataset/images' #directory containing the .tif images (ZxXxYxC=64x416x416x3)
bbox_dir = r'/mnt/2TBData/hemaxi/YOLO3D/FINAL/dataset/bboxes' #directory containing the .npy bboxes (Nx6 array) (xmin, xmax, ymin, ymax, zmin, zmax)
xml_dir = r'/mnt/2TBData/hemaxi/YOLO3D/FINAL/dataset/xml' #directory where the corresponding .xml annotation files will be saved

for img, bbox in zip(os.listdir(img_dir), os.listdir(bbox_dir)):
    image = imread(os.path.join(img_dir, img)) #read the image
    annotation = beginXML(img, np.shape(image)[1], np.shape(image)[2], np.shape(image)[0]) 
    bbox = np.load(os.path.join(bbox_dir, bbox)) #read the bounding box
    
    for b in bbox:
        xmin, xmax, ymin, ymax, zmin, zmax = b[0], b[1], b[2], b[3], b[4], b[5]
        annotation = createXML(xmin, xmax, ymin, ymax, zmin, zmax, img, np.shape(image)[1], np.shape(image)[2], np.shape(image)[0], annotation)
        
    save_path = os.path.join(xml_dir, img.replace('.tif','.xml'))
    with open(save_path, 'wb') as file:
        aux = etree.tostring(annotation, pretty_print=True)
        aux.decode("utf-8")
        file.write(aux)