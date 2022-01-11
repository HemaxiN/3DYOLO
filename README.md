# 3DYOLO
3D Implementation of the Tiny Yolo v2 model to jointly detect nucleus-Golgi pairs in 3D Microscopy Images

## Create the Training Set
Firstly, generate a dataset with input images and corresponding bounding boxes. Input images should be saved as .tif files, with dimensions (Z,X,Y,C). Bounding boxes should be saved in a numpy array (.npy object) of size (N,6), where N denotes the total number of bounding boxes in the corresponding image, and for each bounding box it contains the coordinates: [xmin, xmax, ymin, ymax, zmin, zmax] (in this order).
Thereafter, change the following parameters in `BboxToXml.py` and run it:

* `img_dir`: directory containing the .tif images (Z,X,Y,C)
* `bbox_dir`: directory containing the .npy objects with the bounding boxes (N,6) 
* `xml_dir`: directory where the .xml files will be saved

This will generate the dataset in the appropriate format to train the model.

## Data Augmentation
If you want to perform data augmentation, change the `img_dir`, `bbox_dir`, and `maxpatches` parameter in `dataaugmentation.py` and run it, where: 

* `img_dir`: directory containing the .tif images (Z,X,Y,C)
* `bbox_dir`: directory containing the .npy objects with the bounding boxes (N,6) 
* `maxpatches`: number of augmented image patches

It performs z-axis aligned rotations in the range (0, 360◦) with steps of size 90◦, horizontal flips, vertical flips and intensity variations.

## Dataset: Tree Structure
Organize the training and validation images and annotations as follows: 

```
datasetyolo
├── train
│   ├── images
│   └── annot
└── val
    ├── images
    └── annot
```



