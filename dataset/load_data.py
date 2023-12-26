import os
import sys
import numpy as np
import tensorflow as tf
from xml.etree import ElementTree as ET

# Extend two classes "Config" and "Dataset" to train the model on provided VOC dataset
from mrcnn.config import Config
from mrcnn.utils import Dataset

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor")

# Custom configuration for VOC dataset
class CustomConfig(Config):
    NAME = "VOC"
    GPU_COUNT = 1 # Use one GPU
    IMAGES_PER_GPU = 2 # A 3090 24GB can typically handle 2 images of 1024x1024px
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    BACKBONE = "resnet101" # resnet50, resnet101
    COMPUTE_BACKBONE_SHAPE = None
    BACKBONE_STRIDES = [4, 8, 16, 32, 64] # The strides of each layer of the FPN Pyramid based on Resnet101 backbone
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    TOP_DOWN_PYRAMID_SIZE = 256
    NUM_CLASSES = 21 # 20 classification classes + 1 background
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_STRIDE = 1
    RPN_NMS_THRESHOLD = 0.7 # Non-max suppression threshold to filter RPN proposals
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    PRE_NMS_LIMIT = 6000
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 0
    IMAGE_CHANNEL_COUNT = 3
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    
    TRAIN_ROIS_PER_IMAGE = 200 # Number of ROIs per image to feed to classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33 # Percent of positive ROIs used to train classifier/mask heads
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28] # Shape of output mask, changed with the neural network mask branch
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    DETECTION_MAX_INSTANCES = 100
    DETECTION_MIN_CONFIDENCE = 0.7 # Minimum probability value to accept a detected instance
    DETECTION_NMS_THRESHOLD = 0.3 # Non-maximum suppression threshold for detection
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 0.
    } # Set the wight of mrcnn_mask_loss to 0
    USE_RPN_ROIS = True
    TRAIN_BN = False
    GRADIENT_CLIP_NORM = 5.0 # Gradient norm clipping
    
# Custom class to use VOC dataset
class CustomDataset(Dataset):
    
    def load_voc(self, dataset_dir):
        # Add class names information
        for i in range(1, 21):
            self.add_class("VOC", i, VOC_CLASSES[i-1])
            
        annotations_dir = os.path.join(dataset_dir, 'Annotations')
        image_dir = os.path.join(dataset_dir, 'JPEGImages')
        
        for image_id in os.listdir(annotations_dir):
            annotation_path = os.path.join(annotations_dir, image_id)
            image_path = os.path.join(image_dir, image_id.replace('.xml', '.jpg'))
            
            # Add image information
            self.add_image(
                source="VOC",
                image_id=image_id,
                path=image_path,
                annotation_path=annotation_path
            )
            
    def load_bbox(self, image_id):
        image_info = self.image_info[image_id]
        annotation_path = image_info["annotation_path"]
        
        # Use the ElementTree to parse XML annotations
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Extract bbox and class information
        instance_bboxes = np.zeros([len(root.findall(".//object")), 4], dtype=np.int32)
        class_ids = []
        i = 0
        for obj in root.findall("object"):
            name = obj.find("name").text
            class_ids.append(VOC_CLASSES.index(name) + 1)
            
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            instance_bboxes[i] = np.array([ymin, xmin, ymax, xmax])
            i+=1
            
        class_ids = np.array(class_ids, dtype=np.int32)
        return instance_bboxes, class_ids
