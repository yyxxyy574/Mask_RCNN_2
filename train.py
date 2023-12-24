import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/stu7/prml23/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from dataset.load_data import CustomConfig
from dataset.load_data import CustomDataset
from mrcnn import utils
import mrcnn.model as modelib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
# Configurations
config = CustomConfig()
config.display()

# Training dataset
dataset_train = CustomDataset()
dataset_train.load_voc("/home/stu7/prml23/data/train")
dataset_train.prepare()

# Validation dataset
dataset_val = CustomDataset()
dataset_val.load_voc("/home/stu7/prml23/data/validation")
dataset_val.prepare()

# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    modelib.load_image_gt(dataset_train, config, image_id)
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))

# Create model in tranining mode
model = modelib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"

model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "Mrcnn_bbox", "mrcnn_mask"])

# Train the head branches
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers="heads")

# Fine tune all layers
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,
            layers="all")

# Detection
class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modelib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

# Get path to saved weights
model_path = model.find_last()

# Load trained weights
print("Loading weights from", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    modelib.load_image_gt(dataset_val, inference_config, image_id)
visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)
r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'])

# Evaluation
image_ids = np.random.chocie(dataset_val.image_ids, 10)
APs = []
for imaga_id in image_ids:
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modelib.load_image_gt(dataset_val, inference_config, image_id)

    results = model.detect([original_image], verbose=1)
    r = results[0]
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                          r["rois"], r["class_ids"], r["scores"], r["masks"])
    APs.append(AP)

print("mAP:", np.mean(APs))