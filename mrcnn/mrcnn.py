# Import necessary libraries
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import imgaug

# Root directory of the project
ROOT_DIR = os.path.abspath(r"D:\Newfolder\Mask_RCNN-master\Mask_RCNN-master")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Path to the dataset
DATASET_DIR = os.path.join(ROOT_DIR, "balloon_dataset", "balloon")

# Configuration
class CustomConfig(Config):
    NAME = "custom"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # Background + Number of classes

config = CustomConfig()
config.display()

# Create model
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Load COCO pre-trained weights
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Load your dataset
class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        # Add your classes here
        self.add_class("custom", 1, "balloon")
        self.add_class("custom", 2, "non_balloon")

        # Load annotations
        annotations = json.load(open(os.path.join(dataset_dir, subset, "via_region_data.json")))
        annotations = list(annotations.values())

        # Add images and annotations to the dataset
        for a in annotations:
            polygons = a['regions']
            image_path = os.path.join(dataset_dir, subset, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "custom",
                image_id=a['filename'],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons
            )
# original
    #def load_mask(self, image_id):
        # Create masks from polygons
     #   info = self.image_info[image_id]
      #  mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
       # for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
        #    rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
         #   mask[rr, cc, i] = 1

       # return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)



    def load_mask(self, image_id):
    # Create masks from polygons
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            if isinstance(p, str):
                # Handle the case where 'p' is a string (additional checks or actions may be needed)
                #print(f"Skipping invalid polygon (not a dictionary): {p}")
                print(f"Skipping invalid polygon (not a dictionary) in image {info['id']}: {p}")

                continue

            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)
    

# Create training and validation datasets
dataset_train = CustomDataset()
dataset_train.load_custom(DATASET_DIR, "train")
dataset_train.prepare()

dataset_val = CustomDataset()
dataset_val.load_custom(DATASET_DIR, "val")
dataset_val.prepare()

# Image Augmentation
augmentation = imgaug.augmenters.Fliplr(0.5)

# Training - Fine-tune the model
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=5,
            layers='heads',
            augmentation=augmentation)

# Save weights (optional)
model_path = os.path.join(MODEL_DIR, "mask_rcnn_custom.h5")
model.keras_model.save_weights(model_path)
