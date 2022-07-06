"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 salpic.py train --dataset=/path/to/salpic/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 salpic.py train --dataset=/path/to/salpic/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 salpic.py train --dataset=/path/to/salpic/dataset --weights=imagenet
    # Apply color splash to an image
    python3 salpic.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color splash to video using the last weights you trained
    python3 salpic.py splash --weights=last --video=<URL or path to file>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt



import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
import random
from random import randint 
import cv2
import math


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/salpic/")

TRAIN_IMAGE_DIR = os.path.join(ROOT_DIR, "images/train")
VAL_IMAGE_DIR = os.path.join(ROOT_DIR, "images/val")

############################################################
#  Configurations
############################################################


class SalpicConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"
    #BACKBONE_STRIDES =
    #BATCH SIZE = is calculated based on the number of GPU
    #BBOX_STD_DEV =
    #COMPUTE_BACK_BONE_SHAPE =

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400
    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between salamander and BG
    DETECTION_MIN_CONFIDENCE = 0
    #DETECTION_NMS_THRESHOLD = 0,3
    #FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    GPU_COUNT = 1
    #GRADIENT_CLIP_NORM
    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 6
    #IMAGE_CHANNEL_COUNT =
    IMAGE_MIN_DIM = 1024
    #IMAGE_META_SIZE
    IMAGE_MAX_DIM = 0
    IMAGE_MIN_SCALE = 0
    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "square"
    #IMAGE_SHAPE
    #LEARNING_RATE
    #LOSS_WEIGHTS
    #MASK_POOL_SIZE
    #MASK_SHAPE
    #MAX_GT_INSTANCES
    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    # Give the configuration a recognizable name
    NAME = "salpic"
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Salamander
    POOL_SIZE = 7
    #POST_NMS_ROIS_INTERFERENCE = 1000
    #POST_NMS_ROIS_TRAINING = 2000
    #PRE_NMS_LIMIT = 6000
    #ROI_POSITIVE_RATIO = 0.33
    #RPN_ANCHORS_RATIOS = [0.5, 1 , 2]

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    #RPN_ANCHOR_STRIDE = 1
    #RPN_BBOX_STD_DEV = [0.1 0.1 0.2 0.2]
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7 #also try 0.9
    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_DIR)) // IMAGES_PER_GPU
    #TOP_DOWN-PYRAMID_SIZE = 256
    #TRAIN_BN = False
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    #USE_RPN_ROIS = True
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_DIR) // IMAGES_PER_GPU)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200



class SalpicInferenceConfig(SalpicConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7



############################################################
#  Dataset
############################################################



class SalpicDataset(utils.Dataset):

    def load_salamander_shape(self, count, height, width):
        """Load a subset of the salpic dataset.
        dataset_dir: Root directory of the dataset
        """
        """Generate the requested number of synthetic images.
                count: number of images to generate.
                height, width: the size of the generated images.
                """
        # Add classes. We have one class.
        # Naming the dataset salpic, and the class salamander
        self.add_class("salpic", 1, "salamander")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, salpic = self.random_image(height, width)
            self.add_image("salpic", image_id=i, path=None,
                           width=width,height=height,
                           bg_color=bg_color, salpic=salpic)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['salpic']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """Return the salpic data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "salpic":
            return info["salpic"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        salpic = info['salpic']
        count = len(salpic)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['salpic']):
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
                                                  shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(
                occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in salpic])
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'salamander':
            image = cv2.rectangle(image, (x - s, y - s),
                                  (x + s, y + s), color, -1)
            image = cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height // 4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        import random 
        from random import randint
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y - s, x - s, y + s, x + s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(
            np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes







############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = SalpicDataset()
    dataset_train.load_salamander_shape(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SalpicDataset()
    dataset_val.load_salamander_shape(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='all')




############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)



############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = SalpicDataset()
    dataset.load_salamander_shape(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)

    ############################################################
    #  Command Line
    ############################################################

    if __name__ == '__main__':
        import argparse

        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Mask R-CNN for nuclei counting and segmentation')
        parser.add_argument("command",
                            metavar="<command>",
                            help="'train' or 'detect'")
        parser.add_argument('--dataset', required=False,
                            metavar="/path/to/dataset/",
                            help='Root directory of the dataset')
        parser.add_argument('--weights', required=True,
                            metavar="/path/to/weights.h5",
                            help="Path to weights .h5 file or 'coco'")
        parser.add_argument('--logs', required=False,
                            default=DEFAULT_LOGS_DIR,
                            metavar="/path/to/logs/",
                            help='Logs and checkpoints directory (default=logs/)')
        parser.add_argument('--subset', required=False,
                            metavar="Dataset sub-directory",
                            help="Subset of dataset to run prediction on")
        args = parser.parse_args()

        # Validate arguments
        if args.command == "train":
            assert args.dataset, "Argument --dataset is required for training"
        elif args.command == "detect":
            assert args.subset, "Provide --subset to run prediction on"

        print("Weights: ", args.weights)
        print("Dataset: ", args.dataset)
        if args.subset:
            print("Subset: ", args.subset)
        print("Logs: ", args.logs)

        # Configurations
        if args.command == "train":
            config = NucleusConfig()
        else:
            config = NucleusInferenceConfig()
        config.display()

        # Create model
        if args.command == "train":
            model = modellib.MaskRCNN(mode="training", config=config,
                                      model_dir=args.logs)
        else:
            model = modellib.MaskRCNN(mode="inference", config=config,
                                      model_dir=args.logs)

        # Select weights file to load
        if args.weights.lower() == "coco":
            weights_path = COCO_WEIGHTS_PATH
            # Download weights file
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)
        elif args.weights.lower() == "last":
            # Find last trained weights
            weights_path = model.find_last()
        elif args.weights.lower() == "imagenet":
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()
        else:
            weights_path = args.weights

        # Load weights
        print("Loading weights ", weights_path)
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)

        # Train or evaluate
        if args.command == "train":
            train(model, args.dataset, args.subset)
        elif args.command == "detect":
            detect(model, args.dataset, args.subset)
        else:
            print("'{}' is not recognized. "
                  "Use 'train' or 'detect'".format(args.command))



