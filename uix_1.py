from flask import Flask, render_template, request
import pickle as p
import pandas as pd
import json
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import os
from flask import Flask, redirect, render_template, request, session, url_for
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
from mrcnn.model import log


import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time

from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model

#%matplotlib inline

from os import listdir
from xml.etree import ElementTree



app = Flask(__name__, template_folder = 'templates')

UPLOAD_FOLDER = os.path.basename('uploads')

dropzone = Dropzone(app)
##Dropzone Settings

app.config['DROPZONE_UPLOAD_MULTIPLE'] = False
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'
app.config['SECRET_KEY'] = 'oh_so_secret'
#Uploaded IMAGES
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'
app.config['SECRET_KEY'] = 'supersecretkeygoeshere'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)

file_urls = []

filename = None


#configrations
class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"

    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # number of classes (we would normally add +1 for the background)
     # kangaroo + BG
    NUM_CLASSES = 1+1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 20

    # Learning rate
    LEARNING_RATE=0.006

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # setting Max ground truth instances
    MAX_GT_INSTANCES=10

class RoadDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):

        # Add classes. We have only one class to add.
        self.add_class("dataset", 1, "crack")

        # define data locations for images and annotations
        images_dir = dataset_dir + 'images/'
        annotations_dir = dataset_dir + 'annots/'

        # Iterate through all files in the folder to
        #add class, images and annotaions
        for filename in listdir(images_dir):

            # extract image id
            image_id = filename[:-4]

            # skip bad images
            if image_id in ['00090']:
                continue
            # skip all images after 150 if we are building the train set
            if is_train and int(image_id) >= 55:
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and int(image_id) < 55:
                continue

            # setting image file
            img_path = images_dir + filename

            # setting annotations file
            ann_path = annotations_dir + image_id + '.xml'

            # adding images and annotations to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):

        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)

        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
     """
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]

        # define anntation  file location
        path = info['annotation']

        # load XML
        boxes, w, h = self.extract_boxes(path)

        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('crack'))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    #Return the path of the image."""
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']


config = myMaskRCNNConfig()
model_path = 'mask_rcnn(final)_.1579767785.0369418.h5'
model = MaskRCNN(mode='training', model_dir='./', config=config)
model.load_weights(model_path, by_name=True)

#train_set = RoadDataset()
#train_set.load_dataset('/home/amboo/Documents/road-master/', is_train=True)
#train_set.prepare()
#print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
#test_set = RoadDataset()
#test_set.load_dataset('/home/amboo/Documents/road-master/', is_train=False)
#test_set.prepare()
#print('Test: %d' % len(test_set.image_ids))


@app.route("/")
def hello():
    response_string = """Welcome to Mask R-CNN
    Endpoints:
    POST /changemodel - For selecting which model is running
        Json Body with:
	        "modelName": "Project_Christoppher--- MASK-RCNN",
            "base64Image": "base64 encoded image data"
    """
    return response_string



@app.route("/visualize", methods=['POST'])
def return_visualized_image():
    # Get image from request and change to array
    image = fh.image_from_request(request)
    image = fh.image_to_array(image)

    # Run detection
    results = MODEL.detect([image])
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                             CLASS_NAMES, r['scores'])

    buf = BytesIO()
    plt.savefig(buf, format='jpg')

    response = Response()
    response.set_data(buf.getvalue())
    response.headers['Content-Type'] = 'image/jpeg'
    return response


if __name__ == '__main__':

    app.run(port=9000, debug=True)
