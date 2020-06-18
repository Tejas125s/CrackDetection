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
import matplotlib.pyplot as plt

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


import os
import sys
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

import matplotlib
matplotlib.use('Agg')

app = Flask(__name__, template_folder = 'templates')

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



def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha,
                                  image[:, :, c])
    return image


@app.route('/')
def home():
    return render_template('home.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index', methods=['GET', 'POST'])
def index():

    global filename

    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']    # handle image upload from Dropzone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)

            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename
            )            # append image urls
            file_urls.append(photos.url(filename))

        session['file_urls'] = file_urls
        return "uploading..."    # return dropzone template on GET request
    return render_template('index.html')


@app.route('/results', methods = ['POST', 'GET'])
def results():

    def random_colors(N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors


    #model.load_weights(model_path, by_name=True)
    global filename
    global final_image
    image_id1 = int(input('Enter Image ID'))

    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(test_set, config,
                           image_id1, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    def display_instances(image, boxes, masks, class_ids, class_names,scores=None, title="",figsize=(16, 16), ax=None,show_mask=True, show_bbox=True,colors=None, captions=None):

        """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        title: (optional) Figure title
        show_mask, show_bbox: To show masks and bounding boxes or not
        figsize: (optional) the size of the image
        colors: (optional) An array or colors to use with each object
        captions: (optional) A list of strings to use as captions for each object
        """
        # Number of instances
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        # If no axis is passed, create one and automatically call show()
        auto_show = False
        if not ax:
            _, ax = plt.subplots(1, figsize=figsize)
            auto_show = True

        # Generate random colors
        colors = random_colors(N)

        # Show area outside image boundaries.
        height, width = image.shape[:2]
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')
        ax.set_title(title)

        masked_image = image.astype(np.uint32).copy()
        for i in range(N):
            #color = colors[i]

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                    alpha=0.7, linestyle="dashed",
                                     facecolor='none')
                ax.add_patch(p)

            # Label
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            ax.text(x1, y1 + 8, caption, size=11, backgroundcolor="none")

            # Mask
            mask = masks[:, :, i]
            if show_mask:
                masked_image = apply_mask(masked_image, mask)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none",)
                ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8))
        if auto_show:
            plt.savefig('/home/amboo/Desktop/Road/static/crack.png')
            plt.savefig('/home/amboo/Desktop/Road/foo.png')
    pic_ = 'crack.png'

    display_instances(original_image, gt_bbox, gt_mask, gt_class_id, train_set.class_names, figsize=(8, 8))


    return render_template('results1.html', name = pic_)





    #show = visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                        #    train_set.class_names, figsize=(8, 8))




    # redirect to home if no images to display
    #if "file_urls" not in session or session['file_urls'] == []:
        #return redirect(url_for('index'))

    # set the file_urls and remove the session variable
    #file_urls = session['file_urls']
    #session.pop('file_urls', None)


if __name__ == '__main__':

    app.run(port=5000, debug=True)
