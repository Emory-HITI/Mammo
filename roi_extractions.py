import os
import time
import datetime
# libraries for checking mammo image to ROI image
import SimpleITK as sitk
from skimage.transform import resize
from skimage.util import compare_images

# other libraries for ROI detection and inference
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import seaborn as sn
import pathlib
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import random
import io
import cv2
import imageio
import glob
import math
import shutil
import scipy.misc
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
#from object_detection.utils import colab_utils
from object_detection.builders import model_builder

import wandb

#%matplotlib inline

# class for ROI extractions
class ROI_extraction():
    
    def __init__(self, model_dep_path):
        
        sys.path.insert(1, model_dep_path + '/models/')

        self.chosen_model = 'efficientdet-d0'
        self.pbtxt_fname = model_dep_path + '/OD_Files/label_map.pbtxt' #Label map file 
        self.pipeline_config = model_dep_path + '/OD_Files/ssd_efficientdet_d0_512x512_coco17_tpu-8.config' #Config file required for the model
        self.ckpt_dir = model_dep_path + '/training/efficientdet-d0/ckpt-20'  # update checkpoint when available
        self.class_mapping = {1: 'ROI'}
        
        self.load_detection_fnt()
        self.load_metrics()
    
    def load_detection_fnt(self):
        detection_model = self.load_model()

        label_map = label_map_util.load_labelmap(self.pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        self.detect_fn = self.get_model_det_fun(detection_model)
    
    # ---functions from Raman's code---
    def load_model(self):
        configs = config_util.get_configs_from_pipeline_file(self.pipeline_config)
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(self.ckpt_dir))

        return detection_model

    def get_model_det_fun(self, model):

        @tf.function(experimental_relax_shapes=True)
        def detect_fn(image):
            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            detections = model.postprocess(prediction_dict, shapes)

            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn

    def load_image_into_numpy_array(self, path, cvt_to_grayscale=False):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
        path: the file path to the image

        Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
        """
        #img_data = tf.io.gfile.GFile(path, 'rb').read()
        #image = Image.open(BytesIO(img_data))
        #(im_width, im_height) = image.size
        image = cv2.imread(path)

        #Converting Grayscale to RGB
        if(cvt_to_grayscale):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image # removed: .astype(np.uint8) since it should read as 16 bit but will change to .astype(np.uint16) if it doesn't work
        # since data is loaded from 16bit, maybe change to uint16

        return image

    def get_img_dims(self, path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image.shape[0], image.shape[1]

    def get_coordinates(self, img_name, detections, height, width, threshold=0.5, 
                        verbose=False, plot_image=False, save_image=False, 
                        chosen_model=None, cvt_to_grayscale=False, flip_flag=0,
                       return_image=False):

        color = (0, 255, 0)
        thickness = 4
        label_id_offset = 1
        image_np = self.load_image_into_numpy_array(img_name)
        if(cvt_to_grayscale):
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        if(flip_flag==1):
            image_np = cv2.flip(image_np, 1)
        image_np = image_np.astype(np.uint8)  # shouldn't matter because it's coordinates
        score_array = detections['detection_scores'][0].numpy()
        boxes = detections['detection_boxes'][0].numpy()
        classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)

        positions = []
        true_classes = []
        true_boxes = []
        coords = []
        scores = []

        for i, score in enumerate(score_array):
            if(score > threshold):
                positions.append(i)
                scores.append(score)

        if(len(positions) == 0):
            #No Detection
            return [], []

        for i, box in enumerate(boxes):
            if(i in positions):
                true_boxes.append(box)
                true_classes.append(self.class_mapping[classes[i]])

        for i in range(len(true_boxes)):
            box = true_boxes[i]
            ymin, xmin, ymax, xmax = int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)
            coords.append([ymin, xmin, ymax, xmax])

        for i in range(len(coords)):
            ymin, xmin, ymax, xmax = coords[i][0], coords[i][1], coords[i][2], coords[i][3]
            if(verbose):
                print("CLASS DETECTED: ", true_classes[i])
                print("Confidence: ", scores[i])
                print("ymin={}, xmin={}, ymax={}, xmax={}".format(ymin, xmin, ymax, xmax))

            if(i==0):
                image = cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color, thickness)
            else:
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

        if(plot_image):
            plt.figure(figsize=(12,16))
            plt.imshow(image)

        if(save_image):
            if(chosen_model == None):
                raise ValueError('Please provide a value for "Chosen Model". ')
            else:
                #cv2.imwrite('annotated_Results_TF/'+chosen_model+'/'+image_paths[i].split('/')[-1], image)
                cv2.imwrite('mammo_images_Hari/good_ROIs/no_detections/'+img_name.split('/')[-2]+'_'+img_name.split('/')[-1], image)

        if return_image:
            return coords, true_classes, image
        else:
            return coords, true_classes

    def get_detections(self, image_path, cvt_to_grayscale=False, flip_flag=0):
        '''
        A function that takes in an image path and return the detections from the model
        '''

        image_np = self.load_image_into_numpy_array(image_path)
        #if(cvt_to_grayscale):
        #    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        if(flip_flag==1):
            image_np = cv2.flip(image_np, 1)
        #image_np = image_np.astype(np.uint8)

        input_tensor = tf.convert_to_tensor(
                       np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = self.detect_fn(input_tensor)
        label_id_offset = 1

        return detections

    def create_label_map(self):
        configs = config_util.get_configs_from_pipeline_file(self.pipeline_config)
        label_map_path = configs['eval_input_config'].label_map_path
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

        return label_map_dict

    def visualize_quick(self, img_path):

        image_np = self.load_image_into_numpy_array(img_path)
        if(image_np.shape != 3):
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = self.detect_fn(input_tensor)
        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.2, #EXPERIMENT WITH THIS THRESHOLD
          agnostic_mode=False)


        plt.figure(figsize=(10,10))
        plt.imshow(image_np_with_detections)
        plt.show()

    def checkImageSimilarity(self, roi_path, mammo_paths, sim_metric, hist_matcher, img_on=False):
        # reading the ROI image
        roi_img_path = roi_path  # directly put the path
        roi_sitk = sitk.ReadImage(roi_img_path)
        roi_np = sitk.GetArrayFromImage(roi_sitk)

        # flipping the ROI image
        roi_np_flip = np.flip(roi_np, axis=1)

        # initializing for choosing the best
        best_sim = 0
        best_match = None
        flip_flag = 0

        # for all other files
        for i in mammo_paths:
            # loads and shows the next image
            img_path = i # directly put the path
            img_sitk = sitk.ReadImage(img_path)
            img_sitk = hist_matcher.Execute(img_sitk, roi_sitk)
            img_np = sitk.GetArrayFromImage(img_sitk)

            # check the aspect ratio of the image (sanity check and speeds up code by filtering out images)
            if np.round(roi_np.shape[0]/roi_np.shape[1], 2) != np.round(img_np.shape[0]/img_np.shape[1], 2):
                continue

            # resizing and changing the datatype from float (result of resize) to unit8 for similarity check compatability
            img_np_resize = resize(img_np, roi_np.shape, preserve_range=True).astype(np.uint16)

            # converts to sitk images again to do similarity check
            roi_img = sitk.GetImageFromArray(roi_np)
            img_img = sitk.GetImageFromArray(img_np_resize)
            roi_flip_img = sitk.GetImageFromArray(roi_np_flip)

            # gets similarity matrix of original
            sim_metric.Execute(roi_img, img_img)
            sim_o = sim_metric.GetSimilarityIndex()  # calculate signature
            # (pixel by pixel or extracted features if extracted features, extract index of the image not the full data)

            # gets similarity matrix of flipped
            sim_metric.Execute(roi_flip_img, img_img)
            sim_f = sim_metric.GetSimilarityIndex()

            # get scale factor (between the roi image and the mammo)
            x_ratio = img_np.shape[1]/roi_np.shape[1]
            y_ratio = img_np.shape[0]/roi_np.shape[0]
            scale_factor = np.mean([x_ratio, y_ratio])

            # get max similarity for the image
            if sim_o > sim_f:
                img_sim_max = sim_o
                flip_flag = 0
            else:
                img_sim_max = sim_f
                flip_flag = 1

            # if the image similarity is better than previous ones and update with better ones
            if img_sim_max > best_sim: 
                best_sim = img_sim_max
                best_match = i
                best_img = img_np_resize
                final_flip = flip_flag
                final_scale = scale_factor
                if sim_o == best_sim:
                    best_roi = roi_np
                else:
                    best_roi = roi_np_flip
            # if images should be shown:
            if img_on:
                # shows all figures side by side:
                f = plt.figure(figsize=(10,15))
                f.add_subplot(1,3, 1)
                plt.imshow(roi_np, cmap='gray')
                plt.title('ROI')
                f.add_subplot(1,3, 2)
                plt.imshow(img_np_resize, cmap='gray')
                plt.title('Mammo')
                f.add_subplot(1,3, 3)
                plt.imshow(roi_np_flip, cmap='gray')
                plt.title('Flipped ROI')
                plt.show(block=True)
                print('Similarity: {}\n'.format(img_sim_max))

            #print(i)
        #print(best_match)

        return best_sim, best_match, best_img, best_roi, final_flip, scale_factor

    def extrapolate_coords(self, roi_path, coords, flip_flag, scale_factor, verbose=False):
        """
        returns the screensave and mammogram coordinates based on the image, flip flag, and scale factor
        """
        # read roi image
        if verbose:
            print('reading {} images'.format(roi_path))
        roi_sitk = sitk.ReadImage(roi_path)
        if verbose:
            print('making array from image')
        roi_np = sitk.GetArrayFromImage(roi_sitk)
        # if flip_flag = 1, flip the roi image
        if verbose:
            print('flipping the image')
        if flip_flag == 1:
            roi_np = np.flip(roi_np, axis=1)
            roi_width = roi_np.shape[1]
            coords = [coords[0], roi_width - coords[3], coords[2], roi_width - coords[1]]
        # scaling up the roi_coordinates to the mammogram image
        if verbose:
            print('scaling up the coordinates')
        mam_coord = [int(np.round(i*scale_factor)) for i in coords]
        if verbose:
            print(mam_coord)

        return coords, mam_coord
    
    def load_metrics(self):
        """
        loads the metrics required for image similarity checks with SITK
        """
        # setting up metrics
        self.metric = sitk.SimilarityIndexImageFilter()  # set similarity metric as SimilarityIndexImageFilter - precalculate simIndex
        # set up the histogram matcher
        self.matcher = sitk.HistogramMatchingImageFilter()
        self.matcher.SetNumberOfHistogramLevels(256)
        self.matcher.SetNumberOfMatchPoints(7)
        self.matcher.ThresholdAtMeanIntensityOn()

    # actual ROI extraction step in a function
    def ROI_Extract(self, roi_filename, roi_dict, start_time, save_path=None, verbose=False):
        """
        for a given ROI_ssc image, obtains the closest matching mammogram (with image similarity check), extracts the ROI coordinates for the ROI_ssc and mammogram, and returns the roi path, roi coordinates, mammogram path, and mammogram coordinates.
        
        really just needs an roi filename and folder path of the roi that is not mammograms
        """
        if verbose:
            print(datetime.datetime.now() - start_time)
            print('extracting files from ROI directory...')
        # gets the roi_path from joining the path and filename from the same column
        roi_path = os.path.join(roi_dict[roi_filename], roi_filename)
        # gets the list of files from first getting a list of files with the same path - need to change
        mammo_files = [i for i in os.listdir(roi_dict[roi_filename]) if i not in roi_dict.keys()]  # gets the filenames for everything that's not an ROI_ssc
        mammo_paths = [os.path.join(roi_dict[roi_filename], file) for file in mammo_files]
        # runs the similarity check
        if verbose:
            print(datetime.datetime.now() - start_time)
            print('running similarity check...')
        sim_score, match_path, img_array, roi_array, flip_flag, scale_factor = self.checkImageSimilarity(roi_path, mammo_paths, self.metric, self.matcher, False)
        if verbose:
            print(datetime.datetime.now() - start_time)
            print('checking scaling factor...')
        # double checking scale_factor
        img_sitk = sitk.ReadImage(match_path)
        img_np = sitk.GetArrayFromImage(img_sitk)
        #print('mammogram bit depth: {}'.format(img_np.dtype))
        roi_sitk = sitk.ReadImage(roi_path)
        roi_np = sitk.GetArrayFromImage(roi_sitk)
        #print('screen save bit depth: {}'.format(roi_np.dtype))

        # if scale factor is wrong, correct to the right scale factor
        if np.round(scale_factor, 3) != np.round(img_np.shape[0]/roi_np.shape[0], 3):
            print('Scale factor is off, correcting')
            scale_factor = img_np.shape[0]/roi_np.shape[0]

        # uses Raman's code to extract the ROI from the screen save image
        img = self.load_image_into_numpy_array(roi_path) #Load the image in a numpy array
        if(len(img.shape) != 3):
            img = self.load_image_into_numpy_array(roi_path, cvt_to_grayscale=True)
            shape_change_flag = 1
        if verbose:
            print(datetime.datetime.now() - start_time)
            print('getting detections...')
        #This return different objects that a model detects in an image
        detections = self.get_detections(roi_path, cvt_to_grayscale=True)
        if verbose:
            print(datetime.datetime.now() - start_time)
            print('getting coordinates of detected ROIs...')
        #This function return the coordinates and the classes
        coords, classes = self.get_coordinates(roi_path, detections, img.shape[0], img.shape[1], threshold=0.5, 
                                              verbose=False, plot_image=False, save_image=False, chosen_model=self.chosen_model, 
                                              cvt_to_grayscale=False, return_image=False)
        if verbose:
            print(datetime.datetime.now() - start_time)
            print('extrapolating and scaling up SSC coordinates to Mammograms...')
            print(coords, classes)
        # ROI coordinates:
        extracted_mam_coord = []
        matching_mammo = []
        # going through all detected ROIs
        for i in range(len(classes)):
            # if the detection is an ROI
            if classes[i] == 'ROI':
                # extract and save the ROI visualization
                ss_coord, mam_coord = self.extrapolate_coords(roi_path, coords[i], flip_flag, scale_factor, verbose=verbose)
                # add to list of coordinates
                extracted_mam_coord.append(mam_coord)
                matching_mammo.append(match_path)
        if verbose:
            print(ss_coord)
            print(roi_path)
            print(extracted_mam_coord)
            print(matching_mammo)

        return ss_coord, roi_path, extracted_mam_coord, matching_mammo
    
    def run_extractions(self, ss_dict, save_path=None, verbose=False):
        # test on cohort_1 to check memory
        ROI_coords_ssc = []
        ROI_coords_mammo = []
        ROI_matching_ssc = []
        ROI_matching_mammo = []

        compare_save_path = save_path  # necessary if output images desired

        start = datetime.datetime.now()  # check start time

        n = 0
        for i in ss_dict.keys():  # HITIdf_ssc.shape[0]
            try:
                extracted_ssc_coord, ssc_path, extracted_mam_coord, matching_mammo = self.ROI_Extract(i, ss_dict, start_time=start, save_path=save_path, verbose=verbose)
                if verbose:
                    print(datetime.datetime.now() - start)  # check elapsed time
            except:
                if verbose:
                    print('No ROI detected')
                extracted_ssc_coord = []
                extracted_mam_coord = []
                ssc_path = []
                matching_mammo = []
                if verbose:
                    print(datetime.datetime.now() - start)  # check elapsed time
            if verbose:
                print('Progress:{}/{}'.format(n+1, len(ss_dict.keys())))
                print('SSC_coord: {}'.format(extracted_ssc_coord))
                print('Mammo_coord: {}'.format(extracted_mam_coord))
                print('SSC_path: {}'.format(ssc_path))
                print('Mammo_path: {}'.format(matching_mammo))
            n += 1
            ROI_coords_ssc.append(extracted_ssc_coord)
            ROI_coords_mammo.append(extracted_mam_coord)
            ROI_matching_ssc.append(ssc_path)
            ROI_matching_mammo.append(matching_mammo)

        time_elapse = datetime.datetime.now() - start  # check elapsed time
        if verbose:
            print(time_elapse)
        
        return ROI_coords_ssc, ROI_coords_mammo, ROI_matching_ssc, ROI_matching_mammo