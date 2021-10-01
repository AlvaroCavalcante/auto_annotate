import os
import numpy as np
import argparse
from PIL import Image, ImageOps
import cv2

import tensorflow as tf

from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util


def infer_images(detect_fn, image, category_index, file_name, draw_image=True):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    image_np_with_detections = image_np.copy()

    if draw_image:
      viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          file_name=file_name,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    return image_np_with_detections

parser = argparse.ArgumentParser(description='Auto annotation arguments.')
parser.add_argument('--labelmap', help='The path of the label_map file.')
parser.add_argument('--saved_model', help='The path of the saved model folder.')
parser.add_argument('--imgs', help='The path of the images that will be annotated.')

args = parser.parse_args()

category_index = label_map_util.create_category_index_from_labelmap(args.labelmap,
                                                         use_display_name=True)

detect_fn = tf.saved_model.load(args.saved_model)

for img in os.listdir(args.imgs):
    try:
        file_name = img.split('.')[0]
        img = np.array(ImageOps.exif_transpose(Image.open(args.imgs+'/'+img)))
        result_img = infer_images(detect_fn, img, category_index, file_name)
                
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./results/'+ file_name + '.jpg', result_img)
    except Exception as e:
        print('Error to process image {}'.format(file_name))
        print(e)