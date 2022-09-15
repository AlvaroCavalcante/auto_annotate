"""Generate xml annotations for your TensorFlow Object Detection models.

This library is used to generate xml annotations and simplify the process
of creating a labeled dataset. It uses a pre-trained model to detect objects
and generates the annotations for each detected object based on a threshold.

Example:

    auto_ann = AutoAnnotate(
        'path/to/saved_mdel',
        'path/to/label_map.pbtxt',
        'path/to/images'
    )
    auto_ann.generate_annotations()
"""

import glob
import argparse

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tqdm import tqdm

import label_map_util
import generate_xml


class AutoAnnotate():
    """Class used to generate xml annotations for TensorFlow object detection models.

    Attributes:
        model: A TensorFlow model used to detect objects.
        category_index: A dictionary containing the class names from label map.
        images_path: A string containing the path to the images to annotate.
        images: A list containing the complete path of all images to annotate.
        xml_path: A string containing the path to save the xml annotations.
        detection_threshold: A float containing the threshold to filter detections.
        xml_generator: An instance of GenerateXml class.
    """

    def __init__(self, saved_model_path: str, label_map_path: str, images_path: str, xml_path: str = None, detection_threshold: float = 0.5) -> None:
        self.model = load_detection_model(saved_model_path)
        self.category_index = load_label_map(label_map_path)
        self.images_path = images_path
        self.images = glob.glob(self.images_path+'/*')
        self.xml_path = xml_path if xml_path else self.images_path
        self.detection_threshold = detection_threshold
        self.xml_generator = generate_xml.GenerateXml(self.xml_path)

    def generate_annotations(self) -> None:
        """Iterates over all images and generates the annotations for each one."""

        print(f'Found {len(self.images)} images to annotate.')

        for image in tqdm(self.images, colour='green'):
            try:
                img = np.array(ImageOps.exif_transpose(Image.open(image)))
                im_height, im_width, _ = img.shape

                detections = self._get_model_detections(img)
                class_names, bounding_boxes = self._filter_detections_by_threshold(
                    detections['detection_scores'],
                    detections['detection_classes'],
                    detections['detection_boxes'],
                    im_height,
                    im_width
                )
                file_name = image.split('/')[-1]
                self.xml_generator.generate_xml_annotation(
                    file_name,
                    bounding_boxes,
                    im_width,
                    im_height,
                    class_names
                )
            except Exception as error:
                print(error)

    def _get_model_detections(self, image) -> dict:
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.model(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(
            np.int64)

        return detections

    def _filter_detections_by_threshold(self, scores: list, classes: list, boxes: list, heigth: int, width: int) -> tuple:
        class_names = []
        bounding_boxes = []

        for index in np.where(scores > self.detection_threshold)[0]:
            class_names.append(self.category_index[classes[index]].get('name'))
            xmin, xmax, ymin, ymax = self._get_box_coordinates(
                boxes, heigth, width, index)

            output_boxes = {'xmin': xmin, 'xmax': xmax,
                            'ymin': ymin, 'ymax': ymax}
            bounding_boxes.append(output_boxes)

        return class_names, bounding_boxes

    def _get_box_coordinates(self, boxes: list, heigth: int, width: int, index: int) -> tuple:
        xmin, xmax, ymin, ymax = boxes[index][1], boxes[index][3], boxes[index][0], boxes[index][2]
        xmin, xmax, ymin, ymax = int(
            xmin * width), int(xmax * width), int(ymin * heigth), int(ymax * heigth)

        return (xmin, xmax, ymin, ymax)


def load_detection_model(saved_model_path: str) -> tf.saved_model.load:
    """Loads an object detection model from path

    Args: saved_model_path (str): Path to saved model directory.

    Returns:
        A tf.saved_model.load object.
    """
    try:
        print('Loading model into memory...')
        return tf.saved_model.load(saved_model_path)
    except Exception as error:
        print(f'Error loading model: {error}')
        raise error


def load_label_map(label_map_path: str) -> dict:
    """Loads label map from pbtxt file.

    Args:
        label_map_path (str): Path to label map file.

    Returns:
        A dict containing the class names that are in the label map.
    """
    try:
        print('Loading label map...')
        return label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
    except Exception as error:
        print(f'Error loading label map: {error}')
        raise error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto annotation arguments.')
    parser.add_argument('--label_map_path', type=str, help='The path of the label_map.pbtxt file containing the classes.')
    parser.add_argument('--saved_model_path', type=str, help='The path of the saved model folder.')
    parser.add_argument('--imgs_path', type=str, help='The path of the images that will be annotated.')
    parser.add_argument('--xml_path', type=str, help='The path where the xml files will be saved.', default=None)
    parser.add_argument('--threshold', type=float, help='The path where the xml files will be saved.', default=0.5)

    args = parser.parse_args()

    AutoAnnotate(args.saved_model_path,
                 args.label_map_path,
                 args.imgs_path,
                 args.xml_path,
                 args.threshold).generate_annotations()
