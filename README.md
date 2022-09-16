![auto-annotate-logo](https://raw.githubusercontent.com/Lucs1590/auto_annotate/master/images/logo.png)
# Auto Annotation Tool for TensorFlow Object Detection
Are you tired to label your images by hand when working with object detection? Have hundreds or thousands of images to label? Then this project will make your life easier, just create some annotations and let the machine do the rest for you!

# Contents
- [How it works](#how)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
    - [Command line](#command-line)
    - [Code](#code)
- [Contribute](#contribute)

# :thinking: How it works <a id="how"></a>
This auto annotation tool is based on the idea of a semi-supervised architecture, where a model trained with a small amount of labeled data is used to produce the new labels for the rest of the dataset.

As simple as that, the library uses an initial and simplified object detection model to generate the XML files with the image annotations (considering the PASCAL VOC format).
Besides that, it's possible to define a confidence threshold for the detector, acting as a trade-off for the generated predictions.

If you want to know more technical details about the project, please, refer to my Medium article (*coming soon*).

# 📝 Prerequisites <a id="prerequisites"></a>
To use this library you will need a pre-trained object detection model with a subsample of your dataset. As a semi-supervised solution, it's impossible to avoid manual annotation, but you'll need to label just a small amount of your data.

It's hard to determine the number of images to label manually, once it depends on the complexity of your problem. If you want to detect dogs and cats and have 2000 images in your dataset, for example, probably 200 images are enough (100 per class). On the other hand, if you have dozens of classes or objects that are hard to detect, you should need more manual annotations to see the benefits of the semi-supervised approach.

After training this initial model, export your best checkpoint to the [SavedModel](https://www.tensorflow.org/guide/saved_model) format and you'll be ready to use the auto annotation tool!

# :floppy_disk: Installation <a id="installation"></a>
It's recommended to use a Python [virtual environment](https://docs.python.org/3/library/venv.html) to avoid any compatibility issue with your TensorFlow version. 

In your environment, you can install the project using pip:
```
$ pip install auto-annotate
```

# :man_scientist: Usage <a id="usage"></a>
You can use this tool either from the command line or directly in your Python code. For both, you'll have the same set of parameters:
- saved_model_path: The path of the **saved_model** folder with the initial model.
- label_map_path: The path of the **label_map.pbtxt** file.
- imgs_path: The path of the folder with your dataset images to label.
- xml_path (**optional**): Path to save the resulting XML files. The default behavior is to save in the same folder of the dataset images.
- threshold: Confidence threshold to accept the detections made by the model. the defaults is 0.5.

## Command line <a id="command-line"></a>
To use this tool from the command line, you just need to run:
```
python -m auto_annotate --label_map_path /example/label_map.pbtxt \
--saved_model_path /example/saved_model \
--imgs_path /example/dataset_images \
--xml_path /example/dataset_labels \
--threshold 0.65
```
## Code <a id="code"></a>
To use this tool from your Python code, check the following code snippet:
```python
from auto_annotate import AutoAnnotate

ann_tool = AutoAnnotate(
              saved_model_path = '/example/saved_model',
              label_map_path = '/example/label_map.pbtxt',
              images_path = '/example/dataset_images',
              xml_path = '/example/dataset_labels',
              detection_threshold = 0.65)

ann_tool.generate_annotations()
```
# :handshake: Contribute <a id="contribute"></a>
Contributions are welcome! Feel free to open a new issue if you have any problem to use the library of find a bug!

You can also use the [discussions](https://github.com/AlvaroCavalcante/auto_annotate/discussions) section to suggest improvements and ask questions!
