![auto-annotate-logo](https://raw.githubusercontent.com/Lucs1590/auto_annotate/master/images/logo.png)
# Auto-annotate
### Welcome to the auto-annotate images for TensorFlow object detection!

You are tired to label your images by hand to work with object detection? So, this project will make your life easier, just annotate some images and let the machine do the rest for you!

## Requirements
- You will need to [clone the TensorFlow repository](https://github.com/tensorflow/models)
- Install the [dependencies](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/install.html) for object detection

**note:** This project is compatible with TF>=1.4
## How to run
- Copy and paste the file generate_xml.py and visualization_utils.py into the **research/object_detection/utils** in the tensorflow repo.
- Add your pre-treined model and label map into the 'graphs' folder.
- Add the images you want to label into the images folder
- Change the xml path in generate_xml.py to put your own local path.
- Inside the auto_annotate folder run: **python3 scripts/detection_images.py**
- If everything is ok you will see the inference results and the xml in your respective folders!
- If you have trouble or doubt check my [tutorial on medium](https://medium.com/@alvaroleandrocavalcante/auto-annotate-images-for-tensorflow-object-detection-19b59f31c4d9?sk=0a189a8af4874462c1977c6f6738d759)

## Any trouble?
Don't worry, open a issue and I'll hep you!
