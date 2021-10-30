![auto-annotate-logo](https://raw.githubusercontent.com/Lucs1590/auto_annotate/master/images/logo.png)
# Auto Annotation Tool for TensorFlow Object Detection

Are you tired to label your images by hand to work with object detection? Have hundreds or thousands of images to label? Then this project will make your life easier, just create some annotations and let the machine do the rest for you!

## Contents
- [Requirements](#requirements)
- [How to run](#usage)
    - [TensorFlow < 2.x](#minor)
    - [TensorFlow >= 2.x](#greater)
- [Any trouble?](#trouble)


## Requirements <a id="requirements"></a>
- You will need to [clone the TensorFlow repository](https://github.com/tensorflow/models)
- Install the [dependencies](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/install.html) for object detection

## How to run <a id="usage"></a>
- Copy and paste the files **generate_xml.py** and **visualization_utils.py** into the **research/object_detection/utils** in the tensorflow repo.
- Change the xml path in generate_xml.py to put your own local path.
- Add the images you want to label into the images folder

### TensorFlow < 2.x <a id="minor"></a>
- Add your pre-treined model (as a fronzen inference graph) and label map into the 'graphs' folder.
- Inside the auto_annotate folder run: **python3 scripts/detection_images.py**
### TensorFlow >= 2.x <a id="greater"></a>
- If you have TF 2.x, just run the following command:
```
python3 scripts/detection_img_tf2.py --saved_model /path-saved-model --labelmap /path-label-map.pbtxt --imgs /path-of-the-imgs
```

- If it runs correctly, you will see the inference results and the xml in your respective folders!

## Any trouble? <a id="trouble"></a>
If you have trouble or doubt check my [tutorial on medium](https://medium.com/@alvaroleandrocavalcante/auto-annotate-images-for-tensorflow-object-detection-19b59f31c4d9?sk=0a189a8af4874462c1977c6f6738d759). You can also open an issue and I'll hep you!
