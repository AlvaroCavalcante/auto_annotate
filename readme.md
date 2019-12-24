### Welcome to the auto-annotate images for object detection!

## Prerequisites
- You will need to clone the tensorflow repository: https://github.com/tensorflow/models
- Install the dependencies: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
- Install the xElementTree dependency via pip
## How to run
- Copy and paste the file generate_xml.py and visualization_utils.py into the **research/object_detection/utils** in the tensorflow repo.
- Add your pre-treined model and label map into the 'graphs' folder.
- Add the images you want to label into the images folder
- Change the xml path in generate_xml.py to put your own local path.