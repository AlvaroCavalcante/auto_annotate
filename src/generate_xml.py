import xml.etree.cElementTree as ET


class GenerateXml(object):
    def __init__(self, xml_path: str) -> None:
        self.xml_path = xml_path

    def generate_xml_annotation(self, file_name: str, bboxes: list, im_width: int, im_height: int, classes: list) -> None:
        try:
            annotation = ET.Element('annotation')
            ET.SubElement(annotation, 'filename').text = file_name
            size = ET.SubElement(annotation, 'size')
            ET.SubElement(size, 'width').text = str(im_width)
            ET.SubElement(size, 'height').text = str(im_height)
            ET.SubElement(size, 'depth').text = '3'

            for index, box in enumerate(bboxes):
                objectBox = ET.SubElement(annotation, 'object')
                ET.SubElement(objectBox, 'name').text = classes[index]
                ET.SubElement(objectBox, 'pose').text = 'Unspecified'
                ET.SubElement(objectBox, 'truncated').text = '0'
                ET.SubElement(objectBox, 'difficult').text = '0'
                bndBox = ET.SubElement(objectBox, 'bndbox')
                ET.SubElement(bndBox, 'xmin').text = str(box['xmin'])
                ET.SubElement(bndBox, 'ymin').text = str(box['ymin'])
                ET.SubElement(bndBox, 'xmax').text = str(box['xmax'])
                ET.SubElement(bndBox, 'ymax').text = str(box['ymax'])

            arquivo = ET.ElementTree(annotation)
            arquivo.write(f'{self.xml_path}/{file_name.split(".")[0]}.xml')
        except Exception as e:
            print('Error to generate the XML for image {}'.format(file_name))
            print(e)
