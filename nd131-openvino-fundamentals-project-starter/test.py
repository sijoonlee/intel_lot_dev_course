import os
from labelMap import labels_map


model_xml = os.path.join('.',  '/yolov3-ir/frozen_darknet_yolov3_model.xml')
if __name__ == "__main__":
    print(model_xml)
    print(labels_map)

