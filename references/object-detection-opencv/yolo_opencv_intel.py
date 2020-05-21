#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np
from time import time
from yoloSupport import YoloParams, parse_yolo_region, intersection_over_union
from inference import Network

DEFAULT_MODEL_PATH = '/home/sijoonlee/Documents/intel-openvino-projects/yolov3/model/frozen_darknet_yolov3_model.xml'

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-m', '--model',
                help = 'path to yolo model', default=DEFAULT_MODEL_PATH)
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


start_time = time()    
image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


########################
prob_threshold = 0.5
iou_threshold = 0.5
infer_network = Network()
infer_network.load_model(args.model, 'CPU', None) # model_path, device, extension
net_input_shape = infer_network.get_input_shape()

image_for_inference = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
image_for_inference = image_for_inference.transpose((2,0,1)) # h,w,c -> c,h,w
image_for_inference = image_for_inference.reshape(1,*image_for_inference.shape)
infer_network.exec_net(image_for_inference)
objects = list()
if infer_network.wait(0) == 0:
    ### TODO: Get the results of the inference request ###
    output = infer_network.get_output(0)
    ### TODO: Extract any desired stats from the results ###
    for layer_name, out_blob in output.items():
        out_blob = out_blob.reshape(infer_network.network.layers[infer_network.network.layers[layer_name].parents[0]].out_data[0].shape)
        layer_params = YoloParams(infer_network.network.layers[layer_name].params, out_blob.shape[2])
        layer_params.log_params()
        objects += parse_yolo_region(out_blob, image_for_inference.shape[2:],
                                        image.shape[:-1], layer_params,
                                        prob_threshold)

objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
for i in range(len(objects)):
    if objects[i]['confidence'] == 0:
        continue
    for j in range(i + 1, len(objects)):
        if intersection_over_union(objects[i], objects[j]) > iou_threshold:
            objects[j]['confidence'] = 0

objects = [obj for obj in objects if obj['confidence'] >= prob_threshold]


if len(objects):
    print(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")
    
origin_im_size = image.shape[:-1]


for obj in objects:
    # Validation bbox of detected object
    if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
        continue
    color = (int(min(obj['class_id'] * 12.5, 255)), min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
    det_label = classes[obj['class_id']] if classes and len(classes) >= obj['class_id'] else str(obj['class_id'])
    cv2.rectangle(image, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
    cv2.putText(image,
                "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)


    print("{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(
        det_label, obj['confidence'], obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], color))

print(time() - start_time)

cv2.imshow("object detection", image)
cv2.waitKey()
cv2.imwrite("object-detection-post-conversion.jpg", image)
cv2.destroyAllWindows()
############################