"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from yoloSupport import YoloParams, parse_yolo_region, intersection_over_union
from labelMap import labels_map

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-it", "--iou_threshold", default=0.4, type=float,
                        help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = None

    return client



def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    iou_threshold = args.iou_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension) # model_path, device, extension
    net_input_shape = infer_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    #cap.open(args.i)
    width = int(cap.get(3)) # 768
    height = int(cap.get(4)) # 432
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    #fourcc = 0x00000021 # not working since I didn't include it when I compile OpenCV
    out = cv2.VideoWriter('./resources/out.mp4', fourcc, 30, (width, height))

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        request_id = 0
        in_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        in_frame = in_frame.transpose((2,0,1))
        in_frame = in_frame.reshape(1,*in_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(in_frame)

        ### TODO: Wait for the result ###

        # Collecting object detection results
        objects = list()
        if infer_network.wait(request_id) == 0:
            ### TODO: Get the results of the inference request ###
            output = infer_network.get_output(request_id)
            ### TODO: Extract any desired stats from the results ###
            for layer_name, out_blob in output.items():
                out_blob = out_blob.reshape(infer_network.network.layers[infer_network.network.layers[layer_name].parents[0]].out_data[0].shape)
                layer_params = YoloParams(infer_network.network.layers[layer_name].params, out_blob.shape[2])
                layer_params.log_params()
                objects += parse_yolo_region(out_blob, in_frame.shape[2:],
                                             frame.shape[:-1], layer_params,
                                             prob_threshold)
            
        # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
        objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
        for i in range(len(objects)):
            if objects[i]['confidence'] == 0:
                continue
            for j in range(i + 1, len(objects)):
                if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                    objects[j]['confidence'] = 0

        # Drawing objects with respect to the --prob_threshold CLI parameter
        objects = [obj for obj in objects if obj['confidence'] >= prob_threshold]

        if len(objects):
            print("\nDetected boxes for batch {}:".format(1))
            print(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")
            
        origin_im_size = frame.shape[:-1]
        
        person_counter = 0
        for obj in objects:
            # Validation bbox of detected object
            if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
                continue
            color = (int(min(obj['class_id'] * 12.5, 255)), min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
            det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else str(obj['class_id'])
            if det_label == 'person':
                person_counter += 1
            cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
            cv2.putText(frame,
                        "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                        (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            
            print("# of People: {}", person_counter)
            print("{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(
                det_label, obj['confidence'], obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], color))
        out.write(frame)

        # ESC key
        if key_pressed == 27:
            break
        
    out.release()
    cap.release()
    cv2.destroyAllWindows()            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
    # client = connect_mqtt()
    # /usr/bin/python3 /home/sijoonlee/Documents/intel_lot_dev_course/nd131-openvino-fundamentals-project-starter/main.py -m /home/sijoonlee/Documents/intel-openvino-projects/yolov3/model/frozen_darknet_yolov3_model.xml -d CPU -pt 0.8 -it 0.8 -i /home/sijoonlee/Documents/intel_lot_dev_course/nd131-openvino-fundamentals-project-starter//resources/Pedestrian_Detect_2_1_1.mp4