#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore
from labelMap import labels_map
from yoloSupport import YoloParams, parse_yolo_region, intersection_over_union

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        return


    def load_model(self, model=os.path.join('/home/sijoonlee/Documents/intel-openvino-projects/yolov3/model',
                        'frozen_darknet_yolov3_model.xml'), device = 'CPU', cpu_extenstion=None):
        ### TODO: Load the model ###
        model_bin = os.path.splitext(model)[0] + ".bin"

        ### TODO: Add any necessary extensions ###
        self.plugin = IECore()
        if cpu_extenstion and 'CPU' in device:
            self.plugin.add_extension(cpu_extenstion, 'CPU')
        
        print("Loading network files:\n\t{}\n\t{}".format(model, model_bin))
        # self.network = IENetwork(model = model, weights = model_bin) # Deprecation
        self.network = self.plugin.read_network(model = model, weights = model_bin)
        
        ### TODO: Check for supported layers ###
        if "CPU" in device:
            supported_layers = self.plugin.query_network(self.network, "CPU")
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                        format(device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                        "or --cpu_extension command line argument")
                sys.exit(1)

        assert len(self.network.inputs.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"

        ### TODO: Return the loaded inference plugin ###
        self.exec_network = self.plugin.load_network(self.network, num_requests=1 ,device_name = device)
        
        ### set input/output blob
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        n, c, h, w = self.network.inputs[self.input_blob].shape
        return (n, c, h, w)

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        return

    def wait(self, request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_network.requests[request_id].wait(-1)
        return status

    def get_output(self, request_id):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[request_id].outputs



if __name__ == '__main__':
    prob_threshold = 0.6
    iou_threshold = 0.6
    infer_network = Network()

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model()

    ## infer_network.batch_size = 1
    # infer_network.input_blob = next(iter(infer_network.network.inputs))
    # n, c, h, w = infer_network.network.inputs[infer_network.input_blob].shape
    # infer_network.output_blob = next(iter(infer_network.network.outputs))
    n, c, h, w = infer_network.get_input_shape()

    # if args.labels:
    #     with open(args.labels, 'r') as f:
    #         labels_map = [x.strip() for x in f]
    # else:

    ### TODO: Handle the input stream ###
    # is_async_mode = True
    
    cap = cv2.VideoCapture('./resources/Pedestrian_Detect_2_1_1.mp4')
    #cap.open('./resources/Pedestrian_Detect_2_1_1.mp4')

    # number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 1394
    # number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames

    # Number of frames in picture is 1 and this will be read in cycle. Sync mode is default value for this case
    wait_key_code = 1
    # if number_input_frames != 1:
    # ret, frame = cap.read()
    # else:
    #     is_async_mode = False
    #     wait_key_code = 0

    width = int(cap.get(3)) # 768
    height = int(cap.get(4)) # 432
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    #fourcc = 0x00000021 # not working since I didn't include it when I compile OpenCV
    out = cv2.VideoWriter('./resources/out.mp4', fourcc, 30, (width, height))

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        # Here is the first asynchronous point: in the Async mode, we capture frame to populate the NEXT infer request
        # in the regular mode, we capture frame to the CURRENT infer request
      
        ### TODO: Read from the video capture ###
        # if is_async_mode:
        #     ret, next_frame = cap.read()
        # else:
        ret, frame = cap.read()

        if not ret:
            break

        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        cur_request_id = 0
        next_request_id = 1
        # if is_async_mode:
        #     request_id = next_request_id
        #     in_frame = cv2.resize(next_frame, (w, h))
        # else:
        request_id = cur_request_id
        in_frame = cv2.resize(frame, (w, h))

        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        # print(in_frame.shape) # (1, 3, 416, 416)

        ### TODO: Start asynchronous inference for specified request ###        
        infer_network.exec_network.start_async(request_id=0, inputs={infer_network.input_blob: in_frame})

        
        ### TODO: Wait for the result ###
        
        # Collecting object detection results
        objects = list()

        
        if infer_network.exec_network.requests[cur_request_id].wait(-1) == 0:
            output = infer_network.exec_network.requests[cur_request_id].outputs

            for layer_name, out_blob in output.items():
                # out_blob = out_blob.reshape(infer_network.network.layers[infer_network.network.layers[layer_name].parents[0]].shape) # Deprecation
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

        # print(frame.shape) # 432, 768, 3
        origin_im_size = frame.shape[:-1]
        
        for obj in objects:
            # Validation bbox of detected object
            if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
                continue
            color = (int(min(obj['class_id'] * 12.5, 255)), min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
            det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else str(obj['class_id'])
            cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
            cv2.putText(frame,
                        "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                        (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            
            print("{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(
                det_label, obj['confidence'], obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], color))
        out.write(frame)
        key = cv2.waitKey(wait_key_code)

        # ESC key
        if key == 27:
            break
        
    out.release()
    cap.release()
    cv2.destroyAllWindows()





