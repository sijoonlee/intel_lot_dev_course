# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

### The process behind converting custom layers involves...
(ANSWER)   
1. Building model  
2. Creating the custom layer  
  - generate the extension template files using the model extension generator(one of the deployment tools)  
3. Generating IR files
  - use model optimizer
4. Executing the model with the custom layer(as extension)

### The poteintial reasons for handling custom layers in a trained model
(ANSWER)   
Customized layer would be needed for experimental or newly developed activation functions


## Comparing Model Performance

### My method(s) to compare models before and after conversion to Intermediate Representations
were...
(ANSWER) 
It is to compare the amount of time and the level of accuracy of inference. To achieve this goal, I modifed a [code repo](https://github.com/arunponnusamy/object-detection-opencv) by [Arun Ponnusamy](http://www.arunponnusamy.com)  
For pre-conversion model, I used OpenCV's yolov3 support with the same weight that I have used for this project.  
And performed a simple task to predict classes and draw binding boxes in a single image.  
The result is below

 
|          |Intel OpenVINO      | OpenCV yolov3 support|
|----------|--------------------|----------------------|
|Class     | Confidence         | Confidence           |
|dog       |   0.998923         | 0.99790170           |
|bicycle   |   0.996617         | 0.99009365           |
|truck     |   0.927796         | 0.93702490           |
|Time Spent|   1.65             | 1.99                 |

[Image - Post-conversion](https://github.com/sijoonlee/intel_lot_dev_course/blob/master/nd131-openvino-fundamentals-project-starter/images/object-detection-pre-conversion.jpg)  
[Image - Post-conversion](https://github.com/sijoonlee/intel_lot_dev_course/blob/master/nd131-openvino-fundamentals-project-starter/images/object-detection-pre-conversion.jpg)    
[Modified Code](https://github.com/sijoonlee/intel_lot_dev_course/tree/master/references/object-detection-opencv)


### The difference between model accuracy pre- and post-conversion was...  
(ANSWER)  
In term of accuracy, pre-conversion and post-conversion predicts very well - both of them made right prediction for each class - dog, bicycle, truck.  

However, the precision of confidence has decreased after convesion.
For example, OpenCV's yolov3 model produced 8 precision
On the other hand, OpenVINO model produced 6 precision

### The size of the model pre- and post-conversion was...  
(ANSWER)  
Pretrained Yolov3 weight file : 248,007,048 bytes  
Post-conversion model : 247,691,416 bytes  
The file size is decreased down, but really small difference  

###  The inference time of the model pre- and post-conversion was...  
(ANSWER)  
pre-conversion : 1.99 (s)  
post-conversion: 1.65 (s)  
It's about 17% decrease  
  
## Assess Model Use Cases

Some of the potential use cases of the people counter app are...
(ANSWER) Class attendence checking app  

Each of these use cases would be useful because...
(ANSWER) it would decrease the amount of time required for teachers to check the attendence

## Assess Effects on End User Needs

### Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...  
(ANSWER)
The constraints that the model is trained upon would affect the model's effectiveness for end-users
For example, if users need to use the model with a camera that is around 50 meters away from objects,  
it would be effective if using the model trained on such a constraint. 
For other instance, if the model is not trained properly on dark images,  
it would be no use for night-time images or video streams
Last but not least, image size and model accuracy are closely related with the speed of inference.  
If the model is for a scientific research that is required to be very precise, it would be better to use  
the moddel with large image and high level accuracy. On the other hand, if the model is for a simple  
task that doesn't need to be precise, it doesn't need large size of image input nor high accuracy.
