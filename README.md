# AI-POWERED VISION ASSISTANCE FOR VISUALLY CHALLENGED

## Description
The YOLO (You Only Look Once) v3 is an object detection algorithm that can be used for
various computer vision applications, including object detection for blind people. The system
architecture of object detection for blind people using YOLOv3 typically involves the following
components:


#### Input:
The system takes input from a camera or a pre-recorded video stream. In the case of blind
people, this input is usually captured by a wearable camera or a smartphone camera.


#### Preprocessing: 
The input frames captured by the camera are preprocessed to enhance the quality of
the images and normalize them for better detection performance. Preprocessing techniques may
include resizing, normalization, and noise reduction. 

#### YOLOv3 Model: 
The core component of the system is the YOLOv3 model, which is responsible for detecting objects in the input frames.
YOLOv3 uses a deep convolutional neural network (CNN) architecture with multiple layers. It
divides the input image into a grid and predicts bounding boxes and class probabilities for each grid
cell.


#### Training: 
Before deploying the system, the YOLOv3 model needs to be trained on a large dataset of
labeled images. The training process involves optimizing the model’s parameters to accurately detect
objects in various scenarios.


#### 
Object Detection: During the inference phase, the YOLOv3 model applies a series of convolutional
and pooling layers to extract features from the input image. These features are then used to predict
bounding boxes and class probabilities for potential objects in the image.


#### Audio Feedback: 
In the context of object detection for blind people, the final step is to convert the
visual information into audio feedback. The system use text-to speech synthesis to
communicate the detected objects to the user. The audio feedback include relevant information
such as object class, location, size and direction.
