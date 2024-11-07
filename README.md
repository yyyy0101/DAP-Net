# DAP-Net
-----Code----------------------
The source code of the proposed algorithm in this paper are given in this folder.
## Catalogue
1.Performance
2.Environment
3.Dataset processing
4.How to train
5.How to predict
6.Dataset Source
7.Key code interpretation 
8.Others
9.Conclusion
## Performance
| Training dataset | Weight file name | test dataset | Input image size | mAP 0.5:0.95 | mAP 0.5 |
| OpenSARShip-training | [self-training.h5]| OpenSARShip-test | 640x640 |mAP 0.5|
## Environment
|Package  |  Version |
|scipy==1.4.1.|
|numpy==1.18.4|
|matplotlib==3.2.1|
|opencv_python==4.2.0.34|
|tensorflow_gpu==2.2.0|
|tqdm==4.46.1|
|Pillow==8.2.0|
|h5py==2.10.0|
|tensorflow-gpu-estimator==2.2.0|
|termcolor==2.3.0|
|threadpoolctl==3.1.0|
|tqdm==4.46.1|
|opencv-python==4.2.0.34|
## Dataset processing
The training is conducted using the VOC format. 
We have provided examples of the .xml file in the trainingdataset_record.
## How to train
Attention mechanism modules and backbone networks have been given.
After configuring the environment parameters and preparing the training dataset, you are ready to train. 
## How to predict
Run predict.py for detection, and then enter the image path for detection.
model_path points to the trained weight file. 
Run get_map.py to obtain the evaluation results, which will be saved in the map_out folder.
## DataSet Source
The OpenSARShip dataset we used in this paper comes from:
Huang, L., Liu, B., Li, B., Guo, W., Yu, W., Zhang, Z., Yu, W.: OpenSARShip: A dataset dedicated to Sentinel-1 ship interpretation. IEEE J. Sel. Top. Appl. EarthObs. Remote Sens. 11(1), 195–208 (2018) 
https://doi.org/10.1109/JSTARS.2017.2755672
## Key code interpretation 
BackboneNet.py : Code for the backbone network.
attention.py : Code for the attention mechanism module used in this paper.
yolo.py : The full implementation of a YOLO-based object detection model with support for two images as input.
yolo_training.py : This code is used to compute the loss function for the YOLO (You Only Look Once) object detection model and define the learning rate scheduler.
## Others
Support different size model training, respectively, s, m, l, x version of yolov5.
## Conclusion
The training process involves feeding images into the YOLO model, where forward propagation generates predictions such as bounding boxes, class probabilities, and confidence scores. Then, the loss function calculates the discrepancy between the predictions and the true labels. The loss function includes location loss (such as CIoU), confidence loss, and class loss. An improved version of the loss function is also provided. 
Afterward, backpropagation computes the gradients, and the optimizer (such as Adam or SGD) updates the network's weights. This process is repeated for multiple iterations until the loss converges or the training reaches a predefined number of epochs. Once training is completed, the model is evaluated on the test set, and an optimized model capable of performing object detection is generated.
In this file, we provide the source code, including key modules such as dual input, attention mechanisms, loss functions, and backbone network modules. Additionally, we also provide the prediction code, MAP (mean average precision) calculation code, network output inspection code, and loss function comparison code.

