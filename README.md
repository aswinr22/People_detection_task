# Aerial People_detection_task

#### This task project based on opencv haar_caascade
run `python people_detection_task.py`

## problem
Take the aerial image attached. Detect and count the human in the image.
Compute net average IOU of the detections against the ground truth (Image
is not pre-labeled so candidate can generate the ground truth).
![alt text](https://github.com/aswinr22/People_detection_task/blob/main/Task-2.jpg)

## Approach
 1.First Select ROI so here iam masking unwated areas from the image and output look like this
![alt text](https://github.com/aswinr22/People_detection_task/blob/main/cropped.png)


##### 2.Crop peoples images from aerial image for training

#### Model
Iam using haar_cascade based detection the reason i chose because it takes short amount of time to train as well as detection and the catch is it won't overload the cpu.

#### Training 
These are the commands to training custom caascades

```
datay <cascade_dir_name>  : Where the trained classifier should be stored. This folder should be created manually beforehand.
vec <vec_file_name> : vec-file with positive samples (created by opencv_createsamples utility).
bg <background_file_name> : Background description file. This is the file containing the negative sample images.
numPos <number_of_positive_samples> : Number of positive samples used in training for every classifier stage.
numNeg <number_of_negative_samples> : Number of negative samples used in training for every classifier stage.
numStages <number_of_stages> : Number of cascade stages to be trained.
```
Here positve sample is only peoples and negatives are rest of other object in the images
Training is completed in lest than 15 minutes for 25 stages

#### post training
##### 1.Create blank image with same size and shape as of original 
##### 2.Draw bounding boxes on blank image 
##### 3.Blend it with original image using opencv `addedWeight()`function for final output and it looks like this.
![alt text](https://github.com/aswinr22/People_detection_task/blob/main/detection.png)

#### IOU Calculation
Generated ground thruth values :`(887,1530,958,1629),(1228,1668,1291,1770),(204,1662,285,1768),(3900,1838,3948,1947),(3543,1881,3617,2021),(2530,1179,2593,1245),(3200,1129,3238,1189),(2278,2395,2366,2488),(3595,1272,3633,1349),(1129,1735,1184,1834),(1504,1334,1558,1411),(3774,1265,3823,1337),(3224,1439,3284,1526),(3557,1258,3601,1333),(3494,1269,3545,1339),(3601,1370,3653,1445),(3520,1805,3600,1885),(1400,1648,1469,1749),(3500,1437,3579,1532),(2683,1379,2728,1458)`

#### Now getting good IOU value is bit challeging for me because the detection is in single image with multiple objects so detection order might change and it ruins the iou calculations so i couldnt find a better soultion for that till now but calculation is done and can be seen while running on cmd. 
