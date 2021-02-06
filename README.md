# People_detection_task

This is TAsk project based o opencv haar_caascade

## problem
Take the aerial image attached. Detect and count the human in the image.
Compute net average IOU of the detections against the ground truth (Image
is not pre-labeled so candidate can generate the ground truth).

## Approach
 1.First Select ROI so here iam masking unwated areas from the image and output look like this
![alt text](https://github.com/aswinr22/People_detection_task/blob/main/cropped.png)


##### 2.Crop peoples images from aerial image for train

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
1.Create blank image with same size and shape as of original 
2.Draw bounding boxes on black image 
3.Blend it with original image for final output and it looks like this
