##created by Aswin R

#importing libraries
import matplotlib.pylab as plt
import numpy as np
import cv2

#loading haarcaascade
body_cascade=cv2.CascadeClassifier("C:/Users/pc/Desktop/cascade_for_people.xml")


start_point=28,400
end_point=891,400
color=2550,0,0
thickness=2
alpha=0.85
beta=1.0
IOU=[]
roi=[(0,3000),(10,1656),(1719,1056),(4020,1130),(4000,3000)]

#Selecting Roi to avoid unwanted detection
def masking(im,vertices):
	mask=np.zeros_like(im)
	channel_cout=im.shape[2]
	match_mask_color=(255,)*channel_cout
	cv2.fillPoly(mask,vertices,match_mask_color)
	masked_image=cv2.bitwise_and(im,mask)
	return masked_image

#function 'unique' to creating blank image
#purpose for blank image is blend masked and detected image with black image to show output detection 
def uniq(val):
	uniq.dup_img=np.copy(val)
	uniq.blacnk_img=np.zeros((uniq.dup_img.shape[0],uniq.dup_img.shape[1],3),dtype=np.uint8)
	
def detect_body(frame):
	uniq(simulator.im)
	count=0

	#Detection
	detect_body.body=body_cascade.detectMultiScale(frame,1.15,4)
	for(x,y,w,h) in detect_body.body:
		ox=cv2.rectangle(uniq.blacnk_img,(x,y),(x+w,y+h),(0,0,255),3)
		
		cv2.putText(ox,'People',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 2)
		
		#blending the images 
		detect_body.frame=cv2.addWeighted(uniq.dup_img,alpha,uniq.blacnk_img,beta,0.0)
		count=count+1
	print("Total count is {}".format(count))

	return frame

# iou Calculation
def iou_calculation():
	
	#ground truth values as gt
	gt=(887,1530,958,1629),(1228,1668,1291,1770),(204,1662,285,1768),(3900,1838,3948,1947),(3543,1881,3617,2021),(2530,1179,2593,1245),(3200,1129,3238,1189),(2278,2395,2366,2488),(3595,1272,3633,1349),(1129,1735,1184,1834),(1504,1334,1558,1411),(3774,1265,3823,1337),(3224,1439,3284,1526),(3557,1258,3601,1333),(3494,1269,3545,1339),(3601,1370,3653,1445),(3520,1805,3600,1885),(1400,1648,1469,1749),(3500,1437,3579,1532),(2683,1379,2728,1458)
	for(x1,y1,w1,h1),(truth) in zip(detect_body.body , gt) :
		predicted_box=x1,y1,x1+w1,y1+h1
				
		ground_truth=truth
				
		xA = max(ground_truth[0], predicted_box[0])
		yA = max(ground_truth[1], predicted_box[1])
		xB = min(ground_truth[2], predicted_box[2])
		yB = min(ground_truth[3], predicted_box[3])

		# compute the area of intersection rectangle
		interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
		# compute the area of both the prediction and ground-truth
					
		boxAArea = (ground_truth[2] - ground_truth[0] + 1) * (ground_truth[3] - ground_truth[1] + 1)
		boxBArea = (predicted_box[2] - predicted_box[0] + 1) * (predicted_box[3] - predicted_box[1] + 1)

		# compute the intersection over union by taking the intersection
		
		iou = interArea / float(boxAArea + boxBArea - interArea)
		
		IOU.append(iou)
	# Detected bounding box is interchanging so iou value will  change in each exceution and affects its value as well	
	print("AVG iou")
	print(sum(IOU)/20)			
	return iou
	
#main function
def simulator():
	
	
	simulator.im=cv2.imread("C:/Users/pc/Desktop/Task-2.jpg")
	im=simulator.im
	
	#cropping/masking/defining ROI
	cropped_image=masking(im,np.array([roi],np.int32))
	
	
	detect_body(cropped_image)
	
	
	cv2.imshow('frame',detect_body.frame)
	iou_calculation()
	controlkey=cv2.waitKey(0)
	
	cv2.destroyAllWindows()

if __name__=='__main__':
	simulator()

