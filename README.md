# Viola-Jones-face-detector
Implementation of Viola Jones Face detection algorithmn 
Implemented Viola Jones Detector: 
5 Haar types features were taken: 
Where haar==1 :5.
This was defined in the calculatehaar.m file. The integralImg.m function calculates the integral values for the points taken with the changing dimensions of the harr feature. The haar diensions changes as per the command line  haarX = dimX:dimX:window-pixelX and haarY = dimY:dimY:window-pixelY inside the window size pixelX = 2:window-dimX and pixelY = 2:window-dimY.  
Image weights are normalized by dividing 1/(facesize+nonfacesize) where face size is the number of face images and nonface is the number of non face images. 
Image weights are updated in the adaboost function on the basis of the false positives and false negatives. 
Adaboost assigns a capture value 1 to all correct captures on the basis of the harrVal that it generates. Also classifier weights are updated as per the error and new image weights  are passed on to the main script. 
Weakclassifiers  obtained is 221  
And strong classifiers are obtained by combining these weak classifiers to give one strong classifier. 
For training purpose 507 face images were taken and 1870 non face images from the same data set was chosen.  
These images were then resized to 24X24 size in order to get haar features on just that window. haarVector1 stores all the values for haar one in all the face images. And haarVector2 stores all the values for haar one in all non face images.  The loops take care that the total error is stays less than 50% for all the time.  
Results: 
False negative Capture Rate : 42/07=0.082 
False positive Capture Rate: 542/1860=0.291 
Accuracy 57% (approx.) 
For testing purpose the classifiers were run through the 24X24 sub image window and a union of the windows with high probability of face were taken into account generating the following results.  
Following are some of the images form test data : 
 
