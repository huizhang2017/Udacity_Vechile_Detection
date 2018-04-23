## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./my_mexamples/car_not_car.png
[image2]: ./my_examples/HOG_example.jpg
[image3]: ./my_examples/sliding_windows.jpg
[image4]: ./my_examples/sliding_window.jpg
[image5]: ./my_examples/bboxes_and_heat.png
[image6]: ./my_examples/labels_map.png
[image7]: ./my_examples/output_bboxes.png
[video1]: ./output_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

############### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 9th code cell of the IPython notebook submit_P5_object_detect.ipynb.  

I started by reading in all the `vehicle` and `non-vehicle` images, and I also add some false positive examples from the video.  
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found only when I set the hog_channel to be "ALL", the test accuray is enough high (at least > 0.98), I also changed other
parameters such as 'pixels_per_cell', 'cells_per_block' etc, but the accuracy doesn't change much.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the HOG feature, binned color features and color histogram features. I also set the C parameter in SVM to be 14, 
which I want to classify more training points correctly. The code for this step is contained in 12th cell of the Ipython notebook.

################# Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?


I set window size to be 64 by 64 pixels, when I extract the sub-image from the input image, I resized the sub-image with different scales, this method corresponds to use
different window sizes, and when I constrain the position of the boxes, I recover the size of the sub-image. Please check this method in the 13th cell (find_cars function).
I slide the window with 2 cells per step, that corresponds 75% of the window are overlaped. I adjusted this parameter, but it doesn't make the performance better, too large
step will decrease the chances to find the car. Here is the example of the sliding windows:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which 
provided a relatively nice result.  

In order to get good results, I set decision_function of the trained model to be larger than 1.2, this will remove part of the false positive.
Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. I overlaped 6 consecutive frame and build the heat-map from these detections in order to combine
overlapping detections and remove false positives. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I  assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


In this project, first I used SVM to train the vechiles and non-vechiles examples. For both of these examples, I selected the color space 'YUV' after many tests, I used spatially binned color and histograms of color in the feature vector, the I also extracted histogram of the orient gradient feature from the samples. These features are combined to one vector for
each examples and was normalized before the training. In order to get rid of false positive results, I used decision functon of SVM to only selet the positive examples 
with the score larger than 1.5, I used the sliding windows with three different scales to search the vechiles, I overlapped 6 consective frames from the video and used the 
heat map method to constrain the real positions. But I think the finnal result is not satisfied me. I spent a lot of hours to adjust the parameters to train, but there are
still some false positive examples. I think, the key problems caused such results is the accuracy of the result from SVM. Maybe we will get better results from the convolutional neural network. If the accuracy of the classify is not high enough, then the pipeline will fail, because there are too many false positives and sometimes the heat of the false positive may even hotter than the object.


