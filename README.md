# Real-time Pill Counting using Computer Vision

Pill counting is one of indispensable steps of distributing medicine in medical facilities. Generally, pills are counted manually and this counting method takes a lot of time for pharmacists and patients. These systems provide contactless control which eliminates the influence of human factors, reduces errors. As a result, the measurement will give faster results with higher accuracy. In this project, we propose a real-time computer vision program in order to count the number of pills via video captured by a removable camera connected to the computer. Our program is based on Otsu's threshold method and the image segmentation of Watershed transformation which can count pills without considering other factors such as shape and size. <p
align="center">
<img src="images/system.png" width="600"> </p>

## Feature

### Snapshot image from webcam

After set up the system, we got the image from webcam and load data in to `mycam` variable.

```python
mycam = cv2.VideoCapture(0)
```

### Image preprocessing

In this program, we will work with digital images. The captured image of the camera is stored as a multi-dimensional array of data mxnx3 which includes 3 matrices with 3 color bands that defines the red (R), green (G) and blue (B) color values stored at pixels’ location in the color plane. In this form, we cannot apply image processing tools yet, but we must convert it into a binary image.

First of all, we will convert color image into grey image. The primary color components used in the video and television industries incorporate a correction to compensate for the nonlinearity of the video monitors. According to ITU-R BT.601 standard, the average luminance of the reconstruction color space of a television monitor is computed:

***Y = 0.299R + 0.587G + 0.114B***

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

<img src="images/RGB.png" width="200"> 
<img src="images/GRAY.png" width="200">

Now we will only work with 1 matrix (a x b). Things have become simpler but it isn’t the result we expect yet. From the grey image above, we will convert it to binary image with a threshold given by Otsu's threshold method.
Otsu's threshold method is an adaptive way for binarization algorithms in image processing. Firstly, the statistical data of an image is used to build a histogram to describe the distribution of grey scale in pixels as in.

<img src="images/histogram.png" width="300">

The next step is to calculate the “Within-Class Variance” of every threshold value and find the threshold value where the “Within-Class Variance” of  both foreground and background are at their minimum to make sure the homogeneity of each region. A faster approach is that we can calculate what is called the “Between Class Variance” and try to find the threshold value where the "Between Class Variance" is at its maximum to ensure the distinction between foreground and background Now we have a binary image as in Fig. 7(c), this is an important matrix for further processing steps.

```python
BW = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
```

Now we have a binary image, this is an important matrix for further processing steps.

<img src="images/BINARY.png" width="300">

### Watershed Transform

### Processing area

### Calculate the number of pill

## Installation

## Usage