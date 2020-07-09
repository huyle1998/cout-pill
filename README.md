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

<img src="images/RGB.png" width="200"> <img src="images/GRAY.png" width="200">

Now we will only work with 1 matrix (a x b). Things have become simpler but it isn’t the result we expect yet. From the grey image above, we will convert it to binary image with a threshold given by Otsu's threshold method.
Otsu's threshold method is an adaptive way for binarization algorithms in image processing. Firstly, the statistical data of an image is used to build a histogram to describe the distribution of grey scale in pixels as in.

<img src="images/histogram.png" width="300">

The next step is to calculate the “Within-Class Variance” of every threshold value and find the threshold value where the “Within-Class Variance” of  both foreground and background are at their minimum to make sure the homogeneity of each region. A faster approach is that we can calculate what is called the “Between Class Variance” and try to find the threshold value where the "Between Class Variance" is at its maximum to ensure the distinction between foreground and background Now we have a binary image as in Fig. 7(c), this is an important matrix for further processing steps.

```python
BW = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
```

Now we have a binary image, this is an important matrix for further processing steps.

<p align="center"> <img src="images/BINARY.png" width="250"> </p>

### Watershed Transform

Watershed transformation is a computer analysis of objects in digital image which helps to define objects in a picture. Computer analysis of objects starts by finding which pixel belongs to the object. This is called image segmentation, the process of separating objects from the background as well as from each other.

From the binary image, we will convert it into “distance transformation of binary image” by using another image processing called “Distance transform” to compute the distance from every pixel in the region of objects to the nearest zero-valued pixel. The farther the distance is, which means the closer the pixel is to the center of the object, the darker it will be and vice versa.

Any “distance transformation of binary image” can be considered as a terrain surface. Dark areas with homogeneous grey level are considered low areas (catchment basins) and bright areas are considered high areas (watershed lines). If we flood this surface from its minima and, if we prevent the merging of the waters coming from different sources, we can divide the image into two different sets: the catchment basins and the watershed lines which are the objects that we need to define and their boundaries.

This algorithm is a great tool to count pills that have multitudinous shapes. We process on similar regions that have approximately the same area without bothering their shape. It helps to overcome the limitations of "Hough Transform" which counts pills by detecting objects described with its model which is similar to the shape simulated earlier, and so it can only handle analytically defined shapes such as circle or ellipse.

Nevertheless, there is a common circumstance in watershed segmentation which is called oversegmentation. Generally, there are some noise spots in  “distance transformation'' image because of the uneven distribution of light on the surface of the object due to smoothness or the structure of its surface. Besides, this phenomenon is immensely common at the asymmetrical object since the distance between the boundary pixels and the center pixels are quite different, leading to the dark area (low area) in “distance transformation'' image not gathering in the middle of objects. As result,  there are more than one local minimum in the object's region. Each local minimum, even if it is inconsiderable, forms a catchment basin and then a watershed region around them. Subsequently, there are a host of watershed regions in a object's region.


<p align="center"> <img src="images/NOTWATER.png" width="250"> </p>

One solution here is modifying the image to filter out tiny local minima or remove minima that are too shallow. This is called "minima imposition" and the result:

```python
def seg_watershed(BW, gray):
    # Watershed Transform
    D = ndimage.distance_transform_edt(BW)
    ret, mask = cv2.threshold(D, 0.4 * D.max(), 255, 0)
    mask = np.uint8(mask)

    # Marker labeling Watershed Line ==> line
    ret, markers = cv2.connectedComponents(mask)
    labels = watershed(-D, markers, mask=gray, watershed_line=True)
    line = np.zeros(BW.shape, dtype=np.uint8)
    line[labels == 0] = 255
    line = cv2.dilate(line, np.ones((2, 2), np.uint8), iterations=1)

    # Creating BW2
    BW2 = BW.copy()
    BW2[line == 255] = 0
    return BW2
```

<p align="center"> <img src="images/WATER.png" width="250"> </p>

### Processing area

### Calculate the number of pill

## Installation

## Usage