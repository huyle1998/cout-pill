import cv2
import numpy as np
import time
from skimage.measure import label, regionprops
from skimage.morphology import watershed
from scipy import ndimage


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


def caculate_pill(BW2):
    label_image = label(BW2)
    A = [r.area for r in regionprops(label_image)]
    A.sort()

    num = 0
    S = 0
    num_pill = 0
    warn = False
    # Find minArea
    for i in range(len(A)):
        rateArea = A[i] / A[0]
        if rateArea < 1.15:
            num = num + 1
            S = S + A[i]
    if num != 0:
        minArea = S / num

    # Calculate num_pill
    for i in range(len(A)):
        rate = A[i] / minArea
        appro_rate = round(rate, 0)
        delta_rate = abs(rate - appro_rate)

        if delta_rate < 0.3:
            num_pill = num_pill + appro_rate
        else:
            warn = True

    if num_pill == 0:
        warn = True
    return num_pill, warn


def main():
    mycam = cv2.VideoCapture(0)
    while True:
        t = time.time()
        ret, img = mycam.read()
        img[0:50, 0:250] = (0, 0, 0)
        # Pre-processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        BW = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        BW2 = seg_watershed(BW, gray)
        num_pill, warn = caculate_pill(BW2)
        if warn:
            cv2.putText(img, 'Warning', (10, 450), 4, 1, (0, 255, 255), 1)
            num_pill = 0
        cv2.putText(img, 'Result: ' + str(int(num_pill)), (5, 30), 4, 1, (0, 255, 255), 1)
        fps = round((1 / (time.time() - t)), 2)
        cv2.putText(img, 'Fps: ' + str(int(fps)), (500, 30), 4, 1, (0, 255, 255), 1)
        cv2.imshow('Counting Pill', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    mycam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
