import math
from time import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from image_segment import image_segment


def preprocessing(img_name, sigma, ksize):
    img = cv2.imread(img_name)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_g = cv2.GaussianBlur(img_hsv, ksize=(ksize, ksize), sigmaX=sigma)
    return img, img_g


def show_img(img, cvt, title):
    """
    Print BGR encoded image through matplotlib
    :param img: BGR image
    :return: None
    """
    if cvt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # Read input image & image pre-processing
    img_name = "grain.jpg"
    sigma = 0.9
    ksize = int(math.ceil(4 * sigma)) + 1
    img, img_g = preprocessing(img_name, sigma, ksize)

    # Perform image segmentations
    k = 3000
    min_size = 100
    t_start = time()
    img_mask = image_segment(img_g, k, min_size)
    t_end = time()
    print(f">> Time spent: {t_end - t_start}")

    # Show segmented image
    # show_img(img, cvt=True, title="Original Image")
    # show_img(img_mask, cvt=False, title="Segmented Image")

    img_mask = img_mask.astype(np.uint16)
    # cv2.imwrite("Results/beach-grid-graph.jpg", img_mask)
    # cv2.imwrite("Results/beach-hsv-grid-graph.jpg", img_mask)
    # cv2.imwrite("Results/beach-nearest-neighbor.jpg", img_mask)
    # cv2.imwrite("Results/beach-hsv-nearest-neighbor.jpg", img_mask)

    # cv2.imwrite("Results/grain-grid-graph.jpg", img_mask)
    cv2.imwrite("Results/grain-hsv-nearest-neighbor.jpg", img_mask)