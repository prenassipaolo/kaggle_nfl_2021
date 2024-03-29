import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

DIR_NFL = 'data/nfl-health-and-safety-helmet-assignment/'
EXAMPLE_IMG_PATH = DIR_NFL + 'images/57502_001570_Sideline_frame1395.jpg'
#EXAMPLE_IMG_PATH = DIR_NFL + 'images/57503_004294_Endzone_frame0578.jpg'
#EXAMPLE_IMG_PATH = DIR_NFL + 'images/57503_001581_Endzone_frame327.jpg'


def path_to_img(img_path):
    img = cv2.imread(img_path)
    return img


def show_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    

img = path_to_img(EXAMPLE_IMG_PATH)
#show_image(img)

def image_preprocessing(img):
    # convert to HLS image
    # LIGHTNESS is unique for white color distribution of an object. But SATURATION and HUE may be vary.
    # This make the HLS color space the most suitable color space for color based image segmentation related to white objects.
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # HARD CODED EDGES COLOR
    # define range of white color in HLS
    hls_lower_color = np.array([0,160,0])
    hls_upper_color = np.array([180,255,255])
    # Threshold the HLS image to get only white colors
    hls_mask = cv2.inRange(hls_img, hls_lower_color, hls_upper_color)
    # masking of original image
    res = cv2.bitwise_and(img,img, mask= hls_mask)

    # gray scaler
    gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

    # gaussian blur
    kernel_size = 5
    blur = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0) 

    # Canny edge detection
    low_threshold = 10
    high_threshold = 200
    edges = cv2.Canny(blur, low_threshold, high_threshold)

    # dilation of edges
    dilated = cv2.dilate(edges, np.ones((2,2), dtype=np.uint8))

    return dilated






show_image(~image_preprocessing(img))


