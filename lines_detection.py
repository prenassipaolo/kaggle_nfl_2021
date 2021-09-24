import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

DIR_NFL = 'data/nfl-health-and-safety-helmet-assignment/'


def show_image(path):
    img = cv2.imread(path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

example_image_path = DIR_NFL + 'images/57502_001570_Sideline_frame1395.jpg'

show_image(example_image_path)
