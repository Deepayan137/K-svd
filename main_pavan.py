from time import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- P A T H T O T H E I M A G E-------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
