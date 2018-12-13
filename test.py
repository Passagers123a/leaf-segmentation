import os
import numpy as np
import cv2

from utils import *
from background_marker import *
from review import *

def generate_background_marker(file):
   
    # check file name validity
    if not os.path.isfile(file):
        raise ValueError('{}: is not a file'.format(file))

    original_image = read_image(file)

    marker = np.full((original_image.shape[0], original_image.shape[1]), True)

#  update marker based on vegetation color index technique
#    marker=color_index_marker(index_diff(original_image), marker)
    marker = texture_filter(original_image, marker)
    marker=remove_whites(original_image, marker)
    marker=remove_blacks(original_image, marker)
    marker=remove_blues(original_image, marker)
  
    
    return original_image,marker


def segment_leaf(image_file, filling_mode, smooth_boundary, marker_intensity):
    # get background marker and original image
    original, marker = generate_background_marker(image_file)

    # set up binary image for futher processing
    bin_image = np.zeros((original.shape[0], original.shape[1]))
    bin_image[marker] = 255
    bin_image = bin_image.astype(np.uint8)
    cv2.imshow("bin_image",bin_image); cv2.waitKey()
    # further processing of image, filling holes, smoothing edges
#    largest_mask = \
#        select_largest_obj(bin_image, fill_mode=filling_mode,
#                           smooth_boundary=smooth_boundary)
#    cv2.imshow("largest_mask",largest_mask); cv2.waitKey()
     
    if marker_intensity > 0:
        bin_image[bin_image != 0] = marker_intensity
        image = bin_image
    else:
        # apply marker to original image
        image = original.copy()
        image[bin_image == 0] = np.array([0, 0, 0])
    cv2.imshow("image",image); cv2.waitKey()
    
    
    return original, image

if __name__ == '__main__':
    
    image_file="./testing_files/4ac89748-5322-447e-a97d-d1ef03c46893.jpg"
 
    
    original, image1 = segment_leaf(image_file, filling_mode='FLOOD', smooth_boundary=True, marker_intensity=0)
    cv2.imshow("image1",image1)
    cv2.waitKey()
    ret_val, segmented_image=segment_with_otsu(image1, background = 0)
    cv2.imshow("segmented_image",segmented_image)
    cv2.waitKey()
    
#    original_image,masker = generate_background_marker(image_file)
#    background=0
#    mask = np.logical_not(masker)
#    new_image = original_image.copy()
#    new_image[mask] = background
#    new_image[~mask] = 255
#    cv2.imshow("new_image",new_image)
#    cv2.waitKey()
    
    #
    #cv2.waitKey()