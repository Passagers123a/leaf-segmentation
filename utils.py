import numpy as np
import cv2


# error message when image could not be read
IMAGE_NOT_READ = 'IMAGE_NOT_READ'

# error message when image is not colored while it should be
NOT_COLOR_IMAGE = 'NOT_COLOR_IMAGE'


def read_image(file_path, read_mode=cv2.IMREAD_COLOR):
    """
    Read image file with all preprocessing needed

    Args:
        file_path: absolute file_path of an image file
        read_mode: whether image reading mode is rgb, grayscale or somethin

    Returns:
        np.ndarray of the read image or None if couldn't read

    Raises: 
        ValueError if image could not be read with message IMAGE_NOT_READ
    """
    # read image file in grayscale
    image = cv2.imread(file_path, read_mode)
    
    if image is None:
        raise ValueError(IMAGE_NOT_READ)
    else:
        return image


def ensure_color(image):
    """
    Ensure that an image is colored
    Args:
        image: image to be checked for

    Returns:
        nothing

    Raises:
        ValueError with message code if image is not colored
    """
    if len(image.shape) < 3:
        raise ValueError(NOT_COLOR_IMAGE)


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.true_divide(a, b)
        q[ ~ np.isfinite(q) ] = 0  # -inf inf NaN

    return q

def remove_whites(image, marker):
    """
    Remove pixels resembling white from marker as background
    Args:
        image:
        marker: to be overloaded with white pixels to be removed

    Returns:
        nothing
    """
    # setup the white remover to process logical_and in place
    white_remover = np.full((image.shape[0], image.shape[1]), True)

    # below line same as: white_remover = np.logical_and(white_remover,  image[:, :, 0] > 200)
    white_remover[image[:, :, 0] <= 200] = False # blue channel

    # below line same as: white_remover = np.logical_and(white_remover,  image[:, :, 1] > 220)
    white_remover[image[:, :, 1] <= 220] = False  # green channel

    # below line same as: white_remover = np.logical_and(white_remover,  image[:, :, 2] > 200)
    white_remover[image[:, :, 2] <= 200] = False  # red channel

    # remove whites from marker
    marker[white_remover] = False

    return marker

def remove_blacks(image, marker):
    """
    Remove pixels resembling black from marker as background
    Args:
        image:
        marker: to be overloaded with black pixels to be removed

    Returns:
        nothing
    """
    # setup the black remover to process logical_and in place
    black_remover = np.full((image.shape[0], image.shape[1]), True)

    # below line same as: black_remover = np.logical_and(black_remover,  image[:, :, 0] < 30)
    black_remover[image[:, :, 0] >= 50] = False  # blue channel

    # below line same as: black_remover = np.logical_and(black_remover,  image[:, :, 1] < 30)
    black_remover[image[:, :, 1] >= 50] = False  # green channel

    # below line same as: black_remover = np.logical_and(black_remover,  image[:, :, 2] < 30)
    black_remover[image[:, :, 2] >= 50] = False  # red channel

    # remove blacks from marker
    marker[black_remover] = False

    return marker
    
def remove_blues(image, marker):
    """
    Remove pixels resembling blues better than green from marker as background
    Args:
        image:
        marker: to be overloaded with blue pixels to be removed

    Returns:
        nothing
    """
    # choose pixels that have higher blue than green
    blue_remover = image[:, :, 0] > image[:, :, 1]

    # remove blues from marker
    marker[blue_remover] = False

    return marker
    
def excess_green(image, scale = 2):
    """
    Compute excess green index for colored image

    Args:
        image: image to be converted
        scale: number to scale green channel of the image

    Returns:
        new image with excess green
    """

    ensure_color(image)

    bgr_sum = np.sum(image, axis=2)
    debug(bgr_sum, 'green bgr sum')

    blues = div0(image[:, :, 0], bgr_sum)
    greens = div0(image[:, :, 1], bgr_sum)
    reds = div0(image[:, :, 2], bgr_sum)

    index = scale * greens - (reds + blues)

    return index


def excess_red(image, scale=1.4):
    """
    Compute excess red index for colored image

    Args:
        image: image to be converted
        scale: number to scale red channel of the image

    Returns:
        new image with excess red
    """

    ensure_color(image)

    bgr_sum = np.sum(image, axis=2)

#    blues = div0(image[:, :, 0], bgr_sum)
    greens = div0(image[:, :, 1], bgr_sum)
    reds = div0(image[:, :, 2], bgr_sum)

    index = scale * reds - greens

    return index


def index_diff(image, green_scale=2, red_scale=1.3):#red_scale=1.4  2  2.2  1.3

    ensure_color(image)

    bgr_sum = np.sum(image, axis=2)

    blues = div0(image[:, :, 0], bgr_sum)
    greens = div0(image[:, :, 1], bgr_sum)
    reds = div0(image[:, :, 2], bgr_sum)

    green_index = green_scale * greens - (reds + blues)
    red_index = red_scale * reds - (greens)

    return green_index - red_index

def color_index_marker(color_index_diff, marker):
    """
    Differentiate marker based on the difference of the color indexes
    Threshold below some number(found empirically based on testing on 5 photos,bad)
    If threshold number is getting less, more non-green image
     will be included and vice versa
    Args:
        color_index_diff: color index difference based on green index minus red index
        marker: marker to be updated

    Returns:
        nothing
    """
    marker[color_index_diff <= -0.05] = False
    return marker
    
def debug(value, name=None):
    if isinstance(value, np.ndarray):
        name = 'ndarray' if name is None else name

        print("{}: {}".format(name, value))
        print("{} shape: {}".format(name, value.shape))
    else:
        name = 'value' if name is None else name

        print("{} type: {}".format(name, type(value)))
        print("{}: {}".format(name, value))

