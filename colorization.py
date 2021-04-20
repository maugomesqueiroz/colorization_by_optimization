import cv2
import numpy as np

def access_pixel_data(image):
    rows, cols, _ = image.shape

    for i in range(rows):
        for j in range(cols):
            pixel = image[i,j]
            print(pixel)

def convert_to_yuv(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return yuv

def convert_to_rgb(image):
    bgr = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    return bgr

def convert_to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def colorize(input_image, marked_image):
    marked_yuv = convert_to_yuv(marked_image)
    access_pixel_data(marked_yuv)
    return marked_yuv

def marked_pixels(input_image, marked_image):
    diff = marked_image - input_image

    y_diff = diff[:,:,0]
    u_diff = diff[:,:,1]
    v_diff = diff[:,:,2]

    colored_u = list(zip(*np.nonzero(u_diff)))
    colored_v = list(zip(*np.nonzero(v_diff)))
    unique_colored_locations = list(set(colored_u+colored_v))
    return unique_colored_locations

def affinity_around(i,j, intensity):
    """ Affinity around r=(i,j). Using window size = (3,3)
    """

    window_size = 3
    height, width = intensity.shape

    intensity_window = intensity[window((i,j), height, width)]
    std = intensity_window.std()
    mean = intensity_window.mean()

    affinity_window_matrix = np.zeros(shape=(3,3)) #wrs for Neighborhood
    yr = window[1,1]
    for i in range(window_size):
        for j in range(window_size):
            ys = intensity_window[i,j]
            affinity_window_matrix[i,j] = _affinity_function(yr,ys,std)
    return affinity_window_matrix

def window(point, height, width, size=1):
    ''' Returns a sclice object of the neighborhood of given point
    '''
    i,j = point

    left_border = max((0, j-size))
    right_border = min((width, j+size+1))
    upper_border = max((0, i-size))
    bottom_border = min((height, i+size+1))

    return slice(upper_border,bottom_border), slice(left_border,right_border)
 
def _affinity_function(yr, ys, std):
    if 2*std**2 < 0.01:
        return 0.0
    return np.exp( -(yr - ys)**2/(2*std**2) )