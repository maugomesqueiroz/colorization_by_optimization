import cv2
import numpy as np
from scipy import sparse
from scipy.linalg import solve_banded
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def convert_to_yuv(image, is_gray=False):
    if is_gray:
        image = convert_to_rgb_gray(image)

    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return yuv

def convert_to_rgb(image):
    bgr = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    return bgr

def convert_to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def convert_to_rgb_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return gray

def colorize(input_image, marked_image):
    ''' Main process of colorization
    '''

    n = input_image.shape[0]
    m = input_image.shape[1]
    size = n*m

    # _idx is a index mapping. _idx[7,5] -> sparse_idx
    _idx = np.arange(size).reshape(n, m)


    input_yuv = convert_to_yuv(input_image, is_gray=True)
    marked_yuv = convert_to_yuv(marked_image)

    marked_locations, marked_mask = marked_pixels(input_yuv, marked_yuv)
    masked_data = cv2.bitwise_and(marked_image, marked_image, mask=marked_mask)

    # sparse_weights, sparse_idx = affinity_sparse_matrix(input_yuv)

    # u = np.zeros((sparse_weights.shape[0]))
    # v = np.zeros((sparse_weights.shape[0]))
    # for (x, y) in marked_locations:
    #     u[sparse_idx[x,y]] = marked_yuv[x,y,1]
    #     v[sparse_idx[x,y]] = marked_yuv[x,y,2]

    # im_weights = np.zeros(input_yuv.shape[:2])
    # for x in range(n):
    #     for y in range(m):
    #         a,b = sparse_idx[(50,100),(x,y)]
    #         im_weights[x,y] = sparse_weights[a,b]


    print('HERE')
    print(f'shape: {u.shape}, \n shapesparse: {sparse_weights.shape}')

    # new_vals = sparse.linalg.spsolve(sparse_weights, u)
    # new_vals = sparse.linalg.lsqr(sparse_weights, u)[0]


    #TODO make marked matrix (size x 1)
    #TODO compute colored optimization
    #TODO transform optimization into image again
    # new_vals = linalg.spsolve(A, b)

    return masked_data

def marked_pixels(input_image, marked_image):
    ''' Finds out what pixels are marked by subtracting marked from input
    returns pixel locations and a numpy matrix that can be used as mask.
    '''

    diff = marked_image - input_image

    y_diff = diff[:,:,0]
    u_diff = diff[:,:,1]
    v_diff = diff[:,:,2]

    colored_u = list(zip(*np.nonzero(u_diff)))
    colored_v = list(zip(*np.nonzero(v_diff)))
    unique_colored_locations = list(set(colored_u+colored_v))

    marked_mask = np.zeros(marked_image.shape[:2], dtype='uint8')
    for (i,j) in unique_colored_locations:
        marked_mask[i,j] = 255

    return unique_colored_locations, marked_mask

def affinity_around(i,j, intensity):
    """ Affinity around r=(i,j). Using window size = (3,3)
    """

    window_size = 3
    height, width = intensity.shape

    intensity_window = intensity[window((i,j), height, width)]
    std = intensity_window.std()
    mean = intensity_window.mean()

    affinity_window_matrix = np.zeros(shape=(3,3), dtype=np.float64) #wrs for Neighborhood

    yr = intensity_window[1,1]/255 #center point intensity
    for i in range(window_size):
        for j in range(window_size):
            try:
                ys = intensity_window[i,j]/255
                affinity_window_matrix[i,j] = _affinity_function(yr,ys,std)
            except IndexError:
                affinity_window_matrix[i,j] = 0.0
    return affinity_window_matrix

def window(point, height, width, size=1):
    ''' Returns a slice object of the neighborhood of given point
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

def affinity_sparse_matrix(yuv):

    Y = yuv[:,:,0] # First channel only

    n = Y.shape[0]
    m = Y.shape[1]
    size = m*n

    sparse_idx = SparseIndex(n,m)
    sparce_weights = sparse.lil_matrix((size, size))

    for row in range(n):
        for col in range(m):

            center_point = row, col
            center_idx = sparse_idx[center_point]

            affinity_matrix = affinity_around(row,col,Y)
            affinity_window = window((row,col), n, m)
            affinity_idx = sparse_idx[affinity_window]

            for k in range(affinity_idx.shape[0]):
                for l in range(affinity_idx.shape[1]):
                    if k != l: # when k==l, w=1 (the affinity of center with itself)
                        sparce_weights[center_idx,affinity_idx[k,l]] = -affinity_matrix[k,l]

            if col>10 and col <15 and row >10 and row <15:
                print(affinity_matrix)
                print(affinity_idx, affinity_window)

    return sparce_weights, sparse_idx

def marked_places(input_image, marked_image):
    ''' Main process of colorization
    '''

    n = input_image.shape[0]
    m = input_image.shape[1]
    size = n*m

    input_yuv = convert_to_yuv(input_image, is_gray=True)
    marked_yuv = convert_to_yuv(marked_image)

    marked_locations, marked_mask = marked_pixels(input_yuv, marked_yuv)
    masked_data = cv2.bitwise_and(marked_image, marked_image, mask=marked_mask)
    return masked_data

def weightmap(point, input_image):
    ''' Produces a heat map of weights around point.
    '''
    input_yuv = convert_to_yuv(input_image, is_gray=True)

    x,y = point

    n = input_image.shape[0]
    m = input_image.shape[1]

    Y = input_yuv[:,:,0] # First channel only
    height, width = Y.shape

    std = Y.std()
    mean = Y.mean()

    affinity_window_matrix = np.zeros(shape=(height,width), dtype=np.float64) #wrs for Neighborhood

    yr = Y[y,x]/255 #center point intensity

    for i in range(n):
        for j in range(m):
            ys = Y[i,j]/255
            affinity_window_matrix[i,j] = _affinity_function(yr,ys,std)

    plt.matshow(affinity_window_matrix)
    plt.show()

class SparseIndex:

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.size = n*m
        self._idx = np.arange(self.size).reshape(n, m)

    def __getitem__(self, key):

        x, y = key

        if isinstance(x, slice):
            x_vals = np.arange(*x.indices(self.size))
            y_vals = np.arange(*y.indices(self.size))

            indices = []
            for i in x_vals:
                for j in y_vals:
                    indices.append(self._idx[i,j])

            return np.array(indices).reshape(len(x_vals),len(y_vals))
        else:
            return self._idx[x,y]

