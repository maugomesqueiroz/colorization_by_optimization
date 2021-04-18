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

