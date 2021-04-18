import cv2
import sys

from colorization import (
    colorize,
    convert_to_rgb,
    convert_to_yuv,
    convert_to_gray
)

def show_in_output(image):
     cv2.namedWindow("Output")
     cv2.imshow("Output", image)

input_image = cv2.imread(sys.argv[1])
marked_image = cv2.imread(sys.argv[2])

cv2.namedWindow("Input")
cv2.namedWindow("Marked")

input_image = convert_to_gray(input_image)
output_image = marked_image
while True:

    cv2.imshow("Input", input_image)
    cv2.imshow("Marked", marked_image)

    key = cv2.waitKey(1)

    if key == ord('c'):
        print('Colorizing Image!')
        output_image = colorize(input_image, marked_image)
        show_in_output(output_image)

    if key == ord('b'):
        print('Converting YUV->RGB')
        output_image = convert_to_rgb(output_image)
        show_in_output(output_image)

    if key == ord('y'):
        print('Converting RGB->YUV')
        output_image = convert_to_yuv(output_image)
        show_in_output(output_image)

    if key == ord('r'):
        print('Reseting Image!')
        output_image = marked_image
        show_in_output(output_image)

    if key == 27:
        break

cv2.destroyAllWindows()
