
import cv2 as cv
import numpy as np
import math

def calculate_template_space(temp_side_length):
        return int(temp_side_length/2)

def dilation(image, template_side_length, template):
    new_image = np.zeros(image.shape, image.dtype)
    # Coordinates are provided as (y,x), where the origin is at the top left of the image
    # So always remember that (-) is used instead of (+) to iterate
    template_space = calculate_template_space(template_side_length)
    half_template = int((template_side_length - 1) / 2)

    for x in range(template_space, new_image.shape[1] - template_space):
        for y in range(template_space, new_image.shape[0] - template_space):
            maximum = 0
            for c in range(0, template_side_length):
                for d in range(0, template_side_length):
                    a = x - half_template - 1 + c
                    b = y - half_template - 1 + d
                    sub = image[b, a] - template[d, c]
                    if sub > maximum:
                        if sub > 0:
                            maximum = sub
            new_image[y, x] = int(maximum)
    return new_image

img = cv.imread("VA_p1/PruebaVA/Captura.png", cv.IMREAD_GRAYSCALE)
filter_size = 9
kernel = np.ones((9, 9), np.uint8)
temp = np.zeros(img.shape, img.dtype)
new_img = dilation(img, filter_size, temp)
cv2_img = cv.dilate(img, kernel, iterations=1)
cv.imshow('image', img)
cv.waitKey(0)
cv.imshow('Imageorigi', new_img)
cv.waitKey(0)
cv.imshow('Imagecv', cv2_img)
cv.waitKey(0)
cv.destroyAllWindows()