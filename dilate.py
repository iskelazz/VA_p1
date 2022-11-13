
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

#Equivalente a operacion cv2.dilate. Funcional pero posiblemente mejorable, se come los margenes 
def dilate(inImage, SE, center=[]):
    if (len(center) < 2 or len(center)>2):
        center = [int(SE.shape[0]/2 + 1), int(SE.shape[1]/2 + 1)]
    desplazamiento_pInitoCenter = center[0] - 1
    desplazamiento_pCentertoEnd = SE.shape[0] - center[0]
    desplazamiento_qInitoCenter = center[1] - 1
    desplazamiento_qCentertoEnd = SE.shape[1] - center[1]
    image_result = np.zeros([inImage.shape[0], inImage.shape[1]], inImage.dtype)
    for x in range(inImage.shape[0]):
        for y in range (inImage.shape[1]):
            max = 0
            x1 = x-desplazamiento_pInitoCenter
            x2 = x+desplazamiento_qCentertoEnd
            y1 = y-desplazamiento_qInitoCenter
            y2 = y+desplazamiento_pCentertoEnd
				# structure modification
            se_x1 = 0
            se_x2 = SE.shape[0]
            se_y1 = 0
            se_y2 = SE.shape[1]

            if x1 < 0:
                se_x1 = -x1
                x1 = 0
            if x2 > inImage.shape[0]-1:
                se_x2 = se_x2 - (x2 - (inImage.shape[0]-1))
                x2 = inImage.shape[0]-1
            if y1 < 0:
                se_y1 = -y1
                y1 = 0
            if y2 > inImage.shape[1]-1:
                se_y2 = se_y2 - (y2 - (inImage.shape[1]-1))
                y2 = inImage.shape[1]-1
            local_matrix = inImage[x1:x2+1,y1:y2+1]
            temporal_SE = SE[se_x1:se_x2,se_y1:se_y2]
            for a in range(temporal_SE.shape[0]):
                for b in range(temporal_SE.shape[1]):
                    if (SE[a, b] == 1):
                        if(local_matrix[a, b] > max):
                            max = local_matrix[a,b]
            image_result[x,y] = max
    return image_result

def erode (inImage, SE, center=[]):
    if (len(center) < 2 or len(center)>2):
        center = [int(SE.shape[0]/2 + 1), int(SE.shape[1]/2 + 1)]
    desplazamiento_pInitoCenter = center[0] - 1
    desplazamiento_pCentertoEnd = SE.shape[0] - center[0]
    desplazamiento_qInitoCenter = center[1] - 1
    desplazamiento_qCentertoEnd = SE.shape[1] - center[1]
    image_result = np.zeros([inImage.shape[0], inImage.shape[1]], inImage.dtype)
    for x in range(inImage.shape[0]):
        for y in range (inImage.shape[1]):
            min = 256
            x1 = x-desplazamiento_pInitoCenter
            x2 = x+desplazamiento_qCentertoEnd
            y1 = y-desplazamiento_qInitoCenter
            y2 = y+desplazamiento_pCentertoEnd
            if x1 < 0 or x2 > inImage.shape[0]-1 or y1 < 0 or y2 > inImage.shape[1]-1:
                image_result[x,y] = 0
            else:
                local_matrix = inImage[x1:x2+1,y1:y2+1]
                for a in range(SE.shape[0]):
                    for b in range(SE.shape[1]):
                        if (SE[a, b] == 1):
                            if(local_matrix[a, b] < min):
                                min = local_matrix[a,b]
                image_result[x,y] = min
    return image_result

def opening (inImage, SE, center=[]):
    erode_image = erode(inImage,SE,center)
    image_result = dilate(erode_image,SE,center)
    return image_result  

def closing (inImage, SE, center=[]):
    dilate_image = dilate(inImage,SE,center)
    image_result = erode(dilate_image,SE,center)
    return image_result  


#img = cv.imread("PruebaVA/Captura.png", cv.IMREAD_GRAYSCALE)
#kernel = np.ones((9, 9), np.uint8)
#dilate(img, SE=kernel)

img = cv.imread("PruebaVA/morph.png", cv.IMREAD_GRAYSCALE)
filter_size = 9
kernel = np.ones((9, 9), np.uint8)
temp = np.zeros(img.shape, img.dtype)
cringe = cv.morphologyEx(img,cv.MORPH_CLOSE ,kernel)
new_img = closing(img, kernel)
cv.imshow('image', img)
cv.waitKey(0)
cv.imshow('Imageorigi', new_img)
cv.waitKey(0)
cv.imshow('Imagecvmbre', cringe)
cv.waitKey(0)
cv.destroyAllWindows()