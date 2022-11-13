import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure, feature, filters, transform, morphology
import math

##   SUPPORT OPERATIONS

#Pinta histogramas con valores entre 0 y 1
def plot_grayhist(hist,bins):
    plt.figure()
    plt.title("Histograma de escala de grises")
    plt.xlabel("valor de grises")
    plt.ylabel("numero de pixels")
    plt.xlim([0.0, 1.0]) 
    plt.plot(bins[0:-1], hist)  
    plt.show()

#Genera un kernel con un valor constante para todos los terminos del tamaño nxm
def auto_kernel(n,m,valor=-1.0):
    if (valor<=1 and valor>=0):
         return np.ones((n, m), dtype="float") * valor
    return np.ones((n, m), dtype="float") * (1.0 / (n * m))

##  3.1 HISTOGRAM: CONTRAST ENHANCEMENT

#Ajusta la escala de grises que usa la imagen pasada por parametro
def adjustIntensity (inImage, inRange=[], outRange=[0, 1]):
    if (inRange == []):
        inRange = [np.amin(inImage),np.amax(inImage)]
    temp = (inImage-inRange[0])/(inRange[1]-inRange[0])
    modificador = outRange[1]-outRange[0]
    resultadoFloat = temp * modificador
    resultadoFloat = resultadoFloat + outRange[0]
    return resultadoFloat

#Ecualiza una imagen con un numero de bins pasado por parametro(por defecto 256)
def equalizeIntensity(inImage, nBins=256):
    width,height = inImage.shape
    hist,bins = np.histogram(inImage,nBins,(0,1))
    plot_grayhist(hist,bins)
    size=width*height
    percent_hist = hist/size
    acum=[]
    total=0
    for i in percent_hist:
        total=total+i
        acum.append(total)
    acum = np.array(acum)
    return np.interp(inImage, bins[:-1],acum)

##  3.2 SPATIAL FILTERING: SMOOTHING AND ENHANCEMENT

#Aplica un filtro de convolucion a la imagen pasada por parametro con el kernel, tambien pasado por parametro
def filterImage(inImage, kernel):
    #Center
    p = int(kernel.shape[0]/2 + 1)
    q = int(kernel.shape[1]/2 + 1)
    #Paddings
    desplazamiento_pInitoCenter = p - 1
    desplazamiento_pCentertoEnd = kernel.shape[0] - p
    desplazamiento_qInitoCenter = q - 1
    desplazamiento_qCentertoEnd = kernel.shape[1] - q
    #imagen con padding
    fPad = np.pad(inImage, ((desplazamiento_pInitoCenter, desplazamiento_pCentertoEnd),(desplazamiento_qInitoCenter, desplazamiento_qCentertoEnd)), mode = 'reflect')
    width, height = inImage.shape
    image_result = np.zeros([inImage.shape[0], inImage.shape[1]], 'float')
    #Operacion de filtrado
    for x in range(0,width):
        for y in range(0,height):
            local_matrix = fPad[x:x+desplazamiento_pInitoCenter+desplazamiento_pCentertoEnd
                +1,y:y+desplazamiento_qInitoCenter+desplazamiento_qCentertoEnd+1]
            acum = np.sum(local_matrix*kernel)
            image_result[x][y] = acum
    return image_result

#equivalente a cv2.getGaussianKernel(size,sigma) 
#crea un kernel a partir de la variable sigma con la longitud 1xN donde N = 2*ceil(3*sigma) + 1
def gaussKernel1D(sigma):
    n = 2*math.ceil(3*sigma) + 1
    rango = np.linspace(-int(n/2),int(n/2),n)
    kernel=[]
    for x in rango:
        kernel = np.hstack((kernel, 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2))))
    kernel = kernel/np.sum(kernel)
    kernel = np.array([kernel])
    return kernel.T

#equivalente a cv2.GaussianBlur(image,(0,0),4)
# Seguimos 100% las instrucciones del pdf de la practica
def gaussianFilter(inImage,sigma):
    kernel = gaussKernel1D(sigma)
    semi_processedPic = filterImage(inImage,kernel)
    image_result = filterImage(semi_processedPic,kernel.T)
    return image_result

#equivalente al cv2.medianBlur(img, kernel), parecido al filtro2D
def medianFilter(inImage, filterSize):
     #Center
    p = int(filterSize/2 + 1)
    #Paddings
    desplazamiento_InitoCenter = p - 1
    desplazamiento_CentertoEnd = filterSize - p
    #imagen con padding
    fPad = np.pad(inImage, ((desplazamiento_InitoCenter, desplazamiento_CentertoEnd),(desplazamiento_InitoCenter, desplazamiento_CentertoEnd)), mode = 'reflect')
    width, height = inImage.shape
    image_result = np.zeros([inImage.shape[0], inImage.shape[1]], 'float')
    #Operacion de filtrado
    for x in range(0,width):
        for y in range(0,height):
            local_matrix = fPad[x:x+desplazamiento_InitoCenter+desplazamiento_CentertoEnd
                +1,y:y+desplazamiento_InitoCenter+desplazamiento_CentertoEnd+1]
            median = np.median(local_matrix)
            image_result[x][y] = median
    return image_result

#Explanations in https://theailearner.com/2019/05/14/unsharp-masking-and-highboost-filtering/
def highBoost(inImage,A,method,param):
    if (method == 'gaussian'):
        blur_image = gaussianFilter(inImage,param)
    else:
        if (method == 'median'):
            blur_image = medianFilter(inImage,param)
        else:
             raise ValueError("Método no válido")
    mask = inImage - blur_image
    image_result = inImage + A*mask
    return image_result

##  3.3 MORPHOLOGICAL OPERATORS

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
            min = 1
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

def opening (inImage, SE, center=[]):
    erode_image = erode(inImage,SE,center)
    image_result = dilate(erode_image,SE,center)
    return image_result  

def closing (inImage, SE, center=[]):
    dilate_image = dilate(inImage,SE,center)
    image_result = erode(dilate_image,SE,center)
    return image_result  


def fill (inImage, seeds, SE=[], center=[]):
    if (len(SE) < 1): 
        SE = np.array([[0,1,0],[1, 1, 1],[0, 1, 0]])
    if (len(center) < 2 or len(center)>2):
        center = [int(SE.shape[0]/2 + 1), int(SE.shape[1]/2 + 1)]   
    stack = set(((seeds[0][0], seeds[0][1]),))
    desplazamiento_pInitoCenter = center[0] - 1
    desplazamiento_pCentertoEnd = SE.shape[0] - center[0]
    desplazamiento_qInitoCenter = center[1] - 1
    desplazamiento_qCentertoEnd = SE.shape[1] - center[1]
    image_result = np.array(inImage, copy=True)
    if (inImage.dtype == "uint8"): newColor=255
    else: newColor=1
    count = 0 
    for x in range(0, len(seeds)):
        if (255 == inImage[seeds[x][0],seeds[x][1]]): continue
        stack.add((seeds[x][0], seeds[x][1]))
        while (stack):    
            new_pos_x,new_pos_y = stack.pop()
            if (inImage[new_pos_x,new_pos_y]!=inImage[seeds[x][0],seeds[x][1]]): continue
            else: image_result[new_pos_x,new_pos_y]=newColor
            x1 = new_pos_x-desplazamiento_pInitoCenter
            x2 = new_pos_x+desplazamiento_qCentertoEnd
            y1 = new_pos_y-desplazamiento_qInitoCenter
            y2 = new_pos_y+desplazamiento_pCentertoEnd
            local_matrix = image_result[x1:x2+1,y1:y2+1]
            if x1 < 0 or x2 > inImage.shape[0]-1 or y1 < 0 or y2 > inImage.shape[1]-1:
                    continue
            for a in range(SE.shape[0]):
                for b in range(SE.shape[1]):
                    if (SE[a, b] == 1):
                        if(local_matrix[a,b]==inImage[seeds[x][0],seeds[x][1]]):
                            pos1 = a-desplazamiento_pInitoCenter
                            pos2 = b-desplazamiento_qInitoCenter
                            if (pos1 != 0 or pos2 != 0):
                                count = count +1 
                                stack.add((new_pos_x+pos1,new_pos_y+pos2))

    return image_result


def gradientImage (inImage, operator):
    return inImage # Por desarrollar

def edgeCanny (inImage, sigma, tlow, thigh):
    return inImage # Por desarrollar

# Load
img = cv2.imread("PruebaVA/circles.png", cv2.IMREAD_GRAYSCALE)
img_float = img/255
filter_size = 9
SE = np.array([[1,1,1],[1, 1, 1],[1, 1, 1]])
new_img = fill(img_float, [[150,150],[395,395]],SE=SE) #[100,100]
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.imshow('Imageorigi', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()