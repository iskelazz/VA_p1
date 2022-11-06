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
        inRange = [np.amin(image),np.amax(image)]
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
    return inImage ##Empty funtion

def dilate (inImage, SE, center=[]):
    return inImage ##Empty funtion    

def opening (inImage, SE, center=[]):
    return inImage ##Empty funtion  

def closing (inImage, SE, center=[]):
    return inImage ##Empty funtion  

# Load
image_path = 'PruebaVA/Captura.png'
image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
image_float = image/255
# Show image
cv2.imshow('Image', image_float)
cv2.waitKey(0)

rango = [np.amin(image),np.amax(image)]
print(rango)

normalizeImage = highBoost(image_float,1,'median',7)
mask_image = highBoost(image_float,7,'median',7)


rango2 = [np.amin(normalizeImage),np.amax(normalizeImage)]
print(rango2)

rango4 = [np.amin(mask_image),np.amax(mask_image)]
print(rango4)

cv2.imshow('normalizeImage', normalizeImage)
cv2.waitKey(0)

cv2.imshow('mask', mask_image)
cv2.waitKey(0)
"""norhist,bins = np.histogram(normalizeImage,256,(0,255))
plt.plot(norhist)
plt.xlim([0, 255])
plt.show()"""

cv2.destroyAllWindows()