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

## Funcion que analiza los vecinos de un pixel y establece en ese pixel el menor valor encontrado entre ellos. Puede dar algún problema en los margenes, mirar eso. Equivalente a cv2.erode
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

## Funcion que analiza los vecinos de un pixel y establece en ese pixel el mayor valor encontrado entre ellos. Equivale a cv2.dilate
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
            # Analiza margenes y corrige si se sale de ellos
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

#Consiste en aplicar erode y luego dilate sobre el resultado
def opening (inImage, SE, center=[]):
    erode_image = erode(inImage,SE,center)
    image_result = dilate(erode_image,SE,center)
    return image_result  

#Consiste en aplicar dilate y luego erode sobre el resultado
def closing (inImage, SE, center=[]):
    dilate_image = dilate(inImage,SE,center)
    image_result = erode(dilate_image,SE,center)
    return image_result  

#El objetivo de esta funcion es rellenar huecos con un color (blanco en este caso). Valida para imagenes "uint8" ademas de para los valores de entre [0,1]
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


##  3.4 EDGE DETECTION

def sobel (inImage):
    kernel_x = np.array([[-1,0,1],[-2, 0, 2],[-1, 0, 1]])
    kernel_y = np.array([[-1,-2,-1],[0, 0, 0],[1, 2, 1]])
    return filterImage(inImage,kernel_y),filterImage(inImage,kernel_x)

def prewitt (inImage):
    kernel_x = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernel_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    return filterImage(inImage,kernel_y),filterImage(inImage,kernel_x)

#Revisar si hacerlo con mascara 3x3
def roberts (inImage):
    kernel_x = np.array([[1,0],[0,-1]])
    kernel_y = np.array([[0,1],[-1,0]])
    return filterImage(inImage,kernel_y),filterImage(inImage,kernel_x)

def centralDiff(inImage):
    kernel_x = np.array([[0,1,0],[0,0,0],[0, -1, 0]])
    kernel_y = np.array([[0, 0, 0],[-1, 0, 1],[0, 0, 0]])
    return filterImage(inImage,kernel_y),filterImage(inImage,kernel_x)

def gradientImage (inImage, operator):
    if (operator=="Roberts"): return roberts(inImage)
    if (operator=="CentralDiff"): return centralDiff(inImage)
    if (operator=="Prewitt"): return prewitt(inImage)
    if (operator=="Sobel"): return sobel(inImage)
    raise ValueError("El operador pasado por parámetro no es válido")


def nonMaximumSuppression(inImage, orientationMatrix):
    height, width = inImage.shape
    result_image = np.zeros((height,width), dtype=np.float64)
    
    for x in range(height):
        for y in range(width):
            maxNeighbourA = 1
            maxNeighbourB = 1
            max = 1

            if (0 <= orientationMatrix[x,y] < 22.5) or (157.5 <= orientationMatrix[x,y] <= 180):
                if ((y+1)<width): maxNeighbourA = inImage[x, y + 1]
                if ((y-1)>0): maxNeighbourB = inImage[x, y - 1]
            
            if (22.5 <= orientationMatrix[x,y] < 67.5):
                if (((y-1)>0) and ((x-1)>0)): maxNeighbourA = inImage[x-1,y-1]
                if (((y+1)<width) and ((x+1)<height)): maxNeighbourB = inImage[x+1,y+1]

            if (67.5 <= orientationMatrix[x, y] < 112.5):
                if ((x+1)<height): maxNeighbourA = inImage[x + 1, y]
                if ((x-1)>0): maxNeighbourB = inImage[x - 1, y]

            if (112.5 <= orientationMatrix[x, y] < 157.5):
                if (((y-1)>0) and ((x+1)<height)): maxNeighbourA = inImage[x+1,y-1]
                if (((x-1)>0) and ((y+1)<width)): maxNeighbourB = inImage[x-1,y+1]


            if (maxNeighbourA>maxNeighbourB): max = maxNeighbourA
            else: max = maxNeighbourB

            if (inImage[x,y] >= max): result_image[x,y] = inImage[x,y]
            else: result_image[x,y] = 0
   
    return result_image

def threshold (inImage, tlow, thigh):
    
    height, width = inImage.shape
    result_image = np.zeros((height,width), dtype=np.float64)
    for x in range (height):
        for y in range (width):
            if (inImage[x,y] >= thigh): result_image[x,y] = 1
            elif (inImage[x,y] < tlow): result_image[x,y] = 0
            else: result_image[x,y] = 0.5   

    return result_image

def hysteresis(inImage):
    height, width = inImage.shape
    SE = np.array([[1,1,1],[1, 1, 1],[1, 1, 1]])
    for x in range(height-1):
        for y in range(width-1):
            if(inImage[x,y] == 0.5):
                if 1 in [inImage[x,y-1],inImage[x,y+1],inImage[x+1,y-1],inImage[x+1,y],inImage[x+1,y+1], 
                    inImage[x-1,y-1],inImage[x-1,y],inImage[x-1,y+1]]:
                        inImage = fill(inImage,seeds=[[x,y]],SE=SE)
                else:
                    inImage[x,y] = 0
    return inImage #Por desarrollar


def edgeCanny (inImage, sigma, tlow, thigh):
    #Suavizado para quitar el ruido
    smoothImage = gaussianFilter(inImage, sigma)
    #Gradientes de la imagen suavizada
    Image_Gy, Image_Gx = sobel(smoothImage)
    Image_G = np.sqrt((Image_Gx**2)+(Image_Gy**2))
    G_direction = np.arctan2(Image_Gy,Image_Gx)
    angle = (G_direction * 180 /np.pi)%180
    #Non-Maximum Suppression para "adelgazar" los bordes
    result = nonMaximumSuppression(Image_G,angle)
    #Umbralizacion para definir los roles de cada pixel
    result = hysteresis (threshold(result,tlow,thigh))
    return result # Por desarrollar, falta hysterisis

# Load
img = cv2.imread("PruebaVA/emma.png", cv2.IMREAD_GRAYSCALE)
img_float = img/255
filter_size = 9
gx = cv2.Canny(img,25,40,L2gradient=False)
gxf = edgeCanny(img_float,1,0.04,thigh =0.3)

cv2.imshow('Imageorigi', img)
cv2.waitKey(0)
cv2.imshow('gx', gx)
cv2.waitKey(0)
cv2.imshow('gxf', gxf)
cv2.waitKey(0)
cv2.imshow('gxy', gxy)
cv2.waitKey(0)
#cv2.imshow('gy', img_prewitty)
#cv2.waitKey(0)
#cv2.imshow('gxy', gxy)
#cv2.waitKey(0)
cv2.destroyAllWindows()