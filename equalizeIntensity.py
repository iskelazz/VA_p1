import cv2
import numpy as np
from matplotlib import pyplot as plt    

def adjustIntensity (inImage, inRange=[], outRange=[0, 255]):
    if (inRange == []):
        inRange = [np.amin(image),np.amax(image)]
    temp = (inImage-inRange[0])/(inRange[1]-inRange[0])
    modificador = outRange[1]-outRange[0]
    resultadoFloat = temp * modificador
    resultadoFloat = resultadoFloat + outRange[0]
    resultado = np.uint8(resultadoFloat)
    return resultado

def equalizeIntensity(inImage, nBins=256):
    height, width = inImage.shape[::1]
    hist,bins = np.histogram(inImage.flatten(),nBins,[0,256])
    size=width*height
    percent_hist = hist/size
    acum=[]
    total=0
    for i in percent_hist:
        total=total+i
        acum.append(total)
    acum = np.array(acum)
    acum_255scale = np.round(acum*255)
    equ_h=[]
    for i in range(0,255):
        temp=0
        result=0
        for j in acum_255scale:
            if (j==i):
                result=result+hist[temp]
            temp=temp+1
        equ_h.append(result)
    equ_h = np.array(equ_h,dtype=object)    
    plt.plot(equ_h)
    plt.xlim([0, 255])
    plt.show()
    print(inImage.astype('uint8'))
    imgEq = acum_255scale[inImage.astype('uint8')]
    return np.uint8(imgEq)
    

def equalizeIntensityPlot (inImage):
    hist,bins = np.histogram(inImage.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(inImage.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

# Load
image_path = 'PruebaVA/Captura.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# Show image
cv2.imshow('Image', image)
cv2.waitKey(0)
print(image)
rango = [np.amin(image),np.amax(image)]
print(rango)
hist = cv2.calcHist([image], channels=[0], mask=None, histSize=[256], ranges=[0,255])
normalizeImage = equalizeIntensity(image)
#normalizeImage = cv2.equalizeHist(image)
# Histogram
rango2 = [np.amin(normalizeImage),np.amax(normalizeImage)]
print(rango2)
cv2.imshow('normalizeImage', normalizeImage)
cv2.waitKey(0)
hist = cv2.calcHist([image], channels=[0], mask=None, histSize=[256], ranges=[0,255])
#Equalize histogram and show image
# Plot histogram image
plt.plot(hist)
plt.xlim([0, 255])
plt.show()

norhist = cv2.calcHist([normalizeImage], channels=[0], mask=None, histSize=[256], ranges=[0,255])

plt.plot(norhist)
plt.xlim([0, 255])
plt.show()

cv2.destroyAllWindows()