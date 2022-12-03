from practica01 import *

PATH = "PruebaVA/circles.png"

img = load_image(PATH)
# lo
# adjustIntensity
#responseImage = adjustIntensity(img)

# equalizeIntensity
responseImage = equalizeIntensity(img,10)

# filterImage
#kernel = auto_kernel(3,3)
#responseImage= filterImage(img,kernel)

# gaussKernel1D
#kernel = gaussKernel1D(1.2)

# gaussianFilter
#responseImage = gaussianFilter(img,1.4)

# medianFilter
#responseImage = medianFilter(img,3)

# highBoost
#responseImage = highBoost(img,1.4, "gaussian", 0.9)
#responseImage = highBoost(img,2.2, "median", 5)

# Erode
#SE = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
#responseImage = erode(img,SE)

# Dilate
#responseImage = dilate(img,SE)

# Opening
#responseImage = opening(img,SE)

# Closing
#responseImage = closing(img,SE)

# Fill
#responseImage = fill (img, [[1,1]])
#responseImage = fill (img, [[1,1],[150,150]])

# gradientImage
#responseImageY, responseImageX = gradientImage(img, "Sobel")
#responseImageY, responseImageX = gradientImage(img, "centralDiff")

# edgeCanny
#responseImage = edgeCanny(img,1.1, 0.1, 0.35)

#Mostrar imagenes de Entrada y salida
plot_image("ImagenEntrada", img)
plot_image("ImagenSalida", responseImage)

#Guardar imagen de salida en la ruta..
save_image("saveImages/response.png", responseImage)




