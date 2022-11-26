import cv2
from practica01 import *
import numpy as np
import sys
import argparse

# Valores entre corchetes opcionales
# Los primeros 2 valores simpre deben ser la imagen de entrada de primero y la funcion a ejecutar de segundo, a partir del segundo el orden no es de importancia
# Si lleva "--plot" muestra por pantalla la imagen de entrada y la de salida
# Si lleva "--image_output value" se define la ruta de salida, desde el directorio actual
#####OPERACIONES

#  adjustIntensity -> python3 main.py <Ruta_imagen> adjustIntensity [--inRange minValue maxValue] [--outRange minValue maxValue]
#  equalizeIntensity -> python3 main.py <Ruta_imagen> equalizeIntensity [--bins value]

#  filterImage -> python3 main.py <Ruta_imagen> filterImage --numberRows number --kernel value1 value2 ...
#  gaussFilter -> python3 main.py <Ruta_imagen> gaussKernel --sigma value
#  medianFilter -> python3 main.py <Ruta_imagen> medianFilter --filterSize value
#  highBoost -> python3 main.py <Ruta_imagen> highBoost --A value --method method --param value

#  erode -> python3 main.py <Ruta_imagen> erode --numberRows number --SE value1 value2 ... [--center value1 value2]
#  dilate -> python3 main.py <Ruta_imagen> dilate --numberRows number --SE value1 value2 ... [--center value1 value2]
#  opening -> python3 main.py <Ruta_imagen> opening --numberRows number --SE value1 value2 ... [--center value1 value2]
#  closing -> python3 main.py <Ruta_imagen> closing --numberRows number --SE value1 value2 ... [--center value1 value2]
#  fill -> python3 main.py <Ruta_imagen> fill --seeds value1 value2 .. [--center value1 value2] [--numberRows number] [--SE value1 value2 ...]
#
#  gradientImage -> python3 main.py <Ruta_imagen> gradientImage --operator operator
#  edgeCanny -> python3 main.py <Ruta_imagen> edgeCanny --sigma sigma --tlow value --thigh value
#

if __name__ == '__main__':

    if (len(sys.argv) < 2):
        print("Uso:    python main.py <ruta_imagen> <funcion> arg1 arg2 ...#Lee imagen de memoria")
        exit(-1)

    #Lectura Imagen
    parser = argparse.ArgumentParser(
    description='Example',
    )
    parser.add_argument('--image_output',action="store",dest="image_output",type=str)
    parser.add_argument('--plot',action="store_true",dest="plot")

    if (sys.argv.__contains__("--plot")):plot = True
    else: plot=False
    if (sys.argv.__contains__("--image_output")): image_out = True
    else: image_out = False  
    
    img_float = load_image(sys.argv[1])
    
    ### <Ruta_imagen> adjustIntensity [--inRange minValue maxValue] [--outRange minValue maxValue]
    if (sys.argv[2] == "adjustIntensity"):
        parser.add_argument('--inRange',action="store",dest="inRange",type=float, nargs='+')
        parser.add_argument('--outRange',action="store",dest="outRange",type=float, nargs='+')
        if (sys.argv.__contains__("--inRange") or sys.argv.__contains__("--outRange")): 
            args = parser.parse_args(sys.argv[3:])
            if (sys.argv.__contains__("--inRange") and sys.argv.__contains__("--outRange")): 
                response_image = adjustIntensity (img_float,args.inRange,args.outRange)
            elif (sys.argv.__contains__("--inRange")): response_image = adjustIntensity (img_float,args.inRange)
            elif (sys.argv.__contains__("--outRange")): response_image = adjustIntensity (img_float,outRange=args.outRange)
        else: response_image = adjustIntensity (img_float)
    
    ### python3 main.py <Ruta_imagen> equalizeIntensity [--bins value]   
    if (sys.argv[2] == "equalizeIntensity"):
        parser.add_argument('--bins',action="store",dest="nBins",type=int)
        if (sys.argv.__contains__("--bins")): 
            args = parser.parse_args(sys.argv[3:])
            response_image = equalizeIntensity (img_float,args.nBins)
        else: response_image = equalizeIntensity (img_float)
            
    
    ### python3 main.py <Ruta_imagen> filterImage --numberRows number --kernel value1 value2 ...    
    if (sys.argv[2] == "filterImage"):
        parser.add_argument('--numberRows',action="store",dest="numberRows",type=int)
        parser.add_argument('--kernel',action="store",dest="kernel",type=float, nargs='+')
        if (sys.argv.__contains__("--numberRows") and  sys.argv.__contains__("--kernel")):
            args = parser.parse_args(sys.argv[3:])
            kernel = np.array(args.kernel).reshape((args.numberRows, len(args.kernel)//args.numberRows))
            response_image = filterImage (img_float,kernel)
        else: 
            print("No se tienen los argumentos necesarios para ejecutar la operacion")
            exit(-1)

    ### python3 main.py <Ruta_imagen> gaussianKernel --sigma valor 
    if (sys.argv[2] == "gaussianFilter"):
        parser.add_argument('--sigma',action="store",dest="sigma",type=float)
        if (sys.argv.__contains__("--sigma")): 
            args = parser.parse_args(sys.argv[3:])
            response_image = gaussianFilter (img_float,args.sigma)
        else: 
            print("No se tienen los argumentos necesarios para ejecutar la operacion")
            exit(-1)

    ### python3 main.py <Ruta_imagen> medianFilter --filterSize valor 
    if (sys.argv[2] == "medianFilter"):
        parser.add_argument('--filterSize',action="store",dest="filterSize",type=int)
        if (sys.argv.__contains__("--filterSize")): 
            args = parser.parse_args(sys.argv[3:])
            response_image = medianFilter (img_float,args.filterSize)
        else: 
            print("No se tienen los argumentos necesarios para ejecutar la operacion")
            exit(-1)

    ### python3 main.py <Ruta_imagen> highBoost --A valor --method valor --param valor
    if (sys.argv[2] == "highBoost"):
        parser.add_argument('--A',action="store",dest="amplify",type=int)
        parser.add_argument('--method',action="store",dest="method",type=str)
        parser.add_argument('--param',action="store",dest="param",type=float)
        if (sys.argv.__contains__("--A") and  sys.argv.__contains__("--method") and sys.argv.__contains__("--param")):
            args = parser.parse_args(sys.argv[3:])
            response_image = highBoost (img_float,args.amplify,args.method,args.param)
        else: 
            print("No se tienen los argumentos necesarios para ejecutar la operacion")
            exit(-1)

    ###  python3 main.py <Ruta_imagen> erode --numberRows number [--center value1 value2] --SE value1 value2 ...
    if (sys.argv[2] == "erode"):
        parser.add_argument('--numberRows',action="store",dest="numberRows",type=int)
        parser.add_argument('--center',action="store",dest="center",type=int, nargs='+')
        parser.add_argument('--SE',action="store",dest="SE",type=int, nargs='+')
        if (sys.argv.__contains__("--numberRows") and sys.argv.__contains__("--SE")): 
            args = parser.parse_args(sys.argv[3:])
            StrE = np.array(args.SE).reshape((args.numberRows, len(args.SE)//args.numberRows))

            if (sys.argv.__contains__("--center")): response_image = erode (img_float,StrE,args.center)
            else: response_image = erode (img_float,StrE)
        else: 
            print("No se tienen los argumentos necesarios para ejecutar la operacion")
            exit(-1)

    ###  python3 main.py <Ruta_imagen> dilate --numberRows number [--center value1 value2] --SE value1 value2 ...
    if (sys.argv[2] == "dilate"):
        parser.add_argument('--center',action="store",dest="center",type=int, nargs='+')
        parser.add_argument('--SE',action="store",dest="SE",type=int, nargs='+')
        parser.add_argument('--numberRows',action="store",dest="numberRows",type=int)
        if (sys.argv.__contains__("--numberRows") and sys.argv.__contains__("--SE")): 
            args = parser.parse_args(sys.argv[3:])
            StrE = np.array(args.SE).reshape((args.numberRows, len(args.SE)//args.numberRows))

            if (sys.argv.__contains__("--center")): response_image = dilate (img_float,StrE,args.center)
            else: response_image = dilate (img_float,StrE)
        else: 
            print("No se tienen los argumentos necesarios para ejecutar la operacion")
            exit(-1)

    ###  python3 main.py <Ruta_imagen> opening --numberRows number [--center value1 value2] --SE value1 value2 ...
    if (sys.argv[2] == "opening"):
        parser.add_argument('--center',action="store",dest="center",type=int, nargs='+')
        parser.add_argument('--SE',action="store",dest="SE",type=int, nargs='+')
        parser.add_argument('--numberRows',action="store",dest="numberRows",type=int)
        if (sys.argv.__contains__("--numberRows") and sys.argv.__contains__("--SE")): 
            args = parser.parse_args(sys.argv[3:])
            StrE = np.array(args.SE).reshape((args.numberRows, len(args.SE)//args.numberRows))

            if (sys.argv.__contains__("--center")): response_image = dilate (img_float,StrE,args.center)
            else: response_image = dilate (img_float,StrE)
        else: 
            print("No se tienen los argumentos necesarios para ejecutar la operacion")
            exit(-1)
        
    ###  python3 main.py <Ruta_imagen> closing --numberRows number [--center value1 value2] --SE value1 value2 ...
    if (sys.argv[2] == "closing"):
        parser.add_argument('--center',action="store",dest="center",type=int, nargs='+')
        parser.add_argument('--SE',action="store",dest="SE",type=int, nargs='+')
        parser.add_argument('--numberRows',action="store",dest="numberRows",type=int)
        if (sys.argv.__contains__("--numberRows") and sys.argv.__contains__("--SE")): 
            args = parser.parse_args(sys.argv[3:])
            StrE = np.array(args.SE).reshape((args.numberRows, len(args.SE)//args.numberRows))

            if (sys.argv.__contains__("--center")): response_image = closing (img_float,StrE,args.center)
            else: response_image = closing (img_float,StrE)
        else: 
            print("No se tienen los argumentos necesarios para ejecutar la operacion")
            exit(-1)
    
    ###  python3 main.py <Ruta_imagen> fill --seeds value1 value2 .. [--center value1 value2] [--numberRows number] [--SE value1 value2 ...]
    if (sys.argv[2] == "fill"):
        parser.add_argument('--center',action="store",dest="center",type=int, nargs='+')
        parser.add_argument('--SE',action="store",dest="SE",type=int, nargs='+')
        parser.add_argument('--seeds',action="store",dest="seeds",type=int, nargs='+')
        parser.add_argument('--numberRows',action="store",dest="numberRows",type=int)
        if (sys.argv.__contains__("--seeds")): 
            args = parser.parse_args(sys.argv[3:])
            if ((sys.argv.__contains__("--SE")) and (sys.argv.__contains__("--numberRows"))): StrE = np.array(args.SE).reshape((args.numberRows, len(args.SE)//args.numberRows))
            else: StrE = []
            seeds = np.array(args.seeds).reshape((int(len(args.seeds)/2), len(args.seeds)//(int(len(args.seeds)/2))))
            if (sys.argv.__contains__("--center")):center = args.center
            else: center = []
            response_image = fill(img_float,seeds,StrE,center)
        else: 
            print("No se tienen los argumentos necesarios para ejecutar la operacion")
            exit(-1)

     ### python3 main.py <Ruta_imagen> gradientImage --operator operator
    if (sys.argv[2] == "gradientImage"):
        parser.add_argument('--operator',action="store",dest="operator",type=str)
        if (sys.argv.__contains__("--operator")): 
            args = parser.parse_args(sys.argv[3:])
            _,response_image = gradientImage (img_float,args.operator)
        else: 
            print("No se tienen los argumentos necesarios para ejecutar la operacion")
            exit(-1)

     ### python3 main.py <Ruta_imagen> edgeCanny --sigma fvalue --tlow value --thigh value
    if (sys.argv[2] == "edgeCanny"):
        parser.add_argument('--sigma',action="store",dest="sigma",type=float)
        parser.add_argument('--tlow',action="store",dest="tlow",type=float)
        parser.add_argument('--thigh',action="store",dest="thigh",type=float)
        if (sys.argv.__contains__("--sigma") and  sys.argv.__contains__("--tlow") and sys.argv.__contains__("--thigh")):
            args = parser.parse_args(sys.argv[3:])
            response_image = edgeCanny (img_float,args.sigma,args.tlow,args.thigh)
        else: 
            print("No se tienen los argumentos necesarios para ejecutar la operacion")
            exit(-1)

    
    if (args.plot): 
        #Imagen de entrada
        plot_image("imagenEntrada",img_float)
        #Imagen de salida
        plot_image("imagenSalida",response_image)
    if (image_out==True): save_image(args.image_output,response_image)
    cv2.destroyAllWindows()
