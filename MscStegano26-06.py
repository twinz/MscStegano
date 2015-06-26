import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def loadImg(imgfile):
    global iHeight
    global iWidth
    global wndLoad
    global wndName

    img = cv2.imread(imgfile, cv2.CV_LOAD_IMAGE_COLOR)

    wndLoad = "pixeknot-lenna stego-image"
    cv2.namedWindow(wndLoad, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(wndLoad, img)

    iHeight, iWidth = img.shape[:2]
    print "size:", iWidth, "x", iHeight

    # set size to multiply of 8
    if (iWidth % 8) != 0:
        filler = img[:,iWidth-1:,:]
        for i in range(8 - (iWidth % 8)):
            img = np.append(img, filler, 1)

    if (iHeight % 8) != 0:
        filler = img[iHeight-1:,:,:]
        for i in range(8 - (iHeight % 8)):
            img = np.append(img, filler, 0)

    iHeight, iWidth = img.shape[:2]
    print "new size:", iWidth, "x", iHeight
    return img

def dct(coverImage):
    dctList = np.empty(shape=(iHeight, iWidth))
    # ETAPE 1: COLOR PROCESSING
    coverImage = cv2.cvtColor(coverImage, cv2.COLOR_BGR2YCR_CB)
    # ETAPE 2: SUB-SAMPLING
    for startY in range(0, iHeight, 8):
        for startX in range(0, iWidth, 8):
            # ETAPE 3: DIVISION INTO BLOCS
            for c in range(0, 3):
                block = coverImage[startY:startY+8, startX:startX+8, c:c+1].reshape(8,8)

                # ETAPE 4: DCT FOR EACH BLOCS
                blockf = np.float32(block)     # float conversion
                dst = cv2.dct(blockf)          # dct
                dst2 = np.around(dst)
                 # store the result
                for y in range(8):
                    for x in range(8):
                        if (x == 0 and y == 0):
                           pass
                        else:
                            dctList[startY+y, startX+x] = dst2[y, x]
    return dctList

def countDct(dctCoeff):
    mapDct = {}
    i = 0
    j = 0
    while (i <(dctCoeff.size / dctCoeff[i].size ) - 1):
        while (j < (dctCoeff[i].size / 3)):
            if mapDct.has_key(dctCoeff[i][j]):
                mapDct[dctCoeff[i][j]] += 1
            else:
                mapDct[dctCoeff[i][j]] = 1
            j += 1
        j = 0
        i += 1

    return mapDct

def printImg(mapDct, mapDct2):
    X = []
    Y = []

    X2 = []
    Y2 = []

    for key in mapDct:
        if (key >= -8 and key <= 8):
            X.append(key)
            Y.append(mapDct[key])

    for key in mapDct2:
        if (key >= -8 and key <= 8):
            X2.append(key)
            Y2.append(mapDct2[key])

    plt.title("DCT histogram")
    plt.hist(X, 15, weights=Y, color='b', label='stego-image', alpha=0.5)
    plt.hist(X2, 15, weights=Y2, color='r', label='cover-image', alpha=0.5)
    plt.xlabel('DCT')
    plt.ylabel('Number')
    plt.legend(loc='upper right')

    plt.grid(True)
    plt.show()

def modifImage(path):
    jpeg = Image.open(path)
    quantization = jpeg.quantization
    jpeg.save("new_lenna.bmp", "BMP",  bmp_rle=True)
    bmp = Image.open("new_lenna.bmp")
    width, height = bmp.size
    jpeg_modif = bmp.crop((4, 4, width-4, height-4))
    jpeg_modif.save("lenna_modif.jpg", "JPEG", qtables=quantization)



def Hkl(d, mapDct):
    P = mapDct[1] + mapDct[2] + mapDct[3] + mapDct[4] + mapDct[5] + mapDct[6] + mapDct[7] + mapDct[8]
    n = 20 #must be calculate
    if (d == 0):
        H = mapDct[0] + ( (n/P) * mapDct[1] )
    else:
        H = ((1 - (n/P)) * mapDct[d]) + ( (n/P) * mapDct[d + 1] )
    return H

def Beta(mapDct, mapDctModif):
    a = ( mapDctModif[1] * (Hkl(0, mapDct) - mapDctModif[0]) ) +  ( (Hkl(1, mapDct) - mapDctModif[1]) * (mapDctModif[2] - mapDctModif[1]) )
    if ((mapDctModif[2] - mapDctModif[1]) > 0):
        b = (mapDctModif[1] ** .5) + ( (mapDctModif[2] - mapDctModif[1]) ** .5 )
    else:
        b = (mapDctModif[1] ** .5)
    res = a/b
    return res

def MsgLength(beta, mapDct):
    P = mapDct[1] + mapDct[2] + mapDct[3] + mapDct[4] + mapDct[5] + mapDct[6] + mapDct[7] + mapDct[8]
    k = 5 #must be calculate
    m = ((math.pow(2,k)) / (math.pow(2,k) - 1) * k * beta * (P - mapDct[1]))
    return m

if __name__ == "__main__":
    imgLoaded = loadImg("images/pixeknot-lenna.jpg")
    dctArray = dct(imgLoaded)
    mapDct = countDct(dctArray)

    modifImage("images/pixeknot-lenna.jpg")
    imgLoaded2 = loadImg("lenna_modif.jpg")
    dctArray2 = dct(imgLoaded2)
    mapDct2 = countDct(dctArray2)

    printImg(mapDct, mapDct2)

    print "Beta = "
    beta = Beta(mapDct, mapDct2)
    print beta
    print "Message Length = "
    print MsgLength(beta, mapDct)


