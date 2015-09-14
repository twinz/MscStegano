########################################################################
#                       ALEXANDRE MARTENS                              #
#         MSc Project. Breaking PixelKnot (Breaking F5 Algo)           #
#         Implementation of Fridrich Attack and Benford Attack         #
########################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import os

########################################################################
#                   BREAKING THE F5 ALGORITHM                          #
#                       Fridrich Attack                                #
########################################################################

def modifImage(path, name, type):
    jpeg = Image.open(path + name + type)
    quantization = jpeg.quantization
    jpeg.save(path + name + ".bmp", "BMP",  bmp_rle=True)
    bmp = Image.open(path + name + ".bmp")
    width, height = bmp.size
    jpeg_modif = bmp.crop((4, 4, width-4, height-4))
    jpeg_modif.save(path + name + "_modif.jpg", "JPEG", qtables=quantization)

def Beta(mapDct, mapDctModif):
    beta = np.empty((8,8))
    H_0 = np.zeros((8,8))
    H_1 = np.zeros((8,8))
    H_2 = np.zeros((8,8))
    hbarre_0 = np.zeros((8,8))
    hbarre_1 = np.zeros((8,8))
    hbarre_2 = np.zeros((8,8))

    for i in range(0,len(mapDct)):
        for k in range(0,8):
            for l in range(0,8):
                if (mapDct[i][k][l] == '0'):
                    H_0[k][l] += 1
                elif (mapDct[i][k][l] == '1' or mapDct[i][k][l] == '-1'):
                    H_1[k][l] += 1
                elif(mapDct[i][k][l] == '2' or mapDct[i][k][l] == '-2'):
                    H_2[k][l] += 1

    for i in range(0,len(mapDctModif)):
        for k in range(0,8):
            for l in range(0,8):
                if (mapDctModif[i][k][l] == '0'):
                    hbarre_0[k][l] += 1
                elif (mapDctModif[i][k][l] == '1' or mapDctModif[i][k][l] == '-1'):
                    hbarre_1[k][l] += 1
                elif(mapDctModif[i][k][l] == '2' or mapDctModif[i][k][l] == '-2'):
                    hbarre_2[k][l] += 1

    for k in range(0,8):
        for l in range(0,8):
            a = ( hbarre_1[k][l] * (H_0[k][l] - hbarre_0[k][l]) ) +  ( (H_1[k][l] - hbarre_1[k][l]) * (hbarre_2[k][l] - hbarre_1[k][l]) )
            b = ( hbarre_1[k][l] * hbarre_1[k][l] ) + ( hbarre_2[k][l] - hbarre_1[k][l] ) * ( hbarre_2[k][l] - hbarre_1[k][l] )
            beta[k][l] = a/b

    betaAverage = (beta[0][1] + beta[1][0] + beta[1][1]) / 3
    return betaAverage

def MsgLength(beta, mapDctModif):
    nbAcNoNull = 0
    nbCoefEgal1 = 0

    for y in range(8):
        for x in range(8):
            nbCoefEgal1 += mapDctModif[y][x].count('1')
            nbCoefEgal1 +=  mapDctModif[y][x].count('-1')
            for item in mapDctModif[y][x]:
                if (item != '0' and item != '-0'):
                    nbAcNoNull += 1
    k = (math.log(1 / beta) + 1) / math.log(2)
    capacity = np.around(nbAcNoNull - 0.51 * nbCoefEgal1)
    length = (math.pow(k,2) / (pow(k,2) - 1)) * k * beta * capacity
    return length

########################################################################
#                            FOR FRIDRICH                               #
#                 Read DCT on file, Put on a 2D list                    #
########################################################################

def dctFridrich(dctFile):
    dctList = []
    with open(dctFile) as f:
        content = f.readlines()
    for i in range(0, len(content)):
        z = 0
        block = content[i].split()
        dctList.append([])
        for x in range(0,8):
            dctList[i].append([])
            for y in range(0,8):
                dctList[i][x].append(block[z])
                z += 1
    return dctList

########################################################################
#                            FOR BENFORD                               #
#       Read DCT on file, del DC coeff and 0. Put on a list            #
########################################################################

def dctBenford(dctFile):
    dctList = []
    with open(dctFile) as f:
        content = f.readlines()

    for i in range(0, len(content)):
        block = content[i].split()
        del block[0]
        while block.count('0') > 0:
            block.remove('0')
        dctList += block

    return dctList

########################################################################
#                            BENFORD                                   #
########################################################################

def benford_law():
    N = 1.344
    S = -0.376
    q = 1.685

    return [(N * math.log10(1 + (1 / ( S + math.pow(i, q)))))*100.0 for i in xrange(1,10)]

def find_leading_number(line):
    numbers = "123456789"
    line = str(line)
    index = len(line)
    for i in range(0, index):
        if line[i] in numbers:
            return int(line[i])
    return 0

def calc_firstdigit(dctList):
   fdigit = [str(find_leading_number(value)) for value in dctList]

   distr = [fdigit.count(str(i))/float(len(dctList))*100 for i in xrange(1, 10)]
   return distr

def pearson(x,y):
   nx = len(x)
   ny = len(y)
   if nx != ny: return 0
   if nx == 0: return 0
   n = float(nx)
   meanx = sum(x)/n
   meany = sum(y)/n
   sdx = math.sqrt(sum([(a-meanx)*(a-meanx) for a in x])/(n-1) )
   sdy = math.sqrt(sum([(a-meany)*(a-meany) for a in y])/(n-1) )
   normx = [(a-meanx)/sdx for a in x]
   normy = [(a-meany)/sdy for a in y]
   return sum([normx[i]*normy[i] for i in range(nx)])/(n-1)

########################################################################
#                             PLOT                                     #
########################################################################

# FOR BENBORD
def plot_comparative(aset, bset, dataset_label):
   aset = [0] + aset
   bset = [0] + bset
   plt.axis([1, 9, 0, 60])
   plt.plot(aset, linewidth=1.0)
   plt.plot(bset, linewidth=1.0)
   plt.xlabel("First Digit")
   plt.ylabel("Perc. %%")
   plt.title("Benfords's law for %s (Pearson's Corr. %.2f)" % (dataset_label, pearson(aset, bset)))
   plt.legend((dataset_label, "Benford's Law"))
   plt.grid(True)
   return plt.show()

if __name__ == "__main__":

    ## TEST  BENFORD
    file = open("result.txt", "w")
    path = "C:\Users\Alexandre\Dropbox\kent\Project_Research\MscStegano\dctFile\dctFilePure\\"
    tab = {}
    for picture in os.listdir(path):
        name = os.path.splitext(picture)[0]
        bendordLaw = benford_law()
        dctListPure = dctBenford(path + picture)
        pure = calc_firstdigit(dctListPure)
        dctListStego = dctBenford("C:\Users\Alexandre\Dropbox\kent\Project_Research\MscStegano\dctFile\dctFile\\" + picture.replace("pure", "stego"))
        stego = calc_firstdigit(dctListStego)
        i = 0
        diff = 0
        file.write("Picture: " + name + "\n")
        file.write("First Digits\tDeviations(pure)\tDeviations(stego)\tDifference\n")
        for value in bendordLaw:
            file.write(str(i + 1) + "\t\t\t\t" + str(round(pure[i] - bendordLaw[i], 2)) + "\t\t\t\t" + str(round(stego[i] - bendordLaw[i], 2)) + "\t\t\t\t" + str(round( (pure[i] - bendordLaw[i]) - (stego[i] - bendordLaw[i]) , 2)) + "\n")
            diff += abs((pure[i] - bendordLaw[i]) - (stego[i] - bendordLaw[i]))
            i += 1
        tab[name] = round(diff,2)
    print tab

    # TEST  FRIDRICH
    dctArray = dctFridrich("C:\Users\Alexandre\Dropbox\kent\Project_Research\MscStegano\dctFile\dctFilePure\\vinePureDCT.txt")
    dctArray2 = dctFridrich("C:\Users\Alexandre\Dropbox\kent\Project_Research\MscStegano\dctFile\dctFilePureModif\\vineModifPureDCT.txt")
    beta = Beta(dctArray, dctArray2)
    print "Beta = " + str(beta)
    msgLength = MsgLength(beta, dctArray2)
    print "msgLength = " + str(msgLength)
