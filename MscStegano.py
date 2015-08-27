import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import os
from collections import defaultdict

def loadImg(imgfile):
    global iHeight
    global iWidth
    global wndLoad
    global wndName
    global quantizationTable

    img = cv2.imread(imgfile, cv2.CV_LOAD_IMAGE_COLOR)
    quantization =  Image.open(imgfile).quantization
    quantization[2] = quantization[1]
    quantizationTable = []
    quantizationTable.append([])
    quantizationTable.append([])
    quantizationTable.append([])
    l = 0

    # Quantization table
    for i in range(len(quantization)):
        quantizationTable[i].append([])
        k = 0
        for j in range(len(quantization[i])):
            quantizationTable[i][k].append(quantization[i][j])
            l += 1
            if (l == 8):
                l = 0
                k += 1
                if (k != 8):
                    quantizationTable[i].append([])



    iHeight, iWidth = img.shape[:2]

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
    return img

def dct(coverImage):
    dctList = np.empty(shape=(iHeight, iWidth))
    #dctList = np.empty((iHeight, iWidth, 1000))
    dctList = []
    dctDict = {}
    d3_dict = defaultdict(lambda: defaultdict(dict))
    # ETAPE 1: COLOR PROCESSING
    #coverImage = cv2.cvtColor(coverImage, cv2.COLOR_BGR2YCR_CB)"""""""""""""""""""""""""""

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

                #quantization
                blockq = np.around(np.divide(dst, quantizationTable[c])) # ne pas recompresser pour friddish
                # blockq = np.multiply(blockq, quantizationTable[c])

                 # store the result

    #             for y in range(8):
    #                 for x in range(8):
    #                     if (dst2[y,x] <= 8 and dst2[y,x] >= -8):
    #                         if (dst2[y,x] in  d3_dict[y, x] ):
    #                             d3_dict[(y, x)][dst2[y,x]] += 1
    #                         else:
    #                             d3_dict[(y, x)][dst2[y,x]] = 1
    # return d3_dict

                for y in range(8):
                    for x in range(8):
                        #dctDict[startY+y, startX+x] = dst2[y, x]
                        #dctList.append(dst2[y, x])
                        dctList.append(blockq[y,x])

    return dctList



# def countDct(dctCoeff):
#     mapDct = {}
#     i = 0
#     j = 0
#     while (i <(dctCoeff.size / dctCoeff[i].size ) - 1):
#         while (j < (dctCoeff[i].size / 3)):
#             if mapDct.has_key(dctCoeff[i][j]):
#                 mapDct[dctCoeff[i][j]] += 1
#             else:
#                 mapDct[dctCoeff[i][j]] = 1
#             j += 1
#         j = 0
#         i += 1
#
#     return mapDct



def modifImage(path, name, type):
    jpeg = Image.open(path + name + type)
    quantization = jpeg.quantization
    jpeg.save(path + name + ".bmp", "BMP",  bmp_rle=True)
    bmp = Image.open(path + name + ".bmp")
    width, height = bmp.size
    jpeg_modif = bmp.crop((4, 4, width-4, height-4))
    jpeg_modif.save(path + name + "_modif.jpg", "JPEG", qtables=quantization)


########################################################################
#                   BREAKING THE F5 ALGORITHM                          #
#                       Fridrich Attack                                #
########################################################################

def Beta(mapDct, mapDctModif):
    beta = np.empty((8,8))
    for y in range(2):
        for x in range(2):
            if (x == 0 and y == 0):
                pass
            else:
                hbarre_0 = mapDctModif[y, x][0]
                hbarre_1 = mapDctModif[y, x][1] + mapDctModif[y, x][-1]
                hbarre_2 = mapDctModif[y, x][2] + mapDctModif[y, x][-2]
                H_0 = mapDct[y, x][0]
                H_1 = mapDct[y, x][1] + mapDct[y, x][-1]
                H_2 = mapDct[y, x][2] + mapDct[y, x][-2]

                a = ( hbarre_1 * (H_0 - hbarre_0) ) +  ( (H_1 - hbarre_1) * (hbarre_2 - hbarre_1) )
                if ((hbarre_2 - hbarre_1) > 0):
                    b = math.pow(hbarre_1, 2) + (math.pow(hbarre_2 - hbarre_1, 2))
                else:
                    b = math.pow(hbarre_1, 2)

                beta[y, x] = a/b

    betaAverage = (beta[0,1] + beta[1,0] + beta[1,1]) / 3
    return betaAverage

def MsgLength(beta, mapDctModif):
    nbAcNoNull = 0
    nbCoefEgal1 = 0

    for y in range(8):
        for x in range(8):
            if (x == 0 and y == 0):
                pass
            else:
                for item in mapDctModif[y,x]:
                    if (item == 1 or item == -1):
                        nbCoefEgal1 += mapDctModif[y,x][item]
                    if (item != 0 or item != -0):
                        nbAcNoNull += mapDctModif[y,x][item]
    k = (math.log(1 / beta) + 1) / math.log(2)
    capacity = np.around(nbAcNoNull - 0.51 * nbCoefEgal1)
    length = (math.pow(k,2) / (pow(k,2) - 1)) * k * beta * capacity  #0.51h(1) is the estimated loss due to shrinkage
    return length


########################################################################
#                            BENFORD                                   #
########################################################################

def find_leading_number(line):
    """
   Takes in a string
   Goes through the string to find the first occurrance of a number.
   Returns the number.
   """
    numbers = "123456789"
    line = str(line)
    index = len(line)
    for i in range(0, index):
        if line[i] in numbers:
            return int(line[i])
    return 0

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

def benford_law():
    N = 1.344
    S = -0.376
    q = 1.685

    return [(N * math.log10(1 + (1 / ( S + math.pow(i, q)))))*100.0 for i in xrange(1,10)]

def calc_firstdigit(dataset):
   fdigit = [str(find_leading_number(value)) for value in dataset]
   distr = [fdigit.count(str(i))/float(len(dataset))*100 for i in xrange(1, 10)]
   return distr


########################################################################
#                             PLOT                                     #
########################################################################

# FOR BENBORD
def plot_comparative(aset, bset, dataset_label):
   plt.plot(aset, linewidth=1.0)
   plt.plot(bset, linewidth=1.0)
   plt.xlabel("First Digit")
   plt.ylabel("Perc. %%")
   plt.title("Benfords's law for %s (Pearson's Corr. %.2f)" % (dataset_label, pearson(aset, bset)))
   plt.legend((dataset_label, "Benford's Law"))
   plt.grid(True)
   return plt.show()

# FOR BREAKING THE F5 ALGORITHM
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


if __name__ == "__main__":

########################################################################
#                       POUR TESTER UN BENFORD                         #
########################################################################

### Aff graph
    path = "C:\Users\Alexandre\Dropbox\kent\Project_Research\project\pictures\stego\\"
    name = "pixelknot-cave"

    imgLoaded = loadImg(path + name + ".jpg")
    dctArray = dct(imgLoaded)

    bendordLaw = benford_law()
    me = calc_firstdigit(dctArray)
    print me
    plot_comparative(me, benford_law(), name + "-JPEGCoeff")

    ## plot_occurrance_data(dctArray)
    ## plot_occurrance_data(["-0.7", "3.2", "-0.19", "0.25", "-0.5", "-4.5", "5.6"])
    ## plot_probability_data(["-0.7", "3.2", "-0.19", "0.25", "-0.5", "-4.5", "5.6"])




### Print value on file
    # file = open("result.txt", "w")
    # pathPure = "C:\Users\Alexandre\Dropbox\kent\Project_Research\project\pictures\\pure\\"
    # pathStego = "C:\Users\Alexandre\Dropbox\kent\Project_Research\project\pictures\\stego\\"
    # bendordLaw = benford_law()
    #
    # for picture in os.listdir(pathPure):
    #     print picture
    #     file.write("Processing " + picture + "\n")
    #     imgLoaded = loadImg(pathPure + picture)
    #     dctArray = dct(imgLoaded)
    #     bendord = calc_firstdigit(dctArray)
    #     file.write(str(bendord) + "\n")
    #
    #     print pathStego + "pixelknot-" + picture
    #     file.write("Processing " + "pixelknot-" + picture + "\n")
    #     imgLoaded = loadImg(pathStego + "pixelknot-" + picture)
    #     dctArray = dct(imgLoaded)
    #     bendord = calc_firstdigit(dctArray)
    #     file.write(str(bendord) + "\n")
    #
    #
    # file.close()




########################################################################
#                       POUR TESTER UN FICHIER                         #
########################################################################

    # path = "C:\Users\Alexandre\Dropbox\kent\Project_Research\project\pictures\stego\\"
    # picture = "pixelknot-boat.jpg"
    # name = "pixelknot-boat"
    #
    # print path + picture
    # imgLoaded = loadImg(path + picture)
    # dctArray = dct(imgLoaded)
    #
    # modifImage(path, name, ".jpg")
    # imgLoaded2 = loadImg(path + name + "_modif.jpg")
    # dctArray2 = dct(imgLoaded2)
    #
    #
    # beta = Beta(dctArray, dctArray2)
    # print beta
    # msgLength = MsgLength(beta, dctArray2)
    # print msgLength

########################################################################
#                       POUR TESTER UN DOSSIER                         #
########################################################################

    # file = open("result.txt", "a")
    # path = "C:\Users\Alexandre\Dropbox\kent\Project_Research\project\pictures\\stego\\"
    # for picture in os.listdir(path):
    #     print "Processing " + picture
    #     imgLoaded = loadImg(path + picture)
    #     dctArray = dct(imgLoaded)
    #
    #     modifImage(path, os.path.splitext(picture)[0], os.path.splitext(picture)[1])
    #     imgLoaded2 = loadImg(path + os.path.splitext(picture)[0] + "_modif.jpg")
    #     dctArray2 = dct(imgLoaded2)
    #
    #     beta = Beta(dctArray, dctArray2)
    #     msgLength = MsgLength(beta, dctArray2)
    #     file.write(picture + " : \n")
    #     file.write("\t Beta : " + str(beta) + "\n")
    #     print "\t Beta : " + str(beta)
    #     print "\t Msg Length : " + str(msgLength)
    #     file.write("\t Msg Length : " + str(msgLength) + "\n")
    #
    #     os.remove(path + os.path.splitext(picture)[0] + ".bmp")
    #     os.remove(path + os.path.splitext(picture)[0] + "_modif.jpg")
    # file.close()


########################################################################
#                       POUR AFFICHER GRAPH (ANCIEN)                   #
########################################################################

    #mapDct = countDct(dctArray)
    #mapDct2 = countDct(dctArray2)
    #printImg(mapDct, mapDct2)




