from PIL import Image
from plantcv import plantcv as pcv
import cv2
import numpy as np
import os, shutil, glob, os.path
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import glob
from imutils import paths
import argparse
import scipy.stats
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import statistics
import concurrent.futures

total_size_array = []
total_size_avg_array = []
total_size_std_array = []
total_color_array = []
total_color_avg_array = []
total_color_std_array = []
temp_array = []
temp_color = []
image_counter = 0


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=False,
                help="path to input directory of images")
args = vars(ap.parse_args())

def initexecl_liz(): #U platenumber - plate row A-H col number 1-12
    df = pd.DataFrame(
        {'Image processed': (), 'Cluster': (), 'Q1_size': (), 'Q2_size': (), 'Q3_size': (),
         'Q4_size': (),'Avg_size': (), 'Size_stdev': (), 'Q1_Zscore': (),'Q2_Zscore': (), 'Q3_Zscore': (),
         'Q4_Zscore': (), 'Zscore_avg': (), '# of threshold': (), 'modifier': (), 'temp': (),'Hit': (),
         'Q1_colorfullness': (), 'Q2_colorfullness': (), 'Q3_colorfullness': (), 'Q4_colorfullness': (),
         'Avg_color': (),'Color_stdev':(), 'Q1_color_Zscore' : (), 'Q2_color_Zscore': (), 'Q3_color_Zscore': (),
         'Q4_color_Zscore': (),'color_Zscore_avg': (), '# of threshold': (), 'color_modifier': (), 'color_temp': (),
         'colorhit': (),'POSITIVE': ()})
    writer = pd.ExcelWriter("A_test.xlsx", engine='openpyxl')
    df.to_excel(writer,index=False, header=True, startcol=0)
    writer.save()
    # https://medium.com/better-programming/using-python-pandas-with-excel-d5082102ca27

def initexecl_chris(): #U platenumber - plate row A-H col number 1-12
    df = pd.DataFrame(
        {'Image processed': (), 'Cluster': (), 'Q1_size': (), 'Q2_size': (), 'Q3_size': (),
         'Q4_size': (),'Avg_size': (), 'Size_stdev': (), 'Q1_Zscore': (),'Q2_Zscore': (), 'Q3_Zscore': (),
         'Q4_Zscore': (), 'Zscore_avg': (), '# of threshold': (), 'modifier': (), 'temp': (),'POSITIVE': ()})
    writer = pd.ExcelWriter("A_test.xlsx", engine='openpyxl')
    df.to_excel(writer,index=False, header=True, startcol=0)
    writer.save()

print("Liz processing 1 Chris processing 2")
mode = input("press 1 or 2 than press enter: ")
if int(mode) == 1:
    initexecl_liz()
    #print("test")
if int(mode) == 2:
    #print("testplz")
    initexecl_chris()

imagePath = sorted(list(paths.list_images(args["images"])))
for i in range(len(imagePath)):
    img = Image.open(imagePath[i])
    #img.show()
    base = os.path.basename(imagePath[i])
    print("PROCESSING IMAGE %s..." %base)

    def initcrop(img): #change line 70 for threshold adj
        left = 1875  # was 2050
        top = 730  # was 870
        right = 5680
        bottom = 3260  # was 3280
        dire = os.getcwd()
        path = dire + '/Classifyer_dump'
        try:
            os.makedirs(path)
        except OSError:
            pass
        img_crop = img.crop((left, top, right, bottom))
        # img_crop.show()
        img_crop.save(os.path.join(path, 'Cropped_full_yeast.png'))
        circle_me = cv2.imread(os.path.join(path, "Cropped_full_yeast.png"))
        cropped_img = cv2.imread(
            os.path.join(path, "Cropped_full_yeast.png"))  # changed from Yeast_Cluster.%d.png  %counter
        blue_image = pcv.rgb2gray_lab(cropped_img, 'b')  # can do l a or b
        Gaussian_blue = cv2.adaptiveThreshold(blue_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 241,
                                              -1)  # For liz's pictures 241
        cv2.imwrite(os.path.join(path, "blue_test.png"), Gaussian_blue)
        blur_image = pcv.median_blur(Gaussian_blue, 10)
        heavy_fill_blue = pcv.fill(blur_image, 400)  # value 400
        cv2.imwrite(os.path.join(path, "Cropped_Threshold.png"), heavy_fill_blue)


    def cluster_maker():
        counter3 = 0
        counter1 = 0
        dire = os.getcwd()
        path = dire + '/Classifyer_dump'
        path1 = dire + '/Yeast_cluster_inv'
        path2 = dire + '/Yeast_cluster'
        path3 = dire + '/Cells'
        try:
            os.makedirs(path)
        except OSError:
            pass
        try:
            os.makedirs(path1)
        except OSError:
            pass
        try:
            os.makedirs(path2)
        except OSError:
            pass
        try:
            os.makedirs(path3)
        except OSError:
            pass
        counter = 0
        im = Image.open(os.path.join(path, "Cropped_Threshold.png"))  # was "Cropped_full_yeast.png"
        sizeX, sizeY = im.size
        im_sizeX = round(sizeX / 12)
        im_sizeY = round(sizeY / 8)
        for h in range(0, im.height, im_sizeY):
            nim = im.crop((0, h, im.width - 1, min(im.height, h + im_sizeY) - 1))
            nim.save(os.path.join(path, "Yeast_Row." + str(counter) + ".png"))
            counter += 1
        anotherCounter = 0
        for i in range(0, 8):
            columnImage = (os.path.join(path, "Yeast_Row.%d.png" % anotherCounter))
            Each_Image = Image.open(columnImage)
            sizeX2, sizeY2 = Each_Image.size
            Each_Image_sizeX = round(sizeX2 / 12)
            Each_Image_sizeY = round(sizeY2 / 8)
            anotherCounter += 1
            widthCounter1 = 0
            widthCounter2 = Each_Image_sizeX
            for w in range(0, 12):
                Wim = Each_Image.crop(
                    (widthCounter1, w, widthCounter2, min(Each_Image.height, w + Each_Image_sizeX) - 1))
                Wim.save(os.path.join(path1, "Yeast_Cluster." + str(counter1) + ".png"))
                counter1 += 1
                widthCounter1 = widthCounter1 + Each_Image_sizeX
                widthCounter2 = widthCounter2 + Each_Image_sizeX
                print(counter1)
        im = Image.open(os.path.join(path, "Cropped_full_yeast.png"))  # was "Cropped_full_yeast.png"
        counter = 0
        anotherCounter = 0
        counter1 = 0
        sizeX, sizeY = im.size
        im_sizeX = round(sizeX / 12)
        im_sizeY = round(sizeY / 8)
        for h in range(0, im.height, im_sizeY):
            nim = im.crop((0, h, im.width - 1, min(im.height, h + im_sizeY) - 1))
            nim.save(os.path.join(path, "Yeast_Row." + str(counter) + ".png"))
            counter += 1
        anotherCounter = 0
        for i in range(0, 8):
            columnImage = (os.path.join(path, "Yeast_Row.%d.png" % anotherCounter))
            Each_Image = Image.open(columnImage)
            sizeX2, sizeY2 = Each_Image.size
            Each_Image_sizeX = round(sizeX2 / 12)
            Each_Image_sizeY = round(sizeY2 / 8)
            anotherCounter += 1
            widthCounter1 = 0
            widthCounter2 = Each_Image_sizeX
            for w in range(0, 12):
                Wim = Each_Image.crop(
                    (widthCounter1, w, widthCounter2, min(Each_Image.height, w + Each_Image_sizeX) - 1))
                Wim.save(os.path.join(path2, "Yeast_Cluster." + str(counter1) + ".png"))
                counter1 += 1
                widthCounter1 = widthCounter1 + Each_Image_sizeX
                widthCounter2 = widthCounter2 + Each_Image_sizeX
        row_counter_for_save = 0
        row_counter_for_open = 0
        for i in range(0, 96):
            im = Image.open(os.path.join(path2, "Yeast_Cluster.%d.png" % i))
            sizeX, sizeY = im.size
            im_sizeX = round(sizeX / 2)
            im_sizeY = round(sizeY / 2)
            for h in range(0, im.height, im_sizeY):
                nim = im.crop((0, h, im.width - 1, min(im.height, h + im_sizeY) - 1))
                nim.save(os.path.join(path, "ROW_SMALL." + str(row_counter_for_save) + ".png"))
                row_counter_for_save += 1
                if (h >= im_sizeY):
                    break
            for i in range(0, 2):
                rowImage = (os.path.join(path, "ROW_SMALL.%d.png" % row_counter_for_open))
                Each_Image = Image.open(rowImage)
                sizeX2, sizeY2 = Each_Image.size
                Each_Image_sizeX = round(sizeX2 / 2)
                Each_Image_sizeY = round(sizeY2 / 2)
                row_counter_for_open += 1
                widthCounter1 = 0
                widthCounter2 = Each_Image_sizeX
                for w in range(0, 2):
                    Wim = Each_Image.crop(
                        (widthCounter1, w, widthCounter2, min(Each_Image.height, w + Each_Image_sizeX) - 1))
                    Wim.save(os.path.join(path3, "SMALL_CELL." + str(counter3) + ".png"))
                    counter3 += 1
                    widthCounter1 = widthCounter1 + Each_Image_sizeX
                    widthCounter2 = widthCounter2 + Each_Image_sizeX


    def connected_comps_for_liz(counter):
        dire = os.getcwd()
        path = dire + '/Yeast_cluster_inv'
        cropped_img = cv2.imread(os.path.join(path, 'Yeast_Cluster.%d.png' % counter),
                                 cv2.IMREAD_UNCHANGED)  # changed from Yeast_Cluster.%d.png  %counter
        circle_me = cv2.imread(os.path.join(path, "Yeast_Cluster.%d.png" % counter))

        connected_counter = 0

        connectivity = 8

        connectivity = 8  # either 4 or 8
        output = cv2.connectedComponentsWithStats(cropped_img, connectivity)  # Will determine the size of our clusters

        num_labels = output[0]
        labels = output[1]
        stats = output[2]  # for size
        centroids = output[3]  # for location

        #print("Currently on cell %d" % counter)
        cc_size_array = []
        #print(centroids)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 40 and centroids[i][0] <= 110 and centroids[i][1] >= 40 and centroids[i][1] <= 110):
                print("%d is in 1" % i)
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 40 and centroids[i][1] <= 110):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                print("%d is in 2" % i)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 40 and centroids[i][0] <= 110 and centroids[i][1] >= 200 and centroids[i][1] <= 270):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                print("%d is in 3" % i)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 200 and centroids[i][
                1] <= 270):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                print("%d is in 4" % i)

        if (len(stats) < 4):
            print("too few decteted on %d" % counter)
            print((len(stats)))
            for i in range((len(stats)), 5, 1):
                cc_size_array.append(0)

        if (len(cc_size_array) >= 5):
            print("problem on cell %d" % counter)
            exit(-1)

        # total_size_array = total_size_array + cc_size_array
        #print("size data")
        #print(cc_size_array)
        avg_size = np.average(cc_size_array)
        #print(avg_size)
        std = np.std(np.array(cc_size_array))
        #print(std)
        Zscore_array = abs(scipy.stats.zscore(cc_size_array))
        #print(Zscore_array)
        Z_avg = np.average(Zscore_array)
        #print(Z_avg)
        mod = 1.5  # if avg zscore is less than .5 or 40% that is rewarded
        if Z_avg >= .8: # completely random number lol
            mod = 1
        above_size_ther = 0
        for i in range(0, len(cc_size_array)):
            if cc_size_array[i] <= 3500:  # for liz she wants small?
                above_size_ther += 1

        temp = ((10*above_size_ther)-(mod * Z_avg))  # simply alg to tell is positive gets normalized later
        #print(temp)


        #print("end of size data")



        return cc_size_array, avg_size, std, Zscore_array, Z_avg, above_size_ther, mod, temp


    def connected_comps_for_Chris(counter):
        dire = os.getcwd()
        path = dire + '/Yeast_cluster_inv'
        cropped_img = cv2.imread(os.path.join(path, 'Yeast_Cluster.%d.png' % counter),
                                 cv2.IMREAD_UNCHANGED)  # changed from Yeast_Cluster.%d.png  %counter
        circle_me = cv2.imread(os.path.join(path, "Yeast_Cluster.%d.png" % counter))

        connected_counter = 0

        connectivity = 8

        connectivity = 8  # either 4 or 8
        output = cv2.connectedComponentsWithStats(cropped_img, connectivity)  # Will determine the size of our clusters

        num_labels = output[0]
        labels = output[1]
        stats = output[2]  # for size
        centroids = output[3]  # for location

        #print("Currently on cell %d" % counter)
        cc_size_array = []
        #print(centroids)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 40 and centroids[i][0] <= 110 and centroids[i][1] >= 40 and centroids[i][1] <= 110):
                print("%d is in 1" % i)
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 40 and centroids[i][1] <= 110):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                print("%d is in 2" % i)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 40 and centroids[i][0] <= 110 and centroids[i][1] >= 200 and centroids[i][1] <= 270):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                print("%d is in 3" % i)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 200 and centroids[i][
                1] <= 270):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                print("%d is in 4" % i)

        if (len(stats) < 4):
            print("too few decteted on %d" % counter)
            print((len(stats)))
            for i in range((len(stats)), 5, 1):
                cc_size_array.append(0)

        if (len(cc_size_array) >= 5):
            print("problem on cell %d" % counter)
            exit(-1)

        # total_size_array = total_size_array + cc_size_array
        #print("size data")
        #print(cc_size_array)
        avg_size = np.average(cc_size_array)
        #print(avg_size)
        std = np.std(np.array(cc_size_array))
        #print(std)
        Zscore_array = abs(scipy.stats.zscore(cc_size_array))
        #print(Zscore_array)
        Z_avg = np.average(Zscore_array)
        #print(Z_avg)
        mod = 1.5  # if avg zscore is less than .5 or 40% that is rewarded
        if Z_avg >= .8:  # completely random number lol
            mod = 1
        above_size_ther = 0
        for i in range(0, len(cc_size_array)):
            if cc_size_array[i] >= 3500:  # for Chris likes them big
                above_size_ther += 1

        temp = ((10 * above_size_ther) - (mod * Z_avg))  # simply alg to tell is positive gets normalized later
        #print(temp)

        #print("end of size data")

        return cc_size_array, avg_size, std, Zscore_array, Z_avg, above_size_ther, mod, temp


    def image_colorfulness(image):
        # split the image into its respective RGB components
        (B, G, R) = cv2.split(image.astype("float"))
        # compute rg = R - G
        rg = np.absolute(R - G)
        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)
        # compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        # combine the mean and standard deviations
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        # derive the "colorfulness" metric and return it
        return stdRoot + (0.3 * meanRoot)


    def colorful_writer(color_counter):
        dire = os.getcwd()
        path = dire + '/Cells'
        # https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
        color_array = []
        for i in range(0, 4):
            #print("THIS IS COLOR COUNTER")
            #print(color_counter)
            image = cv2.imread(os.path.join(path, "SMALL_CELL.%d.png" % color_counter))
            C = image_colorfulness(image)
            # display the colorfulness score on the image
            color_array.append(C)
           # cv2.putText(image, "{:.2f}".format(C), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            #cv2.imwrite(os.path.join(path, "SMALL_CELL.%d.png" % color_counter), image)
            color_counter = color_counter + 1

        # total_color_array = total_color_array + color_array
        avg_color = (color_array[0] + color_array[1] + color_array[2] + color_array[3]) / 4
        std_color = np.std(np.array(color_array))

        if (len(color_array) < 4):
            print((len(color_array)))
            for i in range((len(color_array)), 5, 1):
                color_array.append(0)

        Zscore_array = abs(scipy.stats.zscore(color_array))
        #print(Zscore_array)
        Z_avg = np.average(Zscore_array)
        #print(Z_avg)
        mod = 1.5  # if avg zscore is less than .5 or 40% that is rewarded
        if Z_avg >= .8:  # completely random number lol
            mod = 1
        above_size_ther = 0
        for i in range(0, len(color_array)):
            if color_array[i] <= 23: #pretty white ones???
                above_size_ther += 1
        temp = ((10*above_size_ther) - (mod * Z_avg))


        #print(color_array)
        return color_array, avg_color, std_color, color_counter, Zscore_array, Z_avg, above_size_ther, mod, temp


    def excel_writer_liz(base,toomanycounter, anothercounter, image_counter, cc_size_array, avg_size, std,
                     Zscore, Zscore_avg, above_thres, mod,temp, color_array, avg_color,
                     std_color, Zscore_color, Zscore_color_avg, above_C_thres, mod_C, temp_C):
        char = chr(toomanycounter + 64)
        plate_name = ("U%d-%c%d" % (image_counter, char, anothercounter))
        print(plate_name)
        new_df = pd.DataFrame(
            {'Image processed':(base),'Cluster' :(plate_name),
             'Q1_size': (cc_size_array[0]), 'Q2_size': (cc_size_array[1]),'Q3_size': (cc_size_array[2]),
             'Q4_size': (cc_size_array[3]),'Avg_size': (avg_size), 'Size_stdev': (std),
             'Q1_Zscore': (Zscore[0]), 'Q2_Zscore': (Zscore[1]), 'Q3_Zscore': (Zscore[2]),
             'Q4_Zscore': (Zscore[3]),'Avg_Zscore': (Zscore_avg), '# above threshold': (above_thres),
             'modifier': (mod), 'temp': (temp), 'hit':(temp),
             'Q1_color': (color_array[0]), 'Q2_color': (color_array[1])
                , 'Q3_color': (color_array[2]), 'Q4_color': (color_array[3]), 'Avg_color': (avg_color),
             'Color_stdev': (std_color), 'Q1_color_Zscore' : (Zscore_color[0]), 'Q2_color_Zscore': (Zscore_color[1]),
             'Q3_color_Zscore': (Zscore_color[2]),'Q4_color_Zscore': (Zscore_color[3]),
             'color_Zscore_avg': (Zscore_color_avg), '# of threshold': (above_C_thres), 'modifier': (mod_C),
             'color_temp': (temp_C),'colorhit': (temp_C), 'POSITIVE': (temp_C)}, index=[0])
        writer = pd.ExcelWriter('A_test.xlsx', engine='openpyxl')
        writer.book = load_workbook('A_test.xlsx')
        writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        reader = pd.read_excel(r'A_test.xlsx')
        new_df.to_excel(writer, index=False, header=False, startcol=0, startrow=len(reader) + 1)
        writer.close()


    def excel_writer_chris(base, toomanycounter, anothercounter, image_counter, cc_size_array, avg_size, std,
                         Zscore, Zscore_avg, above_thres, mod, temp):
        char = chr(toomanycounter + 64)
        plate_name = ("U%d-%c%d" % (image_counter, char, anothercounter))
        print(plate_name)
        new_df = pd.DataFrame(
            {'Image processed': (base), 'Cluster': (plate_name),
             'Q1_size': (cc_size_array[0]), 'Q2_size': (cc_size_array[1]), 'Q3_size': (cc_size_array[2]),
             'Q4_size': (cc_size_array[3]), 'Avg_size': (avg_size), 'Size_stdev': (std),
             'Q1_Zscore': (Zscore[0]), 'Q2_Zscore': (Zscore[1]), 'Q3_Zscore': (Zscore[2]),
             'Q4_Zscore': (Zscore[3]), 'Avg_Zscore': (Zscore_avg), '# above threshold': (above_thres),
             'modifier': (mod), 'temp': (temp),'POSITIVE': (temp)}, index=[0])
        writer = pd.ExcelWriter('A_test.xlsx', engine='openpyxl')
        writer.book = load_workbook('A_test.xlsx')
        writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        reader = pd.read_excel(r'A_test.xlsx')
        new_df.to_excel(writer, index=False, header=False, startcol=0, startrow=len(reader) + 1)
        writer.close()


    toomanycounter = 1
    anothercounter = 1
    color_counter = 0
    plate_size = []
    initcrop(img)
    cluster_maker()
    if int(mode) == 1:
        print("liz")
        image_counter = image_counter + 1
        for c in range(0, 96):
            returned_size = connected_comps_for_liz(c) #inputs is counter for which cluster to process and output is an array with size, avg size, and std
            #print(returned_size)
            returned_color = colorful_writer(color_counter) #input is color_counter so knows which cell to process output is an array with colorfulness, avg color, and std
            #print((returned_color))
            excel_writer_liz(base,toomanycounter, anothercounter, image_counter, returned_size[0], returned_size[1],
                         returned_size[2], returned_size[3], returned_size[4],returned_size[5],returned_size[6],
                         returned_size[7],
                         returned_color[0], returned_color[1], returned_color[2], returned_color[4],returned_color[5],
                         returned_color[6], returned_color[7], returned_color[8])  #outputs excel sheet
            anothercounter = anothercounter + 1
            if anothercounter > 12:
                anothercounter = 1
                toomanycounter = toomanycounter + 1
            color_counter = returned_color[3]
            plate_size.extend(returned_size[0])
            temp_array.append(returned_size[7])
            temp_color.append(returned_color[8])
    if int(mode) == 2:
        print("chris test")
        image_counter = image_counter + 1
        for c in range(0, 96):
            returned_size = connected_comps_for_Chris(c)
            excel_writer_chris(base, toomanycounter, anothercounter, image_counter, returned_size[0], returned_size[1],
                             returned_size[2], returned_size[3], returned_size[4], returned_size[5], returned_size[6],
                             returned_size[7])
            anothercounter = anothercounter + 1
            if anothercounter > 12:
                anothercounter = 1
                toomanycounter = toomanycounter + 1
            plate_size.extend(returned_size[0])
            temp_array.append(returned_size[7])
    dire = os.getcwd()
    path = dire + '/Histograms'
    try:
        os.makedirs(path)
    except OSError:
        pass
    #print("Array for plate size")
    #print(plate_size)
    plt.hist(plate_size, bins=10)
    plt.xlabel('Size of Cell')
    plt.ylabel('Amount in bin')
    plt.title('Hist for Image %s' %base)
    plt.savefig(os.path.join(path,"hist%d.png" %image_counter))
    #plt.show()
    plt.close()
    total_size_array.extend(plate_size)
#print(total_size_array)
plt.hist(total_size_array,bins=10)
plt.xlabel('Size of Cell')
plt.ylabel('Amount in bin')
plt.title('Hist for every plate')
plt.savefig(os.path.join(path,"Hist_Every_plate.png"))
#plt.show()
plt.close()





#beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
#beep(5)

X = temp_array
X_max = max(X)
print(X_max)
normalize_array = []
normalize_array_color = []
pos_array = []
for i in range(0, (image_counter*96)):
    x = (X[i]/X_max) * 100
    if x > 70:
        x = 1
    else:
        x = 0
    normalize_array.append(x)

    X = temp_color
    X_max = max(X)
    #print(X_max)
    x = (X[i]/X_max) * 100
    if x > 70:
        x = 1
    else:
        x = 0
    normalize_array_color.append(x)

    if normalize_array[i] == 1 and normalize_array_color[i] == 1:
        pos_array.append(1)
    else:
        pos_array.append(0)

    new_df = pd.DataFrame(
                {'Hit': (normalize_array[i])}, index=[0])
    new_df2 = pd.DataFrame(
                {'colorhit': (normalize_array_color[i])}, index=[0])
    new_df3 = pd.DataFrame(
                {'POSITIVE': (pos_array[i])}, index=[0])
    writer = pd.ExcelWriter('A_test.xlsx', engine='openpyxl')
    writer.book = load_workbook('A_test.xlsx')
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    reader = pd.read_excel(r'A_test.xlsx')
    new_df.to_excel(writer, sheet_name='Sheet1', index=False, header=False, startcol=16, startrow=(i + 1))
    new_df2.to_excel(writer, index=False, header=False, startcol=30, startrow=(i + 1))
    new_df3.to_excel(writer, index=False, header=False, startcol=31, startrow=(i + 1))
    writer.close()
#print(pos_array)



"""
#kmeans stuff
imdir = '/Users/gregglickert/Documents/GitHub/YeastClassification/Cluster'
targetdir = "/Users/gregglickert/Documents/GitHub/YeastClassification/cluster_test"

filelist = glob.glob(os.path.join(imdir, '*.png'))
filelist.sort()

try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_): #changed from kmeans to dbscan
    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r") #same here
    shutil.move(filelist[i], '/Users/gregglickert/Documents/GitHub/YeastClassification/cluster_test')
"""
#His clustering
"""
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
#dend = shc.dendrogram(shc.linkage(X, method='ward'))
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
model.fit(X)
labels = model.labels_
plt.scatter(X[labels==0, 0], X[labels==0, 1], s=50, marker='o', color='red')
plt.scatter(X[labels==1, 0], X[labels==1, 1], s=50, marker='o', color='blue')
plt.scatter(X[labels==2, 0], X[labels==2, 1], s=50, marker='o', color='green')
plt.scatter(X[labels==3, 0], X[labels==3, 1], s=50, marker='o', color='purple')
plt.scatter(X[labels==4, 0], X[labels==4, 1], s=50, marker='o', color='orange')
plt.show()
"""






"""
x = data[["Avg_size","Avg_color"]]
plt.scatter(x["Avg_size"],x["Avg_color"])
plt.xlabel('Avg_size')
plt.ylabel('Avg_color')
plt.show()
"""
#https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/

"""
for i in range(0,96):
    cluster_array[i] = ([cluster_size_array[i] + cluster_color_std_array[i] + cluster_color_array[i] + cluster_size_std_array[i]])
print("clustered is")
print(cluster_array)
print(len(cluster_array))

imdir = '/Users/gregglickert/PycharmProjects/cc_test/CLUSTERD'
targetdir = "/Users/gregglickert/PycharmProjects/cc_test/testing folder"
number_clusters = 5

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.png'))
filelist.sort()

kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(total_size_array))

try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_): #changed from kmeans to dbscan
    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r") #same here
    shutil.move(filelist[i], '/Users/gregglickert/PycharmProjects/cc_test/testing folder2')

print(len(total_size_array))
print(total_size_array)
print(len(total_color_array))
print(total_color_array)

plt.hist(total_size_array)
plt.show()
"""
