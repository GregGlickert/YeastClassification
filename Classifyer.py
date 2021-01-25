#!/usr/bin/env python
from PIL import Image
from plantcv import plantcv as pcv
import cv2
import numpy as np
import os, shutil, glob, os.path
import pandas as pd
import matplotlib.pyplot as plt
# from openpyxl import load_workbook
# import glob
from imutils import paths
# import argparse
import scipy.stats
# import sy
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import normalize
# import scipy.cluster.hierarchy as shc
# from sklearn.cluster import AgglomerativeClustering
# import statistics
# import concurrent.futures
import easygui
from tqdm import tqdm


total_size_array = []
total_size_avg_array = []
total_size_std_array = []
total_color_array = []
total_color_avg_array = []
total_color_std_array = []
temp_array = []
temp_color = []
base_arr = []
platename_arr = []
Q1_size = []
Q2_size = []
Q3_size = []
Q4_size = []
Z1_size = []
Z2_size = []
Z3_size = []
Z4_size = []
Z_avg = []
above_size_ther = []
mod_size = []
temp = []
Q1_color = []
Q2_color = []
Q3_color = []
Q4_color = []
Z1_color = []
Z2_color = []
Z3_color = []
Z4_color = []
Z_avg_color = []
above_size_ther_color = []
mod_color = []
temp_color = []

image_counter = 0

#GUI#
mode = easygui.indexbox(msg="What do you want to process"
                            "\nNote can not process a single image folder must have two or more images"
                            "\nFolder also must ONLY contain images wanting to processed and nothing else at this time"
                            "\nOutputs will be placed in folder selected",
                        title="Yeast Classifier",
                 choices=("Size and color", "Size"))
folder = easygui.diropenbox()
#Loops over every image in the selected folder
imagePath = sorted(list(paths.list_images(folder)))
for i in tqdm(range(len(imagePath))):
    img = Image.open(imagePath[i])
    #img.show()
    base = os.path.basename(imagePath[i])
    #print("PROCESSING IMAGE %s..." %base)

# crops the image into just the plate
    #can improve to add edge detection
    def initcrop(imagePath):
        dire = folder
        path = dire + '/Classifyer_dump'
        try:
            os.makedirs(path)
        except OSError:
            pass
        image = cv2.imread(imagePath)
        blue_image = pcv.rgb2gray_lab(image, 'l')
        Gaussian_blue = cv2.adaptiveThreshold(blue_image, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 981,
                                              -1)  # 241 is good 981
        cv2.imwrite(os.path.join(path, "blue_test.png"), Gaussian_blue)
        fill = pcv.fill_holes(Gaussian_blue)
        fill_again = pcv.fill(fill, 100000)

        id_objects, obj_hierarchy = pcv.find_objects(img=image, mask=fill_again)  #lazy way to findContours and draw them

        roi1, roi_hierarchy = pcv.roi.rectangle(img=image, x=3000, y=1000, h=200, w=300)

        roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img=image, roi_contour=roi1,
                                                                       roi_hierarchy=roi_hierarchy,
                                                                       object_contour=id_objects,
                                                                       obj_hierarchy=obj_hierarchy,
                                                                       roi_type='partial')
        cv2.imwrite(os.path.join(path, "plate_mask.png"), kept_mask)

        mask = cv2.imread(os.path.join(path, "plate_mask.png"))
        result = image * (mask.astype(image.dtype))
        result = cv2.bitwise_not(result)
        cv2.imwrite(os.path.join(path, "AutoCrop.png"), result)

        gray_image = pcv.rgb2gray_lab(result, 'b')
        plate_threshold = cv2.adaptiveThreshold(gray_image, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 241,
                                                -1)  # 241 is good 981
        cv2.imwrite(os.path.join(path, "plate_threshold.png"), plate_threshold)

        fill_again2 = pcv.fill(plate_threshold, 1000)

        id_objects, obj_hierarchy = pcv.find_objects(img=image, mask=fill_again2)

        roi1, roi_hierarchy = pcv.roi.rectangle(img=image, x=1950, y=800, h=75, w=75)

        roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img=image, roi_contour=roi1,
                                                                       roi_hierarchy=roi_hierarchy,
                                                                       object_contour=id_objects,
                                                                       obj_hierarchy=obj_hierarchy,
                                                                       roi_type='partial')
        cv2.imwrite(os.path.join(path, "test_mask.png"), kept_mask)
        mask = cv2.imread(os.path.join(path, "test_mask.png"))
        result = image * (mask.astype(image.dtype))
        result = cv2.bitwise_not(result)
        cv2.imwrite(os.path.join(path, "TEST.png"), result)

        output = cv2.connectedComponentsWithStats(kept_mask, connectivity=8)

        centroids = output[3]
        centroids_x = (int(centroids[1][0]))
        centroids_y = (int(centroids[1][1]))
        #print(centroids_x)
        #print(centroids_y)

        #print(centroids)

        left = (centroids_x - 70)
        right = (centroids_x + 3750)
        top = (centroids_y - 80)
        bottom = (centroids_y + 2450)
        #print(top)
        #print(bottom)
        image = Image.open(imagePath)
        img_crop = image.crop((left, top, right, bottom))
        img_crop.save(os.path.join(path, "CROPPED.png"))

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

# crops the plate into the clusters
    def cluster_maker():
        counter3 = 0
        counter1 = 0
        dire = folder #used to be os.getcwd()
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
                #print(counter1)
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

# runs CC and is looking for small cells returns stats
    def connected_comps_for_liz(counter):
        dire = folder #used to be os.getcwd()
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
                #print("%d is in 1" % i)
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 40 and centroids[i][1] <= 110):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                #print("%d is in 2" % i)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 40 and centroids[i][0] <= 110 and centroids[i][1] >= 200 and centroids[i][1] <= 270):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                #print("%d is in 3" % i)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 200 and centroids[i][
                1] <= 270):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                #print("%d is in 4" % i)

        if (len(stats) < 4):
            #print("too few decteted on %d" % counter)
            #print((len(stats)))
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

# runs CC and is looking for big cells returns stats
    def connected_comps_for_Chris(counter):
        dire = folder #used to be os.getcwd()
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
                #print("%d is in 1" % i)
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 40 and centroids[i][1] <= 110):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                #print("%d is in 2" % i)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 40 and centroids[i][0] <= 110 and centroids[i][1] >= 200 and centroids[i][1] <= 270):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                #print("%d is in 3" % i)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 200 and centroids[i][
                1] <= 270):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                #print("%d is in 4" % i)

        if (len(stats) < 4):
            #print("too few decteted on %d" % counter)
            #print((len(stats)))
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

# determines how much color is in an image returns "colorfulness" value
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

# crops cluster into cells and feeds to image_colorfulness
    def colorful_writer(color_counter):
        dire = folder # used to be os.getcwd()
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
        avg_color = np.average(color_array)
        std_color = np.std(np.array(color_array))

        if (len(color_array) < 4):
            #print((len(color_array)))
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

# normalizes size data and using max temp value determines if a hit took place
    def size_hit(temp_array):
        X = temp_array
        X_max = max(X)
        normalize_array = []
        for i in range(0, (image_counter * 96)):
            x = (X[i] / X_max) * 100
            if x > 70:
                x = 1
            else:
                x = 0
            normalize_array.append(x)
        return normalize_array

# normalizes color data and using max color temp determines if a hit took place
    def color_hit(temp_color):
        normalize_array_color = []
        X = temp_color
        X_max = max(X)
        for i in range(0, (image_counter * 96)):
            x = (X[i] / X_max) * 100
            if x > 70:
                x = 1
            else:
                x = 0
            normalize_array_color.append(x)
        return normalize_array_color

# uses color and size hit to determine pos or not
    def pos_hit(normalize_array, normalize_array_color):
        pos_array = []
        for i in range(0, (image_counter * 96)):
            if normalize_array[i] == 1 and normalize_array_color[i] == 1:
                pos_array.append(1)
            else:
                pos_array.append(0)
        return pos_array

# sends data to excel
    def excel_writer_liz(base_arr, platename_arr, Q1_size, Q2_size, Q3_size, Q4_size,
                         total_size_avg_array, total_size_std_array, Z1_size, Z2_size, Z3_size, Z4_size, Z_avg,
                         above_size_ther, mod_size, temp_array, Q1_color, Q2_color, Q3_color, Q4_color,
                         total_color_avg_array, total_color_std_array, Z1_color, Z2_color, Z3_color,
                         Z4_color, above_size_ther_color, mod_color, temp_color, size_pos, color_pos, pos_size, dire):
        new_df = pd.DataFrame(
            {'Image processed':(base_arr),'Cluster' :(platename_arr),
             'Q1_size': (Q1_size), 'Q2_size': (Q2_size),'Q3_size': (Q3_size),
             'Q4_size': (Q4_size),'Avg_size': (total_size_avg_array), 'Size_stdev': (total_size_std_array),
             'Q1_Zscore': (Z1_size), 'Q2_Zscore': (Z2_size), 'Q3_Zscore': (Z3_size),
             'Q4_Zscore': (Z4_size),'Avg_Zscore': (Z_avg), 'above_sizethres': (above_size_ther),
             'modifier': (mod_size), 'temp': (temp_array), 'hit': (size_pos), 'Q1_color': (Q1_color), 'Q2_color': (Q2_color)
                , 'Q3_color': (Q3_color), 'Q4_color': (Q4_color), 'Avg_color': (total_color_avg_array),
             'Color_stdev': (total_color_std_array), 'Q1_color_Zscore' : (Z1_color), 'Q2_color_Zscore': (Z2_color),
             'Q3_color_Zscore': (Z3_color),'Q4_color_Zscore': (Z4_color),
             'color_Zscore_avg': (Z_avg_color), 'above color_thres': (above_size_ther_color), 'modifier_color': (mod_color),
             'color_temp': (temp_color), 'colorhit': (color_pos), 'POSITIVE': (pos_size)})
        os.chdir(dire)
        Excel_name = "A_test.xlsx"
        new_df.to_excel(Excel_name)

# sends data to excel
    def excel_writer_chris(base_arr, platename_arr, Q1_size, Q2_size, Q3_size, Q4_size,
                         total_size_avg_array, total_size_std_array, Z1_size, Z2_size, Z3_size, Z4_size, Z_avg,
                         above_size_ther, mod_size, temp_array, pos_size, dire):
        new_df = pd.DataFrame(
            {'Image processed':(base_arr),'Cluster' :(platename_arr),
             'Q1_size': (Q1_size), 'Q2_size': (Q2_size),'Q3_size': (Q3_size),
             'Q4_size': (Q4_size),'Avg_size': (total_size_avg_array), 'Size_stdev': (total_size_std_array),
             'Q1_Zscore': (Z1_size), 'Q2_Zscore': (Z2_size), 'Q3_Zscore': (Z3_size),
             'Q4_Zscore': (Z4_size),'Avg_Zscore': (Z_avg), '# above threshold': (above_size_ther),
             'modifier': (mod_size), 'temp': (temp_array), 'hit': (pos_size)})
        os.chdir(dire)
        Excel_name = "A_test.xlsx"
        new_df.to_excel(Excel_name)

##MAIN##
    toomanycounter = 1
    anothercounter = 1
    color_counter = 0
    plate_size = []
    plate_color = []

    initcrop(imagePath[i])
    cluster_maker()
    # 0 means size and color looking for small and red
    if int(mode) == 0:
        image_counter = image_counter + 1
        for c in range(0, 96):
            #print(c)
            base_arr.append(base)
            char = chr(toomanycounter + 64)
            plate_name = ("U%d-%c%d" % (image_counter, char, anothercounter))
            platename_arr.append(plate_name)
            returned_size = connected_comps_for_liz(c)
            #print(returned_size)
            returned_color = colorful_writer(color_counter)
            anothercounter = anothercounter + 1
            if anothercounter > 12:
                anothercounter = 1
                toomanycounter = toomanycounter + 1
            color_counter = returned_color[3]
            cc = []
            cc = returned_size[0]
            #print(cc)
            Q1_size.append(cc[0])
            Q2_size.append(cc[1])
            Q3_size.append(cc[2])
            Q4_size.append(cc[3])
            total_size_array = (Q1_size + Q2_size + Q3_size + Q4_size)
            total_size_avg_array.append(returned_size[1])
            total_size_std_array.append(returned_size[2])
            Zscore_array = (returned_size[3])
            Z1_size.append(Zscore_array[0])
            Z2_size.append(Zscore_array[1])
            Z3_size.append(Zscore_array[2])
            Z4_size.append(Zscore_array[3])
            Z_avg.append(returned_size[4])
            above_size_ther.append(returned_size[5])
            mod_size.append(returned_size[6])
            temp_array.append(returned_size[7])
            plate_size.extend(returned_size[0])
            color_array = []
            color_array = (returned_color[0])
            Q1_color.append(color_array[0])
            Q2_color.append(color_array[1])
            Q3_color.append(color_array[2])
            Q4_color.append(color_array[3])
            plate_color.extend(color_array)
            total_color_array = (Q1_color + Q2_color + Q3_color + Q4_color)
            total_color_avg_array.append(returned_color[1])
            total_color_std_array.append(returned_color[2])
            Z_color = []
            Z_color = returned_color[4]
            Z1_color.append(Z_color[0])
            Z2_color.append(Z_color[1])
            Z3_color.append(Z_color[2])
            Z4_color.append(Z_color[3])
            Z_avg_color.append(returned_color[5])
            above_size_ther_color.append(returned_color[6])
            mod_color.append(returned_color[7])
            temp_color.append(returned_color[8])
        os.chdir(folder)
        dire = folder
        path_his = dire + '/Hist'
        try:
            os.makedirs(path_his)
        except OSError:
            pass
        os.chdir(path_his)
        plt.title("Size Histogram for Image %s" %base)
        plt.ylabel("Number of cells")
        plt.xlabel("Area of cells")
        plt.hist(plate_size, bins=10)
        plt.savefig("%s_size.png" % base)
        plt.close()
        plt.title("Color Histogram for Image %s" % base)
        plt.ylabel("Number of cells")
        plt.xlabel("colorfulness of cells")
        plt.hist(plate_color, bins=10)
        plt.savefig("%s_color.png" % base)
        plt.close()

    # one is size and looking for large
    if int(mode) == 1:
        image_counter = image_counter + 1
        for c in range(0, 96):
            base_arr.append(base)
            char = chr(toomanycounter + 64)
            plate_name = ("U%d-%c%d" % (image_counter, char, anothercounter))
            platename_arr.append(plate_name)
            returned_size = connected_comps_for_Chris(c)
            # print(returned_size)
            returned_color = colorful_writer(color_counter)
            anothercounter = anothercounter + 1
            if anothercounter > 12:
                anothercounter = 1
                toomanycounter = toomanycounter + 1
            color_counter = returned_color[3]
            cc = []
            cc = returned_size[0]
            # print(cc)
            Q1_size.append(cc[0])
            Q2_size.append(cc[1])
            Q3_size.append(cc[2])
            Q4_size.append(cc[3])
            total_size_avg_array.append(returned_size[1])
            total_size_std_array.append(returned_size[2])
            Zscore_array = (returned_size[3])
            Z1_size.append(Zscore_array[0])
            Z2_size.append(Zscore_array[1])
            Z3_size.append(Zscore_array[2])
            Z4_size.append(Zscore_array[3])
            Z_avg.append(returned_size[4])
            above_size_ther.append(returned_size[5])
            mod_size.append(returned_size[6])
            temp_array.append(returned_size[7])
            plate_size.extend(returned_size[0])
            temp_color.append(returned_color[8])

        total_size_array = (Q1_size + Q2_size + Q3_size + Q4_size)
        os.chdir(folder)
        dire = folder
        path_his = dire + '/Hist'
        try:
            os.makedirs(path_his)
        except OSError:
            pass
        os.chdir(path_his)
        plt.title("Size Histogram for Image %s" % base)
        plt.ylabel("Number of cells")
        plt.xlabel("Area of cells")
        plt.hist(plate_size, bins=10)
        plt.savefig("%s_size.png" % base)
        plt.close()

if int(mode) == 0:
    size_pos = size_hit(temp_array)
    color_pos = color_hit(temp_color)
    pos_size = pos_hit(size_pos, color_pos)
    excel_writer_liz(base_arr, platename_arr, Q1_size, Q2_size, Q3_size, Q4_size,
                     total_size_avg_array, total_size_std_array, Z1_size, Z2_size, Z3_size, Z4_size, Z_avg,
                     above_size_ther, mod_size, temp_array, Q1_color, Q2_color, Q3_color, Q4_color,
                     total_color_avg_array, total_color_std_array, Z1_color, Z2_color, Z3_color,
                     Z4_color, above_size_ther_color, mod_color, temp_color, size_pos, color_pos, pos_size, folder)
if int(mode) == 1:
    pos_size = size_hit(temp_array)
    excel_writer_chris(base_arr, platename_arr, Q1_size, Q2_size, Q3_size, Q4_size,
                       total_size_avg_array, total_size_std_array, Z1_size, Z2_size, Z3_size, Z4_size, Z_avg,
                       above_size_ther, mod_size, temp_array, pos_size, folder)
beep = lambda x: os.system("echo -DONE! '\a';sleep 0.2;" * x)
beep(5)
easygui.msgbox(msg="DONE!!!", title="Yeast Classifier")
os.chdir(path_his)
plt.title("Total Size Histogram")
plt.ylabel("Number of cells")
plt.xlabel("Area of cells")
plt.hist(total_size_array)
plt.savefig("Total_size.png")
plt.close()
plt.title("Total Color Histogram")
plt.ylabel("Number of cells")
plt.xlabel("colorfulness of cells")
plt.hist(total_color_array)
plt.savefig("Total_color.png")


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
