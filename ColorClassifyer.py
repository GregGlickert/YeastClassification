#!/usr/bin/env python
from PIL import Image
from plantcv import plantcv as pcv
import cv2
import numpy as np
import os
import os.path
import pandas as pd
import matplotlib.pyplot as plt
from imutils import paths
import scipy.stats
import easygui
from tqdm import tqdm
import xlrd
from natsort import os_sorted

#some arrays we are going to need
base_arr = []
platename_arr = []
total_size_array = []
red_array = []
total_color_array = []
temp_array = []
#some preset values we need
image_counter = 0
TF_was = 1

# GUI#
excel_or_nah = easygui.indexbox(msg="select what you would like to do",
                                title="Yeast Classifier",
                                choices=('Compare Excel sheets', 'Extract data from pictures'))
if excel_or_nah == 1:
    easygui.msgbox("\nCurrently the Size function is set for 384 well plates with each cluster being"
                   "\nU1-A1 U1-A1"
                   "\nU1-A1 U1-A1")
    TF = easygui.indexbox(msg='Would you like to append TF library', title='Yeast Classifier',
                          choices=("Yes", "No"))
    if(TF == 0):
        df = pd.read_excel("TF.xlsx")
        df = df.set_index(['Clone location (plate-well)'])
        in_order = pd.DataFrame(columns=df.columns)
    easygui.msgbox("On the next menu select the folder that has the images you want to process"
                   "\nThe folder should only have images inside and nothing else"
                   "\nOutput will be placed in the folder you selected as well")
    folder = easygui.diropenbox()
    # Loops over every image in the selected folder
    imagePath = (list(paths.list_images(folder)))
    imagePath = os_sorted(imagePath)
    print("Order that the images will be processed is")
    imageName = []
    for i in range(len(imagePath)):
        base = os.path.basename(imagePath[i])
        imageName.append(base)
    print(imageName)
    for i in tqdm(range(len(imagePath))):
        img = Image.open(imagePath[i])
        # img.show()
        base = os.path.basename(imagePath[i])


        #print("PROCESSING IMAGE %s..." %base)

        # crops the image into just the plate
        # can improve to add edge detection
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

            id_objects, obj_hierarchy = pcv.find_objects(img=image,
                                                         mask=fill_again)  # lazy way to findContours and draw them

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

            output = cv2.connectedComponentsWithStats(kept_mask, connectivity=8)
            stats = output[2]
            left = (stats[1, cv2.CC_STAT_LEFT])
            # print(stats[1, cv2.CC_STAT_TOP])
            # print(stats[1, cv2.CC_STAT_HEIGHT])
            # exit(2)

            L, a, b = cv2.split(result)
            # cv2.imwrite("gray_scale.png", L)
            plate_threshold = cv2.adaptiveThreshold(b, 255,
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 87,
                                                    -1)  # 867 is good 241
            cv2.imwrite(os.path.join(path, "plate_threshold.png"), plate_threshold)

            fill_again2 = pcv.fill(plate_threshold, 1000)

            cv2.imwrite(os.path.join(path, "fill_test.png"), fill_again2)
            # fill = pcv.fill_holes(fill_again2)
            # cv2.imwrite(os.path.join(path, "fill_test2.png"), fill)
            blur_image = pcv.median_blur(fill_again2, 10)
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(blur_image, connectivity=8)
            sizes = stats[1:, -1]
            nb_components = nb_components - 1
            min_size = 20000
            img2 = np.zeros((output.shape))
            for i in range(0, nb_components):
                if sizes[i] <= min_size:
                    img2[output == i + 1] = 255
            cv2.imwrite(os.path.join(path, "remove_20000.png"), img2)  # this can be made better to speed it up
            thresh_image = img2.astype(np.uint8)  # maybe crop to the roi below then do it
            thresh_image = pcv.fill_holes(thresh_image)
            cv2.imwrite("NEWTEST.jpg", thresh_image)
            id_objects, obj_hierarchy = pcv.find_objects(img=image, mask=thresh_image)

            roi1, roi_hierarchy = pcv.roi.rectangle(img=image, x=(left + 380), y=750, h=175,
                                                    w=100)
            try:
                where_cell = 0
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
                stats = output[2]
                centroids = output[3]
                centroids_x = (int(centroids[1][0]))
                centroids_y = (int(centroids[1][1]))
            except:
                where_cell = 1
                print("did this work?")
                roi1, roi_hierarchy = pcv.roi.rectangle(img=image, x=(left + 380), y=3200, h=100, w=100)
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
                stats = output[2]
                centroids = output[3]
                centroids_x = (int(centroids[1][0]))
                centroids_y = (int(centroids[1][1]))
            flag = 0

            # print(stats[1, cv2.CC_STAT_AREA])
            if ((stats[1, cv2.CC_STAT_AREA]) > 4000):
                flag = 30
            # print(centroids_x)
            # print(centroids_y)

            # print(centroids)
            if (where_cell == 0):
                left = (centroids_x - 70)
                right = (centroids_x + 3695 + flag)  # was 3715
                top = (centroids_y - 80)
                bottom = (centroids_y + 2462)
            if (where_cell == 1):
                left = (centroids_x - 70)
                right = (centroids_x + 3715 + flag)
                top = (centroids_y - 2480)
                bottom = (centroids_y + 62)

            # print(top)
            # print(bottom)
            image = Image.open(imagePath)
            img_crop = image.crop((left, top, right, bottom))
            img_crop.save(os.path.join(path, "CROPPED.png"))
            img_crop = img.crop((left, top, right, bottom))
            # img_crop.show()
            img_crop.save(os.path.join(path, 'Cropped_full_yeast.png'))
            circle_me = cv2.imread(os.path.join(path, "Cropped_full_yeast.png"))
            cropped_img = cv2.imread(
                os.path.join(path, "Cropped_full_yeast.png"))  # changed from Yeast_Cluster.%d.png  %counter
            L, a, b = cv2.split(cropped_img)  # can do l a or b
            Gaussian_blue = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 241,
                                                  -1)  # For liz's pictures 241
            cv2.imwrite(os.path.join(path, "blue_test.png"), Gaussian_blue)
            blur_image = pcv.median_blur(Gaussian_blue, 10)
            heavy_fill_blue = pcv.fill(blur_image, 1000)  # value 400
            hole_fill = pcv.fill_holes(heavy_fill_blue)
            cv2.imwrite(os.path.join(path, "Cropped_Threshold.png"), hole_fill)


        # crops the plate into the clusters
        def cluster_maker(image_count):
            counter3 = 0
            counter1 = 0
            dire = folder  # used to be os.getcwd()
            path = dire + '/Classifyer_dump'
            path1 = dire + '/Yeast_cluster_inv'
            path2 = dire + '/Yeast_cluster'
            path3 = dire + '/Cells'
            path4 = dire + '/Binary_cell'
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
            try:
                os.makedirs(path4)
            except OSError:
                pass
            counter = 0
            if (image_count != 6):
                im = Image.open(os.path.join(path, "Cropped_Threshold.png"))  # was "Cropped_full_yeast.png"
                sizeX, sizeY = im.size
                im_sizeX = round(sizeX / 12)
                im_sizeY = round(sizeY / 8)
                for h in range(0, im.height, im_sizeY):
                    nim = im.crop((0, h, im.width - 1, min(im.height, h + im_sizeY) - 1))
                    nim.save(os.path.join(path, "Yeast_Row_Bin." + str(counter) + ".png"))
                    counter += 1
                anotherCounter = 0
                for i in range(0, 8):
                    columnImage = (os.path.join(path, "Yeast_Row_Bin.%d.png" % anotherCounter))
                    Each_Image = Image.open(columnImage)
                    sizeX2, sizeY2 = Each_Image.size
                    Each_Image_sizeX = round(sizeX2 / 12)
                    Each_Image_sizeY = round(sizeY2 / 8)
                    anotherCounter += 1
                    widthCounter1 = 0
                    widthCounter2 = Each_Image_sizeX
                    for w in range(0, 12):
                        Wim = Each_Image.crop(
                            (widthCounter1, w, widthCounter2, round(Each_Image.height, w + Each_Image_sizeX)))
                        Wim.save(os.path.join(path1, "Yeast_Cluster_Bin." + str(counter1) + ".png"))
                        counter1 += 1
                        widthCounter1 = widthCounter1 + Each_Image_sizeX
                        widthCounter2 = widthCounter2 + Each_Image_sizeX
                        # print(counter1)
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
                            (widthCounter1, w, widthCounter2, round(Each_Image.height, w + Each_Image_sizeX)))
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
                counter3 = 0
                for i in range(0, 96):
                    im = Image.open(os.path.join(path1, "Yeast_Cluster_Bin.%d.png" % i))
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
                            Wim.save(os.path.join(path4, "SMALL_CELL." + str(counter3) + ".png"))
                            counter3 += 1
                            widthCounter1 = widthCounter1 + Each_Image_sizeX
                            widthCounter2 = widthCounter2 + Each_Image_sizeX
            # print("end of cluster maker")
            # cheating rn to do the 96 well

            if (image_count == 6):
                im = Image.open(os.path.join(path, "Cropped_Threshold.png"))  # was "Cropped_full_yeast.png"
                sizeX, sizeY = im.size
                im_sizeX = round(sizeX / 12)
                im_sizeY = round(sizeY / 8)
                for h in range(0, im.height, im_sizeY):
                    nim = im.crop((0, h, im.width - 1, min(im.height, h + im_sizeY) - 1))
                    nim.save(os.path.join(path, "Yeast_Row_Bin." + str(counter) + ".png"))
                    counter += 1
                anotherCounter = 0
                for i in range(0, 8):
                    columnImage = (os.path.join(path, "Yeast_Row_Bin.%d.png" % anotherCounter))
                    Each_Image = Image.open(columnImage)
                    sizeX2, sizeY2 = Each_Image.size
                    Each_Image_sizeX = round(sizeX2 / 12)
                    Each_Image_sizeY = round(sizeY2 / 8)
                    anotherCounter += 1
                    widthCounter1 = 0
                    widthCounter2 = Each_Image_sizeX
                    for w in range(0, 12):
                        Wim = Each_Image.crop(
                            (widthCounter1, w, widthCounter2, round(Each_Image.height, w + Each_Image_sizeX)))
                        Wim.save(os.path.join(path1, "Yeast_Cluster_Bin." + str(counter1) + ".png"))
                        counter1 += 1
                        widthCounter1 = widthCounter1 + Each_Image_sizeX
                        widthCounter2 = widthCounter2 + Each_Image_sizeX
                        # print(counter1)
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
                            (widthCounter1, w, widthCounter2, round(Each_Image.height, w + Each_Image_sizeX)))
                        Wim.save(os.path.join(path2, "Yeast_Cluster." + str(counter1) + ".png"))
                        counter1 += 1
                        widthCounter1 = widthCounter1 + Each_Image_sizeX
                        widthCounter2 = widthCounter2 + Each_Image_sizeX
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
                            (widthCounter1, w, widthCounter2, round(Each_Image.height, w + Each_Image_sizeX)))
                        Wim.save(os.path.join(path2, "Yeast_Cluster." + str(counter1) + ".png"))
                        counter1 += 1
                        widthCounter1 = widthCounter1 + Each_Image_sizeX
                        widthCounter2 = widthCounter2 + Each_Image_sizeX


        # runs CC and is looking for small cells returns stats
        def connected_comps_for_liz(counter, flag):
            if flag == 0:
                dire = folder  # used to be os.getcwd()
                path = dire + '/Binary_cell'
                cropped_img = cv2.imread(os.path.join(path, 'SMALL_CELL.%d.png' % counter),
                                         cv2.IMREAD_UNCHANGED)  # changed from Yeast_Cluster.%d.png  %counter
                circle_me = cv2.imread(os.path.join(path, "SMALL_CELL.%d.png" % counter))

                connected_counter = 0

                connectivity = 8

                connectivity = 8  # either 4 or 8
                output = cv2.connectedComponentsWithStats(cropped_img,
                                                          connectivity)  # Will determine the size of our clusters

                num_labels = output[0]
                labels = output[1]
                stats = output[2]  # for size
                centroids = output[3]  # for location

                # print("Currently on cell %d" % counter)
                cc_size_array = []
                # print(centroids)
                """
                for i in range(0, (len(stats)), 1):
                    if (centroids[i][0] >= 40 and centroids[i][0] <= 110 and centroids[i][1] >= 40 and centroids[i][1] <= 140):
                        print("%d is in 1" % i)
                        cc_size_array.append(stats[i, cv2.CC_STAT_AREA])

                for i in range(0, (len(stats)), 1):
                    if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 40 and centroids[i][1] <= 140):
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
        """

                # print("end of size data")
                # cc_size_array.append(stats[1, cv2.CC_STAT_AREA])
                answer = 0
                if len(stats) > 1:
                    answer = (stats[1, cv2.CC_STAT_AREA])
                    if answer < 300:
                        answer = (stats[2, cv2.CC_STAT_AREA])
                        print(answer)
                """
                if answer < 300:
                    print('Issue with cell%d' %c)
                    plt.imshow(circle_me)
                """
                return answer

            if flag == 1:
                dire = folder  # used to be os.getcwd()
                path = dire + '/Yeast_cluster_inv'
                cropped_img = cv2.imread(os.path.join(path, 'Yeast_Cluster_Bin.%d.png' % counter),
                                         cv2.IMREAD_UNCHANGED)  # changed from Yeast_Cluster.%d.png  %counter
                circle_me = cv2.imread(os.path.join(path, "Yeast_Cluster_Bin.%d.png" % counter))

                connected_counter = 0

                connectivity = 8

                connectivity = 8  # either 4 or 8
                output = cv2.connectedComponentsWithStats(cropped_img,
                                                          connectivity)  # Will determine the size of our clusters

                num_labels = output[0]
                labels = output[1]
                stats = output[2]  # for size
                centroids = output[3]  # for location

                # print("Currently on cell %d" % counter)
                cc_size_array = []
                # print(centroids)
                """
                for i in range(0, (len(stats)), 1):
                    if (centroids[i][0] >= 40 and centroids[i][0] <= 110 and centroids[i][1] >= 40 and centroids[i][1] <= 140):
                        print("%d is in 1" % i)
                        cc_size_array.append(stats[i, cv2.CC_STAT_AREA])

                for i in range(0, (len(stats)), 1):
                    if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 40 and centroids[i][1] <= 140):
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
        """

                # print("end of size data")
                # cc_size_array.append(stats[1, cv2.CC_STAT_AREA])
                answer = 0
                if len(stats) > 1:
                    answer = (stats[1, cv2.CC_STAT_AREA])
                return answer


        def cellFinder(c, flag):
            dire = folder
            path = dire + '/Found_cell'
            path2 = dire + '/Binary_cell'
            path3 = dire + '/Cells'
            path4 = dire + '/Yeast_cluster'
            path5 = dire + '/Yeast_cluster_inv'
            try:
                os.makedirs(path)
            except OSError:
                pass
            if (flag == 0):
                mask = cv2.imread(os.path.join(path2, 'SMALL_CELL.%d.png' % c), flags=cv2.IMREAD_GRAYSCALE)
                img = cv2.imread(os.path.join(path3, 'SMALL_CELL.%d.png' % c))
                result = img.copy()
                result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
                result[:, :, 3] = mask

                # save resulting masked image
                cv2.imwrite(os.path.join(path, 'cell.%d.png' % c), result)
            if (flag == 1):
                img = cv2.imread(os.path.join(path4, 'Yeast_Cluster.%d.png' % c))
                mask = cv2.imread(os.path.join(path5, 'Yeast_Cluster_Bin.%d.png' % c), flags=cv2.IMREAD_GRAYSCALE)
                result = img.copy()
                result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
                result[:, :, 3] = mask

                # save resulting masked image
                cv2.imwrite(os.path.join(path, 'cell.%d.png' % c), result)


        def rednessExtractor(c, img):
            red_path = folder + '/red'
            try:
                os.makedirs(red_path)
            except OSError:
                pass

            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # lower mask (0-10)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

            # upper mask (170-180)
            lower_red = np.array([170, 50, 50])
            upper_red = np.array([180, 255, 255])
            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

            # join my masks
            mask = mask0 + mask1

            # set my output img to zero everywhere except my mask
            output_img = img.copy()
            output_img[np.where(mask == 0)] = 0

            # or HSV image
            output_hsv = img_hsv.copy()
            output_hsv[np.where(mask == 0)] = 0

            cv2.imwrite(os.path.join(red_path, 'cell.%d.png' % c), output_img)


        # determines how much color is in an image returns "colorfulness" value
        def image_colorfulness(image):
            # split the image into its respective RGB components
            (B, G, R) = cv2.split(image.astype("float"))  # CV2 works in BGR not RGB
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
            dire = folder  # used to be os.getcwd()
            path = dire + '/Cells'
            # https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
            color_array = []
            for i in range(0, 4):
                # print("THIS IS COLOR COUNTER")
                # print(color_counter)
                image = cv2.imread(os.path.join(path, "SMALL_CELL.%d.png" % color_counter))
                C = image_colorfulness(image)
                # display the colorfulness score on the image
                color_array.append(C)
                # cv2.putText(image, "{:.2f}".format(C), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                # cv2.imwrite(os.path.join(path, "SMALL_CELL.%d.png" % color_counter), image)
                color_counter = color_counter + 1

            # total_color_array = total_color_array + color_array
            avg_color = np.average(color_array)
            std_color = np.std(np.array(color_array))

            if (len(color_array) < 4):
                # print((len(color_array)))
                for i in range((len(color_array)), 5, 1):
                    color_array.append(0)

            Zscore_array = abs(scipy.stats.zscore(color_array))
            # print(Zscore_array)
            Z_avg = np.average(Zscore_array)
            # print(Z_avg)
            mod = 1.5  # if avg zscore is less than .5 or 40% that is rewarded
            if Z_avg >= .8:  # completely random number lol
                mod = 1
            above_size_ther = 0
            for i in range(0, len(color_array)):
                if color_array[i] <= 23:  # pretty white ones???
                    above_size_ther += 1
            temp = ((10 * above_size_ther) - (mod * Z_avg))

            # print(color_array)
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
        def excel_writer_liz(base_arr, platename_arr, size, color, dire, red, TF):
            if (TF == 1):
                # print(len(red))
                # print(len(color))
                new_df = pd.DataFrame(
                    {'Image processed': (base_arr), 'Cluster': (platename_arr),
                     'Size': (size), 'color': (color), 'red': (red)})
                os.chdir(dire)
                Excel_name = "results.xlsx"
                new_df.to_excel(Excel_name)
                return 0
            if (TF == 0):
                new_df = pd.DataFrame(
                    {'Image processed': (base_arr), 'Cluster': (platename_arr),
                     'Size': (size), 'color': (color), 'red': (red)})
                return new_df


        ##MAIN##
        toomanycounter = 1
        anothercounter = (str(1).zfill(2))
        color_counter = 0
        plate_size = []
        plate_color = []

        initcrop(imagePath[i])
        cluster_maker(image_counter)
        # 0 means size and color looking for small and red
        path = folder + '/Cells'
        path2 = folder + '/Found_cell'
        path3 = folder + 'Yeast_cluster'
        red_path = folder + '/red'
        plate_number = 1
        temp = 1
        cc = []
        color_array = []
        image_counter = image_counter + 1
        # will need to be changed when not testing

        if image_counter != 6:
            flag = 0
            plate_number = ((4 * image_counter) - 3)
            plate_number = (str(1).zfill(2))
            for c in range(0, 384):
                # print(c)
                base_arr.append(base)
                char = chr(toomanycounter + 64)
                plate_name = ("U%s-%c%s" % (plate_number, char, anothercounter))
                plate_number = int(plate_number) + 1
                if plate_number > (4 * image_counter):
                    plate_number = (plate_number - 4)
                plate_number = (str(plate_number).zfill(2))
                temp = temp + 1
                if temp > 4:
                    anothercounter = int(anothercounter) + 1
                    temp = 1
                if int(anothercounter) > 12:
                    anothercounter = 1
                    toomanycounter = toomanycounter + 1
                anothercounter = (str(anothercounter).zfill(2))
                platename_arr.append(plate_name)
                if (TF == 0):
                    test = df.loc[plate_name]
                    in_order = in_order.append(test)
                # print(plate_name)
                returned_size = connected_comps_for_liz(c, flag)
                cc.append(returned_size)
                total_size_array.append(returned_size)
                cellFinder(c, flag)
                img = cv2.imread(os.path.join(path2, 'cell.%d.png' % c))
                rednessExtractor(c, img)
                red_img = cv2.imread(os.path.join(red_path, 'cell.%d.png' % c))
                red = image_colorfulness(red_img)
                color = image_colorfulness(img)
                color_array.append(color)
                red_array.append(red)
                total_color_array.append(color)
            # shutil.rmtree(path)
            # shutil.rmtree(path2)
            # shutil.rmtree(path3)
            # print(color)
            # exit(-1)

        if (image_counter == 6):
            flag = 1
            for c in range(0, 96):
                # print(c)
                plate_number = (str(21).zfill(2))
                base_arr.append(base)
                char = chr(toomanycounter + 64)
                plate_name = ("U%s-%c%s" % (plate_number, char, anothercounter))
                platename_arr.append(plate_name)
                anothercounter = int(anothercounter) + 1
                if anothercounter > 12:
                    anothercounter = 1
                    toomanycounter = toomanycounter + 1
                anothercounter = (str(anothercounter).zfill(2))
                # print(plate_name)
                if (plate_name == 'U21-D02'):
                    TF = 1
                if (TF == 0):
                    test = df.loc[plate_name]
                    in_order = in_order.append(test)
                    TF_was = 0
                returned_size = connected_comps_for_liz(c, flag)
                cc.append(returned_size)
                total_size_array.append(returned_size)
                cellFinder(c, flag)
                img = cv2.imread(os.path.join(path2, 'cell.%d.png' % c))
                rednessExtractor(c, img)
                red_img = cv2.imread(os.path.join(red_path, 'cell.%d.png' % c))
                red = image_colorfulness(red_img)
                red_array.append(red)
                color = image_colorfulness(img)
                color_array.append(color)
                total_color_array.append(color)
                # shutil.rmtree(path)
                # shutil.rmtree(path2)
                # shutil.rmtree(path3)
                # print(color)
                # exit(-1)

                # print(cc)
                # print(len(cc))
                # print(color_array)
                # print(len(cc))
                # exit(1)

            os.chdir(folder)
            dire = folder
            path_his = dire + '/Hist'
            try:
                os.makedirs(path_his)
            except OSError:
                pass
            """
            for i in range(len(color_array)):
                if color_array[i] <= 8:
                    color_array.pop(i)
            """

            os.chdir(path_his)
            plt.title("Size Histogram for Image %s" % base)
            plt.ylabel("Number of cells")
            plt.xlabel("Area of cells")
            plt.hist(cc, bins=20)
            plt.savefig("%s_size.png" % base)
            plt.close()
            plt.title("Color Histogram for Image %s" % base)
            plt.ylabel("Number of cells")
            plt.xlabel("colorfulness of cells")
            plt.hist(color_array, bins=20)
            plt.savefig("%s_color.png" % base)
            plt.close()
            os.chdir(folder)

            # print(cc)
            # print(len(cc))
            # print(color_array)
            # print(len(cc))
            # exit(1)

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
            plt.hist(cc, bins=20)
            plt.savefig("%s_size.png" % base)
            plt.close()
            plt.title("Color Histogram for Image %s" % base)
            plt.ylabel("Number of cells")
            plt.xlabel("colorfulness of cells")
            plt.hist(color_array, bins=20)
            plt.savefig("%s_color.png" % base)
            plt.close()




    # size_pos = size_hit(temp_array)
    # color_pos = color_hit(temp_color)
    # pos_size = pos_hit(size_pos, color_pos)
    if (TF_was == 1):
        new_df = excel_writer_liz(base_arr, platename_arr, total_size_array,
                                  total_color_array, folder, red_array, TF_was)
    if (TF_was == 0):
        os.chdir(folder)
        new_df = excel_writer_liz(base_arr, platename_arr, total_size_array,
                                  total_color_array, folder, red_array, TF_was)

        Excel_name = "class.xlsx"
        name = "lib.xlsx"
        new_df.to_excel(Excel_name)
        in_order.to_excel(name)
        df1 = pd.read_excel("class.xlsx", index_col=0)
        df2 = pd.read_excel("lib.xlsx", index_col=0).reset_index()

        new_df = pd.concat([df1, df2], axis=1, join="inner")
        new_df.to_excel("results.xlsx")


    beep = lambda x: os.system("echo -DONE! '\a';sleep 0.2;" * x)
    beep(5)
    easygui.msgbox(msg="DONE!!!", title="Yeast Classifier")
    os.chdir(path_his)
    plt.title("Size and color Histogram")
    plt.ylabel("Number of cells")
    plt.xlabel("Area of cells")
    plt.hist(total_size_array, bins=20, color='k')
    plt.savefig("Total_size.png")
    plt.close()
    plt.title("Total Color Histogram")
    plt.ylabel("Number of cells")
    plt.xlabel("colorfulness of cells")
    plt.hist(total_color_array, bins=20, color='g')
    plt.savefig("Total_color.png")

if excel_or_nah == 0:
    easygui.msgbox(msg="After hitting ok select the two excel sheets you want to compare"
                       "\nright now it is set up to look see if anything is bigger on excel sheet #2 compared to #1"
                       "\nThis can be changed at a later time",
                   title="Yeast Classifier")
    excel_path_1 = easygui.fileopenbox(msg="Find the first Excel Sheet",
                                       title="Yeast Classifier")
    excel_path_2 = easygui.fileopenbox(msg="Find the second Excel Sheet",
                                       title="Yeast Classifier")


    def excel_Compare(path1, path2):

        path = path1
        wb = xlrd.open_workbook(path)
        sheet = wb.sheet_by_index(0)
        sheet.cell_value(0, 0)
        color_array1 = []
        for i in range(sheet.nrows - 1):
            color_array1.append(sheet.cell_value((i + 1), 4))

        path2 = path2
        wb = xlrd.open_workbook(path2)
        sheet = wb.sheet_by_index(0)
        sheet.cell_value(0, 0)
        color_array2 = []
        for i in range(sheet.nrows - 1):
            color_array2.append(sheet.cell_value((i + 1), 4))
        # print(color_array1)
        # print(color_array2)
        different_array = []
        for i in range(len(color_array1)):
            color1 = color_array1[i]
            color2 = color_array2[i]
            if ((color2 - 15 >= color1)):
                different_array.append(1)
            else:
                different_array.append(0)

        return different_array


    def new_excel(different_array, path1, path2, excel_save):
        data = pd.read_excel(path1)
        data2 = pd.read_excel(path2)
        array2 = (data2['color'].to_list())
        data.insert(5, "color 2", array2)
        data.insert(6, "changed", different_array)
        Excel_name = "Compared.xlsx"
        os.chdir(excel_save)
        data.to_excel(Excel_name)


    different_array = excel_Compare(excel_path_1, excel_path_2)
    easygui.msgbox(msg="The comparison is done please select where you would like the new excel file to be saved")
    excel_save = easygui.diropenbox()
    new_excel(different_array, excel_path_1, excel_path_2, excel_save)