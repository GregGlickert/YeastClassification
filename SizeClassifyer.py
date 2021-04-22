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
#some preset values we need
image_counter = 0
TF_was = 1

# GUI#
excel_or_nah = easygui.indexbox(msg="select what you would like to do",
                                title="Yeast Classifier",
                                choices=('Compare Excel sheets', 'Extract data from pictures'))
if excel_or_nah == 1:
    easygui.msgbox("\nCurrently the Size function is set for 384 well plates with each cluster being"
                   "\nU1-A1 U1-A1 "
                   "\nU1-A1 U1-A1")
    TF = easygui.indexbox(msg='Would you like to append TF library', title='Yeast Classifier',
                          choices=("Yes", "No"))
    if(TF == 0):
        df = pd.read_excel("TF.xlsx")
        in_order = df
        #df = df.set_index(['Clone location (plate-well)'])
        #in_order = pd.DataFrame(columns=df.columns)
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
            id_objects, obj_hierarchy = pcv.find_objects(img=image, mask=thresh_image)

            roi1, roi_hierarchy = pcv.roi.rectangle(img=image, x=(left + 380), y=700, h=100,
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


        # runs CC and returns size and some stats
        def connected_comps_for_Chris(counter):
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
            cell_counter1 = 0
            cell_counter2 = 0
            cell_counter3 = 0
            cell_counter4 = 0
            radius = 65
            thickness = 2
            color = (232, 161, 20)
            hit_counter = 1
            for i in range(0, (len(stats)), 1):
                if (centroids[i][0] >= 30 and centroids[i][0] <= 120 and centroids[i][1] >= 40 and centroids[i][
                    1] <= 100):
                    # print("%d is in 1" % i)
                    cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                    centroid = int(centroids[i][0]), int(centroids[i][1])
                    cv2.putText(circle_me, "%d" % hit_counter, centroid, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=color, thickness=thickness)
                    hit_counter = hit_counter + 1
                else:
                    cell_counter1 = cell_counter1 + 1
                if (cell_counter1 == len(stats)):
                    cc_size_array.append(0)
            for i in range(0, (len(stats)), 1):
                if (centroids[i][0] >= 200 and centroids[i][0] <= 280 and centroids[i][1] >= 40 and centroids[i][
                    1] <= 100):
                    cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                    centroid = int(centroids[i][0]), int(centroids[i][1])
                    cv2.putText(circle_me, "%d" % hit_counter, centroid, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=color, thickness=thickness)
                    hit_counter = hit_counter + 1
                    # print("%d is in 2" % i)
                else:
                    cell_counter2 = cell_counter2 + 1
                if (cell_counter2 == len(stats)):
                    cc_size_array.append(0)
            for i in range(0, (len(stats)), 1):
                if (centroids[i][0] >= 30 and centroids[i][0] <= 120 and centroids[i][1] >= 200 and centroids[i][
                    1] <= 280):
                    cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                    centroid = int(centroids[i][0]), int(centroids[i][1])
                    cv2.putText(circle_me, "%d" % hit_counter, centroid, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=color, thickness=thickness)
                    hit_counter = hit_counter + 1
                    # print("%d is in 3" % i)
                else:
                    cell_counter3 = cell_counter3 + 1
                if (cell_counter3 == len(stats)):
                    cc_size_array.append(0)
            for i in range(0, (len(stats)), 1):
                if (centroids[i][0] >= 200 and centroids[i][0] <= 280 and centroids[i][1] >= 200 and centroids[i][1]
                        <= 280):
                    cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                    centroid = int(centroids[i][0]), int(centroids[i][1])
                    cv2.putText(circle_me, "%d" % hit_counter, centroid, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=color, thickness=thickness)
                    hit_counter = hit_counter + 1
                    # print("%d is in 4" % i)
                else:
                    cell_counter4 = cell_counter4 + 1
                if (cell_counter4 == len(stats)):
                    cc_size_array.append(0)
            cv2.imwrite("centroid test.png", circle_me)

            while (len(cc_size_array) >= 5):
                print(cc_size_array)
                print("problem on cell %d" % counter)
                print("REMOVE THE LARGER NUMBER FIRST IF MULTIPLE")
                image = Image.open("centroid test.png")
                image.show()
                removed = input("Enter value: ")
                image.close()
                removed = (int(removed) - 1)
                del cc_size_array[(int(removed))]

            # total_size_array = total_size_array + cc_size_array
            # print("size data")
            # print(cc_size_array)
            avg_size = np.average(cc_size_array)
            # print(avg_size)
            std = np.std(np.array(cc_size_array))
            # print(std)
            Zscore_array = abs(scipy.stats.zscore(cc_size_array))
            # print(Zscore_array)
            Z_avg = np.average(Zscore_array)
            # print(Z_avg)
            mod = 1.5  # if avg zscore is less than .5 or 40% that is rewarded
            if Z_avg >= .8:  # completely random number lol
                mod = 1
            above_size_ther = 0
            for i in range(0, len(cc_size_array)):
                if cc_size_array[i] >= 3500:  # for Chris likes them big
                    above_size_ther += 1

            temp = ((10 * above_size_ther) - (mod * Z_avg))  # simply alg to tell is positive gets normalized later
            # print(temp)

            # print("end of size data")
            # print(cc_size_array)
            return cc_size_array, avg_size, std, Zscore_array, Z_avg, above_size_ther, mod, temp, centroids


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


        def excel_writer_chris(base_arr, platename_arr, Q1_size, Q2_size, Q3_size, Q4_size,
                               total_size_avg_array, total_size_std_array, Z1_size, Z2_size, Z3_size, Z4_size, Z_avg,
                               above_size_ther, mod_size, temp_array, pos_size, dire, tf):
            if (tf == 1):
                new_df = pd.DataFrame(
                    {'Image processed': (base_arr), 'Cluster': (platename_arr),
                     'Q1_size': (Q1_size), 'Q2_size': (Q2_size), 'Q3_size': (Q3_size),
                     'Q4_size': (Q4_size), 'Avg_size': (total_size_avg_array), 'Size_stdev': (total_size_std_array),
                     'Q1_Zscore': (Z1_size), 'Q2_Zscore': (Z2_size), 'Q3_Zscore': (Z3_size),
                     'Q4_Zscore': (Z4_size), 'Avg_Zscore': (Z_avg), '# above threshold': (above_size_ther),
                     'modifier': (mod_size), 'temp': (temp_array), 'hit': (pos_size)})
                os.chdir(dire)
                Excel_name = "A_test.xlsx"
                new_df.to_excel(Excel_name)
                return 0
            if (tf == 0):
                new_df = pd.DataFrame(
                    {'Image processed': (base_arr), 'Cluster': (platename_arr),
                     'Q1_size': (Q1_size), 'Q2_size': (Q2_size), 'Q3_size': (Q3_size),
                     'Q4_size': (Q4_size), 'Avg_size': (total_size_avg_array), 'Size_stdev': (total_size_std_array),
                     'Q1_Zscore': (Z1_size), 'Q2_Zscore': (Z2_size), 'Q3_Zscore': (Z3_size),
                     'Q4_Zscore': (Z4_size), 'Avg_Zscore': (Z_avg), '# above threshold': (above_size_ther),
                     'modifier': (mod_size), 'temp': (temp_array), 'hit': (pos_size)})
                return new_df



        ##MAIN##
        toomanycounter = 1
        anothercounter = (str(1).zfill(2))
        color_counter = 0
        plate_size = []
        plate_color = []

        initcrop(imagePath[i])
        cluster_maker(image_counter)

        image_counter = image_counter + 1
        plate_number = (str(21).zfill(2))
        for c in range(0, 96):
            if (TF == 0):
                TF_was = 0
            base_arr.append(base)
            char = chr(toomanycounter + 64)
            plate_name = ("U%s-%c%s" % (image_counter, char, anothercounter))
            platename_arr.append(plate_name)
            anothercounter = int(anothercounter) + 1
            if anothercounter > 12:
                anothercounter = 1
                toomanycounter = toomanycounter + 1
            anothercounter = (str(anothercounter).zfill(2))
            returned_size = connected_comps_for_Chris(c)
            # print(returned_size)
            cc = []
            cc = returned_size[0]
            print('Image %d' % image_counter)
            print('cluster %d' % c)
            print(cc)
            print(returned_size[8])
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
        path_dump = folder + '/Classifyer_dump'
        # shutil.rmtree(path_dump)

    # size_pos = size_hit(temp_array)
    # color_pos = color_hit(temp_color)
    # pos_size = pos_hit(size_pos, color_pos)

    if (TF_was == 1):
        pos_size = size_hit(temp_array)
        excel_writer_chris(base_arr, platename_arr, Q1_size, Q2_size, Q3_size, Q4_size,
                           total_size_avg_array, total_size_std_array, Z1_size, Z2_size, Z3_size, Z4_size, Z_avg,
                           above_size_ther, mod_size, temp_array, pos_size, folder, TF_was)
    if (TF_was == 0):
        os.chdir(folder)
        pos_size = size_hit(temp_array)
        new_df = excel_writer_chris(base_arr, platename_arr, Q1_size, Q2_size, Q3_size, Q4_size,
                           total_size_avg_array, total_size_std_array, Z1_size, Z2_size, Z3_size, Z4_size, Z_avg,
                           above_size_ther, mod_size, temp_array, pos_size, folder, TF_was)
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