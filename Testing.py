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
import re


excel_or_nah = easygui.indexbox(msg="select what you would like to do",
                                title="Yeast Classifier",
                                choices=('Compare Excel sheets', 'Extract data from pictures','lookup cell'))
if (excel_or_nah == 2):
    easygui.msgbox(title="Yeast Classifier",
                   msg="Next select the folder for the images and the cell location")
    dir = easygui.diropenbox(title="Yeast Classifier")
    cell_loc = easygui.enterbox(title="Yeast Classifier",
                                msg="Enter cell location"
                                    "\nExample U04-C07"
                                    "\nMAKE SURE TO USE CAPITAL LETTERS")

    choose = easygui.indexbox(msg="Which mode would you like?",
                                    title="Yeast Classifier",
                                    choices=('384 format with cluster of 4', '384 format each cell different'))
    loc = (re.findall('\d+', cell_loc))
    row = (cell_loc[4])
    col = int(loc[1])
    plate = (int(loc[0]) - 1)  # list will be zero indexed so minus 1
    if (col > 12):
        print("Wrong value entered")
        exit(-2)
    imagePath = (list(paths.list_images(dir)))
    imagePath = os_sorted(imagePath)
    if (choose == 1): # for when 1-4 are on the same plate
        if (plate == 0 or plate == 1 or plate == 2 or plate == 3):
            newplate = 0
        if (plate == 4 or plate == 5 or plate == 6 or plate == 7):
            newplate = 1
        if (plate == 8 or plate == 9 or plate == 10 or plate == 11):
            newplate = 2
        if (plate == 12 or plate == 13 or plate == 14 or plate == 15):
            newplate = 3
        if (plate == 16 or plate == 17 or plate == 18 or plate == 19):
            newplate = 4
        if (plate == 20):
            newplate = 5
    if choose == 0:
        newplate = plate

    selected_image = imagePath[newplate]
    base = os.path.basename(imagePath[newplate])
    display = Image.open(selected_image)
    display = display.resize((600,400))
    display.save('display.jpg')
    #print(selected_image)
    easygui.msgbox(msg="Press OK to look for %s in image %s" % (cell_loc, base),
                   title="Yeast Classifier",
                   image='display.jpg')
    os.remove('display.jpg')



    def initcrop(imagePath):
        dire = dir
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

    def cluster_maker():
        counter3 = 0
        counter1 = 0
        dire = dir  # used to be os.getcwd()
        path = dire + '/Classifyer_dump'
        path2 = dire + '/Yeast_cluster'
        path3 = dire + '/Cells'
        try:
            os.makedirs(path)
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
    initcrop(selected_image)
    cluster_maker()
    if (choose == 0):
        if (row == 'A'):
            image_num = (col - 1)
            path = dir + '/Yeast_cluster'
            img = (os.path.join(path, "Yeast_Cluster.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)
        if (row == 'B'):
            image_num = ((col - 1) + 12)
            path = dir + '/Yeast_cluster'
            img = (os.path.join(path, "Yeast_Cluster.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)
        if (row == 'C'):
            image_num = ((col - 1) + 24)
            path = dir + '/Yeast_cluster'
            img = (os.path.join(path, "Yeast_Cluster.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)
        if (row == 'D'):
            image_num = ((col - 1) + 36)
            path = dir + '/Yeast_cluster'
            img = (os.path.join(path, "Yeast_Cluster.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)
        if (row == 'E'):
            image_num = ((col - 1) + 48)
            path = dir + '/Yeast_cluster'
            img = (os.path.join(path, "Yeast_Cluster.%d.png" % image_num))
            easygui.msgbox(msg='Found it',
                           title="Yeast Classifier",
                           image=img)
        if (row == 'F'):
            image_num = ((col - 1) + 60)
            path = dir + '/Yeast_cluster'
            img = (os.path.join(path, "Yeast_Cluster.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)
        if (row == 'G'):
            image_num = ((col - 1) + 72)
            path = dir + '/Yeast_cluster'
            img = (os.path.join(path, "Yeast_Cluster.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)
        if (row == 'H'):
            image_num = ((col - 1) + 84)
            path = dir + '/Yeast_cluster'
            img = (os.path.join(path, "Yeast_Cluster.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)

    if (choose == 1):
        if (row == 'A'):
            if (col == 1):
                image_num = ((col-1) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
            if col != 1:
                image_num = ((col-1) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
                image_num = image_num + 1
            path = dir + '/Cells'
            img = (os.path.join(path, "SMALL_CELL.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)
        if (row == 'B'):
            if (col == 1):
                image_num = (((col-1) + 12)* 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
            if col != 1:
                image_num = (((col-1) + 12) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
                image_num = image_num + 1
            path = dir + '/Cells'
            img = (os.path.join(path, "SMALL_CELL.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)
        if (row == 'C'):
            if (col == 1):
                image_num = (((col-1) + 24) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
            if col != 1:
                image_num = (((col-1) + 24) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
                image_num = image_num + 1
            path = dir + '/Cells'
            img = (os.path.join(path, "SMALL_CELL.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)
        if (row == 'D'):
            if (col == 1):
                image_num = (((col-1) + 36) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
            if col != 1:
                image_num = (((col-1) + 36) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
                image_num = image_num + 1
            path = dir + '/Cells'
            img = (os.path.join(path, "SMALL_CELL.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)
        if (row == 'E'):
            if (col == 1):
                image_num = (((col-1) + 48) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
            if col != 1:
                image_num = (((col-1) + 48) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
                image_num = image_num + 1
            path = dir + '/Cells'
            img = (os.path.join(path, "SMALL_CELL.%d.png" % image_num))
            easygui.msgbox(msg='Found it',
                           title="Yeast Classifier",
                           image=img)
        if (row == 'F'):
            if (col == 1):
                image_num = (((col-1) + 60) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
            if col != 1:
                image_num = (((col-1) + 60) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
                image_num = image_num + 1
            path = dir + '/Cells'
            img = (os.path.join(path, "SMALL_CELL.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)
        if (row == 'G'):
            if (col == 1):
                image_num = (((col-1) + 72) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
            if col != 1:
                image_num = (((col-1) + 72) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
                image_num = image_num + 1
            path = dir + '/Cells'
            img = (os.path.join(path, "SMALL_CELL.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)
        if (row == 'H'):
            if (col == 1):
                image_num = (((col-1) + 84) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
            if col != 1:
                image_num = (((col-1) + 84) * 4)
                plate = (plate % 4)
                if plate == 1:
                    pass
                if plate == 2:
                    image_num = image_num + 1
                if plate == 3:
                    image_num = image_num + 2
                if plate == 4:
                    image_num = image_num + 3
                image_num = image_num + 1
            path = dir + '/Cells'
            img = (os.path.join(path, "SMALL_CELL.%d.png" % image_num))
            easygui.msgbox(msg='Found it'
                               '\nHere is %s' % cell_loc,
                           title="Yeast Classifier",
                           image=img)



