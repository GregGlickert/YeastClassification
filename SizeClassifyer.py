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
        df = pd.read_excel("TF.xlsx").reset_index()
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

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(path, "gray.png"), gray)

            ret, thres = cv2.threshold(gray,137,255,cv2.THRESH_BINARY) #127 137

            cv2.imwrite(os.path.join(path, 'Wholepic.png'), thres)

            fill = pcv.fill(thres, 100000)
            cv2.imwrite(os.path.join(path, "noiseReduced.png"), fill)


            im2, contours, hierarchy = cv2.findContours(fill,
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            test = []
            for c in contours:
                peri = cv2.arcLength(c, True)
                test.append(peri)

            index_max = np.argmax(test)
            #print(index_max)
            #print(test[index_max])
            x,y,w,h = cv2.boundingRect(contours[index_max])
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)


            # draw the biggest contour (c) in green
            #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),5)

            image = image[y:y+h, x:x+w]

            cv2.imwrite(os.path.join(path, "cropped.png"), image)
            L, a, b = cv2.split(image)  # can do l a or b
            thres = cv2.adaptiveThreshold(b, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255,-1)  # For liz's pictures 241
            fill = pcv.fill(thres, 1000)
            #blur = cv2.blur(fill, (15, 15), 0)

            fill_hole = cv2.morphologyEx(fill, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)), iterations=2)

            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(fill_hole, connectivity=8)
            new_centroids = []
            sizes = stats[1:, -1]
            nb_components = nb_components - 1
            min_size = 20000
            img2 = np.zeros((output.shape))
            for i in range(0, nb_components):
                if sizes[i] <= min_size:
                    img2[output == i + 1] = 255
                    new_centroids.append(centroids[i])
            cv2.imwrite(os.path.join(path, "filter.png"), img2)

            y = []
            x = []
            for i in range(len(new_centroids)):
                y.append(new_centroids[i][1])
                x.append(new_centroids[i][0])

            XARRAY = sorted(x)
            YARRAY = sorted(y)
            smallX = int(XARRAY[6]) - 60
            bigX = int(XARRAY[len(XARRAY) - 6]) + 60
            smallY = int(YARRAY[6]) - 60
            bigY = int(YARRAY[len(YARRAY) - 6]) + 60

            #print(smallX, smallY, bigX, bigY)

            image = cv2.imread(os.path.join(path,"cropped.png"))
            image = image[smallY:bigY, smallX:bigX]
            cv2.imwrite(os.path.join(path, "Cropped_full_yeast.png"), image)
            circle_me = cv2.imread(os.path.join(path, "Cropped_full_yeast.png"))
            cropped_img = cv2.imread(
                os.path.join(path, "Cropped_full_yeast.png"))
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

            im = cv2.imread(os.path.join(path, "Cropped_full_yeast.png"))
            M = int(im.shape[0] / 8)
            N = int(im.shape[1] / 12)
            # print(M, N)

            tiles = [im[x:(x + M), y:(y + N)] for x in range(0, im.shape[0], M) for y in range(0, im.shape[1], N)]
            indexes = []
            for i in range(len(tiles)):
                if (tiles[i].shape[1] < 10 or tiles[i].shape[0] < 10):
                    indexes.append(i)

            print(indexes)
            for index in sorted(indexes, reverse=True):
                del tiles[index]

            # new_tiles = np.delete(tiles, index, axis=0)

            for i in range(len(tiles)):
                cv2.imwrite(os.path.join(path2, "Yeast_Cluster.%d.png") % i, tiles[i])

            im = cv2.imread(os.path.join(path, "Cropped_Threshold.png"))
            M = int(im.shape[0] / 8)
            N = int(im.shape[1] / 12)
            # print(M, N)

            tiles = [im[x:(x + M), y:(y + N)] for x in range(0, im.shape[0], M) for y in range(0, im.shape[1], N)]
            indexes = []
            for i in range(len(tiles)):
                if (tiles[i].shape[1] < 10 or tiles[i].shape[0] < 10):
                    indexes.append(i)

            print(indexes)
            for index in sorted(indexes, reverse=True):
                del tiles[index]

            # new_tiles = np.delete(tiles, index, axis=0)

            for i in range(len(tiles)):
                cv2.imwrite(os.path.join(path1, "Yeast_Cluster_Bin.%d.png") % i, tiles[i])


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
                     'Q4_Zscore': (Z4_size), 'Avg_Zscore': (Z_avg), '# above size threshold': (above_size_ther),
                     'Above Z-score threshold': (mod_size), 'temp': (temp_array), 'hit': (pos_size)})
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
        df2 = pd.read_excel("lib.xlsx", index_col=0)

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