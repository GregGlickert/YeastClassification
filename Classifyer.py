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

total_size_array = []
total_size_avg_array = []
total_size_std_array = []
total_color_array = []
total_color_avg_array = []
total_color_std_array = []
image_counter = 0


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=False,
                help="path to input directory of images")
args = vars(ap.parse_args())

def initexecl(): #U platenumber - plate row A-H col number 1-12
    df = pd.DataFrame(
        {'Image processed': (), 'Cluster': (), 'Q1_size': (), 'Q2_size': (), 'Q3_size': (),
         'Q4_size': (),'Avg_size': (), 'Size_stdev': (), 'Q1_Zscore': (),'Q2_Zscore': (), 'Q3_Zscore': (),
         'Q4_Zscore': (), 'Zscore_avg': (), '# of threshold': (), 'modifier': (), 'temp': (),'Hit': (),
         'Q1_colorfullness': (), 'Q2_colorfullness': (), 'Q3_colorfullness': (), 'Q4_colorfullness': (),
         'Avg_color': (),'Color_stdev':(), 'Q1_color_Zscore' : (), 'Q2_color_Zscore': (), 'Q3_color_Zscore': (),
         'Q4_color_Zscore': (),'color_Zscore_avg': (), '# of threshold': (), 'color_modifier': (), 'color_temp': (),'colorhit': ()})
    writer = pd.ExcelWriter("A_test.xlsx", engine='openpyxl')
    df.to_excel(writer,index=False, header=True, startcol=0)
    writer.save()
    # https://medium.com/better-programming/using-python-pandas-with-excel-d5082102ca27

initexecl()

imagePath = sorted(list(paths.list_images(args["images"])))
for i in range(len(imagePath)):
    img = Image.open(imagePath[i])
    img.show()
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
        dire = os.getcwd()
        path = dire + '/Classifyer_dump'
        counter = 0
        counter1 = 0  # normally 0
        im = Image.open(os.path.join(path, "Cropped_full_yeast.png"))  # was "Cropped_full_yeast.png"
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
                Wim.save(os.path.join(path, "Yeast_Cluster." + str(counter1) + ".png"))
                counter1 += 1
                widthCounter1 = widthCounter1 + Each_Image_sizeX
                widthCounter2 = widthCounter2 + Each_Image_sizeX

        row_counter_for_save = 0
        row_counter_for_open = 0
        counter1 = 0

        for i in range(0, 96):
            print("Cropping Cluster %d" % i)
            im = Image.open(os.path.join(path, "Yeast_Cluster.%d.png" % i))
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
                    Wim.save(os.path.join(path, "SMALL_CELL." + str(counter1) + ".png"))
                    counter1 += 1
                    widthCounter1 = widthCounter1 + Each_Image_sizeX
                    widthCounter2 = widthCounter2 + Each_Image_sizeX

        counter = 0
        counter1 = 0  # normally 0
        im = Image.open(os.path.join(path, "Cropped_Threshold.png"))  # was "Cropped_full_yeast.png"
        sizeX, sizeY = im.size
        im_sizeX = round(sizeX / 12)
        im_sizeY = round(sizeY / 8)
        for h in range(0, im.height, im_sizeY):
            nim = im.crop((0, h, im.width - 1, min(im.height, h + im_sizeY) - 1))
            nim.save(os.path.join(path, "Yeast_Row_Threshold." + str(counter) + ".png"))
            counter += 1
        anotherCounter = 0
        for i in range(0, 8):
            columnImage = (os.path.join(path, "Yeast_Row_Threshold.%d.png" % anotherCounter))
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
                Wim.save(os.path.join(path, "Yeast_Cluster_Threshold." + str(counter1) + ".png"))
                counter1 += 1
                widthCounter1 = widthCounter1 + Each_Image_sizeX
                widthCounter2 = widthCounter2 + Each_Image_sizeX


    def connected_comps(counter):
        dire = os.getcwd()
        path = dire + '/Classifyer_dump'
        cropped_img = cv2.imread(os.path.join(path, 'Yeast_Cluster_Threshold.%d.png' % counter),
                                 cv2.IMREAD_UNCHANGED)  # changed from Yeast_Cluster.%d.png  %counter
        circle_me = cv2.imread(os.path.join(path, "Yeast_Cluster.%d.png" % counter))

        connected_counter = 0

        connectivity = 8

        output = cv2.connectedComponentsWithStats(cropped_img, connectivity)  # change test

        image_height = cropped_img.shape[0]  # chnage test

        num_labels = output[0]

        labels = output[1]

        stats = output[2]

        centroids = output[3]

        small_cells = 0
        med_cells = 0
        large_cells = 0
        XL_cells = 0

        small_centroids = []
        med_centroids = []

        large_centroids = []
        XL_centroids = []

        resultArrayNames = []
        area_array = []
        area_array_small = []
        area_array_med = []
        area_array_large = []
        area_array_XL = []
        left_array = []
        width_array = []
        small_cell_array_x = []
        small_cell_array_y = []
        med_cells_array_x = []
        med_cells_array_y = []
        large_cell_array_x = []
        large_cell_array_y = []
        XL_cell_array_x = []
        XL_cell_array_y = []
        centroid_array = []
        small_lower_bound = 0
        small_upper_bound = 2500
        med_lower_bound = small_upper_bound + 1
        med_upper_bound = 4000
        large_lower_bound = med_upper_bound + 1
        large_upper_bound = 6000
        XL_lower_bound = large_upper_bound + 1
        XL_upper_bound = 1000000
        cell_id_array = []
        XL_counter = 0
        large_counter = 0

        for i in range(0, len(stats)):
            if stats[i, cv2.CC_STAT_AREA] >= small_lower_bound and stats[
                i, cv2.CC_STAT_AREA] <= small_upper_bound and i >= 1:
                small_cell_array_x.append(centroids[i][0])
                small_cell_array_y.append(centroids[i][1])
                area_array_small.append(stats[i, cv2.CC_STAT_AREA])
                radius = 65
                thickness = 2
                color = (255, 0, 0)  # blue
                small_centroids = (int(small_cell_array_x[small_cells])), (int(small_cell_array_y[small_cells]))
                small_cells = small_cells + 1
                cv2.circle(circle_me, small_centroids, radius, color, thickness)
                cv2.putText(circle_me, "%d" % connected_counter, small_centroids, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=color, thickness=thickness)
                connected_counter = connected_counter + 1
                cell_id_array.append(small_centroids)

        for i in range(0, len(stats)):
            if stats[i, cv2.CC_STAT_AREA] >= med_lower_bound and stats[
                i, cv2.CC_STAT_AREA] <= med_upper_bound and i >= 1:  # 3100 good for removing small ones
                med_cells_array_x.append(centroids[i][0])
                med_cells_array_y.append(centroids[i][1])
                area_array_med.append((stats[i, cv2.CC_STAT_AREA]))
                radius = 65
                thickness = 2
                color = (255, 255, 255)  ##white
                med_centroids = (int(med_cells_array_x[med_cells])), (int(med_cells_array_y[med_cells]))
                med_cells = med_cells + 1
                cv2.circle(circle_me, med_centroids, radius, color, thickness)
                cv2.putText(circle_me, "%d" % connected_counter, med_centroids, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=color, thickness=thickness)
                connected_counter = connected_counter + 1
                cell_id_array.append(med_centroids)

        for i in range(0, len(stats)):
            if stats[i, cv2.CC_STAT_AREA] >= large_lower_bound and stats[
                i, cv2.CC_STAT_AREA] <= large_upper_bound and i >= 1:
                large_cell_array_x.append(centroids[i][0])
                large_cell_array_y.append(centroids[i][1])
                area_array_large.append(stats[i, cv2.CC_STAT_AREA])
                radius = 65
                thickness = 2
                color = (0, 255, 100)  # green
                large_centroids = (int(large_cell_array_x[large_cells])), (int(large_cell_array_y[large_cells]))
                cv2.circle(circle_me, large_centroids, radius, color, thickness)
                cv2.putText(circle_me, "%d" % connected_counter, large_centroids,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=color, thickness=thickness)
                connected_counter = connected_counter + 1
                large_cells = large_cells + 1
                cell_id_array.append(large_centroids)

        for i in range(0, len(stats)):
            if stats[i, cv2.CC_STAT_AREA] >= XL_lower_bound and stats[
                i, cv2.CC_STAT_AREA] <= XL_upper_bound and i >= 1:  # took out and i>1
                XL_cell_array_x.append(centroids[i][0])
                XL_cell_array_y.append(centroids[i][1])
                area_array_XL.append(stats[i, cv2.CC_STAT_AREA])
                radius = 65
                thickness = 2
                color = (0, 0, 0)  # black
                print("centroids")
                print(XL_cell_array_x)
                print((XL_cell_array_y))
                XL_centroids = (int(XL_cell_array_x[XL_cells])), (int(XL_cell_array_y[XL_cells]))
                XL_counter = XL_counter + 1
                cv2.circle(circle_me, XL_centroids, radius, color, thickness)
                cv2.putText(circle_me, "%d" % connected_counter, XL_centroids,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=color, thickness=thickness)
                connected_counter = connected_counter + 1
                XL_cells = XL_cells + 1
                cell_id_array.append(XL_centroids)

        cv2.imwrite(os.path.join(path, "centroid test.png"), circle_me)

        area_array = area_array_small + area_array_med + area_array_large + area_array_XL

        """
        if (len(stats) > 5):
            image = Image.open("centroid test.png")
            image.show()
            print("problem with cluster %d error check centroid test" % counter)
            removed = input("Enter value: ")
            image.close()
            del area_array[(int(removed))]
            print(area_array)
        """
        if (len(stats) < 4):
            print("too few decteted on %d" % counter)
            print((len(stats)))
            for i in range((len(stats)), 5, 1):
                area_array.append(0)
            print(area_array)

        print("Currently on cell %d" % counter)
        cc_size_array = []
        print(centroids)
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
        print("size data")
        print(cc_size_array)
        avg_size = np.average(cc_size_array)
        print(avg_size)
        std = np.std(np.array(cc_size_array))
        print(std)
        Zscore_array = abs(scipy.stats.zscore(cc_size_array))
        print(Zscore_array)
        Z_avg = np.average(Zscore_array)
        print(Z_avg)
        mod = 1.5  # if avg zscore is less than .5 or 40% that is rewarded
        if Z_avg >= .8: # completely random number lol
            mod = 1
        above_size_ther = 0
        for i in range(0, len(cc_size_array)):
            if cc_size_array[i] >= 3000:
                above_size_ther += 1

        temp = (above_size_ther-(mod * Z_avg))  # simply alg to tell is positive gets normalized later
        print(temp)


        print("end of size data")



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
        path = dire + '/Classifyer_dump'
        # https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
        color_array = []
        for i in range(0, 4):
            print("THIS IS COLOR COUNTER")
            print(color_counter)
            image = cv2.imread(os.path.join(path, "SMALL_CELL.%d.png" % color_counter))
            C = image_colorfulness(image)
            # display the colorfulness score on the image
            color_array.append(C)
            cv2.putText(image, "{:.2f}".format(C), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imwrite(os.path.join(path, "SMALL_CELL.%d.png" % color_counter), image)
            color_counter = color_counter + 1

        # total_color_array = total_color_array + color_array
        avg_color = (color_array[0] + color_array[1] + color_array[2] + color_array[3]) / 4
        std_color = np.std(np.array(color_array))

        if (len(color_array) < 4):
            print((len(color_array)))
            for i in range((len(color_array)), 5, 1):
                color_array.append(0)

        Zscore_array = abs(scipy.stats.zscore(color_array))
        print(Zscore_array)
        Z_avg = np.average(Zscore_array)
        print(Z_avg)
        mod = 1.5  # if avg zscore is less than .5 or 40% that is rewarded
        if Z_avg >= .8:  # completely random number lol
            mod = 1
        above_size_ther = 0
        for i in range(0, len(color_array)):
            if color_array[i] >= 23: #pretty red ones
                above_size_ther += 1
        temp = (above_size_ther - (mod * Z_avg))


        #print(color_array)
        return color_array, avg_color, std_color, color_counter, Zscore_array, Z_avg, above_size_ther, mod, temp


    def excel_writer(base,toomanycounter, anothercounter, image_counter, cc_size_array, avg_size, std,
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
             'color_temp': (temp_C),'colorhit': (temp_C)}, index=[0])
        writer = pd.ExcelWriter('A_test.xlsx', engine='openpyxl')
        writer.book = load_workbook('A_test.xlsx')
        writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        reader = pd.read_excel(r'A_test.xlsx')
        new_df.to_excel(writer, index=False, header=False, startcol=0, startrow=len(reader) + 1)
        writer.close()


    toomanycounter = 1
    anothercounter = 1
    color_counter = 0
    initcrop(img)
    cluster_maker()
    image_counter = image_counter + 1
    for c in range(0, 96):
        returned_size = connected_comps(c) #inputs is counter for which cluster to process and output is an array with size, avg size, and std
        #print(returned_size)
        returned_color = colorful_writer(color_counter) #input is color_counter so knows which cell to process output is an array with colorfulness, avg color, and std
        #print((returned_color))
        excel_writer(base,toomanycounter, anothercounter, image_counter, returned_size[0], returned_size[1],
                     returned_size[2], returned_size[3], returned_size[4],returned_size[5],returned_size[6],
                     returned_size[7],
                     returned_color[0], returned_color[1], returned_color[2], returned_color[4],returned_color[5],
                     returned_color[6], returned_color[7], returned_color[8])  #outputs excel sheet
        anothercounter = anothercounter + 1
        if anothercounter > 12:
            anothercounter = 1
            toomanycounter = toomanycounter + 1
        color_counter = returned_color[3]

#beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
#beep(5)

data = pd.read_excel('A_test.xlsx')  # takes in temp and normalizes and top 80% of the best values based on more normal
X = abs(data.iloc[:,[15]].values)
print(X)
X_max = max(X)
print(X_max)
normalize_array = []
for i in range(0, 96):
    x = (X[i]/X_max) * 100
    if x > 80:
        x = 1
    else:
        x = 0
    X[i] = x

print(X)


for i in range(0, (image_counter*96)):
    new_df = pd.DataFrame(
                {'Hit': (X[i])})
    writer = pd.ExcelWriter('A_test.xlsx', engine='openpyxl')
    writer.book = load_workbook('A_test.xlsx')
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    reader = pd.read_excel(r'A_test.xlsx')
    new_df.to_excel(writer, index=False, header=False, startcol=16, startrow=(i+1))
    writer.close()

data = pd.read_excel('A_test.xlsx')  # takes in temp and normalizes and top 80% of the best values based on more normal
X = abs(data.iloc[:,[29]].values)
print(X)
X_max = max(X)
print(X_max)
normalize_array = []
for i in range(0, 96):
    x = (X[i]/X_max) * 100
    if x > 80:
        x = 1
    else:
        x = 0
    X[i] = x

print(X)


for i in range(0, (image_counter*96)):
    new_df = pd.DataFrame(
                {'Hit': (X[i])})
    writer = pd.ExcelWriter('A_test.xlsx', engine='openpyxl')
    writer.book = load_workbook('A_test.xlsx')
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    reader = pd.read_excel(r'A_test.xlsx')
    new_df.to_excel(writer, index=False, header=False, startcol=30, startrow=(i+1))
    writer.close()

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
