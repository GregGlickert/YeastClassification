from PIL import Image, ImageDraw
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
from sklearn.cluster import KMeans
import statistics

big_ass_size = []
big_ass_color = []
"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")
args = vars(ap.parse_args())
"""

image_counter = 1
df = pd.DataFrame({'Yeast row':() ,'Yeast col':() , 'Q1_size':() ,'Q2_size':() ,'Q3_size':() ,'Q4_size':(),'Avg_size':(),'Size_stdev':(),
                   'Q1_colorfullness':(),'Q2_colorfullness':(),'Q3_colorfullness':(),'Q4_colorfullness':(), 'Avg_Color':(),'Color_stdev':()})
writer = pd.ExcelWriter("A_test.xlsx",engine='openpyxl')
df.to_excel(writer,startcol=0)
writer.save()

new_df = pd.DataFrame({'Plate Number %d' % image_counter})
writer = pd.ExcelWriter('A_test.xlsx', engine='openpyxl')
writer.book = load_workbook('A_test.xlsx')
writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
reader = pd.read_excel(r'A_test.xlsx')
new_df.to_excel(writer, index=False, header=False, startcol=0, startrow=len(reader) + 1)
writer.close()
image_counter = image_counter+1

            #https://medium.com/better-programming/using-python-pandas-with-excel-d5082102ca27

#writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter')
#writer.save()

#path = '/Users/gregglickert/PycharmProjects/cc_test/A_folder/*.JPG'
#for filename in glob.glob(path):
"""
for imagePath in paths.list_images(args["images"]):
    img = Image.open((imagePath))
    img.show()
    color_counter = 0
"""
for i in range(0,1):
    img = Image.open("/Users/gregglickert/Documents/GitHub/YeastClassification/Test_images/IMG_0221.JPG")
    color_counter = 0

    left = 1875  # was 2050
    top = 730  # was 870
    right = 5680
    bottom = 3260  # was 3280

    path = '/Users/gregglickert/Documents/GitHub/YeastClassification/Classifyer_dump'
    img_crop = img.crop((left, top, right, bottom))
    # img_crop.show()
    img_crop.save(os.path.join(path,'Cropped_full_yeast.png'))
    circle_me = cv2.imread(os.path.join(path,"Cropped_full_yeast.png"))
    cropped_img = cv2.imread(os.path.join(path,"Cropped_full_yeast.png"))  # changed from Yeast_Cluster.%d.png  %counter
    blue_image = pcv.rgb2gray_lab(cropped_img, 'b')  # can do l a or b
    Gaussian_blue = cv2.adaptiveThreshold(blue_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 241,
                                          -1)  # set to 499 111 241
    cv2.imwrite(os.path.join(path,"blue_test.png"), Gaussian_blue)
    blur_image = pcv.median_blur(Gaussian_blue, 10)
    heavy_fill_blue = pcv.fill(blur_image, 400)  # value 400
    cv2.imwrite(os.path.join(path, "Cropped_Threshold.png"), heavy_fill_blue)

    # image cropping crap
    counter = 0
    counter1 = 0  # normally 0
    im = Image.open(os.path.join(path,"Cropped_full_yeast.png"))  # was "Cropped_full_yeast.png"
    sizeX, sizeY = im.size
    im_sizeX = round(sizeX / 12)
    im_sizeY = round(sizeY / 8)
    for h in range(0, im.height, im_sizeY):
        nim = im.crop((0, h, im.width - 1, min(im.height, h + im_sizeY) - 1))
        nim.save(os.path.join(path,"Yeast_Row." + str(counter) + ".png"))
        counter += 1
    anotherCounter = 0
    for i in range(0, 8):
        columnImage = (os.path.join(path,"Yeast_Row.%d.png" % anotherCounter))
        Each_Image = Image.open(columnImage)
        sizeX2, sizeY2 = Each_Image.size
        Each_Image_sizeX = round(sizeX2 / 12)
        Each_Image_sizeY = round(sizeY2 / 8)
        anotherCounter += 1
        widthCounter1 = 0
        widthCounter2 = Each_Image_sizeX
        for w in range(0, 12):
            Wim = Each_Image.crop((widthCounter1, w, widthCounter2, min(Each_Image.height, w + Each_Image_sizeX) - 1))
            Wim.save(os.path.join(path,"Yeast_Cluster." + str(counter1) + ".png"))
            counter1 += 1
            widthCounter1 = widthCounter1 + Each_Image_sizeX
            widthCounter2 = widthCounter2 + Each_Image_sizeX

    row_counter_for_save = 0
    row_counter_for_open = 0
    counter1 = 0

    for i in range(0, 96):
        print("TIMES THRU big loop %d" % i)
        im = Image.open(os.path.join(path,"Yeast_Cluster.%d.png" % i))
        sizeX, sizeY = im.size
        im_sizeX = round(sizeX / 2)
        im_sizeY = round(sizeY / 2)
        for h in range(0, im.height, im_sizeY):
            nim = im.crop((0, h, im.width - 1, min(im.height, h + im_sizeY) - 1))
            nim.save(os.path.join(path,"ROW_SMALL." + str(row_counter_for_save) + ".png"))
            row_counter_for_save += 1
            if (h >= im_sizeY):
                break
        for i in range(0, 2):
            rowImage = (os.path.join(path,"ROW_SMALL.%d.png" % row_counter_for_open))
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
                Wim.save(os.path.join(path,"SMALL_CELL." + str(counter1) + ".png"))
                counter1 += 1
                widthCounter1 = widthCounter1 + Each_Image_sizeX
                widthCounter2 = widthCounter2 + Each_Image_sizeX

    counter = 0
    counter1 = 0  # normally 0
    im = Image.open(os.path.join(path,"Cropped_Threshold.png"))  # was "Cropped_full_yeast.png"
    sizeX, sizeY = im.size
    im_sizeX = round(sizeX / 12)
    im_sizeY = round(sizeY / 8)
    for h in range(0, im.height, im_sizeY):
        nim = im.crop((0, h, im.width - 1, min(im.height, h + im_sizeY) - 1))
        nim.save(os.path.join(path,"Yeast_Row_Threshold." + str(counter) + ".png"))
        counter += 1
    anotherCounter = 0
    for i in range(0, 8):
        columnImage = (os.path.join(path,"Yeast_Row_Threshold.%d.png" % anotherCounter))
        Each_Image = Image.open(columnImage)
        sizeX2, sizeY2 = Each_Image.size
        Each_Image_sizeX = round(sizeX2 / 12)
        Each_Image_sizeY = round(sizeY2 / 8)
        anotherCounter += 1
        widthCounter1 = 0
        widthCounter2 = Each_Image_sizeX
        for w in range(0, 12):
            Wim = Each_Image.crop((widthCounter1, w, widthCounter2, min(Each_Image.height, w + Each_Image_sizeX) - 1))
            Wim.save(os.path.join(path,"Yeast_Cluster_Threshold." + str(counter1) + ".png"))
            counter1 += 1
            widthCounter1 = widthCounter1 + Each_Image_sizeX
            widthCounter2 = widthCounter2 + Each_Image_sizeX

    #pcv.params.debug ="print"
    counter = 0
    toomanycounter = 1
    anothercounter = 1
    for i in range(0,96): # was 96
        cropped_img = cv2.imread(os.path.join(path,'Yeast_Cluster_Threshold.%d.png' %counter), cv2.IMREAD_UNCHANGED) #changed from Yeast_Cluster.%d.png  %counter
        #blue_image = pcv.rgb2gray_lab(cropped_img, 'b') # can do l a or b
        #Gaussian_blue = cv2.adaptiveThreshold(blue_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,241,-1)#set to 499 111 241
        #cv2.imwrite("blue_test.png", Gaussian_blue)
        #blur_image = pcv.median_blur(Gaussian_blue,10)
        #heavy_fill_blue = pcv.fill(blur_image, 400)#value 400
        #cv2.imwrite("Threshold_img_crop.png",heavy_fill_blue)
        circle_me = cv2.imread(os.path.join(path,"Yeast_Cluster.%d.png" %counter))
        """
        id_objects, obj_hierarchy = pcv.find_objects(img=cropped_img, mask=heavy_fill_blue)
        roi_contour, roi_hierarchy = pcv.roi.rectangle(cropped_img,x=0, y=0,h=2530,w=3820)
        roi_objects, roi_obj_hierarchy, kept_mask, obj_area = pcv.roi_objects(img=cropped_img, roi_contour=roi_contour,
                                                                                  roi_hierarchy=roi_hierarchy,
                                                                                  object_contour=id_objects,
                                                                                  obj_hierarchy=obj_hierarchy,
                                                                                  roi_type='partial')
        clusters_i, contours, hierarchies = pcv.cluster_contours(img=cropped_img, roi_objects=roi_objects,
                                                                 roi_obj_hierarchy=roi_obj_hierarchy,
                                                                 nrow=8, ncol=12)
        out = '/Users/gregglickert/PycharmProjects/cc_test'
        output_path, imgs, masks = pcv.cluster_contour_splitimg(cropped_img, clusters_i, contours,
                                                                hierarchies, out, file="test",
                                                                filenames=None)
        """



        #result = circle_me.copy()
        #result = cv2.bitwise_and(result, result, mask=heavy_fill_blue)
        #cv2.imwrite("A_new_Theshold.png",result)

        #test = cv2.imread("Yeast_Cluster.400.png",cv2.IMREAD_UNCHANGED)
        #test = pcv.fill(test,1)


        #num_labels, labels_matrix = cv2.connectedComponents(heavy_fill_blue) #change test

        connected_counter = 0

        connectivity = 8

        output = cv2.connectedComponentsWithStats(cropped_img, connectivity) #change test

        image_height = cropped_img.shape[0] #chnage test

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


        for i in range(0,len(stats)):
            if stats[i,cv2.CC_STAT_AREA] >= small_lower_bound and stats[i,cv2.CC_STAT_AREA] <= small_upper_bound and i >= 1:
                small_cell_array_x.append(centroids[i][0])
                small_cell_array_y.append(centroids[i][1])
                area_array_small.append(stats[i,cv2.CC_STAT_AREA])
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


        for i in range(0,len(stats)):
            if stats[i,cv2.CC_STAT_AREA] >= med_lower_bound and stats[i,cv2.CC_STAT_AREA] <= med_upper_bound and i >= 1: #3100 good for removing small ones
                med_cells_array_x.append(centroids[i][0])
                med_cells_array_y.append(centroids[i][1])
                area_array_med.append((stats[i,cv2.CC_STAT_AREA]))
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


        for i in range(0,len(stats)):
            if stats[i,cv2.CC_STAT_AREA] >= large_lower_bound and stats[i,cv2.CC_STAT_AREA] <= large_upper_bound and i >= 1:
                large_cells = large_cells + 1
                large_cell_array_x.append(centroids[i][0])
                large_cell_array_y.append(centroids[i][1])
                area_array_large.append(stats[i,cv2.CC_STAT_AREA])
                radius = 65
                thickness = 2
                color = (0, 255, 100)  # green
                large_centroids = (int(large_cell_array_x[large_counter])), (int(large_cell_array_y[large_counter]))
                large_counter =+ 1
                cv2.circle(circle_me, large_centroids, radius, color, thickness)
                cv2.putText(circle_me, "%d" % connected_counter, large_centroids,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=color, thickness=thickness)
                connected_counter = connected_counter + 1
                cell_id_array.append(large_centroids)


        for i in range(0,len(stats)):
            if stats[i,cv2.CC_STAT_AREA] >= XL_lower_bound and stats[i,cv2.CC_STAT_AREA] <= XL_upper_bound and i >=1: #took out and i>1
                XL_cells = XL_cells + 1
                XL_cell_array_x.append(centroids[i][0])
                XL_cell_array_y.append(centroids[i][1])
                area_array_XL.append(stats[i,cv2.CC_STAT_AREA])
                radius = 65
                thickness = 2
                color = (0, 0, 0)  # black
                print("centroids")
                print(XL_cell_array_x)
                print((XL_cell_array_y))
                XL_centroids = (int(XL_cell_array_x[XL_counter])), (int(XL_cell_array_y[XL_counter]))
                XL_counter = XL_counter +1
                cv2.circle(circle_me, XL_centroids, radius, color, thickness)
                cv2.putText(circle_me, "%d" % connected_counter, XL_centroids,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=color, thickness=thickness)
                connected_counter = connected_counter + 1
                cell_id_array.append(XL_centroids)


        cv2.imwrite(os.path.join(path,"centroid test.png"), circle_me)

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
        Fuck_arrays = []
        print(centroids)
        for i in range(0,(len(stats)),1):
            if (centroids[i][0] >= 50 and centroids[i][0] <= 100 and centroids[i][1] >= 50 and centroids[i][1] <= 90):
                print("%d is in 1" %i)
                Fuck_arrays.append(stats[i,cv2.CC_STAT_AREA])
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 50 and centroids[i][1] <= 90):
                Fuck_arrays.append(stats[i, cv2.CC_STAT_AREA])
                print("%d is in 2" %i)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 50 and centroids[i][0] <= 100 and centroids[i][1] >= 200 and centroids[i][1] <= 270):
                Fuck_arrays.append(stats[i, cv2.CC_STAT_AREA])
                print("%d is in 3" %i)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 200 and centroids[i][1] <= 270):
                Fuck_arrays.append(stats[i, cv2.CC_STAT_AREA])
                print("%d is in 4" %i)

        if (len(stats) < 4):
            print("too few decteted on %d" % counter)
            print((len(stats)))
            for i in range((len(stats)), 5, 1):
                Fuck_arrays.append(0)
        print(Fuck_arrays)

        big_ass_size = big_ass_size + Fuck_arrays

        avg_size = (Fuck_arrays[0] +Fuck_arrays[1] +Fuck_arrays[2] + Fuck_arrays[3])/4
        print(avg_size)
        std =np.std(np.array(Fuck_arrays))
        print(std)

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


        # https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
        color_array = []
        for i in range(0, 4):
            image = cv2.imread(os.path.join(path,"SMALL_CELL.%d.png" % color_counter))
            C = image_colorfulness(image)
            # display the colorfulness score on the image
            color_array.append(C)
            cv2.putText(image, "{:.2f}".format(C), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imwrite(os.path.join(path,"SMALL_CELL.%d.png" % color_counter), image)
            color_counter = (color_counter + 1)

        big_ass_color = big_ass_color + color_array
        avg_color = (color_array[0] + color_array[1] + color_array[2] + color_array[3])/4
        std_color = np.std(np.array(color_array))

        if (len(color_array) < 4):
            print("too few decteted on %d" % counter)
            print((len(color_array)))
            for i in range((len(color_array)), 5, 1):
                color_array.append(0)

        print(color_array)

        new_df = pd.DataFrame(
            {'Yeast row': (toomanycounter), 'Yeast col': (anothercounter), 'Q1_size': (Fuck_arrays[0]), 'Q2_size': (Fuck_arrays[1]),
             'Q3_size': (Fuck_arrays[2]), 'Q4_size': (Fuck_arrays[3]),'Avg_size':(avg_size),'Size_stdev':(std), 'Q1_color':(color_array[0]), 'Q2_color':(color_array[1])
            ,'Q3_color':(color_array[2]), 'Q4_color':(color_array[3]),'Avg_color':(avg_color),'Color_stdev':(std_color)},index=[0])
        writer = pd.ExcelWriter('A_test.xlsx', engine='openpyxl')
        writer.book = load_workbook('A_test.xlsx')
        writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        reader = pd.read_excel(r'A_test.xlsx')
        new_df.to_excel(writer, index=False, header=False, startcol=1, startrow=len(reader) + 1)
        writer.close()
        anothercounter = anothercounter + 1
        if anothercounter > 12:
            anothercounter = 1
            toomanycounter = toomanycounter + 1

        counter = counter + 1
    new_df = pd.DataFrame({'Plate Number %d' % image_counter})
    writer = pd.ExcelWriter('A_test.xlsx', engine='openpyxl')
    writer.book = load_workbook('A_test.xlsx')
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    reader = pd.read_excel(r'A_test.xlsx')
    new_df.to_excel(writer, index=False, header=False, startcol=0, startrow=len(reader) + 1)
    writer.close()
    image_counter = image_counter+1

"""
imdir = '/Users/gregglickert/PycharmProjects/cc_test/CLUSTERD'
targetdir = "/Users/gregglickert/PycharmProjects/cc_test/testing folder"
number_clusters = 5

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.png'))
filelist.sort()

kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(big_ass_size))

try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_): #changed from kmeans to dbscan
    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r") #same here
    shutil.move(filelist[i], '/Users/gregglickert/PycharmProjects/cc_test/testing folder2')
"""
print(len(big_ass_size))
print(big_ass_size)
print(len(big_ass_color))
print(big_ass_color)

plt.hist(big_ass_size)
plt.show()
