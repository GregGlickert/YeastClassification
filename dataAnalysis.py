import pandas as pd
from tkinter import *
from tkinter import messagebox as msg
from pandastable import Table
from tkinter import filedialog
from ipywidgets import interact
import ipywidgets as widgets
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider
from PyQt5.QtCore import Qt
import easygui

file_name = easygui.fileopenbox()
#https://www.geeksforgeeks.org/create-a-gui-to-convert-csv-file-into-excel-file-using-python/
try:
    df = pd.read_csv(file_name)
except:
    df = pd.read_excel(file_name)

if (len(df) == 0):
    msg.showinfo('No records', 'No records')
else:
    pass

threshold = 3500
z_score_theshold = 0.8
num_above_thres = []
num_above_z_score_theshold = []
length = df['Cluster'].to_list()
for i in range(len(length)):
    row = df.iloc[i]
    num_above_temp = 0
    z_score_temp = 0
    if (int(row[3]) >= threshold):
        num_above_temp = num_above_temp + 1
    if (int(row[4]) >= threshold):
        num_above_temp = num_above_temp + 1
    if (int(row[5]) >= threshold):
        num_above_temp = num_above_temp + 1
    if (int(row[6]) >= threshold):
        num_above_temp = num_above_temp + 1
    if (float(row[9]) >= z_score_theshold):
        z_score_temp += 1
    if (float(row[10]) >= z_score_theshold):
        z_score_temp += 1
    if (float(row[11]) >= z_score_theshold):
        z_score_temp += 1
    if (float(row[12]) >= z_score_theshold):
        z_score_temp += 1


    num_above_thres.append(num_above_temp)
    num_above_z_score_theshold.append(z_score_temp)
    # df.drop(labels="# above threshold", axis='columns',inplace=True)
temp_df = pd.DataFrame({"# above threshold": (num_above_thres)})
df["# above threshold"] = temp_df['# above threshold']
temp_df = pd.DataFrame({"Above Z-score threshold": (num_above_z_score_theshold)})
df['Above Z-score threshold'] = temp_df['Above Z-score threshold']
df.to_excel(file_name, index=False)

#https://www.python-course.eu/tkinter_sliders.php

def quit():
    exit(1)

#master = Tk()
def show_table():
    display_table = Tk()
    display_table.minsize(800, 600)
    # Now display the DF in 'Table' object
    # under'pandastable' module
    f2 = Frame(display_table,height=800, width=500)
    f2.pack(fill=BOTH, expand=1)
    table = Table(f2, dataframe=df, read_only=True)

    table.show()
    def show_values():
        # print(w2.get())
        global threshold
        threshold = w2.get()

    def set_z_score():
        global z_score_theshold
        z_score_theshold = w1.get()

    w2 = Scale(display_table, from_=0, to=10000, orient=HORIZONTAL, label='Threshold slider', length=200)
    w1 = Scale(display_table, from_=0.0, to=4.0, orient=HORIZONTAL, label='Z-score slider', length=200, resolution=0.1)
    w2.pack()
    w1.pack()
    Label(display_table, text = "Confirm threshold then exit CSV to update\n"
                                "Current Size Threshold %d\n"
                                "Current Z-score threshold %0.2f" %(threshold,z_score_theshold)).pack()
    Button(display_table, text='Confirm new Threshold', command=show_values).pack()
    Button(display_table, text="Confirm new Z-score threshold", command=set_z_score).pack()
    Button(display_table, text='Quit analysis', command=quit).pack()

# Driver Code
while(1):
    slider = show_table()
    mainloop()
    if (threshold != 3500 or z_score_theshold != 0.8):
        num_above_thres = []
        num_above_temp = 0
        length = df['Cluster'].to_list()
        for i in range(len(length)):
            row = df.iloc[i]
            num_above_temp = 0
            z_score_temp = 0
            if (int(row[3]) >= threshold):
                num_above_temp = num_above_temp + 1
            if (int(row[4]) >= threshold):
                num_above_temp = num_above_temp + 1
            if (int(row[5]) >= threshold):
                num_above_temp = num_above_temp + 1
            if (int(row[6]) >= threshold):
                num_above_temp = num_above_temp + 1
            if (float(row[9]) >= z_score_theshold):
                z_score_temp += 1
            if (float(row[10]) >= z_score_theshold):
                z_score_temp += 1
            if (float(row[11]) >= z_score_theshold):
                z_score_temp += 1
            if (float(row[12]) >= z_score_theshold):
                z_score_temp += 1

            num_above_thres.append(num_above_temp)
            num_above_z_score_theshold.append(z_score_temp)
            # df.drop(labels="# above threshold", axis='columns',inplace=True)
        temp_df = pd.DataFrame({"# above threshold": (num_above_thres)})
        df["# above threshold"] = temp_df['# above threshold']
        temp_df = pd.DataFrame({"Above Z-score threshold": (num_above_z_score_theshold)})
        df['Above Z-score threshold'] = temp_df['Above Z-score threshold']
        df.to_excel(file_name, index=False)






## Lis might have problems with tinker and therefore may need to use pandas GUI instead
