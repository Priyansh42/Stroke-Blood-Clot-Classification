import os
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.layers import GlobalMaxPooling2D
from IPython.display import clear_output
import math
from keras.callbacks import LearningRateScheduler, EarlyStopping, Callback, ReduceLROnPlateau 
import csv

# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'C:/path/to/openslide-win64-20220811/bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from openslide import OpenSlide

import cv2
from PIL import Image
import torchvision.transforms as transforms

from tkinter import *
import tkinter.ttk as ttk
from tkinter import messagebox,ttk,filedialog
import tkinter as tk
from PIL import Image,ImageTk

class Stroke_Blood_Clot_Origin_CNN:
    def __init__(self,root):
        self.root=root
        #window size
        self.root.geometry("1006x500+0+0")
        self.root.resizable(False, False)
        self.root.title("Image Classification of Stroke Blood Clot Origin")

        img1=Image.open(r"C:/path/to/Image.jpg")
        img1=img1.resize((1006,500),Image.ANTIALIAS)
        #Antialiasing is a technique used in digital imaging to reduce the visual defects that occur when high-resolution images are presented in a lower resolution.
        self.photoimg1=ImageTk.PhotoImage(img1)

        bg_img=Label(self.root,image=self.photoimg1)
        bg_img.place(x=0,y=50,width=1006,height=500)

        # title Label
        title_lbl=Label(text="Image Classification of Stroke Blood Clot Origin",font=("Bradley Hand ITC",30,"bold"),bg="black",fg="skyblue")
        title_lbl.place(x=0,y=0,width=1006,height=50)

        #button 1
        self.b1=Button(text="IMPORT DATA",cursor="hand2",command=self.import_data,font=("Calibri",12,"bold"),bg="white",fg="black")
        self.b1.place(x=80,y=160,width=180,height=30)

        #button 2
        self.b2=Button(text="PREPROCESS DATA",cursor="hand2",command=self.preprocess_data,font=("Calibri",12,"bold"),bg="white",fg="black")
        self.b2.place(x=80,y=220,width=180,height=30)
        self.b2["state"] = "disabled"
        self.b2.config(cursor="arrow")

        #button 3
        self.b3=Button(text="TRAIN DATA",cursor="hand2",command=self.train_data,font=("Calibri",12,"bold"),bg="white",fg="black")
        self.b3.place(x=80,y=280,width=180,height=30)
        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow")

        #button 4
        self.b4=Button(text="TEST DATA",cursor="hand2",command=self.test_data,font=("Calibri",12,"bold"),bg="white",fg="black")
        self.b4.place(x=80,y=340,width=180,height=30)
        self.b4["state"] = "disabled"
        self.b4.config(cursor="arrow")

    def import_data(self):
        self.train_df = pd.read_csv(r'C:/path/to/Stroke Blood Clot Classification/train.csv')
        self.test_df  = pd.read_csv(r'C:/path/to/Stroke Blood Clot Classification/test.csv')
        
        self.train_df.head()
        
        self.train_df["file_path"] = self.train_df["image_id"].apply(lambda x: "C:/path/to/Stroke Blood Clot Classification/train/" + x + ".tif")
        self.test_df["file_path"]  = self.test_df["image_id"].apply(lambda x: "C:/path/to/Stroke Blood Clot Classification/test/" + x + ".tif")
        
        self.train_df["target"] = self.train_df["label"].apply(lambda x : 1 if x=="CE" else 0)
        
        self.train_df.head()

        messagebox.showinfo("Import Data" , "Data Imported Successfully!") 

        self.b1["state"] = "disabled"
        self.b1.config(cursor="arrow") 
        self.b2["state"] = "normal"
        self.b2.config(cursor="hand2") 

        self.traindatacsv = Toplevel(root)
        self.traindatacsv.title("Train Data CSV")
        width = 500
        height = 400
        screen_width = self.traindatacsv.winfo_screenwidth()
        screen_height = self.traindatacsv.winfo_screenheight()
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.traindatacsv.geometry("%dx%d+%d+%d" % (width, height, x, y))
        self.traindatacsv.resizable(0, 0)

        TableMargin = Frame(self.traindatacsv, width=500)
        TableMargin.pack(side=TOP)
        scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
        scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
        tree = ttk.Treeview(TableMargin, columns=("Image_ID", "Center_ID", "Patient_ID","Image_Num","Label"), height=400, selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
        scrollbary.config(command=tree.yview)
        scrollbary.pack(side=RIGHT, fill=Y)
        scrollbarx.config(command=tree.xview)
        scrollbarx.pack(side=BOTTOM, fill=X)
        tree.heading('Image_ID', text="Image_ID", anchor=W)
        tree.heading('Center_ID', text="Center_ID", anchor=W)
        tree.heading('Patient_ID', text="Patient_ID", anchor=W)
        tree.heading('Image_Num', text="Image_Num", anchor=W)
        tree.heading('Label', text="Label", anchor=W)
        tree.column('#0', stretch=NO, minwidth=0, width=0)
        tree.column('#1', stretch=NO, minwidth=0, width=200)
        tree.column('#2', stretch=NO, minwidth=0, width=200)
        tree.column('#3', stretch=NO, minwidth=0, width=200)
        tree.column('#4', stretch=NO, minwidth=0, width=200)
        tree.column('#5', stretch=NO, minwidth=0, width=200)
        tree.pack()

        with open('C:/path/to/Stroke Blood Clot Classification/train.csv') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                image_id = row['image_id']
                center_id = row['center_id']
                patient_id = row['patient_id']
                image_num = row['image_num']
                label = row['label']
                tree.insert("", 0, values=(image_id, center_id, patient_id, image_num, label))

        sample_train = self.train_df[1:2] # image at index 1
        # print(sample_train)

        # first check the path length (here path length is 74)
        # print(sample_train.loc[1,"file_path"])

        # Generating individual Deep Zoom tiles from slide objects
        # now subtract 13 from your path length for reaching to the image file inside the folder, here 74-13=61. Lets say x=61 and join path from x onwards to previous path, here OpenSlide(previous path + previous path[x:]) where x is 61
        slide = OpenSlide(sample_train.loc[1, "file_path"]+sample_train.loc[1,'file_path'][61:]) 
        region = (0, 0)
        size = (10000, 10000)
        # group of connected pixels with similar properties
        region = slide.read_region(region, 0, size)
        plt.figure(figsize=(8, 8)) # tuple of the width and height of the figure in inches
        plt.title(self.train_df["image_id"][1]) # image at index 1
        plt.imshow(region) 
        plt.show() 

        # For displaying multiple images
        # for i in range(1):
        #     # Generating individual Deep Zoom tiles from slide objects
        #     slide = OpenSlide(sample_train.loc[i, "file_path"]+sample_train.loc[i,'file_path'][61:])
        #     region = (0, 0)
        #     size = (10000, 10000)
        #     # group of connected pixels with similar properties
        #     region = slide.read_region(region, 0, size)
        #     plt.figure(figsize=(8, 8)) # tuple of the width and height of the figure in inches
        #     plt.title(self.train_df["image_id"][i])
        #     plt.imshow(region)
        #     plt.show()

    def preprocess_data(self):
        def preprocess(image_path):
            slide = OpenSlide(image_path)
            region = (1000,1000)    
            size = (5000, 5000)
            image = slide.read_region(region, 0, size)
            # transform = transforms.Compose([
            # transforms.PILToTensor()
            # ])
            # self.img_tensor = transform(image)
            image = tf.image.resize(image, (512, 512))
            image = np.array(image)
            return image 

        self.train_x=[]
        for i in tqdm(self.train_df['file_path']):
            # first check the path length (here path length is 74)
            # now subtract 13 from your path length for reaching to the image file inside the folder, here 74-13=61. Lets say x=61 and join path from x onwards to previous path, here preprocess(previous path + previous path[x:length+1]) where x is 61 and length+1=75
            x1=preprocess(i+i[61:75])
            self.train_x.append(x1)

        self.train_x = np.array(self.train_x)/255.0
        self.train_y = self.train_df["target"]

        messagebox.showinfo("Preprocess Data" , "Data Preprocessed Successfully!") 

        self.b2["state"] = "disabled"
        self.b2.config(cursor="arrow") 
        self.b3["state"] = "normal"
        self.b3.config(cursor="hand2")

    def train_data(self):
        self.model = Sequential()
        self.input_shape = (512, 512, 4)
        self.model.add(Conv2D(filters=32, kernel_size = (3,3), strides =2, padding = 'same', activation = 'relu', input_shape = self.input_shape))
        self.model.add(Conv2D(filters=64, kernel_size = (3,3), strides =2, padding = 'same', activation = 'relu'))
        self.model.add(Conv2D(filters=32, kernel_size = (3,3), strides =2, padding = 'same', activation = 'relu'))
        # no pooling layer -- features are too minute, instead we are using a convolution layers with strides of 2
        self.model.add(Flatten())
        self.model.add(Dense(128, activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(0.1)))
        self.model.add(Dropout(0.50))

        self.model.add(Dense(1))

        self.model.compile(
        loss = tf.keras.losses.MeanSquaredError(),    
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), 
                    tf.keras.metrics.BinaryAccuracy(name="accuracy")],
        optimizer = tf.keras.optimizers.Adam(1e-4))

        class PlotLearning(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.metrics = {}
                self.metrics["loss"]=[]
                self.metrics["val_loss"]=[]
                self.metrics["accuracy"]=[]
                self.metrics["val_accuracy"]=[]
                self.metrics["lr"]=[]


            def on_epoch_end(self, epoch, logs={}):
                # Storing metrics
                for metric in logs:
                    if metric in self.metrics:
                        self.metrics[metric].append(logs.get(metric))

                # Plotting
                metrics = [x for x in logs if x in self.metrics and "val" not in x]

                f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
                clear_output(wait=True)

                for i, metric in enumerate(metrics):
                    axs[i].plot(range(1, epoch + 2), 
                            self.metrics[metric], 
                            label=metric)
                    if metric != "lr" and logs['val_' + metric]:
                        axs[i].plot(range(1, epoch + 2), 
                                    self.metrics['val_' + metric], 
                                    label='val_' + metric)

                    axs[i].legend()
                    axs[i].grid()
                    if metric == "lr":
                        axs[i].set_ylim(bottom=0, top=0.0015)
                    else:
                        axs[i].set_ylim(bottom=0, top=1)

                # plt.tight_layout()
                # plt.show()

        self.train_y = self.train_df["target"]
        self.test_size = 0.2
        self.train_x,self.test_x,self.train_y,self.test_y=train_test_split(self.train_x,self.train_y,test_size=0.2) 

        #lrate = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        #                              patience=8, min_lr=0.0001) # not doing anything meaningful for us

        earstop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 16)


        history = self.model.fit(
            self.train_x,
            self.train_y,
            epochs = 100,
            batch_size=32,
            validation_data = (self.test_x,self.test_y),
            shuffle=True,
            verbose = 1,
            callbacks = [PlotLearning(), earstop] #lrate]
        )    

        print(f"Epochs: {len(history.history['accuracy'])}")
        print(f"Accuracy: {history.history['accuracy'][-1]}")
        print(f"Validation Accuracy: {history.history['val_accuracy'][-1]}")
        print(f"Loss: {history.history['loss'][-1]}")
        print(f"Validation Loss: {history.history['val_loss'][-1]}")


        print_data=Label(text="●●●\tT.R.A.I.N.I.N.G\t●●●" + "\nTotal Train Data: " + str(len(self.train_df)) + "\n-------------------------------------" + "\nTrain Validation Split Ratio: " + str(self.test_size) + "\nTrain Data: " + str(int((len(self.train_df)-((len(self.train_df))*(self.test_size))))) + " | Validation Data: " + str(int((len(self.train_df)*(self.test_size)))) +  "\n-------------------------------------" + "\nAccuracy: " + str(round(history.history['accuracy'][-1],2)),font=("Calibri",13,"bold"),bg="skyblue", fg="black")
        print_data.place(x=320,y=160,width=260,height=150) 

        messagebox.showinfo("Train Data" , "Data Trained Successfully!") 

        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow") 
        self.b4["state"] = "normal"
        self.b4.config(cursor="hand2")  

    def test_data(self):
        def preprocess(image_path):
            slide = OpenSlide(image_path)
            region = (1000,1000)    
            size = (5000, 5000)
            image = slide.read_region(region, 0, size)
            # transform = transforms.Compose([
            # transforms.PILToTensor()
            # ])
            # self.img_tensor = transform(image)
            image = tf.image.resize(image, (512, 512))
            image = np.array(image)
            return image 

        print_data_1=Label(text="●●●\tT.E.S.T.I.N.G\t●●●" + "\nTotal Test Data: " + str(len(self.test_df)),font=("Calibri",13,"bold"),bg="lightyellow", fg="black")
        print_data_1.place(x=320,y=320,width=260,height=50)    
        
        test_s=[]
        for i in self.test_df['file_path']:
            # first check the path length (here path length is 74)
            # now subtract 13 from your path length for reaching to the image file inside the folder, here 74-13=61. Lets say x=61 and join path from x onwards to previous path, here preprocess(previous path + previous path[x:length+1]) where x is 61 and length+1=75
            x1=preprocess(i+"/"+i[61:75])
            test_s.append(x1)
        test_s=np.array(test_s)/255.0

        sub_pred=self.model.predict(test_s)

        submission = pd.DataFrame(self.test_df["patient_id"].copy())
        submission["CE"] = sub_pred
        submission["CE"] = submission["CE"].apply(lambda x : 0 if x<0 else x)
        submission["CE"] = submission["CE"].apply(lambda x : 1 if x>1 else x)
        submission["LAA"] = 1- submission["CE"]

        submission = submission.groupby("patient_id").mean()
        submission = submission[["CE", "LAA"]].round(6).reset_index()
        submission

        submission.to_csv("C:/path/to/Stroke Blood Clot Classification/submission.csv", index = False)
        
        self.b5=Button(text="SHOW TESTING RESULTS",cursor="hand2",command=self.showResultsInNewWindow,font=("Calibri",12,"bold"),bg="white",fg="black")
        self.b5.place(x=80,y=400,width=500,height=30)

        messagebox.showinfo("Test Data" , "Data Tested Successfully!") 

        self.b4["state"] = "disabled"
        self.b4.config(cursor="arrow")

        self.testdatacsv = Toplevel(root)
        self.testdatacsv.title("Test Data CSV")
        width = 500
        height = 400
        screen_width = self.testdatacsv.winfo_screenwidth()
        screen_height = self.testdatacsv.winfo_screenheight()
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.testdatacsv.geometry("%dx%d+%d+%d" % (width, height, x, y))
        self.testdatacsv.resizable(0, 0)

        TableMargin = Frame(self.testdatacsv, width=500)
        TableMargin.pack(side=TOP)
        scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
        scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
        tree = ttk.Treeview(TableMargin, columns=("Image_ID", "Center_ID", "Patient_ID","Image_Num"), height=400, selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
        scrollbary.config(command=tree.yview)
        scrollbary.pack(side=RIGHT, fill=Y)
        scrollbarx.config(command=tree.xview)
        scrollbarx.pack(side=BOTTOM, fill=X)
        tree.heading('Image_ID', text="Image_ID", anchor=W)
        tree.heading('Center_ID', text="Center_ID", anchor=W)
        tree.heading('Patient_ID', text="Patient_ID", anchor=W)
        tree.heading('Image_Num', text="Image_Num", anchor=W)
        tree.column('#0', stretch=NO, minwidth=0, width=0)
        tree.column('#1', stretch=NO, minwidth=0, width=200)
        tree.column('#2', stretch=NO, minwidth=0, width=200)
        tree.column('#3', stretch=NO, minwidth=0, width=200)
        tree.column('#4', stretch=NO, minwidth=0, width=200)
        tree.pack()

        with open('C:/path/to/Stroke Blood Clot Classification/test.csv') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                image_id = row['image_id']
                center_id = row['center_id']
                patient_id = row['patient_id']
                image_num = row['image_num']
                tree.insert("", 0, values=(image_id, center_id, patient_id, image_num))


    def showResultsInNewWindow(self):
        self.newWindow = Toplevel(root)
        self.newWindow.title("Test Data Probability")
        width = 500
        height = 400
        screen_width = self.newWindow.winfo_screenwidth()
        screen_height = self.newWindow.winfo_screenheight()
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.newWindow.geometry("%dx%d+%d+%d" % (width, height, x, y))
        self.newWindow.resizable(0, 0)

        TableMargin = Frame(self.newWindow, width=500)
        TableMargin.pack(side=TOP)
        scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
        scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
        tree = ttk.Treeview(TableMargin, columns=("Patient_ID", "CE", "LAA"), height=400, selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
        scrollbary.config(command=tree.yview)
        scrollbary.pack(side=RIGHT, fill=Y)
        scrollbarx.config(command=tree.xview)
        scrollbarx.pack(side=BOTTOM, fill=X)
        tree.heading('Patient_ID', text="Patient_ID", anchor=W)
        tree.heading('CE', text="CE", anchor=W)
        tree.heading('LAA', text="LAA", anchor=W)
        tree.column('#0', stretch=NO, minwidth=0, width=0)
        tree.column('#1', stretch=NO, minwidth=0, width=200)
        tree.column('#2', stretch=NO, minwidth=0, width=200)
        tree.column('#3', stretch=NO, minwidth=0, width=300)
        tree.pack()

        with open('C:/path/to/Stroke Blood Clot Classification/submission.csv') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                patient_id = row['patient_id']
                ce_prob = row['CE']
                laa_prob = row['LAA']
                tree.insert("", 0, values=(patient_id, ce_prob, laa_prob))

# For GUI
if __name__ == "__main__":
        root=Tk()
        obj=Stroke_Blood_Clot_Origin_CNN(root)
        root.mainloop()     