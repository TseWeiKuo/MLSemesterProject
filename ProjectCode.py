import numpy
import numpy as np
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from skimage.transform import resize
from skimage import img_as_ubyte
from keras.layers import Conv2D,MaxPooling2D
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy, Recall, Precision
import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dropout, Conv2D, MaxPooling2D


def LoadData(DataDirectory):
    os.chdir(DataDirectory)
    PatientID = []
    PatientLabelLeft = dict()
    PatientLabelRight = dict()
    PatientImage = dict()

    Label35_Path = Path("label/35label")
    Label100_Path = Path("label/100label")
    Image35_Path = Path("original/35")
    Image100_Path = Path("original/100")

    Label35_Path_List = list(Label35_Path.glob(r"**/*.jpg"))
    Image35_Path_List = list(Image35_Path.glob(r"**/*.jpg"))
    Label100_Path_List = list(Label100_Path.glob(r"**/*.jpg"))
    Image100_Path_List = list(Image100_Path.glob(r"**/*.jpg"))

    Label35_Path_Series = pd.Series(Label35_Path_List, dtype=str,  name="LABEL 35")
    Image35_Path_Series = pd.Series(Image35_Path_List, dtype=str, name="IMAGE 35")
    Label100_Path_Series = pd.Series(Label100_Path_List, dtype=str, name="LABEL 100")
    Image100_Path_Series = pd.Series(Image100_Path_List, dtype=str, name="IMAGE 100")

    for FilePath in Label100_Path_Series:
        LabelDirectory, TargetLabelDirectory = FilePath.split("100label\\")
        Label_Path_Before, Label_Path_Middle, Label_Path_After = TargetLabelDirectory.split("\\")
        Label_Path_Split = Label_Path_After.split("_")
        if Label_Path_Split[3] not in PatientID:
            PatientID.append(Label_Path_Split[3])
            PatientLabelLeft[Label_Path_Split[3]] = []
            PatientLabelRight[Label_Path_Split[3]] = []
            if Label_Path_Split[-2] == "R":
                PatientLabelRight[Label_Path_Split[3]].append(FilePath)
            elif Label_Path_Split[-2] == "L":
                PatientLabelLeft[Label_Path_Split[3]].append(FilePath)
        else:
            if Label_Path_Split[-2] == "R":
                PatientLabelRight[Label_Path_Split[3]].append(FilePath)
            elif Label_Path_Split[-2] == "L":
                PatientLabelLeft[Label_Path_Split[3]].append(FilePath)
    for FilePath in Image100_Path_Series:
        LabelDirectory, TargetLabelDirectory = FilePath.split("100\\")
        Label_Path_Before, Label_Path_After = TargetLabelDirectory.split("\\")
        Label_Path_Split = Label_Path_After.split("_")
        if Label_Path_Split[3] not in PatientImage.keys():
            PatientImage[Label_Path_Split[3]] = []
            PatientImage[Label_Path_Split[3]].append(FilePath)
        else:
            PatientImage[Label_Path_Split[3]].append(FilePath)

    PatientLabelRight.pop("0325")  # Outlier patient, lacking right side of fMRI image.
    PatientLabelLeft.pop("0325")
    PatientImage.pop("0325")
    PatientID.remove("0325")

    for FilePath in Label35_Path_Series:
        LabelDirectory, TargetLabelDirectory = FilePath.split("35label\\")
        Label_Path_Before, Label_Path_Middle, Label_Path_After = TargetLabelDirectory.split("\\")
        Label_Path_Split = Label_Path_After.split("_")
        if Label_Path_Split[3] not in PatientID:
            PatientID.append(Label_Path_Split[3])
            PatientLabelLeft[Label_Path_Split[3]] = []
            PatientLabelRight[Label_Path_Split[3]] = []
            if Label_Path_Split[-2] == "R":
                PatientLabelRight[Label_Path_Split[3]].append(FilePath)
            elif Label_Path_Split[-2] == "L":
                PatientLabelLeft[Label_Path_Split[3]].append(FilePath)
        else:
            if Label_Path_Split[-2] == "R":
                PatientLabelRight[Label_Path_Split[3]].append(FilePath)
            elif Label_Path_Split[-2] == "L":
                PatientLabelLeft[Label_Path_Split[3]].append(FilePath)
    for FilePath in Image35_Path_Series:
        LabelDirectory, TargetLabelDirectory = FilePath.split("35\\")
        Label_Path_Before, Label_Path_After = TargetLabelDirectory.split("\\")
        Label_Path_Split = Label_Path_After.split("_")
        if Label_Path_Split[3] not in PatientImage.keys():
            PatientImage[Label_Path_Split[3]] = []
            PatientImage[Label_Path_Split[3]].append(FilePath)
        else:
            PatientImage[Label_Path_Split[3]].append(FilePath)

    ConcatPatientLabel = dict()
    PatientOriginalImage = dict()
    PatientFileDirectory = dict()

    # Load the data into dictionary as lists of 2d image pixel values
    for ID in PatientID:
        ConcatPatientLabel[ID] = []
        PatientOriginalImage[ID] = []
        FileString = os.path.join(DataDirectory, "Patient_ID_" + str(ID))
        PatientFileDirectory[ID] = FileString
        for i in range(len(PatientLabelLeft[ID])):
            # Load label and original images
            Leftimage = cv2.imread(PatientLabelLeft[ID][i])
            Rightimage = cv2.imread(PatientLabelRight[ID][i])
            Originaliamge = cv2.imread(PatientImage[ID][i])
            # Concat left and right image
            ConcatImage = cv2.addWeighted(Leftimage, 1, Rightimage, 1, 0.2)
            # Gray scale and normalize the image pixel values
            Gray_Scale_Concat = cv2.cvtColor(ConcatImage, cv2.COLOR_BGR2GRAY)
            Gray_Scale_Original = cv2.cvtColor(Originaliamge, cv2.COLOR_BGR2GRAY)
            Gray_Scale_Concat = img_as_ubyte(resize(Gray_Scale_Concat, (Gray_Scale_Concat.shape[0]//3, Gray_Scale_Concat.shape[-1]//3), anti_aliasing=False))
            Gray_Scale_Original = resize(Gray_Scale_Original, (Gray_Scale_Original.shape[0]//3, Gray_Scale_Original.shape[-1]//3), anti_aliasing=False)
            Gray_Scale_Concat[Gray_Scale_Concat <= 150] = 0
            Gray_Scale_Concat[Gray_Scale_Concat > 150] = 1
            Gray_Scale_Original = Gray_Scale_Original / Gray_Scale_Original.max()
            # Append the image to patient dictionary
            ConcatPatientLabel[ID].append(Gray_Scale_Concat)
            PatientOriginalImage[ID].append(Gray_Scale_Original)
        print(f"Converting patient ID: {ID}'s fMRI images to normalized data set")
    return ConcatPatientLabel, PatientOriginalImage, PatientID

def EdgeFilter(shape, dtype=None):
    sobel_x = tf.constant(
        [
            [-3, -2, 0, 2, 3],
            [-8, -8, 0, 8, 6],
            [-8, -18, 0, 18, 8],
            [-6, -8, 0, 8, 6],
            [-3, -2, 0, 2, 3]
        ], dtype=dtype)
    #create the missing dims.
    sobel_x = tf.reshape(sobel_x, (5, 5, 1, 1))
    #tile the last 2 axis to get the expected dims.
    sobel_x = tf.tile(sobel_x, (1, 1, shape[-2],shape[-1]))
    return sobel_x

def Sharpen(shape, dtype=None):
    sharpen_x = tf.constant(
        [
            [-2, -2, -2],
            [-2, 17, -2],
            [-2, -2, -2]
        ], dtype=dtype)
    sharpen_x = tf.reshape(sharpen_x, (3, 3, 1, 1))
    sharpen_x = tf.tile(sharpen_x, (1, 1, shape[-2], shape[-1]))
    return sharpen_x
def RunDropOutCNNModel(Xtrain, Xtest, Ytrain, Ytest, Epochs):
    DropOutModel = models.Sequential()
    DropOutModel.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=Xtrain.shape[1:], padding='same'))
    DropOutModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    DropOutModel.add(Dropout(0.2))
    DropOutModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    DropOutModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    DropOutModel.add(Conv2D(filters=Ytrain.shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same'))
    DropOutModel.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=[BinaryAccuracy(), Recall(), Precision()])
    DropOutModel.fit(Xtrain, Ytrain, epochs=Epochs, batch_size=64, validation_split=0.15)
    DropOutModel.summary()
    decoded_imgs = DropOutModel.predict(Xtest)
    SaveFigure(Xtest, decoded_imgs, Ytest, "Drop Out CNN")
    return RecordMetrices(Ytest, decoded_imgs, "Drop Out")
def RunEdgeSharpenCNNModel(Xtrain, Xtest, Ytrain, Ytest, Epochs):
    EdgeSharpModel = models.Sequential()
    EdgeSharpModel.add(Conv2D(filters=8, kernel_size=(5, 5), kernel_initializer=EdgeFilter, activation='relu',input_shape=Xtrain.shape[1:], padding='same'))
    EdgeSharpModel.add(Conv2D(filters=8, kernel_size=(3, 3), kernel_initializer=Sharpen, activation='relu', padding='same'))
    EdgeSharpModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    EdgeSharpModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    EdgeSharpModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    EdgeSharpModel.add(Conv2D(filters=Ytrain.shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same'))
    EdgeSharpModel.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=[BinaryAccuracy(), Recall(), Precision()])
    EdgeSharpModel.fit(Xtrain, Ytrain, epochs=Epochs, batch_size=64,  validation_split=0.15)
    EdgeSharpModel.summary()
    decoded_imgs = EdgeSharpModel.predict(Xtest)
    SaveFigure(Xtest, decoded_imgs, Ytest, "Edge_Sharpen CNN")
    return RecordMetrices(Ytest, decoded_imgs, "Edge_Sharpen")
def RunBaseCNNModel(Xtrain, Xtest, Ytrain, Ytest, Epochs):
    BaseModel = models.Sequential()
    BaseModel.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=Xtrain.shape[1:], padding='same'))
    BaseModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    BaseModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    BaseModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    BaseModel.add(Conv2D(filters=Ytrain.shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same'))
    BaseModel.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=[BinaryAccuracy(), Recall(), Precision()])
    BaseModel.fit(Xtrain, Ytrain, epochs=Epochs, batch_size=64, validation_split=0.15)
    BaseModel.summary()
    decoded_imgs = BaseModel.predict(Xtest)
    SaveFigure(Xtest, decoded_imgs, Ytest, "Base CNN")
    return RecordMetrices(Ytest, decoded_imgs, "Base")
def RunEdgeFilterCNNModel(Xtrain, Xtest, Ytrain, Ytest, Epochs):
    EdgeFilterModel = models.Sequential()
    EdgeFilterModel.add(Conv2D(filters=8, kernel_size=(5, 5), kernel_initializer=EdgeFilter, activation='relu',input_shape=Xtrain.shape[1:], padding='same'))
    EdgeFilterModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    EdgeFilterModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    EdgeFilterModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    EdgeFilterModel.add(Conv2D(filters=Ytrain.shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same'))
    EdgeFilterModel.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=[BinaryAccuracy(), Recall(), Precision()])
    EdgeFilterModel.fit(Xtrain, Ytrain, epochs=Epochs, batch_size=16, validation_split=0.15)
    EdgeFilterModel.summary()
    decoded_imgs = EdgeFilterModel.predict(Xtest)
    SaveFigure(Xtest, decoded_imgs, Ytest, "Edge Filter CNN")
    return RecordMetrices(Ytest, decoded_imgs, "Edge")
def RunMaxPoolCNNModel(Xtrain, Xtest, Ytrain, Ytest, Epochs):
    EdgeFilterModel = models.Sequential()
    EdgeFilterModel.add(Conv2D(filters=8, kernel_size=(5, 5), activation='relu',input_shape=Xtrain.shape[1:], padding='same'))
    EdgeFilterModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    EdgeFilterModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    EdgeFilterModel.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    EdgeFilterModel.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same"))
    EdgeFilterModel.add(Conv2D(filters=Ytrain.shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same'))
    EdgeFilterModel.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=[BinaryAccuracy(), Recall(), Precision()])
    EdgeFilterModel.fit(Xtrain, Ytrain, epochs=Epochs, batch_size=64, validation_split=0.15)
    EdgeFilterModel.summary()
    decoded_imgs = EdgeFilterModel.predict(Xtest)
    SaveFigure(Xtest, decoded_imgs, Ytest, "Max Pool CNN")
    return RecordMetrices(Ytest, decoded_imgs, "MaxPool")
def SaveFigure(Xtest, decoded_imgs, Ytest, ImageName):
    os.chdir(r"C:\Users\wayne\OneDrive\Desktop\Cpts-437\SemesterProject\Images_Generated")
    n = 15
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display the original images
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(Xtest[i + 55])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display the decoded images
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded_imgs[i + 55])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Display the correct images.
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(Ytest[i + 55])
        ax.get_yaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(ImageName)
    plt.clf()
def RecordMetrices(Ytest, decoded_imgs, ModelName):
    os.chdir(r"C:\Users\wayne\OneDrive\Desktop\Cpts-437\SemesterProject")
    file = open(r"C:\Users\wayne\OneDrive\Desktop\Cpts-437\SemesterProject\Model metrices data record.txt", "a")
    Threshold = 0.1
    Ytest = Ytest.reshape(Ytest.shape[0], np.prod(Ytest.shape[1:]))
    decoded_imgs = decoded_imgs.reshape(decoded_imgs.shape[0], np.prod(decoded_imgs.shape[1:]))
    Ytest = Ytest.ravel().astype(int)
    decoded_imgs = decoded_imgs.ravel()

    decoded_imgs[decoded_imgs > Threshold] = 1
    decoded_imgs[decoded_imgs < Threshold] = 0

    tn, fp, fn, tp = metrics.confusion_matrix(Ytest, decoded_imgs).ravel()
    tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)
    f1_score = (2 * tp) / ((2 * tp) + fp + fn)
    mcc = ((tp * tn) - (fp * fn)) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    file.write(f"{ModelName} performance---f1: {f1_score}\tmcc: {mcc}\ttn: {tn}\tfp: {fp}\tfn: {fn}\ttp: {tp}\n")
    file.close()
    fpr, tpr, _ = roc_curve(Ytest, decoded_imgs, pos_label=1)
    with open(r"C:\Users\wayne\OneDrive\Desktop\Cpts-437\SemesterProject\tpr.txt", "a") as f:
        f.write(ModelName + "\n")
        np.savetxt(f, tpr, delimiter=',')
    with open(r"C:\Users\wayne\OneDrive\Desktop\Cpts-437\SemesterProject\fpr.txt", "a") as f:
        f.write(ModelName + "\n")
        np.savetxt(f, fpr, delimiter=',')
    return ModelName, fpr, tpr

Xtrain = []
Ytrain = []
Xtest = []
Ytest = []
Ratio = 100
Epochs = 20
ConcatPatientLabel, PatientOriginalImage, PatientID = LoadData(r'C:\Users\wayne\OneDrive\Desktop\Cpts-437\SemesterProject\Datafile')
for ID in PatientID:
    Xtrain.append(PatientOriginalImage[ID])
    Ytrain.append(ConcatPatientLabel[ID])
Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
Xtrain, Xtest = Xtrain[0:Ratio], Xtrain[Ratio:]
Ytrain, Ytest = Ytrain[0:Ratio], Ytrain[Ratio:]

Xtrain = Xtrain.reshape(np.prod(Xtrain.shape[0:2]), Xtrain.shape[2], Xtrain.shape[3], 1)
Ytrain = Ytrain.reshape(np.prod(Ytrain.shape[0:2]), Ytrain.shape[2], Ytrain.shape[3], 1)
Xtest = Xtest.reshape(np.prod(Xtest.shape[0:2]), Xtest.shape[2], Xtest.shape[3], 1)
Ytest = Ytest.reshape(np.prod(Ytest.shape[0:2]), Ytest.shape[2], Ytest.shape[3], 1)

# Run Models
#ModelName_D, fpr_D, tpr_D = RunDropOutCNNModel(Xtrain, Xtest, Ytrain, Ytest, Epochs)
#ModelName_B, fpr_B, tpr_B = RunBaseCNNModel(Xtrain, Xtest, Ytrain, Ytest, Epochs)
#ModelName_M, fpr_M, tpr_M = RunMaxPoolCNNModel(Xtrain, Xtest, Ytrain, Ytest, Epochs)
ModelName_E, fpr_E, tpr_E = RunEdgeFilterCNNModel(Xtrain, Xtest, Ytrain, Ytest, Epochs)
#ModelName_S, fpr_S, tpr_S = RunEdgeSharpenCNNModel(Xtrain, Xtest, Ytrain, Ytest, Epochs)

#plt.plot(fpr_B, tpr_B, label=ModelName_B)
#plt.plot(fpr_M, tpr_M, label=ModelName_M)
#plt.plot(fpr_D, tpr_D, label=ModelName_D)
plt.plot(fpr_E, tpr_E, label=ModelName_E)
#plt.plot(fpr_S, tpr_S, label=ModelName_S)

plt.legend(loc="lower right")
plt.savefig("ROC Curve")







