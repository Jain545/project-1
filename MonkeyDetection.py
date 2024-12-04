from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
import cv2
import os
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import webbrowser
from sklearn.metrics import roc_curve
from sklearn import metrics
import pandas as pd

main = tkinter.Tk()
main.title("Monkeypox Detection using Modified VGG16 & Custom CNN Model") 
main.geometry("1300x1200")

global filename, cnn_model, X, Y
global X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore
labels = ['Normal', 'Monkeypox']

#function to return integer label for given plat disease name
def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name: #return integer ID as label for given plant disease name
            index = i
            break
    return index


def uploadDataset(): 
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");


def preprocess():
    global filename, cnn, X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy') #load X and Y data
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j]) #read image
                    img = cv2.resize(img, (32,32)) #resize image
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(32,32,3) #resize as colur image
                    label = getID(name) #get id or label of plant disease
                    X.append(im2arr) #add all image pixel to X array
                    Y.append(label) #add label to Y array
                    print(name+" "+root+"/"+directory[j]+" "+str(label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X) #save X and Y data for future user
        np.save('model/Y.txt',Y)
    X = X.astype('float32') #normalize image pixel with float values
    X = X/255
    indices = np.arange(X.shape[0]) #shuffling the images
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset Preprocessing & Image Normalization Process Completed\n\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% images used to train VGG & CNN algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% images used to test VGG & CNN algorithms : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    test = X[3]
    cv2.imshow("Processed Image",cv2.resize(test,(200,200)))
    cv2.waitKey(0)
       
def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    fpr, tpr, _ = roc_curve(y_test,predict,pos_label=1)
    auc = metrics.auc(fpr, tpr) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" AUC  :  "+str(auc)+"\n")
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def runVGG():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    if os.path.exists('model/vgg_model.json'):
        with open('model/vgg_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            vgg = model_from_json(loaded_model_json)
        json_file.close()    
        vgg.load_weights("model/vgg_model_weights.h5")
        vgg._make_predict_function()       
    else:
        #defining VGG16 object with our own training and test data
        vgg16 = VGG16(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights="imagenet")
        vgg16.trainable = False
        #defining transfer learning model object
        vgg = Sequential()
        #adding VGG16 to tranfer learning model
        vgg.add(vgg16)
        #defining new layers for transfer learning model with 32 layers to filter images
        vgg.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
        #max pooling collect all filtered data from VGG
        vgg.add(MaxPooling2D(pool_size = (1,1)))
        #another layer to refilter image data
        vgg.add(Convolution2D(32, 1, 1, activation = 'relu'))
        vgg.add(MaxPooling2D(pool_size = (1, 1)))
        vgg.add(Flatten())
        vgg.add(Dense(output_dim = 256, activation = 'relu')) #defining output layer
        vgg.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
        vgg.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) #compiling thee model
        hist = vgg.fit(X_train, y_train, batch_size=16, epochs=10, shuffle=True, verbose=2, validation_data=(X_test, y_test)) #start training model
        vgg.save_weights('model/vgg_model_weights.h5')            
        model_json = vgg.to_json()
        with open("model/vgg_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/vgg_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(vgg.summary())
    predict = vgg.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("Modified VGG16", predict, testY)

def runCNN():
    global X_train, X_test, y_train, y_test, cnn_model
    global accuracy, precision, recall, fscore
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn_model = model_from_json(loaded_model_json)
        json_file.close()    
        cnn_model.load_weights("model/model_weights.h5")
        cnn_model._make_predict_function()       
    else:
        cnn_model = Sequential() #define CNN custom model with multiple layers and in this custom CNN we are not using VGG16
        #defining cnn layer with 32 filters and kernel matrix size as 3 X 3
        cnn_model.add(Conv2D(32, (3, 3), padding="same",input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
        cnn_model.add(Activation("relu"))
        cnn_model.add(BatchNormalization(axis=3))
        cnn_model.add(MaxPooling2D(pool_size=(3, 3))) #max pooling collected filtered image pixels from previous CNN layer
        cnn_model.add(Dropout(0.25))
        cnn_model.add(Conv2D(64, (3, 3), padding="same")) #another CNN layer to refilter images pixles so de define multiple layers to filter image data
        cnn_model.add(Activation("relu"))
        cnn_model.add(BatchNormalization(axis=3))
        cnn_model.add(Conv2D(64, (3, 3), padding="same"))
        cnn_model.add(Activation("relu"))
        cnn_model.add(BatchNormalization(axis=3))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
        cnn_model.add(Dropout(0.25))
        cnn_model.add(Conv2D(128, (3, 3), padding="same"))
        cnn_model.add(Activation("relu"))
        cnn_model.add(BatchNormalization(axis=3))
        cnn_model.add(Conv2D(128, (3, 3), padding="same"))
        cnn_model.add(Activation("relu"))
        cnn_model.add(BatchNormalization(axis=3))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
        cnn_model.add(Dropout(0.25))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(1024))
        cnn_model.add(Activation("relu"))
        cnn_model.add(BatchNormalization())
        cnn_model.add(Dropout(0.5))
        cnn_model.add(Dense(y_train.shape[1]))
        cnn_model.add(Activation("softmax"))    
        cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) #compile the model
        hist = cnn_model.fit(X_train, y_train, batch_size=16, epochs=10, shuffle=True, verbose=2, validation_data=(X_test, y_test)) #start traing model
        cnn_model.save_weights('model/model_weights.h5')            
        model_json = cnn_model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(cnn_model.summary())
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("Custom CNN", predict, testY)    


def graph():
    output = "<html><body><table align=center border=1><tr><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th>"
    output+="<th>FSCORE</th></tr>"
    output+="<tr><td>Modified VGG16</td><td>"+str(accuracy[0])+"</td><td>"+str(precision[0])+"</td><td>"+str(recall[0])+"</td><td>"+str(fscore[0])+"</td></tr>"
    output+="<tr><td>Custom CNN</td><td>"+str(accuracy[1])+"</td><td>"+str(precision[1])+"</td><td>"+str(recall[1])+"</td><td>"+str(fscore[1])+"</td></tr>"
    output+="</table></body></html>"
    f = open("table.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("table.html",new=2)
    
    df = pd.DataFrame([['Modified VGG16','Precision',precision[0]],['Modified VGG16','Recall',recall[0]],['Modified VGG16','F1 Score',fscore[0]],['Modified VGG16','Accuracy',accuracy[0]],
                       ['Custom CNN','Precision',precision[1]],['Custom CNN','Recall',recall[1]],['Custom CNN','F1 Score',fscore[1]],['Custom CNN','Accuracy',accuracy[1]],
                      ],columns=['Algorithms','Performance Output','Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.show()


def predict():
    global cnn_model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32,32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = cnn_model.predict(img)
    predict = np.argmax(preds)
    score = np.amax(preds)
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Classification Result: '+labels[predict]+" Detected", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Classification Result: '+labels[predict]+" Detected", img)
    cv2.waitKey(0)


font = ('times', 16, 'bold')
title = Label(main, text='Monkeypox Detection using Modified VGG16 & Custom CNN Model')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Monkeypox Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocess, bg='#ffb3fe')
processButton.place(x=340,y=550)
processButton.config(font=font1) 

vggButton1 = Button(main, text="Run VGG16 Algorithm", command=runVGG, bg='#ffb3fe')
vggButton1.place(x=570,y=550)
vggButton1.config(font=font1) 

cnnButton = Button(main, text="Run Custom CNN Algorithm", command=runCNN, bg='#ffb3fe')
cnnButton.place(x=50,y=600)
cnnButton.config(font=font1) 

graphButton = Button(main, text="Comparison Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=340,y=600)
graphButton.config(font=font1) 

predictButton = Button(main, text="Predict Disease from Test Image", command=predict, bg='#ffb3fe')
predictButton.place(x=570,y=600)
predictButton.config(font=font1) 

main.config(bg='LightSalmon3')
main.mainloop()
