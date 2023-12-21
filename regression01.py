import pandas as pd
import numpy as np
import tensorflow as tf
import os

TF_ENABLE_ONEDNN_OPTS=0



# Change the working directory
os.chdir(r"C:\Users\Lenovo\Desktop\regression course")

# Import the dataset
dataset = pd.read_excel("Folds5x2_pp.xlsx")

X = dataset.iloc[: , :-1].values  
Y = dataset.iloc[: , -1].values  
print(X)
print(Y)                                                               

#splitting the dataset into training set

from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(X,Y, train_size=0.2, random_state=0)


# Building an ANN

#step -1 : Imitializing the ANN

ann = tf.keras.models.Sequential() #library.sub-library.module

#Step 2: Adding input layer and connecting it with first hidden layer
#An activation function in a neural network is a mathematical "gate" that decides whether a neuron in the network should be activated or not.#

ann.add(tf.keras.layers.Dense(units=6 , activation='relu')) #here units represents the no. of hidden layers that are actually needed

#Adding second layer

ann.add(tf.keras.layers.Dense(units=6 , activation='relu'))

#adding the output layer

ann.add(tf.keras.layers.Dense(units=1 , activation='relu'))

#step-3 :- Traiing the ANN

#Compiling the ANN

ann.compile(optimizer ='adam' , loss = 'mean_squared_error')

#training the dataset in training set

ann.fit(X_train , Y_train , batch_size = 32 , epochs = 100  )

#Step : 4 :-

#predicting the set

Y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))
