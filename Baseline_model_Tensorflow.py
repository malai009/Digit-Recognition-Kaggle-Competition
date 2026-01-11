import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense   
from tensorflow.keras.layers import Dropout

train_path = os.path.join("train.csv")
train = pd.read_csv(train_path)
## Visualization of data
# j = 0
# for j in range(10):
#     img = train.iloc[j,1:].values.reshape(28,28)
#     plt.imshow(img, cmap = "gray")
#     plt.title(f"Label: {train.iloc[j, 0]}")
#     plt.axis("off")
#     plt.show()

## Preprocessing of data
# Normalizing pixel values cos, NN loves [0,1], faster training, better gradients
# And reshaping the images
# def norm(train):
#     imgs = []
#     for i, _ in enumerate (train):
#         pixel = train.iloc[i, :]/255
        #imgs.append(pixel.values.reshape(28, 28))
        # plt.imshow(img, cmap = "gray")
        # plt.title(f"Label: {train.iloc[i, 0]}")
        # plt.axis("off")
        # plt.show()
        # return pixel

## Splitting Training data into training and validation
# Standard 80% of data for training and 20% of data for validation
x = train.iloc[:, 1:]
y = train.iloc[:, 0]   # Taking labels seperately
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, 
                                                  random_state= 42, shuffle = True)

#random_state= 42 is for preserving the splitting over multiple runs of code. 42 could be any no.
# print(x_train.shape)
# print(y_train.shape)
# print(x_val.shape)
# print(y_val.shape)

#Normalization
x_val_norm = (x_val)/255
print("Validation data size : ", x_val_norm.shape)
x_train_norm = (x_train)/255
print("Training data size : ", x_train_norm.shape)

## Using Tensorflow model, a baseline MLP
model = Sequential([Dense(256, activation = "relu", input_shape = (784, )), 
                    Dropout(0.4),
                    Dense(128, activation = "relu"),
                    Dense(10, activation = "softmax")])
#Sequential : o/p of one layer is the i/p of another
#The 1st argument 128(standard hidden size) and 10(10 classes) depict the number of neurons
#The image is flattened 28x28 to the shape (784,)
#Activation functions relu for computing output from o/p = relu(weighted sum + bias)
#Softmax is used to compute the probabilities from raw scores
#Dropout=0.3 misses 30% of neurons voluntarity during training to reduce over 
#reliance on few neurons
#Dropout may reduce the accuracy but, it keep the model from overfitting
#But when dropout is increase beyond certain value, it starts to overfitting
#Dropouts are for regularization

## Teaching the model how to learn
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", 
              metrics = ["accuracy"])

# Adam optimizer is the default mostly defines how the model should correct its mistakes

## Training the network
history = model.fit( x_train_norm, y_train, epochs = 30, validation_data = (x_val_norm, y_val))
# epochs = 5, i.e, model sees every training example 5 times
# Initially some random weight and biases are set
# Predict output
# Compute loss against original label
# Backpropagate error
# Update weights
# Evaluate on validation set  

## Visualizing the learning 
plt.plot(history.history["accuracy"], label = "Train Accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation Accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


## Saving the model
#model.save("mnist_model.h5")

## Final test on Test set
#importing test set
test_path = os.path.join("test.csv")
test = pd.read_csv(test_path)
x_test = test.iloc[:, :]/255
preds = model.predict(x_test)
y_pred = preds.argmax(axis = 1)
#print("prediction", y_pred)

## Saving the predictions
image_IDs = range(1, len(y_pred)+ 1)
submission = pd.DataFrame({"Imageid": image_IDs, "Label": y_pred})
submission.to_csv("submission_Tensorflow.csv", index=False)
