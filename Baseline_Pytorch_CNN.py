# MLP model learns statistical pixel combinations, not visual features.
# Convolution layers are what turn pixels into edges, strokes, and shapes

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

train_path = os.path.join("train.csv")
train = pd.read_csv(train_path)

## Splitting Training data into training and validation
# Standard 80% of data for training and 20% of data for validation
x = train.iloc[:, 1:]/255
y = train.iloc[:, 0]   # Taking labels seperately
j = 0

x_train_reshaped = []
x_val_reshaped = []
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, 
                                                  random_state= 42, shuffle = True)
for j,_ in enumerate(x_train.iloc):
     x_train_reshaped.append(x_train.iloc[j, :].values.reshape(1,28,28))
for j,_ in enumerate(x_val.iloc):
     x_val_reshaped.append(x_val.iloc[j, :].values.reshape(1,28,28))
# print("x_train_shape = ", np.shape(x_train_reshaped))
# print("x_val_shape = ", np.shape(x_val_reshaped))
## Converting data into PyTorch Tensor
x_train_t = torch.tensor(x_train_reshaped, dtype = torch.float32)
y_train_t = torch.tensor(y_train.values, dtype = torch.long)        # torch.long means class index
x_val_t = torch.tensor(x_val_reshaped, dtype = torch.float32)
y_val_t = torch.tensor(y_val.values, dtype = torch.long)
# print("x_train_t_shape=", np.shape(x_train_t))
# print("x_val_t_shape=", np.shape(x_val_t))
# print("y_val_t_shape=", np.shape(y_val_t))

## Batched Data loading
train_dataset = TensorDataset(x_train_t, y_train_t)
val_dataset = TensorDataset(x_val_t, y_val_t)
train_loader = DataLoader(train_dataset, batch_size= 64, shuffle= True) 
val_loader = DataLoader(val_dataset, batch_size= 64, shuffle= False)
x_batch, y_batch = next(iter(train_loader))
# print("x_batch_size", x_batch.shape)
# print("y_batch_size", y_batch.shape)
# print("x_batch", x_batch[1,:,:])

## Creating the class
class DigiNet(nn.Module):

    def __init__(self, dropout):
        super().__init__()                #Turns on pytorch
        self.Feature_xtrn = nn.Sequential(nn.Conv2d(in_channels= 1, out_channels= 32, kernel_size= 3, stride=1, padding=1),
                                          nn.ReLU(), nn.MaxPool2d(kernel_size= 2, stride= 2),
                                          nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 3, stride=1, padding=1), 
                                          nn.ReLU(), nn.MaxPool2d(kernel_size= 2, stride= 2))

        self.Classification = nn.Sequential(nn.Flatten(), 
                                            nn.Linear(64*7*7, 128), nn.ReLU(), nn.Dropout(),
                                            nn.Linear(128, 10))

    def forward(self, x):
        x = self.Feature_xtrn(x)
        x = self.Classification(x)
        return x
        
model = DigiNet(dropout = 0.4) #Model is created
optimizer = optim.Adam(model.parameters(), lr = 0.001)

Validation_loss = []
Training_loss = []
Validation_accuracy = []
Training_accuracy = []
a = []
b = []
c = []
d = []
##Training loop
N = 40
for epoch in range(N):
        model.train()                              #Setting the model at training mode
        for x_batch, y_batch in train_loader:      # Batching data
            optimizer.zero_grad()
            pred = model(x_batch)                  # gives the logit scores
            #print(pred.shape)
            #print(y_train_t.shape)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(pred, y_batch)        #Internally applied softmax + cross-entropy
            #print(loss.shape)
            loss.backward()
            optimizer.step()
            #print("count", i)

            #Visualizing
            Training_loss.append(loss.item())
            train_labels = torch.argmax(pred, dim = 1)
            train_acc = (train_labels == y_batch).float().mean()
            Training_accuracy.append(train_acc)

        ##Validation
        model.eval()                               #Model is set to evaluation mode. dropout is off
        for x_val_batch, y_val_batch in val_loader: 
            with torch.no_grad():
                val_pred = model(x_val_batch)
                val_loss = criterion(val_pred, y_val_batch)
                Validation_loss.append(val_loss)
                val_labels = torch.argmax(val_pred, dim = 1)
                val_acc = (val_labels == y_val_batch).float().mean()
                Validation_accuracy.append(val_acc)
        #print("value accuracy size = ",len(Validation_accuracy))
        a.append(np.mean(Training_loss))
        b.append(np.mean(Validation_loss))
        c.append(np.mean(Training_accuracy))
        d.append(np.mean(Validation_accuracy))
        print(f"Epoch {epoch+1}: "
      f"Train Loss = {np.mean(Training_loss):.4f}, "
      f"Val Loss = {np.mean(Validation_loss):.4f}, "
      f"Train Acc = {np.mean(Training_accuracy):.4f}, "
      f"Val Acc = {np.mean(Validation_accuracy):.4f}")
        
##Visualizing loss and accuracy
plt.plot( d, label = "Validation accuracy")
plt.plot( c, label = "Training accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Accuracy Comparision")
plt.legend()
plt.show()

plt.plot(b, label = "Validation loss")
plt.plot(a, label = "Training loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss Comparision")
plt.legend()
plt.show()

##Saving the model
#model.save("CNN_model.h5")
torch.save(model.state_dict(), "CNN_model_prntd.pth")

# Final test on Test set
#importing test set
test_path = os.path.join("test.csv")
test = pd.read_csv(test_path)
test_reshaped = []
predict = []
for j,_ in enumerate(test.iloc):
     test_reshaped.append(test.iloc[j, :].values.reshape(1,28,28)) 
test_t = torch.tensor(test_reshaped, dtype = torch.float32)
test_loader = DataLoader(test_t, batch_size= 64, shuffle= False) 
for test_batch in test_loader: 
    predicted = model(test_batch)
    predict.append(predicted)
predict = torch.cat(predict)
label = torch.argmax(predict, dim = 1)

## Saving the predictions
image_IDs = range(1, len(label)+ 1)
submission = pd.DataFrame({"Imageid": image_IDs, "Label": label})
submission.to_csv("submission_PyTorch_CNN.csv", index=False)
print("submitted")