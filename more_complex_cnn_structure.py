import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import cv2
import pickle
from torchvision import transforms, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.cuda.is_available())

save_path = "hit-uav/saves/downsampled_transformed_data.pkl"

with open(save_path, "rb") as file:
    loaded_transformed_data = pickle.load(file)

train_data = loaded_transformed_data['train']
val_data = loaded_transformed_data['val']
test_data = loaded_transformed_data['test']

print("Loading Data")
batch_size = 32
train_dataloader = DataLoader(dataset = train_data,batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(dataset = val_data, batch_size = batch_size, shuffle=False)
test_dataloader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False)
print("Data Loading is Finished")

#Number of classes
num_classes = 4
#Dropout
dropout = 0.5
#Weight Decay
weight_dcy = 1e-4

print("Defining CNN") 
#CNN architecture
class CNN(nn.Module):
    def __init__(self, num_classes, dropout_prob=dropout, weight_decay=weight_dcy):
        super(CNN, self).__init__()
        """self.bn0 = nn.BatchNorm2d(3)
        self.Conv0 = nn.Conv2d(in_channels = 3, out_channels= 32, kernel_size=1)"""
        #Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        #Activation function
        self.relu = nn.ReLU()
        #self.relu = F.leaky_relu(nn.Conv2d)

        #Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        #Pooling layers
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        #Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)  #Adjusting input size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        #Adding Dropout to Prevent Overfitting
        self.dropout = nn.Dropout(dropout_prob)
        self.weight_decay = weight_decay

    def forward(self, x):
        """x = self.bn0(x)
        x = self.Conv0(x)
        x = F.leaky_relu(x)"""

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


#CREATING A MODEL
model = CNN(num_classes).to(device)

#Defining Loss Function and Optimizer
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate,  weight_decay = weight_dcy)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

#Number of Epochs
num_epochs = 1

#Initializng 
train_total_loss = 0
val_total_loss = 0
test_total_loss = 0

printer_counter = 0

train_avg_loss = 0
val_avg_loss = 0
test_avg_loss = 0

print(len(train_data)) #Checking sizes
print(len(test_data))
print(len(val_data))

print("HERE WE GO!")
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        train_total_loss += loss.item()
        printer_counter += 1

        optimizer.step()

        # Print training progress
        if(printer_counter % 5 == 0):
            if(printer_counter != 0):
                train_avg_loss = train_total_loss / printer_counter
            print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {train_avg_loss}")

#TRAIN LOOP
model.eval()
with torch.no_grad():
    correct_train = 0
    total_train = 0
    true_labels_train = []
    predicted_labels_train = []
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted_train = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

        loss = criterion(outputs, labels)
        train_total_loss += loss.item()

        true_labels_train.extend(labels.cpu().numpy())
        predicted_labels_train.extend(predicted_train.cpu().numpy())

    train_accuracy = 100 * correct_train / total_train
    train_avg_loss = train_total_loss / len(train_dataloader)
    print(f"Training Accuracy: {train_accuracy:.2f}%, Avg Loss: {train_avg_loss:.4f}")


#VALIDATON LOOP
model.eval()
with torch.no_grad():
    correct_valid = 0
    total_valid = 0
    true_labels_val = []
    predicted_labels_val = []
    for images, labels in val_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted_val = torch.max(outputs, 1)
        total_valid += labels.size(0)
        correct_valid += (predicted_val == labels).sum().item()

        loss = criterion(outputs, labels)
        val_total_loss += loss.item()

        true_labels_val.extend(labels.cpu().numpy())
        predicted_labels_val.extend(predicted_val.cpu().numpy())

    val_accuracy = 100 * correct_valid / total_valid
    val_avg_loss = val_total_loss / len(val_dataloader)
    print(f"Validation Accuracy: {val_accuracy:.2f}%, Avg Loss: {val_avg_loss:.4f}")

#TEST LOOP
model.eval()
with torch.no_grad():
    plotting = 0   
    correct_test = 0
    total_test = 0
    true_labels_test = []
    predicted_labels_test = []
    folder_names = 0
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted_test = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted_test == labels).sum().item()

        loss = criterion(outputs, labels)
        test_total_loss += loss.item()

        true_labels_test.extend(labels.cpu().numpy())
        predicted_labels_test.extend(predicted_test.cpu().numpy())

        if(plotting % 6 == 0):
            folder_names += 1
            image = images[0].unsqueeze(0).to(device)
            output = model(image)
            _, predicted_label = torch.max(output, 1)
            predicted_label = predicted_label[0].item()
            image_np = image.squeeze(0).cpu().numpy()
            image_np = np.transpose(image_np, (1, 2, 0))     
            #Displayin Images and Their Labels
            image_filename = f"image_{folder_names}_true_{labels[0].item()}_predicted_{predicted_label}.png"
            saving_path = "Savings/"
            save_path = os.path.join(saving_path, image_filename)
            plt.imsave(save_path, image_np)

        plotting += 1


    test_accuracy = 100 * correct_test / total_test
    test_avg_loss = test_total_loss / len(test_dataloader)
    print(f"Test Accuracy: {test_accuracy:.2f}%, Avg Loss: {test_avg_loss:.4f}")


saving_path = "Savings/"
#CM AND F1 SCORE FOR TRAIN
conf_matrix_train = confusion_matrix(true_labels_train, predicted_labels_train)
f1_train = f1_score(true_labels_train, predicted_labels_train, average='weighted')
confusion_matrix_path = os.path.join(saving_path, "confusion_matrix_train.png")

#Plotting and saving Confusion Matrix and F1 Score
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='BuPu')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Train Confusion Matrix / F1 Score: {f1_train:.4f}')
plt.savefig(confusion_matrix_path)
plt.close()


#CM AND F1 SCORE FOR VAL
conf_matrix_val = confusion_matrix(true_labels_val, predicted_labels_val)
f1_val = f1_score(true_labels_val, predicted_labels_val, average='weighted')  
confusion_matrix_path = os.path.join(saving_path, "confusion_matrix_val.png")

#Plotting and saving Confusion Matrix and F1 Score
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='BuPu')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Val Confusion Matrix / F1 Score: {f1_val:.4f}')
plt.savefig(confusion_matrix_path)
plt.close()


#CM AND F1 SCORE FOR TEST
conf_matrix_test = confusion_matrix(true_labels_test, predicted_labels_test)
f1_test = f1_score(true_labels_test, predicted_labels_test, average='weighted')
confusion_matrix_path = os.path.join(saving_path, "confusion_matrix_test.png")

#Plotting and saving Confusion Matrix and F1 Score
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='BuPu')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Test Confusion Matrix / F1 Score: {f1_test:.4f}')
plt.savefig(confusion_matrix_path)
plt.close()