import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import cv2
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

train_load_path = "hit-uav/saves/augmented_train_image_class_pairs.pkl"
test_load_path = "hit-uav/saves/augmented_test_image_class_pairs.pkl"
val_load_path = "hit-uav/saves/augmented_val_image_class_pairs.pkl"

with open(train_load_path, "rb") as file:
    train_image_class_pairs = pickle.load(file)

with open(test_load_path, "rb") as file:
    test_image_class_pairs = pickle.load(file)

with open(val_load_path, "rb") as file:
    val_image_class_pairs = pickle.load(file)

transform = transforms.Compose([transforms.ToTensor()])

#USING A SMALL PORTOIN OF DATA

downsampling_ratio = 1

small_train_image_class_pairs = []
for i in range(int((len(train_image_class_pairs))/downsampling_ratio)):
    small_train_image_class_pairs.append(train_image_class_pairs[i])
small_test_image_class_pairs = []
for i in range(int((len(test_image_class_pairs))/downsampling_ratio)):
    small_test_image_class_pairs.append(test_image_class_pairs[i])
small_val_image_class_pairs = []
for i in range(int((len(val_image_class_pairs))/downsampling_ratio)):
    small_val_image_class_pairs.append(val_image_class_pairs[i])
print("Downsampling done. Downsampled by factor of: ", downsampling_ratio)

print("Transforming Train")
train_image_class_pairs_transformed = [(transform(image), label) for image, label in small_train_image_class_pairs]
print("Transforming Val")
val_image_class_pairs_transformed = [(transform(image), label) for image, label in small_val_image_class_pairs]
print("Transforming Test")
test_image_class_pairs_transformed = [(transform(image), label) for image, label in small_test_image_class_pairs]

transformed_data = {
    'train': train_image_class_pairs_transformed,
    'val': val_image_class_pairs_transformed,
    'test': test_image_class_pairs_transformed
}

save_path = "hit-uav/saves/downsampled_transformed_data.pkl"
with open(save_path, "wb") as file:
    pickle.dump(transformed_data, file)

print("Finished.")
