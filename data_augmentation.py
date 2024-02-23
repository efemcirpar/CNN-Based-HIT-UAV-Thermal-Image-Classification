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
from augmentation_methods import *

device = "cuda" if torch.cuda.is_available() else "cpu"

train_load_path = "hit-uav/saves/train_image_class_pairs.pkl"
test_load_path = "hit-uav/saves/test_image_class_pairs.pkl"
val_load_path = "hit-uav/saves/val_image_class_pairs.pkl"

with open(train_load_path, "rb") as file:
    train_image_class_pairs = pickle.load(file)

with open(test_load_path, "rb") as file:
    test_image_class_pairs = pickle.load(file)

with open(val_load_path, "rb") as file:
    val_image_class_pairs = pickle.load(file)

print(type(train_image_class_pairs[0][0]))
print(type(train_image_class_pairs[0][1]))

train_class_zero_handler = 17066
train_class_one_handler = 10506
train_class_two_handler = 7260
train_class_three_handler = 424

for i in range(len(train_image_class_pairs)):
    if(train_image_class_pairs[i][1] == 1 and (train_class_zero_handler>=train_class_one_handler+1)):
        class_id = 1
        augmented_image = train_image_class_pairs[i][0]
        augmented_image = apply_random_augmentations(augmented_image)
        train_image_class_pairs.append((augmented_image, class_id))
        train_class_one_handler += 1

for i in range(len(train_image_class_pairs)):
    if(train_image_class_pairs[i][1] == 2 and (train_class_zero_handler>=train_class_two_handler+1)):
        for j in range(2):
            class_id = 2
            augmented_image = train_image_class_pairs[i][0]
            augmented_image = apply_random_augmentations(augmented_image)
            train_image_class_pairs.append((augmented_image, class_id))
            train_class_two_handler += 1

for i in range(len(train_image_class_pairs)):
    if(train_image_class_pairs[i][1] == 3 and (train_class_zero_handler>=train_class_three_handler+1)):
        for j in range(100):
            class_id = 3
            augmented_image = train_image_class_pairs[i][0]
            augmented_image = apply_random_augmentations(augmented_image)
            train_image_class_pairs.append((augmented_image, class_id))
            train_class_three_handler += 1
            if(train_class_zero_handler<=train_class_three_handler+1):
                break

test_class_zero_handler = 5222
test_class_one_handler = 2678
test_class_two_handler = 1592
test_class_three_handler = 130

for i in range(len(test_image_class_pairs)):
    if(test_image_class_pairs[i][1] == 1 and (test_class_zero_handler>=test_class_one_handler+1)):
        class_id = 1
        augmented_image = test_image_class_pairs[i][0]
        augmented_image = apply_random_augmentations(augmented_image)
        test_image_class_pairs.append((augmented_image, class_id))
        test_class_one_handler += 1

for i in range(len(test_image_class_pairs)):
    if(test_image_class_pairs[i][1] == 2 and (test_class_zero_handler>=test_class_two_handler+1)):
        for j in range(3):
            class_id = 2
            augmented_image = test_image_class_pairs[i][0]
            augmented_image = apply_random_augmentations(augmented_image)
            test_image_class_pairs.append((augmented_image, class_id))
            test_class_two_handler += 1
            if(test_class_zero_handler<=test_class_two_handler+1):
                break

for i in range(len(test_image_class_pairs)):
    if(test_image_class_pairs[i][1] == 3 and (test_class_zero_handler>=test_class_three_handler+1)):
        for j in range(50):
            class_id = 3
            augmented_image = test_image_class_pairs[i][0]
            augmented_image = apply_random_augmentations(augmented_image)
            test_image_class_pairs.append((augmented_image, class_id))
            test_class_three_handler += 1
            if(test_class_zero_handler<=test_class_three_handler+1):
                break

val_class_zero_handler = 2336
val_class_one_handler = 1438
val_class_two_handler = 1108
val_class_three_handler = 38

for i in range(len(val_image_class_pairs)):
    if(val_image_class_pairs[i][1] == 1 and (val_class_zero_handler>=val_class_one_handler+1)):
        class_id = 1
        augmented_image = val_image_class_pairs[i][0]
        augmented_image = apply_random_augmentations(augmented_image)
        val_image_class_pairs.append((augmented_image, class_id))
        val_class_one_handler += 1

for i in range(len(val_image_class_pairs)):
    if(val_image_class_pairs[i][1] == 2 and (val_class_zero_handler>=val_class_two_handler+1)):
        for j in range(3):
            class_id = 2
            augmented_image = val_image_class_pairs[i][0]
            augmented_image = apply_random_augmentations(augmented_image)
            val_image_class_pairs.append((augmented_image, class_id))
            val_class_two_handler += 1
            if(val_class_zero_handler<=val_class_two_handler+1):
                break

for i in range(len(val_image_class_pairs)):
    if(val_image_class_pairs[i][1] == 3 and (val_class_zero_handler>=val_class_three_handler+1)):
        for j in range(100):
            class_id = 3
            augmented_image = val_image_class_pairs[i][0]
            augmented_image = apply_random_augmentations(augmented_image)
            val_image_class_pairs.append((augmented_image, class_id))
            val_class_three_handler += 1
            if(val_class_zero_handler<=val_class_three_handler+1):
                break

a = 0
b = 0 
c = 0

for i in range(len(train_image_class_pairs)):
    if(train_image_class_pairs[i][1] == 1):
        a+=1
for i in range(len(train_image_class_pairs)):
    if(train_image_class_pairs[i][1] == 2):
        b+=1
for i in range(len(train_image_class_pairs)):
    if(train_image_class_pairs[i][1] == 3):
        c+=1

print(a,b ,c, len(train_image_class_pairs))

d = 0
e = 0
f = 0

for i in range(len(test_image_class_pairs)):
    if(test_image_class_pairs[i][1] == 1):
        d+=1
for i in range(len(test_image_class_pairs)):
    if(test_image_class_pairs[i][1] == 2):
        e+=1
for i in range(len(test_image_class_pairs)):
    if(test_image_class_pairs[i][1] == 3):
        f+=1

print(d ,e ,f, len(test_image_class_pairs))

g = 0 
h = 0
j = 0

for i in range(len(val_image_class_pairs)):
    if(val_image_class_pairs[i][1] == 1):
        g+=1
for i in range(len(val_image_class_pairs)):
    if(val_image_class_pairs[i][1] == 2):
        h+=1
for i in range(len(val_image_class_pairs)):
    if(val_image_class_pairs[i][1] == 3):
        j+=1

print(g ,h ,j, len(val_image_class_pairs))

print("Ratio of training set in all set: ", len(train_image_class_pairs)/( len(train_image_class_pairs) + len(test_image_class_pairs) + len(val_image_class_pairs)))
print("Ratio of test set in all set: ", len(test_image_class_pairs)/( len(train_image_class_pairs) + len(test_image_class_pairs) + len(val_image_class_pairs)))
print("Ratio of val set in all set: ", len(val_image_class_pairs)/( len(train_image_class_pairs) + len(test_image_class_pairs) + len(val_image_class_pairs)))

train_save_path = "hit-uav/saves/augmented_train_image_class_pairs.pkl"
test_save_path = "hit-uav/saves/augmented_test_image_class_pairs.pkl"
val_save_path = "hit-uav/saves/augmented_val_image_class_pairs.pkl"

with open(train_save_path, "wb") as file:
    pickle.dump(train_image_class_pairs, file)

with open(test_save_path, "wb") as file:
    pickle.dump(test_image_class_pairs, file)

with open(val_save_path, "wb") as file:
    pickle.dump(val_image_class_pairs, file)

print("Augmentation Finished.")