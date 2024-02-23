import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import cv2
import pickle

dataset_path = 'hit-uav/'
folders = os.listdir(dataset_path)
print(folders)

root_directory = dataset_path
train_imgs_dir = 'hit-uav/images/train/'
train_labels_dir = 'hit-uav/labels/train/'
val_imgs_dir = 'hit-uav/images/val/'
val_labels_dir = 'hit-uav/labels/val/'
test_imgs_dir = 'hit-uav/images/test/'
test_labels_dir = 'hit-uav/labels/test/'

image_filenames = os.listdir(train_imgs_dir)
label_filenames = os.listdir(train_labels_dir)
image_filenames.sort()
label_filenames.sort()

train_image_label_matching_pairs = []

###FOR TRAINING
for i in range(len(os.listdir(train_imgs_dir))):
    image_path = train_imgs_dir + image_filenames[i]
    label_path = train_labels_dir + label_filenames[i]
    train_image_label_matching_pairs.append((image_path, label_path))

df_train = pd.DataFrame(train_image_label_matching_pairs)
#print(df_train)

image_filenames = os.listdir(test_imgs_dir)
label_filenames = os.listdir(test_labels_dir)
image_filenames.sort()
label_filenames.sort()

test_image_label_matching_pairs = []

###FOR TEST
for i in range(len(os.listdir(test_imgs_dir))):
    image_path = test_imgs_dir + image_filenames[i]
    label_path = test_labels_dir + label_filenames[i]
    test_image_label_matching_pairs.append((image_path, label_path))

df_test = pd.DataFrame(test_image_label_matching_pairs)
#print(df_test)

image_filenames = os.listdir(val_imgs_dir)
label_filenames = os.listdir(val_labels_dir)
image_filenames.sort()
label_filenames.sort()

val_image_label_matching_pairs = []

###FOR VALIDATION
for i in range(len(os.listdir(val_imgs_dir))):
    image_path = val_imgs_dir + image_filenames[i]
    label_path = val_labels_dir + label_filenames[i]
    val_image_label_matching_pairs.append((image_path, label_path))

df_valid = pd.DataFrame(val_image_label_matching_pairs)
#print(df_valid)

#image = cv2.imread(train_image_label_matching_pairs[0][0])
#print(image.shape)

resizing_factor = 64

###TRAIN
train_image_class_pairs = []

for file_index in range(len(train_image_label_matching_pairs)):
    for image_label_index in range(2):
      label_file_path = train_image_label_matching_pairs[file_index][1]
      with open(label_file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()

            #Class ID Assigning
            class_id = int(parts[0])

            if(class_id == 4): #Concating Class 3 and 4
                class_id = 3

            #Box Assigning
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            image_path = train_image_label_matching_pairs[file_index][0]
            image = cv2.imread(image_path)

            #Calculate bounding box coordinates
            x1 = int((x_center - width / 2) * image.shape[1])
            y1 = int((y_center - height / 2) * image.shape[0])
            x2 = int((x_center + width / 2) * image.shape[1])
            y2 = int((y_center + height / 2) * image.shape[0])

            expand_size = 10

            x1_expanded = max(0, x1 - expand_size)  #Subtract 10 from x1, ensuring it doesn't go below 0
            y1_expanded = max(0, y1 - expand_size)  #Subtract 10 from y1, ensuring it doesn't go below 0
            x2_expanded = min(image.shape[1], x2 + expand_size)  #Add 10 to x2, ensuring it doesn't exceed image width
            y2_expanded = min(image.shape[0], y2 + expand_size)  #Add 10 to y2, ensuring it doesn't exceed image height

            cropped_region = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

            #Resizing
            resized_width = resizing_factor
            resized_height = resizing_factor
            resized_cropped_region = cv2.resize(cropped_region, (resized_width, resized_height))

            """plt.imshow(cv2.cvtColor(resized_cropped_region, cv2.COLOR_BGR2RGB))
            plt.title(class_id)
            plt.axis('off')
            plt.show()"""

            train_image_class_pairs.append((resized_cropped_region, class_id))

print(train_image_class_pairs[0][0].shape)


####TEST
test_image_class_pairs = []

for file_index in range(len(test_image_label_matching_pairs)):
    for image_label_index in range(2):
      label_file_path = test_image_label_matching_pairs[file_index][1]
      with open(label_file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()

            #Class ID Assigning
            class_id = int(parts[0])

            if(class_id == 4): #Concating Class 3 and 4
                class_id = 3


            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            image_path = test_image_label_matching_pairs[file_index][0]
            image = cv2.imread(image_path)

            #Calculate bounding box coordinates
            x1 = int((x_center - width / 2) * image.shape[1])
            y1 = int((y_center - height / 2) * image.shape[0])
            x2 = int((x_center + width / 2) * image.shape[1])
            y2 = int((y_center + height / 2) * image.shape[0])

            expand_size = 10

            x1_expanded = max(0, x1 - expand_size)  #Subtract 10 from x1, ensuring it doesn't go below 0
            y1_expanded = max(0, y1 - expand_size)  #Subtract 10 from y1, ensuring it doesn't go below 0
            x2_expanded = min(image.shape[1], x2 + expand_size)  #Add 10 to x2, ensuring it doesn't exceed image width
            y2_expanded = min(image.shape[0], y2 + expand_size)  #Add 10 to y2, ensuring it doesn't exceed image height

            cropped_region = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

            #Resizing
            resized_width = resizing_factor
            resized_height = resizing_factor
            resized_cropped_region = cv2.resize(cropped_region, (resized_width, resized_height))

            """plt.imshow(cv2.cvtColor(resized_cropped_region, cv2.COLOR_BGR2RGB))
            plt.title(class_id)
            plt.axis('off')
            plt.show()"""

            test_image_class_pairs.append((resized_cropped_region, class_id))

print(test_image_class_pairs[0][0].shape)



#VALID
val_image_class_pairs = []

for file_index in range(len(val_image_label_matching_pairs)):
    for image_label_index in range(2):
      label_file_path = val_image_label_matching_pairs[file_index][1]
      with open(label_file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()

            #Class ID Assigning
            class_id = int(parts[0])

            if(class_id == 4): #Concating Class 3 and 4
                class_id = 3


            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            image_path = val_image_label_matching_pairs[file_index][0]
            image = cv2.imread(image_path)

            #Calculate bounding box coordinates
            x1 = int((x_center - width / 2) * image.shape[1])
            y1 = int((y_center - height / 2) * image.shape[0])
            x2 = int((x_center + width / 2) * image.shape[1])
            y2 = int((y_center + height / 2) * image.shape[0])

            expand_size = 10

            x1_expanded = max(0, x1 - expand_size)  #Subtract 10 from x1, ensuring it doesn't go below 0
            y1_expanded = max(0, y1 - expand_size)  #Subtract 10 from y1, ensuring it doesn't go below 0
            x2_expanded = min(image.shape[1], x2 + expand_size)  #Add 10 to x2, ensuring it doesn't exceed image width
            y2_expanded = min(image.shape[0], y2 + expand_size)  #Add 10 to y2, ensuring it doesn't exceed image height

            cropped_region = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

            #Resizing
            resized_width = resizing_factor
            resized_height = resizing_factor
            resized_cropped_region = cv2.resize(cropped_region, (resized_width, resized_height))

            """plt.imshow(cv2.cvtColor(resized_cropped_region, cv2.COLOR_BGR2RGB))
            plt.title(class_id)
            plt.axis('off')
            plt.show()"""

            val_image_class_pairs.append((resized_cropped_region, class_id))


train_save_path = "hit-uav/saves/train_image_class_pairs.pkl"
test_save_path = "hit-uav/saves/test_image_class_pairs.pkl"
val_save_path = "hit-uav/saves/val_image_class_pairs.pkl"

with open(train_save_path, "wb") as file:
    pickle.dump(train_image_class_pairs, file)

with open(test_save_path, "wb") as file:
    pickle.dump(test_image_class_pairs, file)

with open(val_save_path, "wb") as file:
    pickle.dump(val_image_class_pairs, file)

print(len(train_image_class_pairs))
print(len(test_image_class_pairs))
print(len(val_image_class_pairs))           

print("Ratio of training set in all set: ", len(train_image_class_pairs)/( len(train_image_class_pairs) + len(test_image_class_pairs) + len(val_image_class_pairs)))
print("Ratio of test set in all set: ", len(test_image_class_pairs)/( len(train_image_class_pairs) + len(test_image_class_pairs) + len(val_image_class_pairs)))
print("Ratio of val set in all set: ", len(val_image_class_pairs)/( len(train_image_class_pairs) + len(test_image_class_pairs) + len(val_image_class_pairs)))

print(type(train_image_class_pairs))
print(type(test_image_class_pairs))
print(type(val_image_class_pairs))