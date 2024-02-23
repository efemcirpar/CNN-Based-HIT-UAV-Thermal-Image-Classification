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

def random_rotation(image, max_angle=90):
    angle = np.random.uniform(-max_angle, max_angle)
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def random_affine(image, max_shear=0.2, max_translation=20):
    rows, cols, _ = image.shape
    shear = np.random.uniform(-max_shear, max_shear)
    translation_x = np.random.uniform(-max_translation, max_translation)
    translation_y = np.random.uniform(-max_translation, max_translation)
    
    affine_matrix = np.array([[1, shear, translation_x],
                              [0, 1, translation_y]])
    
    affine_image = cv2.warpAffine(image, affine_matrix, (cols, rows))
    return affine_image

def random_horizontal_flip(image):
    return cv2.flip(image, 1)

def random_vertical_flip(image):
    return cv2.flip(image, 0)

def sharpen_image(image, alpha=1.5, beta=-0.5):
    sharpened_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return sharpened_image

def apply_random_augmentations(image):
    augmentations = []

    if np.random.rand() > 0.5:
        image = random_rotation(image)
    
    if np.random.rand() > 0.5:
        image = random_affine(image)
    
    if np.random.rand() > 0.5:
        image = random_horizontal_flip(image)
    
    if np.random.rand() > 0.5:
        image = random_vertical_flip(image)
    
    if np.random.rand() > 0.5:
        image = sharpen_image(image)
    
    return image
