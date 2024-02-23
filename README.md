# CNN Based HIT-UAV Thermal Image Classification

### This dataset was obtained from: https://www.kaggle.com/datasets/pandrii000/hituav-a-highaltitude-infrared-thermal-dataset, from: https://doi.org/10.1038/s41597-023-02066-6

This project focuses on image classification using convolutional neural networks (CNNs) and augmentation techniques. The dataset contains multiple label inputs (Person, Car, Bicycle, OtherVehicles, DontCares), requiring preprocessing to create a single label input from multi-label input.

## Preprocessing

To preprocess the data, algorithms were developed to consolidate multi-label inputs into single label inputs. I used the same label for OtherVehicles and DontCares, as I considered both of them as 'DontCares'. Additionally, several augmentation methods were randomly implemented to enhance the dataset's diversity and prevent bias. These augmentation methods include:

- Random rotation
- Random affine transformation
- Random horizontal and vertical flips
- Image sharpening

#### Sample Images

Person:
![image_19_true_0_predicted_0](https://github.com/efemcirpar/CNN-Based-HIT-UAV-Thermal-Image-Classification/assets/128602263/6525a297-952c-4a96-b673-ed17b4166878)
Car:
![image_3_true_1_predicted_1](https://github.com/efemcirpar/CNN-Based-HIT-UAV-Thermal-Image-Classification/assets/128602263/7ccd2c0c-f32a-4fa1-88d4-1ef54466063f)
Bicycle:
![image_15_true_2_predicted_2](https://github.com/efemcirpar/CNN-Based-HIT-UAV-Thermal-Image-Classification/assets/128602263/093b708d-6a56-4733-beac-c0aedd2a4973)


## Model

The CNN model consists of three convolutional layers followed by three linear layers. Notably, average pooling is utilized instead of max pooling due to the characteristics of the images.

## Results

The model achieved impressive results during training, validation, and testing:

- Training accuracy: 99.90%
- Validation accuracy: 91.13%
- Test accuracy: 87.32%

## Confussion Matrices

![confusion_matrix_train](https://github.com/efemcirpar/CNN-Based-HIT-UAV-Thermal-Image-Classification/assets/128602263/f70286ea-b46d-40cb-9403-d77fc3194d74)
![confusion_matrix_val](https://github.com/efemcirpar/CNN-Based-HIT-UAV-Thermal-Image-Classification/assets/128602263/64120d29-96d4-41ed-9193-d14ac2efb8a9)
![confusion_matrix_test](https://github.com/efemcirpar/CNN-Based-HIT-UAV-Thermal-Image-Classification/assets/128602263/7e5651b5-7e6b-41f2-8be6-3ecdb3121d28)


