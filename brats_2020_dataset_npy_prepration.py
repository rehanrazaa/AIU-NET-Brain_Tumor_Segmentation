# -*- coding: utf-8 -*-
"""Brats_2020_Dataset_NPY_Prepration.ipynb

## **This notebook is about how to read 3D .nii Files of the BraTS dataset and convert it into 3D npy file for efficient training of the model**
"""

# Importing necesaary Libraries
import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#Here the path where the dataset is stored
TRAIN_DATASET_PATH = '/content/drive/MyDrive/Brats-2020-Dataset/Brats2020_Training_dataset/'

# BraTS 2020 dataset contain 4 diffierent modalities and 1 segmentation mask
# Here we set the path of each modality
t1_list = sorted(glob.glob('path'))
t2_list = sorted(glob.glob('path'))
t1ce_list = sorted(glob.glob('path'))
flair_list = sorted(glob.glob('path'))
mask_list = sorted(glob.glob('path'))

for img in range(len(t2_list)):   #Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)

    temp_image_t1=nib.load(t1_list[img]).get_fdata()
    temp_image_t1=scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)

    temp_image_t2=nib.load(t2_list[img]).get_fdata()
    temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

    temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)

    temp_image_flair=nib.load(flair_list[img]).get_fdata()
    temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

    temp_mask=nib.load(mask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
    print(np.unique(temp_mask))


    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2, temp_image_t1], axis=3)

    #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
    # Also it will resize our dataset from 240x240x155 to 128x128x128 so that  we can efficiently manage the resouces
    #cropping x, y, and z
    # In resize we make sure that there will be no information lost
    temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]

    val, counts = np.unique(temp_mask, return_counts=True)

    if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
        print("Saving the stacking npy file")
        temp_mask= to_categorical(temp_mask, num_classes=4)
        np.save('/content/drive/MyDrive/Brats-2020-Dataset/BraTS2021_NPY_Files/train_images/image_'+str(img)+'.npy', temp_combined_images)
        np.save('/content/drive/MyDrive/Brats-2020-Dataset/BraTS2021_NPY_Files/train_masks/mask_'+str(img)+'.npy', temp_mask)

    else:
        print("Not for any use.")
