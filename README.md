# AIU-NET-Brain_Tumor_Segmentation
AIU-Net: Attention-Based Inception U-Net for Brain Tumor Segmentation from Multimodal MRI Images

Keras implementation.

## Dataset
BraTS 2020  data can be downloaded by requesting and registering at: https://www.med.upenn.edu/cbica/brats2020/registration.html.

## Using the code
Have a look at the LICENSE.

## Resources
This code is used to write a paper titled: "AIU-Net: Attention-Based Inception U-Net for Brain Tumor Segmentation from Multimodal MRI Images". It is currently under publication. 
The study is implemented using a combination of many architectures and deep learning techniques from various research papers on Brain Tumor Segmentation. A novel architecture is formed by taking motivation from different architectures. Some of the best resources used are mentioned below.

- https://arxiv.org/pdf/1802.10508v1.pdf : Unet 3D
- https://link.springer.com/chapter/10.1007/978-3-319-75238-9_30: Inception U-Net


## Task
The task is to segment various parts of the brain i.e. labeling all pixels in the multi-modal 3D  MRI images as one of the following classes:
- Necrosis
- Edema
- Non-enhancing tumor
- Enhancing tumor 
- Everything else

## BraTS Dataset 
This repository used the BraTS 2020 training dataset to analyze the proposed methodology. It consists of authentic patient images from 369 patients created by MICCAI. Each of these folders is then subdivided into High Grade and Low-Grade images. Four modalities(T1, T1-C, T2, and FLAIR) are provided for each patient. The fifth image has ground truth labels for each pixel. The dimensions of the images are (240,240,155) in both.


## Dataset pre-processing 
All the images are normalized into unit variance and zero mean. Before the training all the images were resized into 128x128x128 dimensions due to the limited memory and all modalities are stacked together to take advantage of the comprehensive information present in the different modalities of MRI. The final image shape that was used is 128x128x128x4. The model has been trained only for those slices having all 4 labels(0,1,2,4) to tackle class imbalance and label 4 has been converted into label 3 (so that finally the one-hot encoding has size 4).

## Model Architecture used during the training:
# 1: 3D U-Net (Ablation)
# 2: 3D Attention Inception U-Net (proposed)
# 3: 3D Inception U-Net (Ablation)

## Model architectural diagram and results will be shared once after the paper acceptance.


