# DeepLung

This repository contains the code for a Lung Cancer Classification System using Deep Learning. The model used to train the data is ResNet50. The dataset used is the [Chest CT-Scan Images dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) from Kaggle.

## Methodology

The model was trained twice. Once using the raw image files, and the second time using the masked images obtained by implementing [smoothing](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html) and [Roberts Cross Edge Detector](https://homepages.inf.ed.ac.uk/rbf/HIPR2/roberts.htm).

## Model Architecture

The model used is ResNet50. The model was trained for 100 epochs, with early stopping in kicking in at 25, with a batch size of 32. The model was trained on RTX 3080 Ti.

<p align='center'>
    <img src="Images/resnet50.png" alt='DeepLung Model Architecture'>
</p>

## Results

The model trained on the raw images achieved an accuracy of 84.35%, while the model trained on the masked images achieved an accuracy of 90.13%. This is a good improvement over the raw images. The accuracy can be further improved by training the model for more epochs, and by using a larger dataset.

Accuracy Plot for Raw Images

<p align='center'>
    <img src="Images/raw_accuracy.png" alt='Accuracy Plot for Raw Images' width="75%" height="75%">
</p>

Loss Plot for Raw Images

<p align='center'>
    <img src="Images/raw_loss.png" alt='Loss Plot for Raw Images' width="75%" height="75%">
</p>

Accuracy Plot for Masked Images

<p align='center'>
    <img src="Images/mask_accuracy.png" alt='Accuracy Plot for Masked Images' width="75%" height="75%">
</p>

Loss Plot for Masked Images

<p align='center'>
    <img src="Images/mask_loss.png" alt='Loss Plot for Masked Images' width="75%" height="75%">
</p>