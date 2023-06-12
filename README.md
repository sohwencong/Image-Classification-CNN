# Image Classification using Convolutional Neural Network (CNN or ConvNet)

## Table of Contents
1. [Overview](README.md#overview)
2. [Data Preprocessing and Data Loading](README.md#data-preprocessing-and-data-loading)
3. [Image Classification Model 1: Train from Scratch](README.md#image-classification-model-1-train-from-scratch)
4. [Image Classification Model 2: Utilizing Pre-Train Model](README.md#image-classification-model-2-utilizing-pre-train-model)
5. [Evaluate Models using Test images](README.md#evaluate-models-using-test-images)
6. [Best Model to Perform Classification](README.md#best-model-to-perform-classification)
7. [Summary](README.md#summary)


## Overview
* **Objective**: To build an image classification model to recognize and classify 10 different types of food
* **Framework: Universal Workflow of Machine Learning**
    a. Define the problem and assemble a dataset<br>
    b. Choose a measure of success<br>
    c. Decide on an evaluation protocol<br>
    d. Prepare the data<br>
    e. Develop a model that does better than a baseline<br>
    f. Develop a model that overfits<br>
    g. Regularize the model and tune its hyperparameters<br>
    
* **Type of Problem**: A multiclass classification problem with 10 classes of output
* **Inputs and Outputs**:
    * Inputs (training): 750 food images per type
    * Inputs (validation): 200 food images per type
    * Inputs (testing): 50 food images per type
    * Output: Food labels<br>
* **Measure of Success**: Accuracy
* **Evaluation Protocol**: Maintaining a hold-out validation set (Dataset is huge – images)

## Data Preprocessing and Data Loading
* **Platform**: Google Colab
* **Library / Packages**: Tensorflow, Keras, ImageDataGenerator
* **Data Loading**: Link up folder directory of training, validation and testing dataset
* **Data Preprocessing**: 
    * Rescale pixel values (0 to 255) to [0,1] interval by dividing by 255
    * Standardize image size to 150 x 150 px
  
## Image Classification Model 1: Train from Scratch
* **Library / Packages**: Tensorflow, Keras, Layers, Models, Optimizers 

## Image Classification Model 2: Utilizing Pre-Train Model


## Evaluate Models using Test images


## Best Model to Perform Classification


## Summary
