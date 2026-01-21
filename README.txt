# 🏥 Medical Image Classifier

A comprehensive exploration of computer vision techniques for enhancing medical diagnostics. This project compares traditional machine learning, deep learning (CNNs), and state-of-the-art transfer learning models to identify patterns and anomalies in medical imaging with high precision.

## 📋 Project Description
Medical imaging is vital for disease detection, but the sheer volume of images generated daily poses challenges for manual interpretation. This project develops automated solutions to:
* **Support Healthcare Professionals**: Develop tools to prioritize critical cases and highlight key regions of interest.
* **Enhance Accuracy**: Automate the identification of potential signs of COVID-19 or Viral Pneumonia.
* **Ensure Consistency**: Reduce inconsistencies in patient care caused by varying expertise levels among practitioners.

## 📊 Datasets
We evaluated our models using two distinct datasets to test generalization across different imaging modalities. Due to time and computational constraints, we utilized a subset of **7,000 images** for each dataset:

1. **Medical MNIST (7k Subset)**: Grayscale images (64x64) across six classes: Abdomen CT, Breast MRI, Chest CT, Chest X-Ray, Hand, and Head CT.
2. **COVID-19 Radiography (7k Subset)**: High-resolution images across four classes: Normal, COVID, Viral Pneumonia, and Lung Opacity.



## 🛠️ Technical Approaches
The project systematically compared three distinct methodologies:

### 1. Traditional Computer Vision
Focused on manual feature extraction and classic classification:
* **Features**: Histogram of Oriented Gradients (HoG) and Scale-Invariant Feature Transform (SIFT).
* **Classifiers**: Support Vector Machines (SVMs) and Bag of Words (BoW) models.
* **Results**: Achieved high accuracy on Medical MNIST but faced challenges with the complex internal textures of COVID-19 radiographs.

### 2. Convolutional Neural Networks (CNN)
Deep learning architectures designed to automatically learn hierarchical features:
* **Simple Model**: Two convolutional layers designed for computational efficiency and speed.
* **Advanced Model**: Three convolutional layers with L2 regularization and dropout to prevent overfitting.
* **Results**: The Simple Model proved more robust for these subsets, as increased complexity in the Advanced Model sometimes introduced feature loss.

### 3. Transfer Learning (ResNet-50)
Leveraged a 50-layer deep residual network pre-trained on ImageNet to solve the vanishing gradient problem:
* **Implementation**: Fine-tuned the last 40 layers of ResNet-50 while keeping initial layers frozen to retain pre-trained features.
* **Results**: Delivered the highest accuracy across all experiments, reaching 97.42% for COVID and 99.96% for MNIST.



## 📈 Performance Comparison (7k Subsets)

| Model Approach | Medical MNIST Accuracy | COVID-19 Accuracy |
| :--- | :--- | :--- |
| **SVM (HoG Features)** | 99.50% | 88.25% |
| **Simple CNN** | 99.86% | 92.76% |
| **ResNet-50 (Transfer)**| 99.96% | 97.42% |

## 👥 Group Members
* **Aryansingh Chauhan**: CNN Architectures & Training
* **Tanuj**: Traditional Methods (SVM/KNN)
* **Snigdha**: Transfer Learning (ResNet)

CS 415: Computer Vision, Prof Wei Tang, Fall 2024
University of Illinois Chicago


## References:

* Covid-19 Radiography dataset:

M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676. https://ieeexplore.ieee.org/document/9144185

Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. https://doi.org/10.1016/j.compbiomed.2021.104319


* Medical MNIST dataset:

Polanco, A. (2017). Medical MNIST Classification [GitHub repository]. Retrieved from https://github.com/apolanco3225/Medical-MNIST-Classification.
License: Public Domain

Splash Image Credit: Photo by Hush Naidoo on Unsplash.

https://www.kaggle.com/datasets/andrewmvd/medical-mnist


* ResNet50:

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. arXiv preprint arXiv:1512.03385. https://doi.org/10.48550/arXiv.1512.03385

https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b

ResNet50 — Torchvision. (2017). PyTorch. Retrieved December 12, 2024, from https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html