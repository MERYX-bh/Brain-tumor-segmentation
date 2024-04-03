# Brain Tumor Segmentation Using U-Net

## Introduction
This project focuses on the application of the U-Net convolutional neural network for segmenting brain tumors from MRI scans. It addresses the challenge of accurately identifying tumor boundaries within the complex structures of the brain, crucial for effective treatment planning and monitoring. Leveraging the power of machine learning and the U-Net architecture, this project aims to enhance the accuracy and efficiency of brain tumor segmentation in medical imaging.

## Technologies
- Python
- Flask
- TensorFlow
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Features
- **Brain Tumor Segmentation**: Utilizes a U-Net convolutional neural network for precise segmentation of brain tumors from MRI scans.
  
- **Web App Interface**: Provides an easy-to-use interface for uploading MRI scans and viewing their segmentation results.
- **TensorFlow Lite Implementation**: Ensures model efficiency and faster inference times suitable for web deployment.

## Installation
1. **Clone the Repository**: `git clone https://github.com/MERYX-bh/Brain-tumor-segmentation`
2. **Install Dependencies**: Run `pip install -r requirements.txt` to install required libraries.
3. 3. **Go to the web app directory**: run `cd webapp`
4. **Launch the web app**: run `python app.py`

## Web App Interface
The users can upload images and receive segmentations on the interface in real time.

## Dataset
The dataset used is the **Brain MRI segmentation** dataset on kaggle.
The images were obtained from The Cancer Imaging Archive (TCIA).
They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available.
Here's the link to the dataset: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

![Brain Tumor images from dataset](https://github.com/MERYX-bh/Brain-tumor-segmentation/blob/main/images/dataset.png)


### Image Suggestions:
Here are some examples of MRI scans from the test set, alongside their ground truth and predicted segmentations.
![Brain Tumor images from dataset](https://github.com/MERYX-bh/Brain-tumor-segmentation/blob/main/images/exemple1.png)

![Brain Tumor images from dataset](https://github.com/MERYX-bh/Brain-tumor-segmentation/blob/main/images/exemple2.png)
