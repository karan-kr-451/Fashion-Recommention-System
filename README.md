
# Fashion Recommendation System

### Overview

A brief description of what this project does and who it's for The Fashion Recommendation System is a project aimed at providing personalized fashion recommendations to users based on visual similarities between images of fashion items. Leveraging state-of-the-art deep learning techniques, the system extracts features from images using a pre-trained Convolutional Neural Network (CNN) model, specifically ResNet50, and then recommends visually similar items from a curated dataset.

###  Demo

![Image description](D:\Github Project\Fashion Recommendation\demo\Demo.gif)

### Features
1. Feature Extraction: Utilizes a pre-trained ResNet50 model, fine-tuned for feature extraction, to capture the essential characteristics of fashion items from images.
2. Efficient Image Comparison: Employs feature normalization and Nearest Neighbors algorithms to quickly find and recommend similar items.
3. User-Friendly Interface: Built using Streamlit, the web interface allows users to upload images and receive recommendations in real-time.
4. Optimized Performance: Implements techniques such as batch processing and multiprocessing to enhance the efficiency of feature extraction.

### Technology Stack
1. Python: Core programming language used for developing the backend and processing images.
2. TensorFlow & Keras: Used for building and deploying the ResNet50 model for feature extraction.
3. Streamlit: Provides an interactive web interface for users to upload images and view recommendations.
4. Scikit-learn: Utilized for implementing the Nearest Neighbors algorithm to find similar images.
5. Git LFS: Manages large files efficiently within the Git repository.


### Installation and Setup

1. Clone the Repository:
```cmd
git clone https://github.com/username/fashion-recommendation.git
cd fashion-recommendation
```
2. Install Dependencies:
```cmd
pip install -r requirements.txt
```
3. Set Up Git LFS:
```cmd
git lfs install
git lfs track "*.pkl"
git lfs track "*.h5"
git add .gitattributes
git commit -m "Track large files with Git LFS"
```
4. Run the Streamlit App:
```cmd
Streamlit run app.py
```

### Usage
1. Upload Image: Users can upload an image of a fashion item via the web interface.
2. Receive Recommendations: The system processes the image, extracts its features, and displays similar items from the dataset.


