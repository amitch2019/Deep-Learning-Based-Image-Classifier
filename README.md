# Deep-Learning-Based-Image-Classifier
A Python-based deep learning application for image classification, capable of recognizing 102 flower species. This project showcases a complete machine learning pipeline, including data preprocessing, CNN model implementation with transfer learning, hyperparameter optimization, and inference.  


## Project Overview

This project implements a deep learning-based image classification application that recognizes different species of flowers. It demonstrates the integration of AI into everyday applications, showcasing how deep learning models can be used in practical scenarios like smart phone apps for image recognition.

## Features

- Image classification of 102 different flower species
- Deep learning model trained on a large dataset of flower images
- Command-line interface for easy interaction
- Scalable architecture that can be adapted to other image classification tasks

## Technologies Used

- Python
- Deep Learning frameworks (e.g., TensorFlow or PyTorch)
- Convolutional Neural Networks (CNN)
- Transfer Learning techniques

## Dataset

The project uses the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from the University of Oxford, containing 102 flower categories with each class consisting of between 40 and 258 images.

## Project Structure

The project is divided into three main components:

1. **Data Preprocessing**: 
   - Loading and preprocessing the image dataset
   - Data augmentation and normalization

2. **Model Training**:
   - Implementing a CNN architecture
   - Applying transfer learning from pre-trained models
   - Training the classifier on the flower dataset
   - Hyperparameter tuning and optimization

3. **Inference**:
   - Using the trained classifier for predictions
   - Implementing a command-line interface for user interaction
     
  
## Implementation Details

The project is implemented in Python using PyTorch for deep learning. Here's an overview of the main components:

### Data Preprocessing and Loading
- Uses `torchvision.transforms` for image augmentation and normalization
- Implements separate transformation pipelines for training, validation, and testing datasets
- Utilizes `torchvision.datasets.ImageFolder` for efficient dataset management
- Creates DataLoader objects for batch processing

### Model Architecture
- Employs transfer learning using pre-trained models (default: VGG16)
- Customizes the classifier layer for the specific flower classification task
- Implements a flexible architecture that allows changing the number of hidden units

### Training Process
- Supports both GPU and CPU training
- Uses Cross Entropy Loss as the criterion
- Implements Stochastic Gradient Descent (SGD) optimizer
- Features a configurable learning rate and number of epochs

### Command-line Interface
- Utilizes `argparse` for parsing command-line arguments
- Allows customization of key hyperparameters:
  - Data directory
  - Model architecture
  - Learning rate
  - Number of hidden units
  - Number of epochs
  - GPU/CPU usage

### Training Loop
- Implements a standard training loop with validation
- Prints training loss, validation loss, and validation accuracy at regular intervals

### Key Features
- Modular design for easy expansion and modification
- Efficient use of PyTorch's capabilities for deep learning
- Robust error handling for data directory input

This implementation showcases best practices in deep learning model development, including data preprocessing, transfer learning, and hyperparameter tuning.

This implementation showcases best practices in deep learning model development, including data preprocessing, transfer learning, and hyperparameter tuning.

## Installation

```bash
git clone https://github.com/amitch2019/flower-classifier.git
cd flower-classifier
pip install -r requirements.txt
```

## Sample Output

<img width="1090" alt="image" src="https://github.com/user-attachments/assets/688101af-58be-4eb9-8ecf-18ce894ac8c4">

Contact

For any questions or inquiries, please contact me at [chaubey.amit@gmail.com].

