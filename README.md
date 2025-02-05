# Brain Tumor Detection using Deep Learning

## Overview
This project focuses on brain tumor detection using deep learning techniques. The model processes MRI scan images to classify tumor presence. The implementation leverages Python and popular machine learning libraries for training and evaluation.

## Features
- **Data Preprocessing:** Image resizing, normalization, and augmentation.
- **Model Training:** Utilizes a convolutional neural network (CNN) for feature extraction and classification.
- **Evaluation:** Performance metrics like accuracy, precision, recall, and confusion matrix.
- **Visualization:** Includes model training history, sample predictions, and saliency maps.

## Installation
To set up the environment, install the required dependencies:
```bash
pip install tensorflow keras numpy matplotlib pandas scikit-learn opencv-python
```

## Dataset
The dataset consists of MRI images labeled with tumor presence. Ensure the dataset is structured correctly before training.

## Classification Methodology
The classification process follows these steps:
1. **Data Preprocessing**
   - Images are resized to a fixed shape (e.g., 224x224 pixels).
   - Normalization is applied to scale pixel values between 0 and 1.
   - Augmentation techniques like rotation and flipping are used to improve generalization.

2. **Model Architecture**
   - A Convolutional Neural Network (CNN) is implemented using TensorFlow/Keras.
   - Layers include convolutional, max-pooling, dropout, and fully connected layers.
   - The final layer uses a softmax or sigmoid activation function for classification.

3. **Training and Evaluation**
   - The model is trained using categorical cross-entropy loss and Adam optimizer.
   - Performance is measured using accuracy, precision, recall, and a confusion matrix.
   - The trained model is saved for future inference.

4. **Prediction**
   - An input MRI image is preprocessed and passed through the trained model.
   - The output probability determines whether the tumor is present or absent.

## Usage
Run the following script to train the model:
```bash
python train.py
```
To test the model:
```bash
python predict.py --image path/to/image.jpg
```

## Results
_Add output images and performance metrics here._
![Screenshot 2025-02-05 at 7 59 10 PM](https://github.com/user-attachments/assets/fab632f0-08d6-4b4f-b816-352fe373d692)
![Screenshot 2025-02-05 at 7 59 57 PM](https://github.com/user-attachments/assets/dae9258b-cc21-4c33-86f2-3b6a570cf8b7)


## Conclusion
This model effectively detects brain tumors in MRI images. Further improvements can be made by using more complex architectures and larger datasets.

## Acknowledgments
This project is inspired by medical imaging advancements and deep learning research in healthcare.

