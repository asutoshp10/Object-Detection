# Object-Detection
# Electronic Component Object Detection 🛠️

This project implements an object detection system for five types of electronic components: **Capacitor**, **Transistor**, **IC**, **Resistor**, and **Data Uino**. It leverages a pre-trained **VGG16** model (with `include_top=False`) and a custom dense layer for classification. OpenCV is used for detecting patches in images, which are then classified by the trained model.

## Features

- **Classification of Components**: Identifies and classifies electronic components into five classes.
- **Patch Detection**: Utilizes OpenCV to detect regions of interest (patches) in an image.
- **Pre-trained Model**: Uses the VGG16 model with transfer learning for feature extraction.
- **Custom Layers**: A dense layer added to the VGG16 backbone for accurate classification.
- **Real-time Detection**: Capable of processing images or video streams in real-time.

## Workflow

1. **Patch Detection**:
   - OpenCV detects potential regions containing electronic components in the input image.
2. **Feature Extraction**:
   - The detected patches are resized to fit the input shape required by VGG16.
   - Features are extracted using the pre-trained VGG16 model (with `include_top=False`).
3. **Classification**:
   - Extracted features are passed through the custom dense layer for classification into one of the five classes.
4. **Results**:
   - Output includes the detected component class and bounding box (if applicable).

## Requirements

To run this project, install the following dependencies:

- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib (optional, for visualization)

Install dependencies using pip:

```bash
pip install tensorflow opencv-python numpy matplotlib
