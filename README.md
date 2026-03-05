# DEEPDENT – AI Dental Radiograph Analysis

DEEPDENT is an AI-based system that analyzes panoramic dental X-rays (OPG) to detect dental conditions such as cavities, fillings, implants, impacted teeth, and normal teeth using deep learning.

## Features
- Automatic analysis of dental radiographs
- Deep learning model for dental condition classification
- Tooth region detection using image processing
- Prediction visualization on X-ray images

## Model Used
We used **ResNet (Residual Neural Network)** architecture for image classification.  
ResNet helps in extracting deep features from dental radiographs and improves classification performance.

## Project Pipeline
OPG X-ray Image  
↓  
Tooth Detection (OpenCV)  
↓  
Tooth Cropping  
↓  
Deep Learning Model (**ResNet**)  
↓  
Dental Condition Classification  
↓  
Annotated Output Image  

## Technologies Used
- Python
- PyTorch
- OpenCV
- Deep Learning (ResNet)

## Files
- `dental_dl.py` – Model training code
- `predict_opg.py` – Prediction pipeline
- `tooth_detection.py` – Tooth detection logic
- `dental_model.pth` – Trained model weights

## Future Improvements
- Precise tooth segmentation using U-Net
- Full FDI tooth numbering (1–32)
- Improved multi-tooth detection
