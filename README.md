# CV Project: Fruits Classification

## Project Overview
This project uses deep learning (MobileNetV2, YOLOv8, and GANs) to classify, detect, and generate fruit images from the Fruits-360 dataset. It covers data preprocessing, model training, evaluation, and result visualization.

## Dataset
- **Source:** [Fruits-360 on Kaggle](https://www.kaggle.com/datasets/moltean/fruits)
- **Description:** Contains over 70,000 images of fruits in various classes, split into training and test sets.

## Requirements
- Python 3.7+
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow
- opencv-python
- ultralytics (for YOLO)
- torch, torchvision (for GANs)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## How to Run the Code
1. **Download the dataset** from Kaggle and extract it to the appropriate directory (see code for expected paths).
2. **Run the MobileNetV2 notebook** for classification:
   - Open `Classification/mobilenetv2.ipynb` and run all cells.
3. **Run the YOLO notebook** for detection:
   - Open `YOLO/Fruits_YOLO.ipynb` and run all cells.
4. **Run the GAN notebook** for image generation:
   - Open `GANS/Image_Generation_with_GAN.ipynb` and run all cells.

## Usage
- The code will automatically split the dataset into train/val/test sets.
- Training and validation accuracy/loss plots are generated.
- The best model is saved as `mobilenet_best.h5`.
- YOLOv8 is used for object detection and visualization.
- GANs are used to generate new fruit images.

## Results
- **MobileNetV2 Test Accuracy:** ~99% (see notebook for exact value)
- **YOLOv8 Detection:** Successfully detects and classifies fruits in images.
- **GANs:** Successfully generates realistic fruit images after training (see GAN notebook for samples).

## Report
See `report.pdf` for a detailed summary of methods, results, and observations, including GANs.
