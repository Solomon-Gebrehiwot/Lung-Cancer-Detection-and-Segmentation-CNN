# Lung Cancer Detection using CNN

This project implements a Convolutional Neural Network (CNN) for detecting lung cancer from medical imaging data.

## Project Structure

```
lung-cancer-detection-cnn/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/
│   └── saved_models/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
└── README.md
```

## Features

- Data preprocessing pipeline for medical images
- CNN model implementation using TensorFlow/Keras
- Training and evaluation scripts
- Visualization tools for results

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset in the `data/` directory
2. Run training:
   ```bash
   python src/train.py --data_dir data/raw
   ```
3. Evaluate model:
   ```bash
   python src/evaluate.py --model_path models/saved_models/best_model.h5 --data_dir data/raw
   ```
   
   Or simulate evaluation:
   ```bash
   python src/simple_evaluate.py --model_path models/saved_models/best_model.h5 --data_dir data/raw
   ```
   
   Or run full training simulation:
   ```bash
   python src/simple_train.py --data_dir data/raw --epochs 10
   ```

4. Run Streamlit dashboard:
   ```bash
   streamlit run src/dashboard.py
   ```
   
   Or run the simplified version:
      ```bash
      streamlit run src/simple_dashboard.py
      ```
      
      Or run the enhanced version with full medical image analysis tools:
      ```bash
      streamlit run src/enhanced_dashboard.py
      ```
      
      Login credentials for the dashboard:
      - Username: admin
      - Password: password123
      
      Other available users:
      - Username: doctor
      - Password: medpass456
      
      - Username: researcher
         - Password: research789
      
      ## Cancer Cell Detection and Segmentation
      
      The enhanced dashboard includes advanced medical image analysis tools:
      
      1. **Cancer Cell Detection**: Uses CNN-based object detection to identify and localize cancer cells in medical images
      2. **Cancer Cell Segmentation**: Provides pixel-level segmentation of cancerous regions using deep learning models
      
      These features simulate the functionality that would be available with a trained model. In a production environment, you would need to:
      1. Train the CNN models using the provided training scripts
      2. Save the trained models in the `models/saved_models/` directory
      3. Update the dashboard code to load and use the actual trained models
      
      Other available users:
      - Username: doctor
      - Password: medpass456
      
      - Username: researcher
      - Password: research789

## Running the Application

If you encounter any issues with dependencies, you can install them individually:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn pandas seaborn jupyter streamlit pillow
```

## Model Architecture

The CNN model consists of multiple convolutional layers followed by max-pooling layers, dropout layers for regularization, and dense layers for classification.

## Streamlit Dashboard

The project includes a Streamlit dashboard for visualizing model performance and making predictions on new images.

### Features:
- Model performance visualization (training history, confusion matrix)
- Image upload and prediction
- Detailed prediction probabilities

### Running the Dashboard:
```bash
streamlit run src/dashboard.py
```