# Brain Tumor Classification

This project implements a convolutional neural network (CNN) to classify brain tumor MRI scans into four categories: glioma, meningioma, pituitary tumor, and healthy brain. The model is trained using the dataset available from [Kaggle](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans/data) and utilizes PyTorch for deep learning.

## Project Structure

The project is organized as follows:
```bash
├── data_download.sh # Shell script to download and unzip the dataset 
├── data # Directory containing subfolders for each class (glioma, healthy, meningioma, pituitary) 
│ ├── glioma 
│ ├── healthy 
│ ├── meningioma 
│ └── pituitary 
├── data_preprocessing.py # Preprocessing script to load and transform the dataset 
├── model.py # CNN model definition 
├── train.py # Script to train and validate the model 
├── main.py # Main script to tie everything together (model training, saving) 
└── brain_tumor_cnn.pth # Trained model weights (saved after training)
```

## Dataset

The dataset used in this project consists of MRI scans of the brain, categorized into the following classes:

- **Glioma**
- **Healthy**
- **Meningioma**
- **Pituitary**

You can download the dataset by running the `data_download.sh` script, which fetches and extracts the MRI scan images from Kaggle into the appropriate directories.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/wrenparismoe/TumorClassification.git
   cd TumorClassification
   ```
2. Ensure you have Python 3.x and the necessary dependencies. You can install them using `pip`:
   `pip install -r requirements.txt`
3. Download the dataset using bash:
   ```bash
   chmod +x data_download.sh
   ./data_download.sh
   ```

## Usage

### Step 1: Data Preprocessing
The `data_preprocessing.py` script loads the dataset and prepares it for training, validation, and testing. The `get_data_loaders` function applies transformations (resizing, normalizing, etc.) and splits the data into training, validation, and test sets.

### Step 2: Training the Model
The model is defined in `model.py` as `BrainTumorCNN`. This CNN consists of two convolutional layers followed by fully connected layers. The model is trained using the `train.py` script, which handles the forward and backward passes, loss calculation, and optimizer updates.

To train the model, simply run the `main.py` script within your python environment:
```
python main.py
```

This will:
- Load the data
- Initialize the model
- Train the model for the specified number of epochs
- Save the trained model to `brain_tumor_cnn.pth`

## Results

Sample results:
```bash
Epoch 1/10, Train Loss: 0.8158, Validation Loss: 0.3039, Validation Accuracy: 88.03%
Epoch 2/10, Train Loss: 0.2323, Validation Loss: 0.2520, Validation Accuracy: 91.36%
Epoch 3/10, Train Loss: 0.1011, Validation Loss: 0.2152, Validation Accuracy: 92.69%
Epoch 4/10, Train Loss: 0.0568, Validation Loss: 0.2199, Validation Accuracy: 92.97%
Epoch 5/10, Train Loss: 0.0392, Validation Loss: 0.2792, Validation Accuracy: 93.45%
Epoch 6/10, Train Loss: 0.0366, Validation Loss: 0.2440, Validation Accuracy: 93.26%
Epoch 7/10, Train Loss: 0.0024, Validation Loss: 0.2584, Validation Accuracy: 94.02%
Epoch 8/10, Train Loss: 0.0003, Validation Loss: 0.2563, Validation Accuracy: 94.11%
Epoch 9/10, Train Loss: 0.0001, Validation Loss: 0.2706, Validation Accuracy: 94.21%
Epoch 10/10, Train Loss: 0.0001, Validation Loss: 0.2835, Validation Accuracy: 94.11%
```






