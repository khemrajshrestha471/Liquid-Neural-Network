# Audio Classification using Liquid Neural Network

## Overview
This project implements an audio classification using a **Liquid Neural Network (LNN)**. It processes uploaded audio files, extracts log mel spectrograms, and classifies them into one of ten predefined classes from `UrbanSound8K` dataset using a trained model.

## Features
- Upload `.wav` or `.mp3` audio files.
- Display waveform visualization.
- Perform classification using a Liquid Neural Network.
- Show class probabilities.

## Installation

### Prerequisites
Ensure you have Python installed (recommended: Python 3.8 or later).

### Clone the Repository
```sh
git clone https://github.com/khemrajshrestha471/Liquid-Neural-Network.git
cd Liquid-Neural-Network
```

### Create a Virtual Environment (Optional but Recommended)
```sh
python -m venv lnn
source lnn/Scripts/activate  # On Windows
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Model and Data Preparation
### Download the Pretrained Model
Ensure that the trained model file `model.pth` is present in the project directory. If not, download it from your provided link or train a new model.

## Usage
Run the Streamlit application:
```sh
streamlit run app.py
```

### Workflow
1. Upload an audio file (`.wav` or `.mp3`).
2. View the waveform of the uploaded audio.
3. Click the "Predict" button to classify the audio.
4. View the predicted class and probabilities of all classes.

## Project Structure
```
├── app.py                  # Main Streamlit application
├── model.pth               # Trained Liquid Neural Network model
├── Sample Audio            # Folder contains sample audio to test
├── requirements.txt        # Required dependencies
├── README.md               # Documentation
```

## Dependencies
- `streamlit`
- `torch`
- `librosa`
- `numpy`
- `matplotlib`
- `ncps`

Install all dependencies using:
```sh
pip install -r requirements.txt
```

## Troubleshooting
- If `model.pth` is missing, ensure you have the correct path and file.
- If dependencies fail, try installing them individually.
- For any issues, refer to the documentation or raise an issue in the repository.