import streamlit as st
import torch
import librosa
import numpy as np
from torch.nn.functional import softmax
from ncps.torch import LTC
import torch.nn as nn
import matplotlib.pyplot as plt
import os

class LoadAudio(torch.nn.Module):
    def __init__(self, sr=22050, min_length=4):
        super().__init__()
        self.sr = sr
        self.min_samples = sr * min_length  # Minimum samples required for 4 seconds

    def __call__(self, file_path):
        wave, sr = librosa.load(file_path, sr=self.sr)
        
        # Ensure wave is a NumPy array (Librosa expects it)
        wave = np.array(wave, dtype=np.float32)

        # Pad if the audio length is less than 4 seconds
        if len(wave) < self.min_samples:
            padding = self.min_samples - len(wave)
            wave = np.pad(wave, (0, padding), mode="constant", constant_values=0)

        return wave, sr, 0

class ExtractLogMelSpectrogram(torch.nn.Module):
    def __init__(self, n_mels=8, n_fft=5120, hop_length=2560):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, sample):
        wave, sr, salience = sample
        mel = librosa.feature.melspectrogram(
            y=wave, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        return (logmel, sr, salience)

class NormalizeSpectrogram(torch.nn.Module):
    def __init__(self, mean=-42.265, std=13.133):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        spec, sr, salience = sample
        normalized_spec = (spec - self.mean) / self.std
        return (normalized_spec, sr, salience)

class Transpose(torch.nn.Module):
    def __call__(self, sample):
        return sample[0].T, sample[1], sample[2]

class AddSalience(torch.nn.Module):
    def __call__(self, sample):
        spec, _, salience = sample
        to_add = np.full((spec.shape[0], 1), salience)
        return torch.Tensor(np.hstack((spec, to_add)))

# Define the preprocessing pipeline
pipeline = torch.nn.Sequential(
    LoadAudio(),
    ExtractLogMelSpectrogram(n_mels=8),
    NormalizeSpectrogram(mean=-42.265, std=13.133),
    Transpose(),
    AddSalience(),
)


class Model(nn.Module):
    def __init__(self, n_features=8, model_size=32, n_classes=10, num_layers=1):
        super().__init__()

        self.ltc1 = LTC(input_size=n_features+1, units=model_size)
        self.ltc2 = LTC(input_size=model_size, units=model_size)
        self.dropout = nn.Dropout(p=0.1)
        self.dense = nn.Linear(model_size, n_classes, bias=True)

        self.num_layers = num_layers

    def forward(self, x, l):
        ltc1 = self.ltc1(x)[0]
        ltc2 = self.ltc2(ltc1)[1]
        dropoutted = self.dropout(ltc2)
        logits = self.dense(dropoutted)

        return logits

# Load the trained model
model = Model(n_features=8, model_size=32, n_classes=10, num_layers=1)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

# Function to predict the class of an unseen audio file
def predict_audio_class(audio_file_path):
    # Preprocess the audio file
    preprocessed_data = pipeline(audio_file_path)
    
    # Add batch dimension and convert to tensor
    input_tensor = preprocessed_data.unsqueeze(0)  # Shape: (1, sequence_length, n_features)
    lengths = torch.tensor([input_tensor.shape[1]])  # Sequence length
    
    # Perform inference
    with torch.no_grad():
        logits = model(input_tensor, lengths)
        probabilities = softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Get the probabilities for each class
    class_probabilities = probabilities.squeeze().tolist()
    
    # Create a dictionary of class names and their corresponding probabilities
    class_prob_dict = {class_name: prob for class_name, prob in zip(classes, class_probabilities)}
    
    # Sort the dictionary by probability in descending order
    sorted_class_prob_dict = dict(sorted(class_prob_dict.items(), key=lambda item: item[1], reverse=True))
    
    # Get the predicted class name
    predicted_class_name = classes[predicted_class]
    
    return predicted_class, predicted_class_name, sorted_class_prob_dict

# Generate waveform
def generate_waveform(file_path):
    data, sr = librosa.load(file_path, sr=None)
    time = np.linspace(0, len(data) / sr, len(data))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, data, label="Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform of Uploaded Audio")
    ax.legend(loc="upper right")
    plt.tight_layout()

    return fig

# Streamlit UI
st.title("Audio Classification using Liquid Neural Network")
st.subheader("This model predict the class of audio that are listed below:")
st.write("air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    audio_path = f"temp_{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(uploaded_file, format='audio/wav')
    waveform_fig = generate_waveform(audio_path)
    st.pyplot(waveform_fig)
    if st.button("Predict"):
        predicted_class, predicted_class_name, class_probabilities = predict_audio_class(uploaded_file)
        
        st.write(f"Predicted class index: {predicted_class}")
        st.write(f"Predicted class name: {predicted_class_name}")
        st.write("Class probabilities:")
        
        for class_name, prob in class_probabilities.items():
            st.write(f"{class_name}: {prob:.4f}")
            
        os.remove(audio_path)