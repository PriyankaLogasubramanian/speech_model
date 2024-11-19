import torch
import torchaudio
import numpy as np
import sys

# Add the path to the StreamVC directory (make sure to use the correct path to where StreamVC is located)
sys.path.append('/home/pl1032847/speech_to_speech/hubert/hubert/StreamVC')

from streamvc.f0 import F0Estimator
from streamvc.energy import EnergyEstimator

# Load the pre-trained HuBERT-Soft model
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).eval()

# Define the input and output paths
input_audio_path = "audio3.wav"  # Your input audio file
output_units_path = "audio3_units.npy"  # The output file for extracted speech units

# Load and preprocess the audio file
wav, sr = torchaudio.load(input_audio_path)
wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
wav = wav.unsqueeze(0)  # Add batch dimension

# Initialize the F0 and Energy Estimators
f0_estimator = F0Estimator(sample_rate=16000, frame_length_ms=20)
energy_estimator = EnergyEstimator(sample_rate=16000, frame_length_ms=20)

# Extract the speech units
with torch.no_grad():
    units = hubert.units(wav)

# Estimate the F0 and energy
f0 = f0_estimator(wav)  # Shape: [1, 1, 130, 9]
energy = energy_estimator(wav)  # Shape: [1, 1, 130]

# Print the extracted features
print(f"Shape of extracted HuBERT units: {units.shape}")
print(f"First few extracted units: {units[0, :5]}")  # Print first 5 values of the first feature vector
print(f"F0: {f0[0, :5]}")
print(f"Energy: {energy[0, :5]}")

# Reshape f0 and energy to match the units tensor for concatenation

# f0 has shape [1, 1, 130, 9], we need to squeeze the extra dimension and match the channels with units
f0 = f0.squeeze(1)  # Shape
