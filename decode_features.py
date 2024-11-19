import sys
# Add the path to the StreamVC directory (make sure to use the correct path to where StreamVC is located)
sys.path.append('/home/pl1032847/speech_to_speech/hubert/hubert/StreamVC')

import torch
import numpy as np
from SoundStream.net import Decoder  # Import from your local SoundStream directory

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load preprocessed enhanced speech units from .npy file
enhanced_units = np.load("audio2_units.npy")
enhanced_units = torch.tensor(enhanced_units, dtype=torch.float32).to(device)

# Check the shape of the enhanced units tensor
print(f"Shape of enhanced units: {enhanced_units.shape}")

# Since the shape is [256, 516], adjust the reshaping according to your model input
# Assuming your decoder expects [batch_size, channels, time_steps, features]
# Here, you would want to add a channel dimension and adjust the dimensions accordingly

# Reshape to match the expected format for the decoder
enhanced_units = enhanced_units.unsqueeze(0)  # Add batch dimension: [1, 256, 516]

# Ensure the shape is compatible with your decoder
print(f"Reformatted input shape: {enhanced_units.shape}")

# Initialize SoundStream decoder (adjust 'C' and 'D' according to your model configuration)
decoder = Decoder(C=1, D=256)  # Adjust parameters to fit the expected dimensions of the features
decoder.to(device)
decoder.eval()

# Run the decoder to convert HuBERT units to audio waveform
with torch.no_grad():
    audio_waveform = decoder(enhanced_units)  # This should be of shape [1, 256, 130]

# Check the output shape and first few values of the audio waveform
print(f"Audio waveform shape before squeeze: {audio_waveform.shape}")
print(f"Audio waveform (first few samples): {audio_waveform[0, 0, :10]}")  # Print first 10 samples

# Reshape the output to ensure it's 2D (e.g., [1, num_frames] for mono audio)
audio_waveform = audio_waveform.squeeze(0)  # Remove the batch dimension (if it's 1)

# Save or process audio_waveform further as needed, e.g., save as a .wav file
import torchaudio
output_path = "decoded_audio.wav"
torchaudio.save(output_path, audio_waveform.cpu(), sample_rate=16000)
print(f"Audio successfully saved to {output_path}")
