import librosa
import numpy as np
import joblib
import os

from librosa.util import normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from keras.models import load_model

def extract_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)

    # Extract Mean Energy
    mean_energy = np.mean(librosa.feature.rms(y=y))

    # Extract Speech Rate
    hop_length = 128  # hop length for short-time Fourier transform
    frame_length = 2048  # frame length for short-time Fourier transform

    # Compute the zero-crossing rate (zcr) to identify speech regions
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    speech_rate = np.sum(zcr > 0.02) / (len(zcr) * hop_length / sr)

    # Extract Pause Duration
    # Compute the short-term energy
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    pause_threshold = 0.01  # energy threshold for pause detection
    pauses = np.where(energy < pause_threshold)[0]
    pause_duration = len(pauses) * hop_length / sr

    # epsilon = 1e-10
    # mean_energy += epsilon
    
    #combined_energy = speech_rate + mean_energy
    contrast_feature = speech_rate + pause_duration
    #HarmonicEnergyRate = 2 * (speech_rate * pause_duration) / (speech_rate + pause_duration)
    harmonic_mean = 2 * (speech_rate * mean_energy) / (speech_rate + mean_energy)
    pause_mean = 2 * (pause_duration * mean_energy) / (pause_duration + mean_energy)
    if pause_duration > 0:
        speech_pause_ratio = speech_rate / pause_duration
    else:
        speech_pause_ratio = 0.0
    
    # speech_to_pause_ratio = np.sum(speech_rate) / np.sum((pause_duration + epsilon))
    # energy_entropy = -np.sum(mean_energy * np.log(mean_energy))
    
    features = np.array([speech_rate, pause_duration, mean_energy, contrast_feature, harmonic_mean, pause_mean, speech_pause_ratio])

    #features = np.array([speech_rate, pause_duration, mean_energy, contrast_feature, pause_mean, speech_pause_ratio])
 
    selected_features = [0,1,2,3,4]
    
    features = features[selected_features]
 
    return features

def normalize_data(X, scaler=None):
    X_reshaped = X.reshape(X.shape[0], -1)
    if scaler is None:
        scaler = StandardScaler().fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape[0], X.shape[1], 1)  # Reshape for Conv1D input
    return X_scaled, scaler

def predict_parkinson(audio_file, model_path, scaler_path): #minmax_scaler_path, std_scaler_path
    # Load the saved model
    model = load_model(model_path)
    
    #model.summary()
    
    # Load scaler
    standard_scaler = joblib.load(scaler_path)
    
    # Extract features from the audio file
    features = extract_features(audio_file)
    features = features.reshape(1, -1)
    #print(features)
    
    # Normalize data
    #features, _ = normalize_data(features, scaler)
    #features, _, _ = normalize_data(features, min_max_scaler, standard_scaler)
    features, _ = normalize_data(features, standard_scaler)
    print(features)
    # Make prediction
    prediction = model.predict(features)
    
    # Decode the predictions
    predicted_label = np.argmax(prediction, axis=1)
    
    return predicted_label, prediction

# Path to the saved model file
model_path = "C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/SIDANG/scaler/FIX"

# Path to the saved scaler
scaler_path = 'C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/SIDANG/scaler/FIX.pkl'

new_audio_paths = [
    "C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/KODINGAN FINAL/Validasi audio/validasi/2024-05-13 13-29-59.wav",
    "C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/KODINGAN FINAL/Validasi audio/validasi/2024-05-13 13-30-56.wav",
    "C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/KODINGAN FINAL/Validasi audio/validasi/2024-05-13 13-32-02.wav",
    "C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/KODINGAN FINAL/Validasi audio/validasi/D2rriovbie49M2605161843.wav",
    "C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/KODINGAN FINAL/Validasi audio/validasi/D2RROIBVEI49M240120171903.wav",
    "C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/KODINGAN FINAL/Validasi audio/validasi/D2sncihcio44M1606161719.wav",
    "C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/KODINGAN FINAL/Validasi audio/validasi/test_simulation.wav",
    "C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/KODINGAN FINAL/Validasi audio/validasi/test_simulation2.wav",
    "C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/KODINGAN FINAL/Validasi audio/validasi/test_simulation3.wav"
]

for audio_path in new_audio_paths:
    predicted_label, prediction = predict_parkinson(audio_path, model_path, scaler_path)
    label_map = {0: "Non-Parkinson", 1: "Parkinson"}
    result = label_map[predicted_label[0]]
    print(f"Predicted Label for {os.path.basename(audio_path)}: {result}")
    print(f"Prediction Confidence: {prediction}")