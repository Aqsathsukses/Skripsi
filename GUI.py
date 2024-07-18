import tkinter as tk
import pyaudio
import wave
import time
import numpy as np
import joblib
import librosa

from keras.models import load_model
from tkinter import messagebox
from threading import Thread
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from librosa.util import normalize

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
    if scaler is None:
        scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled, scaler

# Fungsi ekstraksi fitur dan prediksi tidak berubah, hanya bagian GUI yang diubah warnanya

class ParkinsonDetectionApp:
    
    def __init__(self, master):
        self.master = master
        master.title("Sistem Deteksi Parkinson")
        master.geometry("650x420")
        master.configure(bg="#2C3E50")  # Warna latar belakang utama
        
        self.page1 = Page1(master, self)
        self.page2 = Page2(master, self)

        self.show_page1()

    def show_page1(self):
        self.page2.hide()
        self.page1.show()

    def show_page2(self, prediction_result, confidence):
        self.page1.hide()
        self.page2.show(prediction_result, confidence)

class Page1:
    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.frame = tk.Frame(master, bg="#27AE60")  # Warna latar belakang frame (hijau)
        self.frame.pack(expand=True, fill='both')

        self.label = tk.Label(self.frame, text="Rekam suara anda", font=("Helvetica", 28, "bold"), bg="#27AE60", fg="#ECF0F1")  # Warna teks dan latar belakang label
        self.record_button = tk.Button(self.frame, text="Mulai Rekam", font=("Helvetica", 20, "bold"), bg="#F39C12", fg="#ECF0F1", command=self.record_audio, borderwidth=2, relief="raised")  # Warna tombol (kuning)

        self.label.pack(pady=20)
        self.record_button.pack(pady=50)

    def show(self):
        self.frame.pack(expand=True, fill='both')

    def hide(self):
        self.frame.pack_forget()

    def record_audio(self):
        for i in range(3, 0, -1):
            self.label.config(text=f"Perekaman dimulai dalam {i}")
            self.master.update()
            time.sleep(1)

        self.label.config(text="Perekaman suara...")
        self.master.update()

        recording_thread = Thread(target=self.simulate_recording)
        recording_thread.start()

    def simulate_recording(self):
        try:
            duration = 4
            #file_name = "/home/raspi/Documents/aksat/audio1.wav"
            file_name = "C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/SIDANG/audio3.wav"

            p = pyaudio.PyAudio()

            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=22050,
                            input=True,
                            frames_per_buffer=1024)

            frames = []

            for i in range(0, int(44100 / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            with wave.open(file_name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(frames))

            self.label.config(text="Rekaman suara anda berhasil")
            self.master.update()

            # Path to the saved model file
            model_path = "C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/SIDANG/scaler/FIX"

            # Path to the saved scaler
            scaler_path = 'C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/SIDANG/scaler/FIX.pkl'

            predicted_label, prediction = predict_new_audio(file_name, model_path, scaler_path)
            label_map = {0: "Parkinson", 1: "Non-Parkinson"}
            result = label_map[predicted_label[0]]
            confidence = prediction[0][predicted_label[0]]

            self.app.show_page2(result, confidence)

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat merekam audio: {str(e)}")

class Page2:
    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.frame = tk.Frame(master, bg="#27AE60")  # Warna latar belakang frame (hijau)
        self.frame.pack(expand=True, fill='both')

        self.label = tk.Label(self.frame, text="Hasil Prediksi:", font=("Helvetica", 28, "bold"), bg="#27AE60", fg="#ECF0F1")  # Warna teks dan latar belakang label
        self.prediction_label = tk.Label(self.frame, text="", font=("Helvetica", 24), bg="#27AE60", fg="#ECF0F1")  # Warna teks dan latar belakang label
        self.confidence_label = tk.Label(self.frame, text="", font=("Helvetica", 20), bg="#27AE60", fg="#ECF0F1")  # Warna teks dan latar belakang label

        self.back_button = tk.Button(self.frame, text="Back", font=("Helvetica", 20, "bold"), bg="#F39C12", fg="#ECF0F1", command=app.show_page1, borderwidth=2, relief="raised")  # Warna tombol (kuning)

        self.label.pack(pady=20)
        self.prediction_label.pack(pady=10)
        self.confidence_label.pack(pady=10)
        self.back_button.pack(pady=30)

    def show(self, prediction_result, confidence):
        self.prediction_label.config(text=f"{prediction_result}")
        self.confidence_label.config(text=f"Confidence: {confidence * 100:.2f}%")
        self.frame.pack(expand=True, fill='both')

    def hide(self):
        self.frame.pack_forget()

def predict_new_audio(audio_path, model_path, scaler_path):
    # Load model
    model = load_model(model_path)
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Ekstraksi fitur dari file audio baru
    features = extract_features(audio_path)
    
    # Normalisasi data
    features = features.reshape(1, -1)
    features, _ = normalize_data(features, scaler)
    
    # Prediksi
    prediction = model.predict(features)
    
    # Konversi prediksi ke label
    predicted_label = np.argmax(prediction, axis=1)
    
    return predicted_label, prediction

if __name__ == "__main__":
    root = tk.Tk()
    app = ParkinsonDetectionApp(root)
    root.mainloop()
