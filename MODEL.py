import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Flatten, LeakyReLU, Conv1D, MaxPooling1D, BatchNormalization, SpatialDropout1D, GlobalAveragePooling1D, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Nadam
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
 
def augment_data(X, Y):
    augmented_X, augmented_Y = [], []
    for x, y in zip(X, Y):
        augmented_X.append(x)
        augmented_Y.append(y)
        noise = np.random.normal(0, 0.005, x.shape)
        augmented_X.append(x + noise)
        augmented_Y.append(y)
    return np.array(augmented_X), np.array(augmented_Y)

def normalize_data(X, scaler=None):
    X_reshaped = X.reshape(X.shape[0], -1)
    if scaler is None:
        scaler = StandardScaler().fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped)
    return X_scaled.reshape(X.shape[0], X.shape[1], 1), scaler

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

def load_data():
    fitur_normal = np.load('C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/SIDANG/7FITUR_NORMAL.npy')
    label_normal = np.load('C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/SIDANG/LABEL_NORMAL.npy')

    fitur_parkinson = np.load('C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/SIDANG/7FITUR_PARKIN.npy')
    label_parkinson = np.load('C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/SIDANG/LABEL_PARKIN.npy')
    
    selected_features = [0,1,2,3,4]
    print(fitur_normal.shape)
    
    fitur_normal = fitur_normal[:, selected_features]
    fitur_parkinson = fitur_parkinson[:, selected_features]
    print(fitur_normal.shape)
    
    X = np.vstack((fitur_normal, fitur_parkinson))
    y = np.hstack((label_normal, label_parkinson))

    return X, y

def build_conv1d_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(2, strides=1),
        SpatialDropout1D(0.7),
        #Dropout(0.6),
        
        Conv1D(128, 3, padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(2, strides=1),
        SpatialDropout1D(0.7),
        Dropout(0.6),
        
        Conv1D(256, 3, padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        #SpatialDropout1D(0.6),
        Dropout(0.6),
        
        Conv1D(256, 3, padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(2, strides=1),
        #SpatialDropout1D(0.6),
        Dropout(0.6),
        
        #GlobalAveragePooling1D(),
        Flatten(),
        
        Dense(512, kernel_regularizer=l2(0.01)),
        LeakyReLU(alpha=0.1),
        Dropout(0.6),
        
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=Nadam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_and_train(X, y, epochs=250):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = to_categorical(y_encoded)
    
    X, scaler = normalize_data(X.reshape((X.shape[0], X.shape[1], 1)))
    joblib.dump(scaler, 'SIDANG/scaler/FIX.pkl')

    X_augmented, y_augmented = augment_data(X, y_onehot)

    X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented)
    
    model = build_conv1d_model((X_train.shape[1], 1))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint("C:/Users/Aqsath/Downloads/PROPOSAL 2024/KODINGAN SKRIPSI 2024/SIDANG/modelss", monitor='val_accuracy', save_best_only=True)

    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64,
                        callbacks=[early_stopping, reduce_lr, model_checkpoint])

    return model, history

def plot_confusion_matrix_and_report(model, X, y):
    X_scaled, _ = normalize_data(X.reshape((X.shape[0], X.shape[1], 1)))
    y_pred = np.argmax(model.predict(X_scaled), axis=1)
    le = LabelEncoder()
    y_true = le.fit_transform(y)
    cm = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot()

def generate_classification_report(model, X, y):
    X_scaled, _ = normalize_data(X.reshape((X.shape[0], X.shape[1], 1)))
    y_pred = np.argmax(model.predict(X_scaled), axis=1)
    le = LabelEncoder()
    y_true = le.fit_transform(y)
    classes = ['Non-Parkinson', 'Parkinson']
    report = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:")
    print(report)
    return report

def generate_f1_score(model, X, y):
    X_scaled, _ = normalize_data(X.reshape((X.shape[0], X.shape[1], 1)))
    y_pred = np.argmax(model.predict(X_scaled), axis=1)
    le = LabelEncoder()
    y_true = le.fit_transform(y)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"F1-score: {f1}")
    return f1

X, y = load_data()

trained_model, history = preprocess_and_train(X, y)

y_onehot = to_categorical(LabelEncoder().fit_transform(y))

X_scaled, _ = normalize_data(X.reshape((X.shape[0], X.shape[1], 1)))

print(X_scaled.shape)

plot_confusion_matrix_and_report(trained_model, X, y)

classification_report = generate_classification_report(trained_model, X, y)

f1_score = generate_f1_score(trained_model, X, y)

plot_history(history)

evaluation = trained_model.evaluate(X_scaled, y_onehot)
print(f"Final Loss: {evaluation[0]}")
print(f"Final Accuracy: {evaluation[1]}")
