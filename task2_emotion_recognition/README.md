# Task 2: Emotion Recognition from Speech

A machine learning project that classifies human emotion (Happy, Sad, Angry, Neutral, Fearful, Disgust) directly from audio speech recordings. It uses `librosa` to extract 222 acoustic features (MFCCs, Chroma, Mel Spectrogram, etc.) and trains a deep Bi-LSTM neural network.

## Features
- **Acoustic Feature Extraction:** Uses `librosa` to extract MFCCs, Mel Spectrogram, Zero Crossing Rate, RMS, and Chroma features.
- **Deep Learning Models:** Trains and compares Convolutional Neural Networks (CNN), Bidirectional LSTMs, and Multi-Layer Perceptrons (MLP).
- **Audio Preprocessing Backend:** Includes `PyAV` to seamlessly decode `.webm` web-browser recordings into `.wav` formats in real-time.
- **Live Recording Dashboard:** An interactive HTML/JS dashboard that records your voice directly from the browser and sends it to the API for real-time inference.

## Project Structure
```
task2_emotion_recognition/
├── outputs/               # Saved models (Bi-LSTM.keras), scaler.pkl, and graphs
├── src/
│   ├── data_loader.py     # RAVDESS dataset parser and synthetic fallback
│   ├── feature_extractor.py # Audio feature extraction (librosa)
│   ├── models.py          # Keras (CNN/LSTM) and Scikit-learn (MLP) models
│   └── visualizer.py      # Plotting training curves and confusion matrices
├── dashboard/
│   ├── index.html         # Frontend Dashboard UI with Microphone recorder
│   ├── app.js             # Frontend logic, Web Audio API, and FastAPI integration
│   └── style.css          # UI Styling
├── main.py                # Main ML pipeline (Train, evaluate, save)
└── server.py              # FastAPI server for real-time microphone predictions
```

## Step-by-Step Setup & Execution

### 1. Train the Model
Before running the live server, you must train the Bi-LSTM model so that it saves the required weights and data scalers.
```bash
cd task2_emotion_recognition
python main.py
```
*This will create the `outputs/` folder containing `Bi-LSTM.keras` and `scaler.pkl`.*

### 2. Start the Backend API
Run the FastAPI server. This server will handle real-time `.webm` microphone uploads, extract the acoustic features, and predict the emotion.
```bash
python server.py
```
*The server will start at `http://localhost:8002`.*

### 3. Open the Dashboard & Record
Double-click `dashboard/index.html` to open it in your web browser. 
- Scroll down to the **Live Emotion Prediction** section.
- Click **Start Recording** and say a sentence into your microphone with some emotion.
- Click **Stop & Predict**. The API will instantly decode your voice and the Bi-LSTM neural network will output its prediction!
