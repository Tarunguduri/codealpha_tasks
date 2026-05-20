import os
import sys
import tempfile
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_extractor import extract_features_from_file
from src.data_loader import EMOTION_TO_LABEL, SELECTED_EMOTIONS

# Map index back to emotion name based on the sorted SELECTED_EMOTIONS
emotion_names = sorted(SELECTED_EMOTIONS)

app = FastAPI(title="Emotion Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None

@app.on_event("startup")
async def load_assets():
    global model, scaler
    try:
        model_path = os.path.join(PROJECT_ROOT, "outputs", "Bi-LSTM.keras")
        scaler_path = os.path.join(PROJECT_ROOT, "outputs", "scaler.pkl")
        
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"[Server] Successfully loaded model from {model_path}")
        else:
            print(f"[Server] WARNING: Model not found at {model_path}")
            
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"[Server] Successfully loaded scaler from {scaler_path}")
        else:
            print(f"[Server] WARNING: Scaler not found at {scaler_path}")
            
    except Exception as e:
        print(f"[Server] Error loading assets: {e}")

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    # Save uploaded file to temp file
    temp_fd, temp_webm = tempfile.mkstemp(suffix=".webm")
    temp_wav = temp_webm.replace(".webm", ".wav")
    try:
        with os.fdopen(temp_fd, "wb") as f:
            content = await audio.read()
            f.write(content)
            
        # Convert webm to wav using PyAV
        import av
        container = av.open(temp_webm)
        output = av.open(temp_wav, 'w', 'wav')
        if not container.streams.audio:
            raise HTTPException(status_code=400, detail="No audio stream found")
            
        in_stream = container.streams.audio[0]
        out_stream = output.add_stream('pcm_s16le', rate=in_stream.rate)
        
        for frame in container.decode(in_stream):
            for packet in out_stream.encode(frame):
                output.mux(packet)
                
        for packet in out_stream.encode(None):
            output.mux(packet)
            
        output.close()
        container.close()
            
        # Extract features
        features = extract_features_from_file(temp_wav)
        if features is None:
            raise HTTPException(status_code=400, detail="Failed to extract features from audio")
            
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Reshape for Bi-LSTM: (batch, time_steps, features) -> (1, 1, 222)
        X_lstm = features_scaled.reshape((1, 1, features_scaled.shape[1]))
        
        # Predict
        prediction = model.predict(X_lstm, verbose=0)
        class_idx = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        predicted_emotion = emotion_names[class_idx]
        
        return {
            "prediction": predicted_emotion,
            "confidence": confidence,
            "probabilities": {name: float(p) for name, p in zip(emotion_names, prediction[0])}
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_webm):
            os.remove(temp_webm)
        if 'temp_wav' in locals() and os.path.exists(temp_wav):
            os.remove(temp_wav)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=True)
