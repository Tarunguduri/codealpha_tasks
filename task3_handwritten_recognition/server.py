import os
import sys
import base64
import io
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
import tensorflow as tf

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models import build_deep_cnn

app = FastAPI(title="Handwritten Character Recognition API")

# Enable CORS for the dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
metadata = {
    "n_classes": 10,
    "class_names": [str(i) for i in range(10)]
}

@app.on_event("startup")
async def load_model():
    global model
    try:
        print("[Server] Initializing model...")
        model = build_deep_cnn(input_shape=(28, 28, 1), n_classes=metadata["n_classes"])
        weights_path = os.path.join(PROJECT_ROOT, "outputs", "deep_cnn.weights.h5")
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"[Server] Successfully loaded weights from {weights_path}")
        else:
            print(f"[Server] WARNING: Weights file not found at {weights_path}. Model will use random weights.")
    except Exception as e:
        print(f"[Server] Error loading model: {e}")

class PredictRequest(BaseModel):
    image_base64: str

@app.post("/predict")
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    try:
        # Decode base64 image
        # The string might look like "data:image/png;base64,iVBORw0KGgo..."
        if "," in request.image_base64:
            base64_str = request.image_base64.split(",")[1]
        else:
            base64_str = request.image_base64
            
        image_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image_data))
        
        # Preprocess the image
        # 1. Convert to grayscale
        img = img.convert('L')
        
        # 2. Thresholding: The canvas has a dark background (#0d0d20) and cyan stroke (#00d4ff).
        # When converted to grayscale, background is ~15, stroke is ~153.
        # We need the stroke to be white (255) and background to be black (0).
        # We map everything > 50 to 255, and everything else to 0.
        img = img.point(lambda p: 255 if p > 50 else 0)
        
        # 3. Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 4. Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # 5. Reshape for the model: (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # 6. Predict
        prediction = model.predict(img_array, verbose=0)
        class_idx = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        predicted_char = metadata["class_names"][class_idx]
        
        return {
            "prediction": predicted_char,
            "confidence": confidence,
            "probabilities": {str(metadata["class_names"][i]): float(p) for i, p in enumerate(prediction[0])}
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8003, reload=True)
