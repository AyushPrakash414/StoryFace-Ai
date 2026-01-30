import numpy as np
import cv2
import base64
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import model_from_json

app = FastAPI()

# -------------------------
# Load model at startup
# -------------------------
with open("emotiondetector.json", "r") as f:
    model_json = f.read()

model = model_from_json(model_json)
model.load_weights("emotiondetector.weights.h5")

print("✅ Model loaded successfully")

# Emotion labels (edit if yours are different)
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# -------------------------
# Helper: preprocess image
# -------------------------
def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (48, 48))  # typical FER size
    img = img / 255.0
    img = img.reshape(1, 48, 48, 1)

    return img


# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    img = preprocess_image(image_bytes)

    preds = model.predict(img)
    emotion_index = np.argmax(preds)

    emotion = EMOTIONS[emotion_index]

    return {"emotion": emotion}
