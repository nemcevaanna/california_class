import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras

from fastapi import FastAPI
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
from pydantic import BaseModel


app = FastAPI()

# Загрузка модели и скейлера
model_path = "best_keras_model.h5"
model = keras.models.load_model(model_path, compile=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post('/predict')
async def predict(input_data: InputData):
    # Преобразуем входные данные в numpy массив
    X_new = np.array([[
        input_data.MedInc,
        input_data.HouseAge,
        input_data.AveRooms,
        input_data.AveBedrms,
        input_data.Population,
        input_data.AveOccup,
        input_data.Latitude,
        input_data.Longitude
    ]])

    # Применяем скейлер
    X_scaled = scaler.transform(X_new)
    
    # Получаем предсказание
    prediction = model.predict(X_scaled)
    
    return {"prediction": float(prediction[0][0])}