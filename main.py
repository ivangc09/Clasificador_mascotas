from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
import shutil
import os
from utils import preprocess_image
import json

app = FastAPI()

# Configurar CORS si lo usas con frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # !Cambiarlo cuando lo levante
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo
model = tf.keras.models.load_model("modelo_final.h5")

with open("clases.json", "r") as f:
    clases_dict = json.load(f)

CLASSES_IDX = {v: k for k, v in clases_dict.items()}

@app.post("/clasificar/")
async def clasificar_imagen(file: UploadFile = File(...)):
    try:
        # Guardar el archivo temporalmente
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Preprocesar imagen
        img = preprocess_image(temp_path)

        # Predicción
        pred = model.predict(img)
        predicted_index = pred.argmax()
        predicted_class = CLASSES_IDX[predicted_index]
        confidence = float(pred[0][predicted_index])

        # Borrar imagen temporal
        os.remove(temp_path)

        return {"raza": predicted_class, "confianza": round(confidence, 4)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)