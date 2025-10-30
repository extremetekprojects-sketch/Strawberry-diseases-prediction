# # main.py - FastAPI Backend (Sensor + Image + Video) - Render-Optimized
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, JSONResponse
# from pydantic import BaseModel
# import joblib
# from ultralytics import YOLO
# import numpy as np
# from PIL import Image
# import io
# import cv2
# import warnings
# import os
# import shutil
# import uvicorn
# import asyncio
# from contextlib import asynccontextmanager
# import mimetypes
# import uuid
# warnings.filterwarnings("ignore")

# # Global models (lazy-loaded)
# yolo_model = None
# dt_model = None
# scaler = None
# label_map = {0: "Healthy", 1: "Moderate Stress", 2: "High Stress"}

# # Lazy load on first use
# async def load_models():
#     global yolo_model, dt_model, scaler
#     if dt_model is None or scaler is None:
#         print("Loading DT + Scaler...")
#         dt_model = joblib.load("Decision_Tree.pkl")
#         scaler = joblib.load("scaler.pkl")
#         # Fix compatibility for scikit-learn
#         if not hasattr(dt_model, 'monotonic_cst'):
#             try:
#                 dt_model.__dict__['monotonic_cst'] = None
#             except:
#                 pass
#         print("DT + Scaler loaded!")
#     if yolo_model is None:
#         print("Loading YOLO...")
#         yolo_model = YOLO("best.pt")
#         print("YOLO loaded!")

# # Sensor Data Model
# class SensorData(BaseModel):
#     Plant_ID: int = 1
#     Soil_Moisture: float
#     Ambient_Temperature: float
#     Soil_Temperature: float
#     Humidity: float
#     Light_Intensity: float
#     Soil_pH: float
#     Nitrogen_Level: float
#     Phosphorus_Level: float
#     Potassium_Level: float
#     Chlorophyll_Content: float
#     Electrochemical_Signal: float

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     await load_models()
#     yield

# app = FastAPI(title="Strawberry Disease Prediction API", version="1.0.0", lifespan=lifespan)

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/predict/health")
# async def predict_health(data: SensorData):
#     try:
#         await load_models()
        
#         # FIXED: Proper input preparation and confidence calculation
#         input_data = np.array([[
#             data.Plant_ID, data.Soil_Moisture, data.Ambient_Temperature,
#             data.Soil_Temperature, data.Humidity, data.Light_Intensity,
#             data.Soil_pH, data.Nitrogen_Level, data.Phosphorus_Level,
#             data.Potassium_Level, data.Chlorophyll_Content, data.Electrochemical_Signal
#         ]])
        
#         # Scale the input data
#         input_scaled = scaler.transform(input_data)
        
#         # Get prediction
#         pred = dt_model.predict(input_scaled)[0]
        
#         # FIXED: Correct confidence calculation - get max probability from predict_proba
#         # predict_proba returns array of shape (n_samples, n_classes)
#         # We take the maximum probability across classes for the predicted class
#         probabilities = dt_model.predict_proba(input_scaled)[0]
#         confidence = np.max(probabilities) * 100  # Convert to percentage
        
#         # Ensure confidence is reasonable (0-100%)
#         confidence = min(max(confidence, 0), 100)
        
#         health_status = label_map.get(int(pred), "Unknown")
        
#         return {
#             "plant_health_status": health_status,
#             "confidence": f"{confidence:.2f}%",
#             "prediction_code": int(pred),
#             "all_probabilities": {
#                 "Healthy": f"{probabilities[0]*100:.2f}%",
#                 "Moderate Stress": f"{probabilities[1]*100:.2f}%", 
#                 "High Stress": f"{probabilities[2]*100:.2f}%"
#             }
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# @app.post("/detect/image")
# async def detect_image(file: UploadFile = File(...)):
#     try:
#         await load_models()
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents))
#         results = yolo_model(image, verbose=False)
        
#         detections = []
#         for r in results:
#             boxes = r.boxes
#             if boxes is not None:
#                 for box in boxes:
#                     detections.append({
#                         "class": r.names[int(box.cls)],
#                         "confidence": float(box.conf),
#                         "bbox": box.xyxy.tolist()[0]
#                     })
        
#         return {"detections": detections, "total_detections": len(detections)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Image detection failed: {str(e)}")

# @app.post("/detect/video")
# async def detect_video(file: UploadFile = File(...)):
#     try:
#         await load_models()
        
#         # Create temp directory
#         temp_dir = f"temp_video_{uuid.uuid4().hex[:8]}"
#         os.makedirs(temp_dir, exist_ok=True)
        
#         # Save uploaded video with unique name
#         unique_id = str(uuid.uuid4())[:8]
#         input_filename = f"input_{unique_id}.mp4"
#         temp_video_path = os.path.join(temp_dir, input_filename)
        
#         # Save uploaded file
#         contents = await file.read()
#         with open(temp_video_path, "wb") as f:
#             f.write(contents)
        
#         # Run YOLO prediction
#         results = yolo_model.predict(
#             source=temp_video_path,
#             save=True,
#             project="runs/detect",
#             name=f"predict_{unique_id}",
#             exist_ok=True,
#             verbose=False
#         )
        
#         # Find output video path (YOLO saves in runs/detect/predict_[id]/)
#         output_dir = f"runs/detect/predict_{unique_id}"
#         output_files = []
#         for ext in ['.avi', '.mp4']:
#             potential_output = os.path.join(output_dir, f"{input_filename[:-4]}{ext}")
#             if os.path.exists(potential_output):
#                 output_files.append(potential_output)
#                 break
        
#         if not output_files:
#             # If no video found, look for any video file in output directory
#             for root, dirs, files in os.walk(output_dir):
#                 for file in files:
#                     if file.endswith(('.avi', '.mp4', '.mov')):
#                         output_files.append(os.path.join(root, file))
#                         break
        
#         # Cleanup temp input
#         try:
#             os.remove(temp_video_path)
#             shutil.rmtree(temp_dir, ignore_errors=True)
#         except Exception:
#             pass
        
#         if output_files:
#             output_video_path = output_files[0]
#             return {
#                 "message": "Video processed successfully",
#                 "output_video_path": output_video_path,
#                 "total_frames": len(results) if results else 0,
#                 "detections_per_frame": [len(r.boxes) if r.boxes is not None else 0 for r in results] if results else []
#             }
#         else:
#             raise HTTPException(status_code=500, detail="No output video generated")
            
#     except Exception as e:
#         # Cleanup on error
#         try:
#             if 'temp_dir' in locals():
#                 shutil.rmtree(temp_dir, ignore_errors=True)
#         except:
#             pass
#         raise HTTPException(status_code=500, detail=f"Video detection failed: {str(e)}")

# @app.get("/video/{video_path:path}")
# async def get_video(video_path: str):
#     """Serve processed video file for streaming/download."""
#     try:
#         full_path = os.path.join(os.getcwd(), video_path)
        
#         if not os.path.exists(full_path):
#             raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
        
#         # Auto-detect MIME type
#         media_type, _ = mimetypes.guess_type(full_path)
#         if media_type is None:
#             media_type = "application/octet-stream"
        
#         return FileResponse(
#             full_path, 
#             media_type=media_type, 
#             filename=os.path.basename(full_path),
#             headers={"Accept-Ranges": "bytes"}  # Enable video seeking
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to serve video: {str(e)}")

# @app.get("/")
# async def root():
#     return {
#         "message": "Plant Disease Prediction API",
#         "endpoints": [
#             "POST /predict/health - Sensor data prediction",
#             "POST /detect/image - Image detection", 
#             "POST /detect/video - Video detection",
#             "GET /video/{path} - Stream/Download processed video"
#         ]
#     }

# if __name__ == "__main__":
#     port = 4000
#     uvicorn.run(app, host="0.0.0.0", port=port)

# main.py - FastAPI Backend (Sensor + Image + Video) - Fully Fixed & Optimized
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import joblib
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import warnings
import os
import shutil
import uvicorn
import asyncio
from contextlib import asynccontextmanager
import mimetypes
import uuid

warnings.filterwarnings("ignore")

# === Global Models (Lazy Load) ===
yolo_model = None
dt_model = None
scaler = None
label_map = {0: "Healthy", 1: "Moderate Stress", 2: "High Stress"}

async def load_models():
    global yolo_model, dt_model, scaler
    if dt_model is None or scaler is None:
        print("Loading Decision Tree + Scaler...")
        dt_model = joblib.load("Decision_Tree.pkl")
        scaler = joblib.load("scaler.pkl")

        # --- CRITICAL FIX: Remove or safely handle 'monotonic_cst' ---
        # Older scikit-learn versions don't have this. Newer ones do.
        # If present, keep it. If not, don't force it.
        # But if model was saved with it and now missing → error.
        # So we safely inject it only if the class supports it.
        if not hasattr(dt_model, 'monotonic_cst'):
            # Try to set via __dict__ (safest)
            try:
                dt_model.__dict__['monotonic_cst'] = None
            except:
                try:
                    setattr(dt_model, 'monotonic_cst', None)
                except:
                    pass  # Ignore if not possible

        print("DT + Scaler loaded!")

    if yolo_model is None:
        print("Loading YOLO...")
        yolo_model = YOLO("best.pt")
        print("YOLO loaded!")

# === Pydantic Model ===
class SensorData(BaseModel):
    Plant_ID: int = 1
    Soil_Moisture: float
    Ambient_Temperature: float
    Soil_Temperature: float
    Humidity: float
    Light_Intensity: float
    Soil_pH: float
    Nitrogen_Level: float
    Phosphorus_Level: float
    Potassium_Level: float
    Chlorophyll_Content: float
    Electrochemical_Signal: float

# === App Lifespan ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_models()
    yield

app = FastAPI(title="Strawberry Disease Prediction API", version="1.0.0", lifespan=lifespan)

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === /predict/health - FIXED CONFIDENCE + monotonic_cst ===
@app.post("/predict/health")
async def predict_health(data: SensorData):
    try:
        await load_models()

        # Input array
        input_data = np.array([[
            data.Plant_ID, data.Soil_Moisture, data.Ambient_Temperature,
            data.Soil_Temperature, data.Humidity, data.Light_Intensity,
            data.Soil_pH, data.Nitrogen_Level, data.Phosphorus_Level,
            data.Potassium_Level, data.Chlorophyll_Content, data.Electrochemical_Signal
        ]], dtype=np.float64)

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        pred = int(dt_model.predict(input_scaled)[0])

        # --- FIXED: Correct Confidence (0-100%) ---
        probs = dt_model.predict_proba(input_scaled)[0]
        confidence = float(np.max(probs) * 100)
        confidence = min(max(confidence, 0.0), 100.0)  # Clamp

        health_status = label_map.get(pred, "Unknown")

        return {
            "plant_health_status": health_status,
            "confidence": f"{confidence:.2f}%",
            "prediction_code": pred,
            "probabilities": {
                "Healthy": f"{probs[0]*100:.2f}%",
                "Moderate Stress": f"{probs[1]*100:.2f}%",
                "High Stress": f"{probs[2]*100:.2f}%"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# === /detect/image ===
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        await load_models()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = yolo_model(image, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    detections.append({
                        "class": r.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy.tolist()[0]
                    })

        return {"detections": detections, "total_detections": len(detections)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image detection failed: {str(e)}")

# === /detect/video ===
@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    try:
        await load_models()

        # Unique temp dir
        unique_id = uuid.uuid4().hex[:8]
        temp_dir = f"temp_video_{unique_id}"
        os.makedirs(temp_dir, exist_ok=True)

        input_path = os.path.join(temp_dir, f"input_{unique_id}.mp4")
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run YOLO
        results = yolo_model.predict(
            source=input_path,
            save=True,
            project="runs/detect",
            name=f"predict_{unique_id}",
            exist_ok=True,
            verbose=False
        )

        # Find output video
        output_dir = f"runs/detect/predict_{unique_id}"
        output_path = None
        for ext in ['.avi', '.mp4']:
            candidate = os.path.join(output_dir, f"input_{unique_id}{ext}")
            if os.path.exists(candidate):
                output_path = candidate
                break

        if not output_path and os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith(('.avi', '.mp4', '.mov')):
                    output_path = os.path.join(output_dir, f)
                    break

        # Cleanup temp
        try:
            os.remove(input_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

        if not output_path:
            raise HTTPException(status_code=500, detail="YOLO failed to generate output video")

        return {
            "message": "Video processed successfully",
            "output_video_path": output_path,
            "total_frames": len(results),
            "detections_per_frame": [
                len(r.boxes) if r.boxes is not None else 0 for r in results
            ]
        }

    except Exception as e:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Video detection failed: {str(e)}")

# === Serve Video ===
@app.get("/video/{video_path:path}")
async def get_video(video_path: str):
    full_path = os.path.abspath(os.path.join(os.getcwd(), video_path))
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Video not found")

    media_type, _ = mimetypes.guess_type(full_path)
    if media_type is None:
        media_type = "application/octet-stream"

    return FileResponse(
        full_path,
        media_type=media_type,
        filename=os.path.basename(full_path),
        headers={"Accept-Ranges": "bytes"}
    )

# === Root ===
@app.get("/")
async def root():
    return {
        "message": "Plant Disease Prediction API",
        "endpoints": [
            "POST /predict/health",
            "POST /detect/image",
            "POST /detect/video",
            "GET /video/{path}"
        ]
    }

# === Run ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)
