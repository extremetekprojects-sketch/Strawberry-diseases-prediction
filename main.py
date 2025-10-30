# # # main.py - FastAPI Backend (Sensor + Image + Video) - Render-Optimized
# # from fastapi import FastAPI, UploadFile, File, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import FileResponse, JSONResponse
# # from pydantic import BaseModel
# # import joblib
# # from ultralytics import YOLO
# # import numpy as np
# # from PIL import Image
# # import io
# # import cv2
# # import warnings
# # import os
# # import shutil
# # import uvicorn
# # import asyncio
# # from contextlib import asynccontextmanager
# # import mimetypes
# # import uuid
# # warnings.filterwarnings("ignore")

# # # Global models (lazy-loaded)
# # yolo_model = None
# # dt_model = None
# # scaler = None
# # label_map = {0: "Healthy", 1: "Moderate Stress", 2: "High Stress"}

# # # Lazy load on first use
# # async def load_models():
# #     global yolo_model, dt_model, scaler
# #     if dt_model is None or scaler is None:
# #         print("Loading DT + Scaler...")
# #         dt_model = joblib.load("Decision_Tree.pkl")
# #         scaler = joblib.load("scaler.pkl")
# #         # Fix compatibility for scikit-learn
# #         if not hasattr(dt_model, 'monotonic_cst'):
# #             try:
# #                 dt_model.__dict__['monotonic_cst'] = None
# #             except:
# #                 pass
# #         print("DT + Scaler loaded!")
# #     if yolo_model is None:
# #         print("Loading YOLO...")
# #         yolo_model = YOLO("best.pt")
# #         print("YOLO loaded!")

# # # Sensor Data Model
# # class SensorData(BaseModel):
# #     Plant_ID: int = 1
# #     Soil_Moisture: float
# #     Ambient_Temperature: float
# #     Soil_Temperature: float
# #     Humidity: float
# #     Light_Intensity: float
# #     Soil_pH: float
# #     Nitrogen_Level: float
# #     Phosphorus_Level: float
# #     Potassium_Level: float
# #     Chlorophyll_Content: float
# #     Electrochemical_Signal: float

# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     await load_models()
# #     yield

# # app = FastAPI(title="Strawberry Disease Prediction API", version="1.0.0", lifespan=lifespan)

# # # CORS
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # @app.post("/predict/health")
# # async def predict_health(data: SensorData):
# #     try:
# #         await load_models()
        
# #         # FIXED: Proper input preparation and confidence calculation
# #         input_data = np.array([[
# #             data.Plant_ID, data.Soil_Moisture, data.Ambient_Temperature,
# #             data.Soil_Temperature, data.Humidity, data.Light_Intensity,
# #             data.Soil_pH, data.Nitrogen_Level, data.Phosphorus_Level,
# #             data.Potassium_Level, data.Chlorophyll_Content, data.Electrochemical_Signal
# #         ]])
        
# #         # Scale the input data
# #         input_scaled = scaler.transform(input_data)
        
# #         # Get prediction
# #         pred = dt_model.predict(input_scaled)[0]
        
# #         # FIXED: Correct confidence calculation - get max probability from predict_proba
# #         # predict_proba returns array of shape (n_samples, n_classes)
# #         # We take the maximum probability across classes for the predicted class
# #         probabilities = dt_model.predict_proba(input_scaled)[0]
# #         confidence = np.max(probabilities) * 100  # Convert to percentage
        
# #         # Ensure confidence is reasonable (0-100%)
# #         confidence = min(max(confidence, 0), 100)
        
# #         health_status = label_map.get(int(pred), "Unknown")
        
# #         return {
# #             "plant_health_status": health_status,
# #             "confidence": f"{confidence:.2f}%",
# #             "prediction_code": int(pred),
# #             "all_probabilities": {
# #                 "Healthy": f"{probabilities[0]*100:.2f}%",
# #                 "Moderate Stress": f"{probabilities[1]*100:.2f}%", 
# #                 "High Stress": f"{probabilities[2]*100:.2f}%"
# #             }
# #         }
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# # @app.post("/detect/image")
# # async def detect_image(file: UploadFile = File(...)):
# #     try:
# #         await load_models()
# #         contents = await file.read()
# #         image = Image.open(io.BytesIO(contents))
# #         results = yolo_model(image, verbose=False)
        
# #         detections = []
# #         for r in results:
# #             boxes = r.boxes
# #             if boxes is not None:
# #                 for box in boxes:
# #                     detections.append({
# #                         "class": r.names[int(box.cls)],
# #                         "confidence": float(box.conf),
# #                         "bbox": box.xyxy.tolist()[0]
# #                     })
        
# #         return {"detections": detections, "total_detections": len(detections)}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Image detection failed: {str(e)}")

# # @app.post("/detect/video")
# # async def detect_video(file: UploadFile = File(...)):
# #     try:
# #         await load_models()
        
# #         # Create temp directory
# #         temp_dir = f"temp_video_{uuid.uuid4().hex[:8]}"
# #         os.makedirs(temp_dir, exist_ok=True)
        
# #         # Save uploaded video with unique name
# #         unique_id = str(uuid.uuid4())[:8]
# #         input_filename = f"input_{unique_id}.mp4"
# #         temp_video_path = os.path.join(temp_dir, input_filename)
        
# #         # Save uploaded file
# #         contents = await file.read()
# #         with open(temp_video_path, "wb") as f:
# #             f.write(contents)
        
# #         # Run YOLO prediction
# #         results = yolo_model.predict(
# #             source=temp_video_path,
# #             save=True,
# #             project="runs/detect",
# #             name=f"predict_{unique_id}",
# #             exist_ok=True,
# #             verbose=False
# #         )
        
# #         # Find output video path (YOLO saves in runs/detect/predict_[id]/)
# #         output_dir = f"runs/detect/predict_{unique_id}"
# #         output_files = []
# #         for ext in ['.avi', '.mp4']:
# #             potential_output = os.path.join(output_dir, f"{input_filename[:-4]}{ext}")
# #             if os.path.exists(potential_output):
# #                 output_files.append(potential_output)
# #                 break
        
# #         if not output_files:
# #             # If no video found, look for any video file in output directory
# #             for root, dirs, files in os.walk(output_dir):
# #                 for file in files:
# #                     if file.endswith(('.avi', '.mp4', '.mov')):
# #                         output_files.append(os.path.join(root, file))
# #                         break
        
# #         # Cleanup temp input
# #         try:
# #             os.remove(temp_video_path)
# #             shutil.rmtree(temp_dir, ignore_errors=True)
# #         except Exception:
# #             pass
        
# #         if output_files:
# #             output_video_path = output_files[0]
# #             return {
# #                 "message": "Video processed successfully",
# #                 "output_video_path": output_video_path,
# #                 "total_frames": len(results) if results else 0,
# #                 "detections_per_frame": [len(r.boxes) if r.boxes is not None else 0 for r in results] if results else []
# #             }
# #         else:
# #             raise HTTPException(status_code=500, detail="No output video generated")
            
# #     except Exception as e:
# #         # Cleanup on error
# #         try:
# #             if 'temp_dir' in locals():
# #                 shutil.rmtree(temp_dir, ignore_errors=True)
# #         except:
# #             pass
# #         raise HTTPException(status_code=500, detail=f"Video detection failed: {str(e)}")

# # @app.get("/video/{video_path:path}")
# # async def get_video(video_path: str):
# #     """Serve processed video file for streaming/download."""
# #     try:
# #         full_path = os.path.join(os.getcwd(), video_path)
        
# #         if not os.path.exists(full_path):
# #             raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
        
# #         # Auto-detect MIME type
# #         media_type, _ = mimetypes.guess_type(full_path)
# #         if media_type is None:
# #             media_type = "application/octet-stream"
        
# #         return FileResponse(
# #             full_path, 
# #             media_type=media_type, 
# #             filename=os.path.basename(full_path),
# #             headers={"Accept-Ranges": "bytes"}  # Enable video seeking
# #         )
# #     except HTTPException:
# #         raise
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Failed to serve video: {str(e)}")

# # @app.get("/")
# # async def root():
# #     return {
# #         "message": "Plant Disease Prediction API",
# #         "endpoints": [
# #             "POST /predict/health - Sensor data prediction",
# #             "POST /detect/image - Image detection", 
# #             "POST /detect/video - Video detection",
# #             "GET /video/{path} - Stream/Download processed video"
# #         ]
# #     }

# # if __name__ == "__main__":
# #     port = 4000
# #     uvicorn.run(app, host="0.0.0.0", port=port)

# # main.py - Render-Deployable FastAPI (Sensor + Image + Video)
# import os
# import io
# import uuid
# import shutil
# import warnings
# import mimetypes
# import numpy as np
# from PIL import Image
# from contextlib import asynccontextmanager

# import uvicorn
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, JSONResponse
# from pydantic import BaseModel

# import joblib
# from ultralytics import YOLO

# warnings.filterwarnings("ignore")

# # ----------------------------------------------------------------------
# # MODEL PATHS (Ensure these files are in your repo root!)
# # ----------------------------------------------------------------------
# MODEL_FILES = ["best.pt", "Decision_Tree.pkl", "scaler.pkl"]

# for model_file in MODEL_FILES:
#     if not os.path.exists(model_file):
#         raise FileNotFoundError(f"Required model file not found: {model_file}")

# # ----------------------------------------------------------------------
# # GLOBAL MODELS (Lazy Load)
# # ----------------------------------------------------------------------
# yolo_model = None
# dt_model = None
# scaler = None
# label_map = {0: "Healthy", 1: "Moderate Stress", 2: "High Stress"}


# async def load_models():
#     global yolo_model, dt_model, scaler
#     if dt_model is None or scaler is None:
#         print("Loading DT + Scaler...")
#         dt_model = joblib.load("Decision_Tree.pkl")
#         scaler = joblib.load("scaler.pkl")
#         # scikit-learn 1.6+ fix
#         if not hasattr(dt_model, "monotonic_cst"):
#             try:
#                 dt_model.__dict__["monotonic_cst"] = None
#             except:
#                 pass
#         print("DT + Scaler loaded!")

#     if yolo_model is None:
#         print("Loading YOLO...")
#         yolo_model = YOLO("best.pt")
#         print("YOLO loaded!")


# # ----------------------------------------------------------------------
# # SENSOR MODEL
# # ----------------------------------------------------------------------
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


# # ----------------------------------------------------------------------
# # FASTAPI APP
# # ----------------------------------------------------------------------
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     await load_models()
#     yield


# app = FastAPI(title="Strawberry Disease API", lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ----------------------------------------------------------------------
# # 1. HEALTH PREDICTION (Confidence Fixed)
# # ----------------------------------------------------------------------
# @app.post("/predict/health")
# async def predict_health(data: SensorData):
#     try:
#         await load_models()
#         features = np.array([[
#             data.Plant_ID, data.Soil_Moisture, data.Ambient_Temperature,
#             data.Soil_Temperature, data.Humidity, data.Light_Intensity,
#             data.Soil_pH, data.Nitrogen_Level, data.Phosphorus_Level,
#             data.Potassium_Level, data.Chlorophyll_Content, data.Electrochemical_Signal
#         ]])
#         scaled = scaler.transform(features)
#         pred = int(dt_model.predict(scaled)[0])

#         # FIX: Confidence was >100% due to log-proba
#         raw_proba = dt_model.predict_proba(scaled)[0]
#         proba = np.exp(raw_proba) / np.sum(np.exp(raw_proba))
#         confidence = proba.max() * 100

#         return {
#             "plant_health_status": label_map.get(pred, "Unknown"),
#             "confidence": f"{confidence:.2f}%",
#             "prediction_code": pred,
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # ----------------------------------------------------------------------
# # 2. IMAGE DETECTION
# # ----------------------------------------------------------------------
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
#                         "bbox": box.xyxy.tolist()[0],
#                     })
#         return {"detections": detections, "total_detections": len(detections)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # ----------------------------------------------------------------------
# # 3. VIDEO DETECTION
# # ----------------------------------------------------------------------
# @app.post("/detect/video")
# async def detect_video(file: UploadFile = File(...)):
#     try:
#         await load_models()
#         temp_dir = "temp_video"
#         os.makedirs(temp_dir, exist_ok=True)
#         unique_id = str(uuid.uuid4())[:8]
#         input_path = os.path.join(temp_dir, f"input_{unique_id}.mp4")

#         with open(input_path, "wb") as f:
#             shutil.copyfileobj(file.file, f)

#         results = yolo_model.predict(
#             source=input_path,
#             save=True,
#             project="runs/detect",
#             name=f"predict_{unique_id}",
#             exist_ok=True,
#         )

#         video_name = os.path.splitext(os.path.basename(input_path))[0] + ".avi"
#         output_path = os.path.join("runs/detect", f"predict_{unique_id}", video_name)
#         rel_path = os.path.relpath(output_path, os.getcwd())

#         # Cleanup
#         try:
#             os.remove(input_path)
#         except:
#             pass
#         shutil.rmtree(temp_dir, ignore_errors=True)

#         return {
#             "message": "Video processed",
#             "output_video_path": rel_path,
#             "total_frames": len(results),
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # ----------------------------------------------------------------------
# # 4. SERVE VIDEO
# # ----------------------------------------------------------------------
# @app.get("/video/{video_path:path}")
# async def get_video(video_path: str):
#     full_path = os.path.abspath(os.path.join(os.getcwd(), video_path))
#     if not os.path.exists(full_path):
#         return JSONResponse({"error": "Video not found"}, status_code=404)
#     return FileResponse(full_path, media_type="video/avi")


# # ----------------------------------------------------------------------
# # ROOT
# # ----------------------------------------------------------------------
# @app.get("/")
# async def root():
#     return {"message": "API is running!", "endpoints": ["/predict/health", "/detect/image", "/detect/video"]}


# # ----------------------------------------------------------------------
# # RENDER ENTRYPOINT (Critical!)
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     # Render sets $PORT
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")

# main.py – FastAPI (Sensor + Image + Video) – 100% Render-compatible
import os
import io
import uuid
import shutil
import warnings
import mimetypes
import numpy as np
from PIL import Image
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import joblib
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# VERIFY REQUIRED MODEL FILES (fails fast if missing)
# ----------------------------------------------------------------------
for f in ["best.pt", "Decision_Tree.pkl", "scaler.pkl"]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing required model: {f}")

# ----------------------------------------------------------------------
# GLOBAL / LAZY LOADED MODELS
# ----------------------------------------------------------------------
yolo_model = None
dt_model = None
scaler = None
label_map = {0: "Healthy", 1: "Moderate Stress", 2: "High Stress"}


async def load_models():
    global yolo_model, dt_model, scaler
    if dt_model is None or scaler is None:
        print("Loading DT + Scaler...")
        dt_model = joblib.load("Decision_Tree.pkl")
        scaler = joblib.load("scaler.pkl")
        if not hasattr(dt_model, "monotonic_cst"):
            try:
                dt_model.__dict__["monotonic_cst"] = None
            except:
                pass
        print("DT + Scaler loaded!")
    if yolo_model is None:
        print("Loading YOLO...")
        yolo_model = YOLO("best.pt")
        print("YOLO loaded!")


# ----------------------------------------------------------------------
# SENSOR DATA MODEL
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# FASTAPI APP + LIFESPAN
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_models()
    yield


app = FastAPI(title="Strawberry Disease Prediction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# 1. SENSOR HEALTH PREDICTION (unchanged – confidence fixed)
# ----------------------------------------------------------------------
@app.post("/predict/health")
async def predict_health(data: SensorData):
    try:
        await load_models()
        features = np.array([[
            data.Plant_ID, data.Soil_Moisture, data.Ambient_Temperature,
            data.Soil_Temperature, data.Humidity, data.Light_Intensity,
            data.Soil_pH, data.Nitrogen_Level, data.Phosphorus_Level,
            data.Potassium_Level, data.Chlorophyll_Content, data.Electrochemical_Signal
        ]])
        scaled = scaler.transform(features)
        pred = int(dt_model.predict(scaled)[0])

        raw = dt_model.predict_proba(scaled)[0]
        proba = np.exp(raw) / np.sum(np.exp(raw))
        confidence = proba.max() * 100

        return {
            "plant_health_status": label_map.get(pred, "Unknown"),
            "confidence": f"{confidence:.2f}%",
            "prediction_code": pred,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# 2. IMAGE DETECTION (unchanged)
# ----------------------------------------------------------------------
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        await load_models()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        results = yolo_model(image, verbose=False)

        detections = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    detections.append({
                        "class": r.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy.tolist()[0],
                    })

        return {"detections": detections, "total_detections": len(detections)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# 3. VIDEO DETECTION – FIXED: Use results[0].save_path
# ----------------------------------------------------------------------
@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    try:
        await load_models()

        # 1. Save uploaded file
        temp_dir = "temp_video"
        os.makedirs(temp_dir, exist_ok=True)
        uid = str(uuid.uuid4())[:8]
        original_filename = file.filename
        input_path = os.path.join(temp_dir, f"input_{uid}.mp4")

        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. Run YOLO prediction
        results = yolo_model.predict(
            source=input_path,
            save=True,
            project="runs/detect",
            name=f"predict_{uid}",
            exist_ok=True,
        )

        # 3. Get actual saved video path from YOLO results
        if not results or len(results) == 0:
            raise ValueError("YOLO failed to process video")

        saved_video_path = results[0].save_path  # <-- This is the correct path!
        if not os.path.exists(saved_video_path):
            raise FileNotFoundError(f"YOLO saved video not found: {saved_video_path}")

        # 4. Return relative path for frontend
        rel_path = os.path.relpath(saved_video_path, os.getcwd())

        # 5. Clean up input
        try:
            os.remove(input_path)
        except:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "message": "Video processed successfully",
            "output_video_path": rel_path,
            "total_frames": len(results),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video detection failed: {str(e)}")


# ----------------------------------------------------------------------
# 4. SERVE PROCESSED VIDEO – Reliable FileResponse
# ----------------------------------------------------------------------
@app.get("/video/{video_path:path}")
async def get_video(video_path: str):
    full_path = os.path.abspath(os.path.join(os.getcwd(), video_path))

    if not os.path.exists(full_path):
        return JSONResponse({"error": "Video not found"}, status_code=404)

    media_type = mimetypes.guess_type(full_path)[0] or "video/avi"
    return FileResponse(
        full_path,
        media_type=media_type,
        filename=os.path.basename(full_path),
    )


# ----------------------------------------------------------------------
# ROOT
# ----------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "API is running",
        "endpoints": [
            "POST /predict/health",
            "POST /detect/image",
            "POST /detect/video",
            "GET  /video/{path}",
        ],
    }


# ----------------------------------------------------------------------
# RENDER ENTRYPOINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
