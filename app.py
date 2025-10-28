
# # app.py - Streamlit Dashboard with FULL Video Support
# import streamlit as st
# import requests
# import json
# from PIL import Image
# import os
# import time

# # CONFIGURATION
# # API_URL = "http://127.0.0.1:8000"
# API_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
# st.set_page_config(page_title="🌱 Strawberry Disease Predictor", layout="wide", initial_sidebar_state="expanded")

# def check_api_status():
#     try:
#         response = requests.get(f"{API_URL}", timeout=2)
#         return True
#     except:
#         return False

# # Header with styled markdown
# st.markdown("""
#     <style>
#         .main-header { text-align: center; color: #228B22; }
#         .stButton>button { width: 100%; }
#         .stMetric { text-align: center; }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown('<h1 class="main-header">🌱 Strawberry Disease Prediction Dashboard</h1>', unsafe_allow_html=True)
# st.markdown("---")

# # API Status Check with icons
# if check_api_status():
#     st.success(" **FastAPI Server: CONNECTED**")
# else:
#     st.error(" **FastAPI Server: NOT RUNNING**")
#     st.info("💡 **Fix:** Run `uvicorn main:app --reload` in another terminal")
#     st.stop()

# # Sidebar Navigation with emojis
# st.sidebar.title("📋 Navigation")
# page = st.sidebar.radio("Choose Action", [
#     "📊 Sensor Prediction", 
#     "📸 Image Detection", 
#     "🎥 Video Detection"
# ], label_visibility="collapsed")

# # === SENSOR PREDICTION ===
# if page == "📊 Sensor Prediction":
#     st.header("📊 **Sensor-Based Health Prediction**")
#     st.markdown("Enter sensor readings below to predict plant health status.")
    
#     # Improved input form with expanders for better organization
#     with st.expander("🌡️ **Environmental Parameters**", expanded=True):
#         col1, col2 = st.columns(2)
#         with col1:
#             soil_moisture = st.slider("💧 Soil Moisture (%)", 10.0, 40.0, 25.0)
#             temp = st.slider("🌡️ Ambient Temp (°C)", 18.0, 30.0, 24.0)
#             humidity = st.slider("💨 Humidity (%)", 40.0, 70.0, 55.0)
        
#         with col2:
#             light = st.slider("💡 Light Intensity", 200.0, 1000.0, 600.0)
#             ph = st.slider("🧪 Soil pH", 5.5, 7.5, 6.5)
#             soil_temp = st.slider("🌡️ Soil Temp (°C)", 15.0, 25.0, 20.0)
    
#     with st.expander("🌿 **Nutrient Levels**", expanded=True):
#         col3, col4, col5 = st.columns(3)
#         with col3: nitrogen = st.slider("🟢 Nitrogen", 10.0, 50.0, 30.0)
#         with col4: phosphorus = st.slider("🟡 Phosphorus", 10.0, 50.0, 30.0)
#         with col5: potassium = st.slider("🔴 Potassium", 10.0, 50.0, 30.0)
    
#     with st.expander("🍃 **Health Indicators**", expanded=True):
#         col6, col7 = st.columns(2)
#         with col6: chlorophyll = st.slider("🍃 Chlorophyll", 20.0, 50.0, 35.0)
#         with col7: signal = st.slider("⚡ Electro Signal", 0.0, 2.0, 1.0)
    
#     if st.button("🚀 **PREDICT PLANT HEALTH**", type="primary"):
#         with st.spinner("Analyzing sensor data..."):
#             payload = {
#                 "Plant_ID": 1, "Soil_Moisture": soil_moisture, "Ambient_Temperature": temp,
#                 "Soil_Temperature": soil_temp, "Humidity": humidity, "Light_Intensity": light,
#                 "Soil_pH": ph, "Nitrogen_Level": nitrogen, "Phosphorus_Level": phosphorus,
#                 "Potassium_Level": potassium, "Chlorophyll_Content": chlorophyll,
#                 "Electrochemical_Signal": signal
#             }
#             try:
#                 response = requests.post(f"{API_URL}/predict/health", json=payload, timeout=10)
#                 result = response.json()
                
#                 st.markdown("### Prediction Results")
#                 col_a, col_b = st.columns([3,1])
#                 with col_a:
#                     status = result['plant_health_status']
#                     if "Healthy" in status: st.success(f"🟢 **{status}**")
#                     elif "Moderate" in status: st.warning(f"🟡 **{status}**")
#                     else: st.error(f"🔴 **{status}**")
                
#                 with col_b:
#                     st.metric("Confidence", result['confidence'])
                
#                 with st.expander("📝 Detailed JSON Response"):
#                     st.json(result)
#             except Exception as e:
#                 st.error(f"🚨 **Error:** {str(e)}")

# # === IMAGE DETECTION ===
# elif page == "📸 Image Detection":
#     st.header("📸 **Image-Based Plant Detection**")
#     st.markdown("Upload an image to detect plants and potential diseases.")
#     uploaded_file = st.file_uploader("Upload Plant Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=False, width=400)
        
#         if st.button("🔍 **DETECT PLANTS**", type="primary"):
#             with st.spinner("Detecting objects in image..."):
#                 files = {"file": uploaded_file.getvalue()}
#                 try:
#                     response = requests.post(f"{API_URL}/detect/image", files=files, timeout=30)
#                     result = response.json()
                    
#                     if "detections" in result and result["total_detections"] > 0:
#                         st.success(f"✅ **Found {result['total_detections']} detection(s)**")
#                         for i, detection in enumerate(result["detections"], 1):
#                             with st.expander(f"🌿 **Detection {i}: {detection['class']}**"):
#                                 st.metric("Confidence", f"{detection['confidence']:.1%}")
#                                 st.write(f"**Bounding Box:** {detection['bbox']}")
#                     else:
#                         st.info("🌱 **No plants detected**")
#                 except Exception as e:
#                     st.error(f"🚨 **Error:** {str(e)}")

# # === VIDEO DETECTION ===
# elif page == "🎥 Video Detection":
#     st.header("🎥 **Video Analysis**")
#     st.markdown("Upload a video to process and view annotated results on screen.")
    
#     uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    
#     if uploaded_video is not None:
#         # Store bytes for original video
#         original_video_bytes = uploaded_video.getvalue()
        
#         # Display original with better layout
#         st.subheader("📼 **Original Video**")
#         st.video(original_video_bytes)
        
#         # Process button
#         if st.button("🔍 **PROCESS VIDEO**", type="primary"):
#             with st.spinner("🔍 **Processing Video...** (This may take 1-3 minutes)"):
#                 progress_bar = st.progress(0)
                
#                 # Save temp file
#                 progress_bar.progress(10)
#                 temp_video_path = "temp_input.mp4"
#                 with open(temp_video_path, "wb") as f:
#                     f.write(original_video_bytes)
                
#                 # Call API
#                 progress_bar.progress(40)
#                 with open(temp_video_path, "rb") as f:
#                     files = {"file": f}
#                     response = requests.post(f"{API_URL}/detect/video", files=files, timeout=180)
                
#                 # Process results
#                 progress_bar.progress(70)
#                 result = response.json()
                
#                 if "error" in result:
#                     st.error(f"🚨 **API Error:** {result['error']}")
#                 else:
#                     st.success(" **Video Processed Successfully!**")
                    
#                     # Fetch processed video
#                     progress_bar.progress(85)
#                     video_path = result['output_video_path']
#                     video_response = requests.get(f"{API_URL}/video/{video_path}", stream=True, timeout=30)
                    
#                     if video_response.status_code == 200:
#                         progress_bar.progress(100)
#                         processed_video_bytes = video_response.content
                        
#                         # Display annotated video
#                         st.subheader("🎞️ **Annotated Video**")
#                         st.video(processed_video_bytes)
                        
#                         # Processing Summary
#                         st.subheader("📊 **Processing Summary**")
#                         col_a, col_b = st.columns(2)
#                         with col_a: 
#                             st.metric("Total Frames", result['total_frames'])
#                         with col_b: 
#                             st.metric("Status", "Completed")
                        
#                         # Download button
#                         st.download_button(
#                             label="📥 **Download Annotated Video**",
#                             data=processed_video_bytes,
#                             file_name="annotated_video.avi",
#                             mime="video/avi"
#                         )
#                     else:
#                         st.error(f"🚨 **Failed to fetch video:** {video_response.text}")
                
#                 # Cleanup
#                 if os.path.exists(temp_video_path):
#                     os.remove(temp_video_path)

#                 progress_bar.empty()


# # app.py - Strawberry Disease Prediction Dashboard (FULLY WORKING)
# import streamlit as st
# import requests
# import json
# import os
# from PIL import Image

# # ===================================================================
# # 1. MUST BE THE VERY FIRST STREAMLIT COMMAND
# # ===================================================================
# st.set_page_config(
#     page_title="Strawberry Disease Predictor",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ===================================================================
# # 2. CONFIGURATION
# # ===================================================================
# API_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
# st.sidebar.markdown(f"**Backend URL:** `{API_URL}`")

# # ===================================================================
# # 3. API STATUS CHECK (with 30s timeout)
# # ===================================================================
# def check_api_status():
#     try:
#         resp = requests.get(f"{API_URL}/", timeout=30)
#         resp.raise_for_status()
#         return True
#     except Exception as e:
#         st.error(f"API unreachable: {e}")
#         return False

# # ===================================================================
# # 4. UI HEADER & STYLES
# # ===================================================================
# st.markdown("""
#     <style>
#         .main-header { text-align: center; color: #228B22; font-size: 2.5rem; }
#         .stButton>button { width: 100%; }
#         .stMetric { text-align: center; }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown('<h1 class="main-header">Strawberry Disease Prediction Dashboard</h1>', unsafe_allow_html=True)
# st.markdown("---")

# # Status Check
# if check_api_status():
#     st.success(" **FastAPI Server: CONNECTED**")
# else:
#     st.error(" **FastAPI Server: NOT RUNNING**")
#     st.info("**Fix:** Check `BACKEND_URL` in Streamlit Secrets or wake up Render.")
#     st.stop()

# # ===================================================================
# # 5. NAVIGATION
# # ===================================================================
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Choose Action", [
#     "Sensor Prediction",
#     "Image Detection",
#     "Video Detection"
# ], label_visibility="collapsed")

# # ===================================================================
# # 6. HELPER: Safe JSON parsing with debug
# # ===================================================================
# def safe_json(resp: requests.Response):
#     try:
#         resp.raise_for_status()
#         return resp.json()
#     except requests.exceptions.HTTPError:
#         raise ValueError(f"HTTP {resp.status_code} – {resp.text[:500]}")
#     except json.JSONDecodeError:
#         raise ValueError(f"Invalid JSON – {resp.text[:500]}")

# # ===================================================================
# # 7. SENSOR PREDICTION
# # ===================================================================
# if page == "Sensor Prediction":
#     st.header("**Sensor-Based Health Prediction**")
#     st.markdown("Enter sensor data to predict plant health.")

#     with st.expander("Environmental Parameters", expanded=True):
#         c1, c2 = st.columns(2)
#         with c1:
#             soil_moisture = st.slider("Soil Moisture (%)", 10.0, 40.0, 25.0)
#             temp = st.slider("Ambient Temp (°C)", 18.0, 30.0, 24.0)
#             humidity = st.slider("Humidity (%)", 40.0, 70.0, 55.0)
#         with c2:
#             light = st.slider("Light Intensity", 200.0, 1000.0, 600.0)
#             ph = st.slider("Soil pH", 5.5, 7.5, 6.5)
#             soil_temp = st.slider("Soil Temp (°C)", 15.0, 25.0, 20.0)

#     with st.expander("Nutrient Levels", expanded=True):
#         c3, c4, c5 = st.columns(3)
#         with c3: nitrogen = st.slider("Nitrogen", 10.0, 50.0, 30.0)
#         with c4: phosphorus = st.slider("Phosphorus", 10.0, 50.0, 30.0)
#         with c5: potassium = st.slider("Potassium", 10.0, 50.0, 30.0)

#     with st.expander("Health Indicators", expanded=True):
#         c6, c7 = st.columns(2)
#         with c6: chlorophyll = st.slider("Chlorophyll", 20.0, 50.0, 35.0)
#         with c7: signal = st.slider("Electro Signal", 0.0, 2.0, 1.0)

#     if st.button("**PREDICT PLANT HEALTH**", type="primary"):
#         with st.spinner("Analyzing..."):
#             payload = {
#                 "Plant_ID": 1,
#                 "Soil_Moisture": soil_moisture,
#                 "Ambient_Temperature": temp,
#                 "Soil_Temperature": soil_temp,
#                 "Humidity": humidity,
#                 "Light_Intensity": light,
#                 "Soil_pH": ph,
#                 "Nitrogen_Level": nitrogen,
#                 "Phosphorus_Level": phosphorus,
#                 "Potassium_Level": potassium,
#                 "Chlorophyll_Content": chlorophyll,
#                 "Electrochemical_Signal": signal
#             }
#             try:
#                 resp = requests.post(f"{API_URL}/predict/health", json=payload, timeout=30)
#                 result = safe_json(resp)

#                 st.markdown("### Prediction Results")
#                 col1, col2 = st.columns([3, 1])
#                 with col1:
#                     status = result['plant_health_status']
#                     if "Healthy" in status:
#                         st.success(f"**{status}**")
#                     elif "Moderate" in status:
#                         st.warning(f"**{status}**")
#                     else:
#                         st.error(f"**{status}**")
#                 with col2:
#                     st.metric("Confidence", result['confidence'])

#                 with st.expander("Detailed Response"):
#                     st.json(result)

#             except Exception as e:
#                 st.error(f"**Error:** {e}")

# # ===================================================================
# # 8. IMAGE DETECTION
# # ===================================================================
# elif page == "Image Detection":
#     st.header("**Image-Based Detection**")
#     st.markdown("Upload an image to detect strawberry diseases.")
#     uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

#     if uploaded_file:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", width=400)

#         if st.button("**DETECT DISEASES**", type="primary"):
#             with st.spinner("Detecting..."):
#                 files = {"file": uploaded_file.getvalue()}
#                 try:
#                     resp = requests.post(f"{API_URL}/detect/image", files=files, timeout=60)
#                     result = safe_json(resp)

#                     if result.get("total_detections", 0) > 0:
#                         st.success(f"**{result['total_detections']} detection(s) found**")
#                         for i, d in enumerate(result["detections"], 1):
#                             with st.expander(f"**Detection {i}: {d['class']}**"):
#                                 st.metric("Confidence", f"{d['confidence']:.1%}")
#                                 st.write(f"**Box:** {d['bbox']}")
#                     else:
#                         st.info("**No diseases detected**")
#                 except Exception as e:
#                     st.error(f"**Error:** {e}")

# # ===================================================================
# # 9. VIDEO DETECTION
# # ===================================================================
# elif page == "Video Detection":
#     st.header("**Video Analysis**")
#     st.markdown("Upload a short video (<30s) for disease detection.")
#     uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], label_visibility="collapsed")

#     if uploaded_video:
#         video_bytes = uploaded_video.getvalue()
#         st.subheader("**Original Video**")
#         st.video(video_bytes)

#         if st.button("**PROCESS VIDEO**", type="primary"):
#             with st.spinner("Processing video... (1–3 min)"):
#                 prog = st.progress(0)

#                 # Save temp
#                 prog.progress(10)
#                 tmp_path = "temp_input.mp4"
#                 with open(tmp_path, "wb") as f:
#                     f.write(video_bytes)

#                 # Upload
#                 prog.progress(40)
#                 with open(tmp_path, "rb") as f:
#                     files = {"file": f}
#                     resp = requests.post(f"{API_URL}/detect/video", files=files, timeout=180)

#                 prog.progress(70)
#                 try:
#                     result = safe_json(resp)
#                 except Exception as e:
#                     st.error(f"**API Error:** {e}")
#                     if os.path.exists(tmp_path):
#                         os.remove(tmp_path)
#                     prog.empty()
#                     st.stop()

#                 if "error" in result:
#                     st.error(f"**Backend Error:** {result['error']}")
#                 else:
#                     st.success("**Processing Complete!**")
#                     prog.progress(85)

#                     # Download annotated video
#                     video_path = result["output_video_path"]
#                     vid_resp = requests.get(f"{API_URL}/video/{video_path}", stream=True, timeout=60)

#                     if vid_resp.status_code == 200:
#                         prog.progress(100)
#                         annotated = vid_resp.content

#                         st.subheader("**Annotated Video**")
#                         st.video(annotated)

#                         st.subheader("**Summary**")
#                         c1, c2 = st.columns(2)
#                         c1.metric("Frames", result["total_frames"])
#                         c2.metric("Status", "Completed")

#                         st.download_button(
#                             "Download Annotated Video",
#                             data=annotated,
#                             file_name="annotated_strawberry.avi",
#                             mime="video/avi"
#                         )
#                     else:
#                         st.error(f"Failed to get video: {vid_resp.status_code}")

#                 # Cleanup
#                 if os.path.exists(tmp_path):
#                     os.remove(tmp_path)
#                 prog.empty()

# app.py - Streamlit Dashboard with FULL Video Support (robust JSON handling)
import streamlit as st
import requests
import json
from PIL import Image
import os
import time

# CONFIGURATION
# API_URL = "http://127.0.0.1:8000"
API_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="🌱 Strawberry Disease Predictor", layout="wide", initial_sidebar_state="expanded")

def check_api_status():
    try:
        response = requests.get(f"{API_URL}", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def safe_parse_json(response):
    """
    Try to parse JSON from a requests.Response object.
    If parsing fails, raise an Exception with useful info.
    """
    try:
        return response.json()
    except ValueError:
        # non-JSON response (empty, HTML error, etc.)
        raise Exception(f"Invalid JSON response (status {response.status_code}). Server returned:\n{response.text[:1000]}")

# Header with styled markdown
st.markdown("""
    <style>
        .main-header { text-align: center; color: #228B22; }
        .stButton>button { width: 100%; }
        .stMetric { text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌱 Strawberry Disease Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# API Status Check with icons
if check_api_status():
    st.success(" **FastAPI Server: CONNECTED**")
else:
    st.error(" **FastAPI Server: NOT RUNNING**")
    st.info("💡 **Fix:** Run `uvicorn main:app --reload` in another terminal")
    st.stop()

# Sidebar Navigation with emojis
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Choose Action", [
    "📊 Sensor Prediction", 
    "📸 Image Detection", 
    "🎥 Video Detection"
], label_visibility="collapsed")

# === SENSOR PREDICTION ===
if page == "📊 Sensor Prediction":
    st.header("📊 **Sensor-Based Health Prediction**")
    st.markdown("Enter sensor readings below to predict plant health status.")
    
    # Improved input form with expanders for better organization
    with st.expander("🌡️ **Environmental Parameters**", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            soil_moisture = st.slider("💧 Soil Moisture (%)", 10.0, 40.0, 25.0)
            temp = st.slider("🌡️ Ambient Temp (°C)", 18.0, 30.0, 24.0)
            humidity = st.slider("💨 Humidity (%)", 40.0, 70.0, 55.0)
        
        with col2:
            light = st.slider("💡 Light Intensity", 200.0, 1000.0, 600.0)
            ph = st.slider("🧪 Soil pH", 5.5, 7.5, 6.5)
            soil_temp = st.slider("🌡️ Soil Temp (°C)", 15.0, 25.0, 20.0)
    
    with st.expander("🌿 **Nutrient Levels**", expanded=True):
        col3, col4, col5 = st.columns(3)
        with col3: nitrogen = st.slider("🟢 Nitrogen", 10.0, 50.0, 30.0)
        with col4: phosphorus = st.slider("🟡 Phosphorus", 10.0, 50.0, 30.0)
        with col5: potassium = st.slider("🔴 Potassium", 10.0, 50.0, 30.0)
    
    with st.expander("🍃 **Health Indicators**", expanded=True):
        col6, col7 = st.columns(2)
        with col6: chlorophyll = st.slider("🍃 Chlorophyll", 20.0, 50.0, 35.0)
        with col7: signal = st.slider("⚡ Electro Signal", 0.0, 2.0, 1.0)
    
    if st.button("🚀 **PREDICT PLANT HEALTH**", type="primary"):
        with st.spinner("Analyzing sensor data..."):
            payload = {
                "Plant_ID": 1, "Soil_Moisture": soil_moisture, "Ambient_Temperature": temp,
                "Soil_Temperature": soil_temp, "Humidity": humidity, "Light_Intensity": light,
                "Soil_pH": ph, "Nitrogen_Level": nitrogen, "Phosphorus_Level": phosphorus,
                "Potassium_Level": potassium, "Chlorophyll_Content": chlorophyll,
                "Electrochemical_Signal": signal
            }
            try:
                response = requests.post(f"{API_URL}/predict/health", json=payload, timeout=15)
                if response.status_code != 200:
                    # show server message if not OK
                    st.error(f"🚨 Server returned status {response.status_code}: {response.text}")
                else:
                    result = safe_parse_json(response)
                    st.markdown("### Prediction Results")
                    col_a, col_b = st.columns([3,1])
                    with col_a:
                        status = result.get('plant_health_status', 'Unknown')
                        if "Healthy" in status: st.success(f"🟢 **{status}**")
                        elif "Moderate" in status: st.warning(f"🟡 **{status}**")
                        else: st.error(f"🔴 **{status}**")
                    
                    with col_b:
                        st.metric("Confidence", result.get('confidence', 'N/A'))
                    
                    with st.expander("📝 Detailed JSON Response"):
                        st.json(result)
            except Exception as e:
                st.error(f"🚨 **Error:** {str(e)}")

# === IMAGE DETECTION ===
elif page == "📸 Image Detection":
    st.header("📸 **Image-Based Plant Detection**")
    st.markdown("Upload an image to detect plants and potential diseases.")
    uploaded_file = st.file_uploader("Upload Plant Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=False, width=400)
        
        if st.button("🔍 **DETECT PLANTS**", type="primary"):
            with st.spinner("Detecting objects in image..."):
                # Create proper multipart file tuple: (filename, bytes, content_type)
                try:
                    filename = getattr(uploaded_file, "name", "uploaded.jpg")
                    content_type = getattr(uploaded_file, "type", "image/jpeg")
                    files = {"file": (filename, uploaded_file.getvalue(), content_type)}
                    
                    response = requests.post(f"{API_URL}/detect/image", files=files, timeout=60)
                    if response.status_code != 200:
                        st.error(f"🚨 Server returned status {response.status_code}: {response.text}")
                    else:
                        result = safe_parse_json(response)
                        if "detections" in result and result.get("total_detections", 0) > 0:
                            st.success(f"✅ **Found {result['total_detections']} detection(s)**")
                            for i, detection in enumerate(result["detections"], 1):
                                with st.expander(f"🌿 **Detection {i}: {detection.get('class','unknown')}**"):
                                    conf = detection.get('confidence', 0.0)
                                    try:
                                        st.metric("Confidence", f"{conf:.1%}")
                                    except Exception:
                                        st.metric("Confidence", conf)
                                    st.write(f"**Bounding Box:** {detection.get('bbox')}")
                        else:
                            st.info("🌱 **No plants detected**")
                except Exception as e:
                    st.error(f"🚨 **Error:** {str(e)}")

# === VIDEO DETECTION ===
elif page == "🎥 Video Detection":
    st.header("🎥 **Video Analysis**")
    st.markdown("Upload a video to process and view annotated results on screen.")
    
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    
    if uploaded_video is not None:
        # Store bytes for original video
        original_video_bytes = uploaded_video.getvalue()
        
        # Display original with better layout
        st.subheader("📼 **Original Video**")
        st.video(original_video_bytes)
        
        # Process button
        if st.button("🔍 **PROCESS VIDEO**", type="primary"):
            with st.spinner("🔍 **Processing Video...** (This may take 1-3 minutes)"):

                progress_bar = st.progress(0)
                
                # Save temp file
                progress_bar.progress(10)
                temp_video_path = "temp_input.mp4"
                try:
                    with open(temp_video_path, "wb") as f:
                        f.write(original_video_bytes)
                    
                    # Call API
                    progress_bar.progress(40)
                    with open(temp_video_path, "rb") as f:
                        files = {"file": ("input.mp4", f, "video/mp4")}
                        response = requests.post(f"{API_URL}/detect/video", files=files, timeout=600)
                    
                    progress_bar.progress(70)
                    if response.status_code != 200:
                        st.error(f"🚨 Server returned status {response.status_code}: {response.text}")
                    else:
                        result = safe_parse_json(response)
                        if "error" in result:
                            st.error(f"🚨 **API Error:** {result['error']}")
                        else:
                            st.success(" **Video Processed Successfully!**")
                            
                            # Fetch processed video
                            progress_bar.progress(85)
                            video_path = result.get('output_video_path')
                            if not video_path:
                                st.error("🚨 API did not return an output_video_path.")
                            else:
                                # Request the video file
                                video_response = requests.get(f"{API_URL}/video/{video_path}", stream=True, timeout=120)
                                
                                if video_response.status_code == 200:
                                    progress_bar.progress(100)
                                    processed_video_bytes = video_response.content
                                    
                                    # Display annotated video
                                    st.subheader("🎞️ **Annotated Video**")
                                    st.video(processed_video_bytes)
                                    
                                    # Processing Summary
                                    st.subheader("📊 **Processing Summary**")
                                    col_a, col_b = st.columns(2)
                                    with col_a: 
                                        st.metric("Total Frames", result.get('total_frames', 'N/A'))
                                    with col_b: 
                                        st.metric("Status", "Completed")
                                    
                                    # Download button
                                    st.download_button(
                                        label="📥 **Download Annotated Video**",
                                        data=processed_video_bytes,
                                        file_name="annotated_video.avi",
                                        mime="video/avi"
                                    )
                                else:
                                    st.error(f"🚨 **Failed to fetch video (status {video_response.status_code}):** {video_response.text}")
                except Exception as e:
                    st.error(f"🚨 **Error:** {str(e)}")
                finally:
                    # Cleanup
                    if os.path.exists(temp_video_path):
                        try:
                            os.remove(temp_video_path)
                        except Exception:
                            pass

                progress_bar.empty()





