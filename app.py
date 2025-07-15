import streamlit as st
import subprocess
import os
import torch
from PIL import Image
import tempfile
import os
import cv2
import numpy as np
from urllib.request import urlopen
st.set_page_config(page_title="Helmet Detection", layout="centered")
# -------------------------------
# Load YOLOv5 model
# -------------------------------
@st.cache_resource
def load_model():
    return torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

model = load_model()

# -------------------------------
# App UI
# -------------------------------
# st.set_page_config(page_title="Helmet Detection", layout="centered")
st.title("🪖 Helmet Detection App")
st.markdown("Upload an image, use webcam, paste an image URL, or upload a video to detect helmets using your YOLOv5 model.")

# -------------------------------
# Inputs
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
image_url = st.text_input("Or paste image URL here:")
use_webcam = st.checkbox("Use Webcam")
video_file = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])

# -------------------------------
# Image Upload / Webcam / URL
# -------------------------------
img = None

if use_webcam:
    picture = st.camera_input("Take a picture")
    if picture:
        img = Image.open(picture)

elif uploaded_file:
    img = Image.open(uploaded_file)

elif image_url:
    try:
        img = Image.open(urlopen(image_url))
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")

# -------------------------------
# Process image if available
# -------------------------------
if img:
    st.image(img, caption="Original Image", use_column_width=True)
    with st.spinner("Detecting..."):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            img.save(temp.name)
            results = model(temp.name)
            results.render()
            st.image(results.ims[0], caption="Detection Result", use_column_width=True)

            if st.checkbox("Save detection result"):
                os.makedirs("runs", exist_ok=True)
                save_path = os.path.join("runs", "detection_result.jpg")
                Image.fromarray(results.ims[0]).save(save_path)
                st.success(f"Saved to {save_path}")

# -------------------------------
# Video detection
# -------------------------------
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    # Get original video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and output video path
    os.makedirs("runs", exist_ok=True)
    output_path = "runs/output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stframe = st.empty()
    st.info("Processing video and saving result...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        results.render()

        processed_frame = results.ims[0]  # BGR image with boxes

        # Show on Streamlit
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        # Write frame to output video
        out.write(processed_frame)

    cap.release()
    out.release()

    st.success("✅ Video processing complete.")
    st.video(output_path)
    with open(output_path, "rb") as file:
        st.download_button(label="📥 Download Processed Video", data=file, file_name="helmet_output.mp4", mime="video/mp4")


st.markdown("Upload an image/video or run real-time detection using your webcam.")

# ✅ Webcam Toggle Button
if st.button("🎥 Start Real-Time Webcam Detection"):
    st.warning("Launching OpenCV window... Press 'Q' in the new window to stop.")
    subprocess.Popen(["python", "webcam.py"])