import os
import asyncio
import streamlit as st
import cv2
import numpy as np
import time
import yt_dlp
from ultralytics import YOLO

# Suppress unnecessary PyTorch debugging messages
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"

# Fix asyncio event loop conflict
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.title("YOLO Object Detection on YouTube Videos")

# Input for YouTube video
youtube_url = st.text_input("Enter YouTube video URL:")

def extract_youtube_video_id(url):
    """Extract the video ID from a YouTube URL."""
    if "youtube.com" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return None

def get_youtube_video_url(youtube_url):
    """Extracts the best available video stream URL using yt-dlp."""
    ydl_opts = {
        "quiet": True,
        "noplaylist": True,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_info = ydl.extract_info(youtube_url, download=False)

            # Check if a valid URL exists
            if "url" in video_info:
                return video_info["url"]
            elif "formats" in video_info:
                return video_info["formats"][-1]["url"]  # Get the last available format
            else:
                st.error("Failed to extract video stream URL. Video may be restricted or unavailable.")
                return None
    except Exception as e:
        st.error(f"Error fetching video: {e}")
        return None

if youtube_url:  # Only load YOLO when a URL is provided
    video_id = extract_youtube_video_id(youtube_url)

    if video_id:
        st.markdown(f"""
            <iframe width="700" height="400" src="https://www.youtube.com/embed/{video_id}" 
            frameborder="0" allowfullscreen></iframe>
        """, unsafe_allow_html=True)

        st.success("Video embedded successfully! 🎥 Now processing frames...")

        # Load YOLO Model only after URL is entered
        model = YOLO("yolov8n.pt")

        # Get direct video URL
        video_url = get_youtube_video_url(youtube_url)

        if video_url:
            cap = cv2.VideoCapture(video_url)
            frame_window = st.empty()  # Placeholder for displaying frames

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)[0]

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = results.names[cls]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB")

                time.sleep(1)  # Process every 1 second to reduce load

            cap.release()
        else:
            st.error("Could not retrieve the video stream. Try a different YouTube video.")
    else:
        st.error("Invalid YouTube URL. Please enter a valid video link.")
