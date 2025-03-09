import os
import streamlit as st
import cv2
import numpy as np
import time
import yt_dlp
import imageio
from ultralytics import YOLO

# Suppress unnecessary PyTorch debugging messages
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"

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
            return video_info.get("url", None)
    except Exception as e:
        st.error(f"Error fetching video: {e}")
        return None

if youtube_url:
    video_id = extract_youtube_video_id(youtube_url)

    if video_id:
        st.markdown(f"""
            <iframe width="700" height="400" src="https://www.youtube.com/embed/{video_id}" 
            frameborder="0" allowfullscreen></iframe>
        """, unsafe_allow_html=True)

        st.success("Video embedded successfully! 🎥 Now processing frames...")

        # Load YOLO Model
        model = YOLO("yolov8n.pt")

        # Get direct video URL
        video_url = get_youtube_video_url(youtube_url)

        if video_url:
            try:
                video_reader = imageio.get_reader(video_url, format='ffmpeg')
                frame_window = st.empty()  # Placeholder for displaying frames
                
                for frame in video_reader:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
                    
                    # Run YOLO on the frame
                    results = model(frame)[0]

                    # Draw bounding boxes
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

                    time.sleep(1)  # Process every second to reduce load

            except Exception as e:
                st.error(f"Error processing video: {e}")
        else:
            st.error("Could not retrieve the video stream. Try a different YouTube video.")
    else:
        st.error("Invalid YouTube URL. Please enter a valid video link.")
