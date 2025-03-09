import os
import asyncio
import streamlit as st
import cv2
import numpy as np
import time
import yt_dlp
import ffmpeg
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


def extract_youtube_video_id(url: str):
    """Extract the video ID from a YouTube URL."""
    if "youtube.com" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return None


def get_youtube_video_url(youtube_url: str):
    """
    Extracts a direct video stream URL using yt-dlp.
    Tries to get an MP4-based format for simpler FFmpeg parsing.
    """
    # Primary approach: bestvideo[ext=mp4] + bestaudio
    ydl_opts_main = {
        "quiet": True,
        "noplaylist": True,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
    }

    # Fallback: just grab any mp4 format if the above fails
    ydl_opts_fallback = {
        "quiet": True,
        "noplaylist": True,
        "format": "mp4",
    }

    # Try main approach first
    for ydl_opts in (ydl_opts_main, ydl_opts_fallback):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                video_info = ydl.extract_info(youtube_url, download=False)
                if "url" in video_info:
                    return video_info["url"]
                elif "formats" in video_info:
                    # Last available format
                    return video_info["formats"][-1]["url"]
                else:
                    st.error("Failed to extract a valid video stream URL.")
        except Exception as e:
            st.warning(f"Attempt with format={ydl_opts['format']} failed: {e}")

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
            # ---------- FFmpeg-based frame reading with better error handling ----------
            try:
                # Probe the stream to get width/height
                probe = ffmpeg.probe(video_url)
                video_stream = next(
                    (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
                    None
                )
                if not video_stream:
                    st.error("No valid video stream found!")
                else:
                    width = int(video_stream["width"])
                    height = int(video_stream["height"])

                    # Create an FFmpeg process to pipe raw video frames
                    # NOTE: Some DRM or adaptive streams may still fail here.
                    process = (
                        ffmpeg
                        .input(video_url)
                        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
                        .run_async(pipe_stdout=True, pipe_stderr=True)
                    )

                    frame_window = st.empty()  # Placeholder for displaying frames

                    while True:
                        in_bytes = process.stdout.read(width * height * 3)
                        if not in_bytes:
                            # End of stream or no data
                            break

                        # Make a writable NumPy array from raw bytes
                        frame = np.frombuffer(in_bytes, np.uint8).copy().reshape((height, width, 3))

                        # YOLO inference
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

                        # Convert for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_window.image(frame_rgb, channels="RGB")
                        time.sleep(1)  # reduce CPU usage

                    process.wait()

            except ffmpeg.Error as e:
                # Show the full stderr message for debugging
                error_output = e.stderr.decode('utf-8', errors='replace')
                st.error(f"FFmpeg streaming error: {error_output}")

            except Exception as e:
                # Catch other, non-ffmpeg exceptions
                st.error(f"Unexpected error: {e}")
            # --------------------------------------------------------------------------
        else:
            st.error("Could not retrieve the video stream. Try a different YouTube video.")
    else:
        st.error("Invalid YouTube URL. Please enter a valid video link.")
