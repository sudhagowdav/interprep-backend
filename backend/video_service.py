"""
Video Service Module - Handles video streaming and serving
Provides real-time video playback from recordings folder
"""

import os
import random
from pathlib import Path
from config import RECORDINGS_FOLDER

def get_video_list():
    """Get list of all available videos from recordings folder"""
    try:
        video_extensions = {'.mp4', '.webm', '.avi', '.mov', '.mkv'}
        video_files = []
        
        if os.path.exists(RECORDINGS_FOLDER):
            for file in os.listdir(RECORDINGS_FOLDER):
                file_path = os.path.join(RECORDINGS_FOLDER, file)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file)
                    if ext.lower() in video_extensions:
                        video_files.append({
                            'filename': file,
                            'path': file_path,
                            'size': os.path.getsize(file_path),
                            'extension': ext.lower()
                        })
        
        return sorted(video_files, key=lambda x: x['filename'])
    except Exception as e:
        print(f"Error getting video list: {e}")
        return []


def get_random_video():
    """Get a random video from recordings folder for interviewer"""
    try:
        videos = get_video_list()
        if videos:
            return random.choice(videos)
        return None
    except Exception as e:
        print(f"Error getting random video: {e}")
        return None


def get_video_by_filename(filename):
    """Get video by filename"""
    try:
        video_path = os.path.join(RECORDINGS_FOLDER, filename)
        if os.path.exists(video_path):
            return {
                'filename': filename,
                'path': video_path,
                'size': os.path.getsize(video_path)
            }
        return None
    except Exception as e:
        print(f"Error getting video: {e}")
        return None


def get_video_stream_url(video_filename):
    """Generate a stream URL for video"""
    return f"/api/videos/stream/{video_filename}"


def get_video_thumbnail(video_filename):
    """Get thumbnail for video (placeholder)"""
    return f"/api/videos/thumbnail/{video_filename}"
