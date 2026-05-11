"""
Configuration module for InterviewX
Stores configuration variables and settings
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-70b-versatile"

# Database Configuration
DATABASE_PATH = "interviewx.db"

# Flask/Server Configuration
FLASK_ENV = os.getenv("FLASK_ENV", "development")
FLASK_DEBUG = os.getenv("FLASK_DEBUG", True)
SERVER_PORT = int(os.getenv("SERVER_PORT", 5000))
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")

# File Upload Configuration
UPLOAD_FOLDER = "uploads"
RECORDINGS_FOLDER = "recordings"
ALLOWED_EXTENSIONS = {'pdf'}
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB

# Interview Configuration
NUM_INTERVIEW_QUESTIONS = 5
RESPONSE_TIME_LIMIT = 300  # seconds (5 minutes)

# CORS Configuration
CORS_ORIGINS = "*"

# Create required directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
