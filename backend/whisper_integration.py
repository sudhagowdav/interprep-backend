"""
OpenAI Whisper Integration for InterviewX
Advanced speech-to-text conversion for video analysis
"""

import os
import json
import requests
from typing import Dict, Any, Optional
from datetime import datetime

class WhisperAPI:
    """OpenAI Whisper API integration for speech-to-text conversion"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
    
    def transcribe_audio_file(
        self,
        audio_file_path: str,
        model: str = "whisper-1"
    ) -> Dict[str, Any]:
        """Transcribe audio file using OpenAI Whisper API"""
        try:
            if not self.api_key:
                return self._generate_fallback_transcription()
            
            # Read audio file
            with open(audio_file_path, 'rb') as audio_file:
                files = {
                    'file': (audio_file_path, audio_file.read(), 'audio/wav')
                }
                
                data = {
                    'model': model,
                    'language': 'en',
                    'response_format': 'json'
                }
                
                headers = {
                    'Authorization': f'Bearer {self.api_key}'
                }
                
                response = requests.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                result = response.json()
                transcription = result.get('text', '')
                
                return {
                    "success": True,
                    "transcription": transcription,
                    "model": model,
                    "duration": result.get('duration', 0),
                    "language": result.get('language', 'en'),
                    "transcribed_by": "whisper",
                    "confidence": result.get('avg_logprob', 0),
                    "words_per_minute": self._calculate_wpm(transcription, result.get('duration', 60))
                }
            else:
                print(f"Whisper API error: {response.status_code} - {response.text}")
                return self._generate_fallback_transcription()
                
        except Exception as e:
            print(f"Error transcribing with Whisper: {e}")
            return self._generate_fallback_transcription()
    
    def transcribe_video_audio(
        self,
        video_path: str,
        model: str = "whisper-1"
    ) -> Dict[str, Any]:
        """Extract audio from video and transcribe using Whisper"""
        try:
            if not self.api_key:
                return self._generate_fallback_transcription()
            
            # Extract audio from video using ffmpeg
            import tempfile
            import subprocess
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                # Extract audio from video
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vn', '-acodec', 'pcm_s16le',
                    '-ar', '16000', '-ac', '1',
                    temp_audio.name, '-y'
                ]
                subprocess.run(cmd, capture_output=True, check=False)
                
                # Transcribe the extracted audio
                result = self.transcribe_audio_file(temp_audio.name, model)
                
                # Clean up temp file
                if os.path.exists(temp_audio.name):
                    os.unlink(temp_audio.name)
                
                return result
                
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            return self._generate_fallback_transcription()
    
    def transcribe_with_diarization(
        self,
        audio_file_path: str,
        num_speakers: int = 2,
        model: str = "whisper-1"
    ) -> Dict[str, Any]:
        """Transcribe with speaker diarization"""
        try:
            if not self.api_key:
                return self._generate_fallback_transcription()
            
            with open(audio_file_path, 'rb') as audio_file:
                files = {
                    'file': (audio_file_path, audio_file.read(), 'audio/wav')
                }
                
                data = {
                    'model': model,
                    'language': 'en',
                    'response_format': 'json',
                    'temperature': 0.0,
                    'diarize': True,
                    'max_speakers': num_speakers
                }
                
                headers = {
                    'Authorization': f'Bearer {self.api_key}'
                }
                
                response = requests.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Process diarization results
                segments = result.get('segments', [])
                speakers = {}
                
                for segment in segments:
                    speaker = segment.get('speaker', 'Unknown')
                    if speaker not in speakers:
                        speakers[speaker] = []
                    speakers[speaker].append({
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'text': segment.get('text', ''),
                        'confidence': segment.get('avg_logprob', 0)
                    })
                
                return {
                    "success": True,
                    "transcription": result.get('text', ''),
                    "segments": segments,
                    "speakers": speakers,
                    "model": model,
                    "diarized": True,
                    "transcribed_by": "whisper"
                }
            else:
                print(f"Whisper diarization API error: {response.status_code} - {response.text}")
                return self._generate_fallback_transcription()
                
        except Exception as e:
            print(f"Error in diarization: {e}")
            return self._generate_fallback_transcription()
    
    def _calculate_wpm(self, transcription: str, duration_seconds: float) -> float:
        """Calculate words per minute"""
        if not transcription or duration_seconds <= 0:
            return 0
        
        words = len(transcription.split())
        minutes = duration_seconds / 60
        wpm = words / minutes if minutes > 0 else 0
        
        return round(wpm, 1)
    
    def _generate_fallback_transcription(self) -> Dict[str, Any]:
        """Generate fallback transcription when API is unavailable"""
        return {
            "success": False,
            "transcription": "[Speech recognition unavailable]",
            "transcribed_by": "fallback",
            "error": "Whisper API not available"
        }

# Global instance
whisper_api = WhisperAPI()
