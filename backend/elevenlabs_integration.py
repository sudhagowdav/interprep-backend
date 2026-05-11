"""
ElevenLabs API Integration for InterviewX
Realistic voice generation for AI interviewer
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

class ElevenLabsAPI:
    """ElevenLabs API integration for realistic voice generation"""
    
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.base_url = "https://api.elevenlabs.io/v1"
        
        if not self.api_key:
            print("Warning: ELEVENLABS_API_KEY not found in environment variables")
    
    def generate_speech_from_text(
        self,
        text: str,
        voice_id: str = "rachel",
        model_id: str = "eleven_multilingual_v2",
        stability: float = 0.75,
        similarity_boost: float = 0.75,
        style: str = "narrator",
        optimize_streaming_latency: int = 4,
        output_format: str = "mp3_44100_192",
        sample_rate: int = 44100
    ) -> Dict[str, Any]:
        """Convert text to realistic speech using ElevenLabs"""
        try:
            if not self.api_key:
                return self._generate_fallback_speech(text)
            
            headers = {
                "Accept": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "voice_id": voice_id,
                "model_id": model_id,
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                    "style": style,
                    "optimize_streaming_latency": optimize_streaming_latency,
                    "speaker_boost": "none"
                },
                "output_format": output_format
            }
            
            response = requests.post(
                f"{self.base_url}/text-to-speech",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                return {
                    "success": True,
                    "audio_url": result.get("audio_url", ""),
                    "voice_id": voice_id,
                    "model_id": model_id,
                    "text": text,
                    "duration_seconds": result.get("duration_seconds", 0),
                    "size_bytes": result.get("size_bytes", 0),
                    "generated_by": "elevenlabs"
                }
            else:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return self._generate_fallback_speech(text)
                
        except Exception as e:
            print(f"Error generating speech with ElevenLabs: {e}")
            return self._generate_fallback_speech(text)
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available voices from ElevenLabs"""
        try:
            if not self.api_key:
                return self._get_fallback_voices()
            
            headers = {
                "Accept": "application/json",
                "xi-api-key": self.api_key
            }
            
            response = requests.get(
                f"{self.base_url}/voices",
                headers=headers
            )
            
            if response.status_code == 200:
                voices = response.json()
                
                return {
                    "success": True,
                    "voices": voices.get("voices", []),
                    "generated_by": "elevenlabs"
                }
            else:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return self._get_fallback_voices()
                
        except Exception as e:
            print(f"Error getting voices: {e}")
            return self._get_fallback_voices()
    
    def generate_interviewer_script(
        self,
        script_type: str = "interview",
        role: str = "interviewer",
        voice_id: str = "rachel",
        model_id: str = "eleven_multilingual_v2",
        style: str = "professional",
        num_questions: int = 5,
        difficulty: str = "intermediate"
    ) -> Dict[str, Any]:
        """Generate complete interviewer script"""
        try:
            if not self.api_key:
                return self._generate_fallback_script(script_type, role)
            
            # Get available voices first
            voices_response = self.get_available_voices()
            if not voices_response["success"]:
                return self._generate_fallback_script(script_type, role)
            
            voices = voices_response["voices"]
            interviewer_voice = None
            for voice in voices:
                if voice.get("voice_id") == voice_id:
                    interviewer_voice = voice
                    break
            
            if not interviewer_voice:
                interviewer_voice = voices[0] if voices else {"voice_id": "rachel"}
            
            prompt = f"""
            Generate a complete {script_type} script for a {role} position.
            
            Role: {role}
            Script Type: {script_type}
            Number of Questions: {num_questions}
            Difficulty: {difficulty}
            Voice: {interviewer_voice.get('name', 'Unknown Voice')}
            
            Generate {num_questions} interview questions covering:
            1. Introduction and icebreaker
            2. Technical questions
            3. Behavioral questions
            4. Problem-solving scenarios
            5. Closing questions
            
            For each question, include:
            - Clear question text
            - Estimated response time
            - Follow-up suggestions
            - Interviewer notes
            
            Make it sound natural and professional. Use the {style} style.
            
            Return ONLY a JSON object:
            {{
                "script_type": "{script_type}",
                "role": "{role}",
                "voice_id": "{voice_id}",
                "model_id": "{model_id}",
                "style": "{style}",
                "num_questions": {num_questions},
                "difficulty": "{difficulty}",
                "questions": [
                    {{
                        "id": 1,
                        "type": "introduction",
                        "text": "Clear question text here",
                        "estimated_time": "2-3 minutes",
                        "follow_up": ["What specific experience interests you most?"]
                    }},
                    ...
                ],
                "introduction": "Opening greeting and setup",
                "closing": "Professional closing and next steps"
            }}
            """
            
            headers = {
                "Accept": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": prompt,
                "voice_id": voice_id,
                "model_id": model_id,
                "voice_settings": {
                    "stability": 0.8,
                    "similarity_boost": 0.75,
                    "style": style,
                    "optimize_streaming_latency": 4
                },
                "output_format": "mp3_44100_192"
            }
            
            response = requests.post(
                f"{self.base_url}/text-to-speech",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                return {
                    "success": True,
                    "script": result.get("text", ""),
                    "audio_url": result.get("audio_url", ""),
                    "voice_id": voice_id,
                    "model_id": model_id,
                    "script_type": script_type,
                    "role": role,
                    "generated_by": "elevenlabs"
                }
            else:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return self._generate_fallback_script(script_type, role)
                
        except Exception as e:
            print(f"Error generating script: {e}")
            return self._generate_fallback_script(script_type, role)
    
    def _generate_fallback_speech(self, text: str) -> Dict[str, Any]:
        """Generate fallback speech when API is unavailable"""
        return {
            "success": False,
            "text": text,
            "audio_url": "",
            "generated_by": "fallback",
            "error": "ElevenLabs API not available"
        }
    
    def _get_fallback_voices(self) -> Dict[str, Any]:
        """Get fallback voices when API is unavailable"""
        return {
            "success": False,
            "voices": [
                {
                    "voice_id": "rachel",
                    "name": "Rachel (Professional)",
                    "gender": "female",
                    "age": "young adult"
                },
                {
                    "voice_id": "domi",
                    "name": "Domi (Friendly)",
                    "gender": "male",
                    "age": "middle aged"
                }
            ],
            "generated_by": "fallback"
        }
    
    def _generate_fallback_script(self, script_type: str, role: str) -> Dict[str, Any]:
        """Generate fallback script when API is unavailable"""
        return {
            "success": False,
            "script_type": script_type,
            "role": role,
            "script": f"Sample {script_type} script for {role} position would go here...",
            "generated_by": "fallback"
        }

# Global instance
elevenlabs_api = ElevenLabsAPI()
