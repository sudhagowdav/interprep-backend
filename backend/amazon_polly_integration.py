"""
Amazon Polly Integration for InterviewX
Text-to-speech conversion using AWS Polly
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

class AmazonPollyAPI:
    """Amazon Polly API integration for text-to-speech conversion"""
    
    def __init__(self):
        self.access_key = os.getenv("AWS_ACCESS_KEY")
        self.secret_key = os.getenv("AWS_SECRET_KEY")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.base_url = f"https://polly.{self.region}.amazonaws.com"
        
        if not self.access_key or not self.secret_key:
            print("Warning: AWS credentials not found in environment variables")
    
    def generate_speech_from_text(
        self,
        text: str,
        voice_id: str = "Joanna",
        output_format: str = "mp3",
        sample_rate: int = 22050,
        language_code: str = "en-US",
        engine: str = "neural",
        text_type: str = "ssml"
    ) -> Dict[str, Any]:
        """Convert text to speech using Amazon Polly"""
        try:
            if not self.access_key or not self.secret_key:
                return self._generate_fallback_speech(text)
            
            # Import boto3 for AWS Polly
            import boto3
            
            client = boto3.client(
                'polly',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
            
            # Prepare synthesis request
            response = client.synthesize_speech(
                Text=text,
                OutputFormat=output_format,
                VoiceId=voice_id,
                SampleRate=sample_rate,
                LanguageCode=language_code,
                Engine=engine,
                TextType=text_type
            )
            
            if response.get('AudioStream'):
                # Save audio to file
                import tempfile
                import base64
                
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                    temp_file.write(response['AudioStream'].read())
                    temp_file_path = temp_file.name
                
                # Convert to base64 for response
                with open(temp_file_path, 'rb') as audio_file:
                    audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
                
                return {
                    "success": True,
                    "audio_base64": audio_base64,
                    "voice_id": voice_id,
                    "engine": engine,
                    "language_code": language_code,
                    "output_format": output_format,
                    "sample_rate": sample_rate,
                    "text": text,
                    "content_type": response.get('ContentType', 'audio/mpeg'),
                    "request_characters": response.get('RequestCharacters', 0),
                    "generated_by": "amazon_polly"
                }
            else:
                print("No audio stream received from Polly")
                return self._generate_fallback_speech(text)
                
        except Exception as e:
            print(f"Error generating speech with Amazon Polly: {e}")
            return self._generate_fallback_speech(text)
    
    def get_available_voices(self) -> Dict[str, Any]:
        """Get available voices from Amazon Polly"""
        try:
            if not self.access_key or not self.secret_key:
                return self._get_fallback_voices()
            
            import boto3
            
            client = boto3.client(
                'polly',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
            
            response = client.describe_voices()
            voices = response.get('Voices', [])
            
            # Filter and format voices
            formatted_voices = []
            for voice in voices:
                if voice.get('LanguageCode', '').startswith('en'):
                    formatted_voices.append({
                        "voice_id": voice.get('Id', ''),
                        "name": voice.get('Name', ''),
                        "language_code": voice.get('LanguageCode', ''),
                        "gender": voice.get('Gender', ''),
                        "engine": voice.get('SupportedEngines', []),
                        "language_name": voice.get('LanguageName', ''),
                        "neural": voice.get('Neural', False)
                    })
            
            return {
                "success": True,
                "voices": formatted_voices,
                "total_count": len(formatted_voices),
                "generated_by": "amazon_polly"
            }
            
        except Exception as e:
            print(f"Error getting voices from Amazon Polly: {e}")
            return self._get_fallback_voices()
    
    def generate_speech_with_ssml(
        self,
        ssml_text: str,
        voice_id: str = "Joanna",
        output_format: str = "mp3",
        sample_rate: int = 22050
    ) -> Dict[str, Any]:
        """Generate speech with SSML markup using Amazon Polly"""
        try:
            if not self.access_key or not self.secret_key:
                return self._generate_fallback_speech(ssml_text)
            
            import boto3
            
            client = boto3.client(
                'polly',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
            
            response = client.synthesize_speech(
                Text=ssml_text,
                OutputFormat=output_format,
                VoiceId=voice_id,
                SampleRate=sample_rate,
                TextType='ssml'
            )
            
            if response.get('AudioStream'):
                # Save audio to file
                import tempfile
                import base64
                
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                    temp_file.write(response['AudioStream'].read())
                    temp_file_path = temp_file.name
                
                # Convert to base64 for response
                with open(temp_file_path, 'rb') as audio_file:
                    audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
                
                return {
                    "success": True,
                    "audio_base64": audio_base64,
                    "voice_id": voice_id,
                    "output_format": output_format,
                    "sample_rate": sample_rate,
                    "ssml_text": ssml_text,
                    "content_type": response.get('ContentType', 'audio/mpeg'),
                    "request_characters": response.get('RequestCharacters', 0),
                    "generated_by": "amazon_polly"
                }
            else:
                print("No audio stream received from Polly")
                return self._generate_fallback_speech(ssml_text)
                
        except Exception as e:
            print(f"Error generating SSML speech with Amazon Polly: {e}")
            return self._generate_fallback_speech(ssml_text)
    
    def _generate_fallback_speech(self, text: str) -> Dict[str, Any]:
        """Generate fallback speech when API is unavailable"""
        return {
            "success": False,
            "text": text,
            "audio_base64": "",
            "generated_by": "fallback",
            "error": "Amazon Polly API not available"
        }
    
    def _get_fallback_voices(self) -> Dict[str, Any]:
        """Get fallback voices when API is unavailable"""
        return {
            "success": False,
            "voices": [
                {
                    "voice_id": "Joanna",
                    "name": "Joanna (Professional Female)",
                    "language_code": "en-US",
                    "gender": "Female",
                    "engine": ["neural", "standard"],
                    "language_name": "US English",
                    "neural": True
                },
                {
                    "voice_id": "Matthew",
                    "name": "Matthew (Professional Male)",
                    "language_code": "en-US",
                    "gender": "Male",
                    "engine": ["neural", "standard"],
                    "language_name": "US English",
                    "neural": True
                },
                {
                    "voice_id": "Amy",
                    "name": "Amy (British Female)",
                    "language_code": "en-GB",
                    "gender": "Female",
                    "engine": ["neural", "standard"],
                    "language_name": "British English",
                    "neural": True
                }
            ],
            "generated_by": "fallback"
        }

# Global instance
amazon_polly_api = AmazonPollyAPI()
