"""
Enhanced Video Analyzer for InterviewX
Uses proper speech recognition and AI analysis with Groq API
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, Optional
from processor import client, MODEL

# Global whisper model to avoid reloading on every request
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            print("Loading Whisper model (base)...")
            _whisper_model = whisper.load_model("base")
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Whisper model loading failed: {e}")
            _whisper_model = False # Mark as failed to avoid retrying every time
    return _whisper_model if _whisper_model is not False else None

class EnhancedVideoAnalyzer:
    """Enhanced video analysis with proper speech detection and AI scoring"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.whisper_model = get_whisper_model()
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze video with proper speech detection and AI scoring"""
        try:
            results = {
                'transcription': '',
                'speech_detected': False,
                'duration': 0,
                'has_audio': False,
                'fluency_score': 0,
                'eye_contact_score': 0,
                'posture_score': 0,
                'confidence_score': 0,
                'clarity_score': 0,
                'relevance_score': 0
            }
            
            # Extract audio and transcribe
            transcription = self._transcribe_audio()
            results['transcription'] = transcription
            
            # Check if speech was actually detected and it's not an error message
            is_valid_transcription = transcription and len(transcription.strip()) > 5 and not transcription.startswith("[")
            
            if is_valid_transcription:
                results['speech_detected'] = True
                results['has_audio'] = True
                
                # Analyze speech patterns
                speech_analysis = self._analyze_speech_patterns(transcription)
                results.update(speech_analysis)
                
                # Get AI-based analysis if Groq is available
                if client:
                    ai_analysis = self._get_ai_analysis(transcription)
                    results.update(ai_analysis)
            else:
                # No meaningful speech detected
                results['speech_detected'] = False
                results['fluency_score'] = 0
                results['confidence_score'] = 0
                results['clarity_score'] = 0
            
            # Visual analysis (basic)
            visual_analysis = self._analyze_visual_features()
            results.update(visual_analysis)
            
            return results
            
        except Exception as e:
            print(f"Error in video analysis: {e}")
            return {
                'transcription': '[Analysis failed]',
                'speech_detected': False,
                'fluency_score': 0,
                'confidence_score': 0,
                'clarity_score': 0,
                'eye_contact_score': 0,
                'posture_score': 0,
                'error': str(e)
            }
    
    def _transcribe_audio(self) -> str:
        """Extract and transcribe audio from video"""
        try:
            # Check if video file exists first
            if not os.path.exists(self.video_path):
                print(f"Video file does not exist: {self.video_path}")
                return "[Video file not found]"
            
            # Extract audio using ffmpeg
            import tempfile
            import subprocess
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_name = temp_audio.name
            
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-i', self.video_path, 
                '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '16000', '-ac', '1', 
                temp_audio_name, '-y'
            ]
            ffmpeg_result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if not os.path.exists(temp_audio_name) or os.path.getsize(temp_audio_name) < 100:
                print(f"ffmpeg extraction failed or produced empty file. Error: {ffmpeg_result.stderr}")
                return "[Audio extraction failed]"

            # Transcribe using Whisper if available
            transcription = ""
            if self.whisper_model:
                try:
                    result = self.whisper_model.transcribe(temp_audio_name)
                    transcription = result.get('text', '').strip()
                    print(f"Whisper transcription successful: {transcription[:50]}...")
                except Exception as whisper_error:
                    print(f"Whisper transcription failed: {whisper_error}")
            
            # Use Google Speech Recognition as fallback if Whisper failed
            if not transcription or transcription == "[Transcription failed]":
                try:
                    import speech_recognition as sr
                    r = sr.Recognizer()
                    
                    with sr.AudioFile(temp_audio_name) as source:
                        audio_data = r.record(source)
                        transcription = r.recognize_google(audio_data)
                    print(f"Google Speech Recognition result: {transcription[:50] if transcription else 'No speech detected'}...")
                    
                except Exception as speech_error:
                    print(f"Google Speech Recognition failed: {speech_error}")
                    if not transcription:
                        transcription = "[Speech recognition unavailable]"
            
            # Clean up temp file
            try:
                if os.path.exists(temp_audio_name):
                    os.unlink(temp_audio_name)
            except:
                pass  # Ignore cleanup errors
                
            return transcription if transcription else "[Transcription failed]"
                
        except Exception as e:
            print(f"Error in transcription: {e}")
            return "[Transcription failed]"
    
    def _analyze_speech_patterns(self, transcription: str) -> Dict[str, Any]:
        """Analyze speech patterns for fluency and confidence"""
        try:
            # Count filler words
            filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'so', 'well']
            filler_count = sum(1 for word in filler_words if word.lower() in transcription.lower())
            
            # Calculate words per minute (rough estimate)
            words = transcription.split()
            word_count = len(words)
            
            # Basic fluency scoring based on filler words and length
            if word_count > 0:
                filler_ratio = filler_count / word_count
                fluency_score = max(0, 10 - (filler_ratio * 20))  # Penalty for filler words
                
                # Confidence based on response length
                if word_count < 10:
                    confidence_score = 3  # Very short response
                elif word_count < 30:
                    confidence_score = 6  # Short response
                else:
                    confidence_score = 8  # Good response
                    
                clarity_score = max(0, fluency_score - 2)  # Clarity slightly lower than fluency
            else:
                fluency_score = 0
                confidence_score = 0
                clarity_score = 0
            
            return {
                'fluency_score': round(fluency_score, 1),
                'confidence_score': round(confidence_score, 1),
                'clarity_score': round(clarity_score, 1),
                'word_count': word_count,
                'filler_count': filler_count,
                'filler_ratio': round(filler_ratio * 100, 1) if word_count > 0 else 0
            }
            
        except Exception as e:
            print(f"Error analyzing speech patterns: {e}")
            return {
                'fluency_score': 0,
                'confidence_score': 0,
                'clarity_score': 0,
                'word_count': 0,
                'filler_count': 0,
                'filler_ratio': 0
            }
    
    def _get_ai_analysis(self, transcription: str) -> Dict[str, Any]:
        """Get AI-based analysis using Groq"""
        try:
            prompt = f"""
            Analyze this interview response and provide scores for communication skills:
            
            Response: "{transcription}"
            
            Rate the following on a scale of 1-10:
            1. Fluency: How smoothly and naturally they speak
            2. Confidence: How confident and assured they sound
            3. Clarity: How clear and understandable their speech is
            4. Professionalism: How professional their communication style is
            
            Return ONLY a JSON object:
            {{
                "fluency_score": <1-10>,
                "confidence_score": <1-10>,
                "clarity_score": <1-10>,
                "professionalism_score": <1-10>,
                "strengths": "<brief list of communication strengths>",
                "areas_for_improvement": "<brief suggestions for improvement>"
            }}
            """
            
            message = client.chat.completions.create(
                model=MODEL,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": "You are an expert communication analyst for job interviews."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = message.choices[0].message.content
            
            # Parse JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
            
            return {
                'fluency_score': analysis.get('fluency_score', 5),
                'confidence_score': analysis.get('confidence_score', 5),
                'clarity_score': analysis.get('clarity_score', 5),
                'professionalism_score': analysis.get('professionalism_score', 5),
                'ai_strengths': analysis.get('strengths', ''),
                'ai_improvements': analysis.get('areas_for_improvement', '')
            }
            
        except Exception as e:
            print(f"Error in AI analysis: {e}")
            return {
                'fluency_score': 5,
                'confidence_score': 5,
                'clarity_score': 5,
                'professionalism_score': 5,
                'ai_strengths': 'Analysis unavailable',
                'ai_improvements': 'Analysis unavailable'
            }
    
    def _analyze_visual_features(self) -> Dict[str, Any]:
        """Analyze visual features like eye contact and posture"""
        try:
            # Basic visual analysis using OpenCV
            cap = cv2.VideoCapture(self.video_path)
            
            if not cap.isOpened():
                return {
                    'eye_contact_score': np.random.randint(3, 7),
                    'posture_score': np.random.randint(3, 7),
                    'visual_analysis': 'Video could not be opened'
                }
            
            frame_count = 0
            eye_contact_frames = 0
            good_posture_frames = 0
            
            # Load face detection model
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except:
                return {
                    'eye_contact_score': np.random.randint(3, 8),
                    'posture_score': np.random.randint(3, 8),
                    'visual_analysis': 'Face detection unavailable'
                }
            
            while frame_count < 100:  # Analyze first 100 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Face detected - assume good eye contact and posture
                    eye_contact_frames += 1
                    good_posture_frames += 1
                
                frame_count += 1
            
            cap.release()
            
            if frame_count > 0:
                eye_contact_score = min(10, (eye_contact_frames / frame_count) * 10)
                posture_score = min(10, (good_posture_frames / frame_count) * 10)
            else:
                eye_contact_score = np.random.randint(4, 9)
                posture_score = np.random.randint(4, 9)
            
            return {
                'eye_contact_score': round(eye_contact_score, 1),
                'posture_score': round(posture_score, 1),
                'frames_analyzed': frame_count,
                'faces_detected': eye_contact_frames
            }
            
        except Exception as e:
            print(f"Error in visual analysis: {e}")
            return {
                'eye_contact_score': 5,
                'posture_score': 5,
                'visual_analysis': f'Visual analysis failed: {str(e)}'
            }
    
    @staticmethod
    def get_word_suggestions(text: str) -> list:
        """Get word suggestions for improvement"""
        try:
            # Simple word suggestions based on common interview vocabulary
            professional_words = [
                'excellent', 'outstanding', 'significant', 'effective',
                'successful', 'achieved', 'implemented', 'developed',
                'managed', 'led', 'collaborated', 'optimized'
            ]
            
            suggestions = []
            text_lower = text.lower()
            
            for word in professional_words:
                if word not in text_lower:
                    suggestions.append(word)
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            print(f"Error getting word suggestions: {e}")
            return []
