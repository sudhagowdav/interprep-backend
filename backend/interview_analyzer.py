"""
Video and Voice Analysis Module for Interview Processing
Analyzes communication skills, posture, filler words, and speech quality using Pre-trained Models
"""

import cv2
import numpy as np
import mediapipe as mp
import librosa
import speech_recognition as sr
from scipy import signal
import os
import tempfile
import uuid
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Initialize mediapipe
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
face_detection = mp_face_detection.FaceDetection()

# Import Whisper API integration
from whisper_integration import whisper_api
WHISPER_AVAILABLE = True # API is always available if configured

# Enhanced filler words dictionary
FILLER_WORDS = {
    'umm', 'uhh', 'uh', 'um', 'erm', 'err', 'hmm', 'huh', 'ah', 'ahh', 'erm',
    'like', 'you know', 'actually', 'basically', 'literally', 'honestly',
    'right', 'okay', 'so', 'well', 'i mean', 'you see', 'i think',
    'kind of', 'sort of', 'just', 'really', 'very', 'actually',
}

# Word upgrades for better vocabulary
WORD_UPGRADES = {
    'good': ['excellent', 'exceptional', 'outstanding'],
    'bad': ['suboptimal', 'challenging', 'areas for improvement'],
    'think': ['believe', 'consider', 'recognize'],
    'maybe': ['potentially', 'possibly', 'conceivably'],
    'really': ['significantly', 'substantially', 'considerably'],
    'like': ['such as', 'for example', 'including'],
    'stuff': ['components', 'elements', 'aspects'],
    'things': ['strategies', 'initiatives', 'implementations'],
    'a lot': ['substantially', 'extensively', 'considerably'],
    'pretty': ['quite', 'considerably', 'notably'],
    'got': ['obtained', 'acquired', 'achieved'],
    'okay': ['understood', 'agreed', 'confirmed'],
}


class InterviewAnalyzer:
    """Analyzes interview videos and audio for communication quality using pre-trained models"""

    def __init__(self, video_path):
        self.video_path = video_path
        self.results = {
            'posture_score': 0,
            'eye_contact_score': 0,
            'fluency_score': 0,
            'filler_words': [],
            'filler_count': 0,
            'words_per_minute': 0,
            'overall_score': 0,
            'suggestions': [],
            'transcription': '',
            'pronunciation_issues': [],
            'clarity_score': 5,
            'confidence_score': 5,
            'relevance_score': 5,
            'feedback': 'Analysis in progress...'
        }

    def analyze(self):
        """Run complete analysis"""
        try:
            # Analyze video for posture and eye contact
            self._analyze_video()
            # Extract and analyze audio
            self._analyze_audio()
            # Calculate overall score
            self._calculate_overall_score()
            # Generate AI feedback
            self._generate_ai_feedback()
            return self.results
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            self.results['suggestions'].append(f"Analysis error: {str(e)}")
        
        return self.results

    def _analyze_video(self):
        """Analyze video for posture, eye contact, and movement"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {self.video_path}")
                return

            frame_count = 0
            posture_scores = []
            eye_contact_frames = 0
            total_face_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # Sample every 10th frame for speed
                if frame_count % 10 != 0:
                    continue

                # Detect pose and faces
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    continue

                # Pose detection
                pose_results = pose.process(rgb_frame)
                if pose_results.pose_landmarks:
                    posture_score = self._evaluate_posture(
                        pose_results.pose_landmarks)
                    posture_scores.append(posture_score)

                # Face detection for eye contact
                face_results = face_detection.process(rgb_frame)
                if face_results.detections:
                    total_face_frames += 1
                    if self._has_good_eye_contact(face_results.detections[0]):
                        eye_contact_frames += 1

            cap.release()

            # Calculate scores - more stringent evaluation
            if posture_scores:
                avg_posture = np.mean(posture_scores)
                self.results['posture_score'] = int(avg_posture * 10)
                self.results['posture_score'] = min(10, max(0, self.results['posture_score']))
            else:
                self.results['posture_score'] = 5

            if total_face_frames > 0:
                eye_contact_percentage = (
                    eye_contact_frames / total_face_frames) * 100
                self.results['eye_contact_score'] = min(10, int(eye_contact_percentage / 10))
            else:
                self.results['eye_contact_score'] = 5

            # Add video-based suggestions
            if self.results['posture_score'] < 6:
                self.results['suggestions'].append(
                    "✓ Posture: Keep your back straight, shoulders relaxed, and maintain an upright position throughout"
                )
            if self.results['eye_contact_score'] < 6:
                self.results['suggestions'].append(
                    "✓ Eye Contact: Look directly at the camera more frequently to build stronger connection"
                )

        except Exception as e:
            print(f"Error analyzing video: {str(e)}")

    def _evaluate_posture(self, landmarks):
        """Evaluate posture based on MediaPipe landmarks (0-1 score)"""
        try:
            # Get key points
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_ear = landmarks[7]
            right_ear = landmarks[8]
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            # Check if shoulders are level (good posture indicator)
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            
            # Check if hips are level
            hip_diff = abs(left_hip.y - right_hip.y)

            # Check ear alignment with shoulder
            ear_shoulder_alignment = min(
                abs(left_ear.y - left_shoulder.y),
                abs(right_ear.y - right_shoulder.y)
            )

            # More strict scoring
            posture_score = 1.0
            posture_score -= min(shoulder_diff * 8, 0.4)  # Stricter shoulder check
            posture_score -= min(hip_diff * 5, 0.3)      # Hip alignment
            posture_score -= min(ear_shoulder_alignment * 3, 0.3)  # Ear-shoulder alignment
            
            return max(0, min(1, posture_score))
        except Exception:
            return 0.5

    def _has_good_eye_contact(self, detection):
        """Check if face is looking at camera"""
        try:
            bounding_box = detection.location_data.relative_bounding_box
            center_x = bounding_box.xmin + bounding_box.width / 2
            center_y = bounding_box.ymin + bounding_box.height / 2

            # Stricter eye contact detection (more centered = better)
            if 0.25 < center_x < 0.75 and 0.15 < center_y < 0.6:
                return True
            return False
        except Exception:
            return False

    def _analyze_audio(self):
        """Extract audio and analyze speech using Whisper if available"""
        try:
            audio_path = self._extract_audio_from_video()
            
            if not audio_path:
                self.results['suggestions'].append(
                    "✓ Audio: No audio detected in video. Please ensure your microphone is working."
                )
                return

            # Load audio
            y, sr_rate = librosa.load(audio_path, sr=None)
            
            # Check if audio is silent
            if self._is_audio_silent(y):
                self.results['suggestions'].append(
                    "✓ Audio: The audio appears to be silent or very low volume. Please speak clearly and ensure proper microphone levels."
                )
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                return

            # Transcribe using Whisper if available, else use Google Speech Recognition
            self._transcribe_audio_improved(audio_path, y, sr_rate)
            
            # Analyze only if transcription is valid
            if self.results['transcription'] and len(self.results['transcription'].strip()) > 5:
                self._detect_filler_words()
                self._analyze_speech_rate(y, sr_rate)
                self._analyze_fluency_improved(y, sr_rate)
                self._analyze_clarity()
            
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)

        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            self.results['transcription'] = "[Audio analysis failed]"

    def _is_audio_silent(self, y, threshold=0.01):
        """Check if audio is essentially silent"""
        rms = np.sqrt(np.mean(y**2))
        return rms < threshold

    def _extract_audio_from_video(self):
        """Extract audio from video file"""
        try:
            import subprocess
            temp_dir = tempfile.gettempdir()
            temp_audio = os.path.join(temp_dir, f"audio_{uuid.uuid4().hex[:8]}.wav")

            cmd = [
                'ffmpeg', '-y', '-i', self.video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                temp_audio
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0 or not os.path.exists(temp_audio):
                return None

            # Check file size
            if os.path.getsize(temp_audio) < 1000:  # Less than 1KB = likely silent
                os.remove(temp_audio)
                return None
                
            return temp_audio
        except Exception as e:
            print(f"Error extracting audio: {str(e)}")
            return None

    def _transcribe_audio_improved(self, audio_path, y, sr_rate):
        """Transcribe audio using Whisper API (most accurate) or fallback to Google"""
        try:
            # Use Whisper API for more accurate transcription
            result = whisper_api.transcribe_audio_file(audio_path)
            
            if result.get('success') and result.get('transcription'):
                self.results['transcription'] = result['transcription'].strip()
                print(f"✓ Whisper API transcription: {self.results['transcription'][:100]}")
            else:
                # Fallback to Google Speech Recognition
                print("Falling back to Google Speech Recognition...")
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_path) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)
                    self.results['transcription'] = text.strip()
                    print(f"✓ Google transcription: {self.results['transcription'][:100]}")
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            self.results['transcription'] = ""

    def _analyze_clarity(self):
        """Analyze speech clarity and articulation from transcription"""
        text = self.results['transcription'].lower()
        
        # Check vocabulary quality
        complex_words = sum(1 for word in text.split() if len(word) > 8)
        total_words = len(text.split())
        
        clarity_score = 5
        if total_words > 0:
            vocabulary_ratio = complex_words / total_words
            clarity_score = int(5 + (vocabulary_ratio * 3))  # 5-8 range
        
        # Penalize for repetition
        words = text.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            clarity_score = int(clarity_score * unique_ratio)
        
        self.results['clarity_score'] = min(10, max(3, clarity_score))

    def _detect_filler_words(self):
        """Detect filler words from transcription with better accuracy"""
        text = self.results['transcription'].lower()
        filler_words_found = []
        
        # Split into words for more accurate detection
        words = text.split()
        
        for word in FILLER_WORDS:
            count = 0
            for w in words:
                # Exact word matching
                if w.strip('.,!?;:') == word or (len(w) > 3 and word in w):
                    count += 1
            if count > 0:
                filler_words_found.extend([word] * count)

        self.results['filler_words'] = list(set(filler_words_found))
        self.results['filler_count'] = len(filler_words_found)

        # Penalize confidence score for filler words
        if self.results['filler_count'] > 0:
            penalty = min(self.results['filler_count'] * 0.5, 3)
            self.results['confidence_score'] = max(2, 8 - int(penalty))
            
            if self.results['filler_count'] > 5:
                self.results['suggestions'].append(
                    f"✓ Filler Words: Reduce usage of '{', '.join(self.results['filler_words'][:3])}'. Replace with brief pauses instead."
                )

    def _analyze_speech_rate(self, y, sr_rate):
        """Calculate words per minute with improved accuracy"""
        try:
            duration = len(y) / sr_rate  # Duration in seconds
            word_count = len(self.results['transcription'].split())
            wpm = (word_count / duration) * 60 if duration > 0 else 0

            self.results['words_per_minute'] = int(wpm)

            # Normal speech is 120-150 WPM, optimal is 140
            if wpm < 100:
                self.results['suggestions'].append(
                    "✓ Speech Rate: Speak a bit faster to maintain better listener engagement (target: 120-150 WPM)"
                )
            elif wpm > 180:
                self.results['suggestions'].append(
                    "✓ Speech Rate: Slow down slightly for better clarity (target: 120-150 WPM)"
                )
            else:
                self.results['suggestions'].append(
                    f"✓ Speech Rate: Excellent pace ({int(wpm)} WPM). Maintain this natural rhythm."
                )

        except Exception as e:
            print(f"Error calculating speech rate: {str(e)}")

    def _analyze_fluency_improved(self, y, sr_rate):
        """Analyze fluency using multiple metrics"""
        try:
            # Energy-based analysis
            S = np.abs(librosa.stft(y))
            energy = np.sqrt((S**2).mean(axis=0))
            threshold = np.mean(energy) * 0.3
            
            pauses = np.where(energy < threshold)[0]
            pause_density = len(pauses) / len(energy) if len(energy) > 0 else 0
            
            # Calculate fluency score (0-10)
            # Lower pause density = better fluency
            fluency_base = 10 - (pause_density * 5)
            fluency_base = max(2, min(10, fluency_base))
            
            self.results['fluency_score'] = int(fluency_base)
            
            # Add suggestion if too many pauses
            if pause_density > 0.3:
                self.results['suggestions'].append(
                    "✓ Fluency: Minimize pauses and speak more continuously to improve flow"
                )

        except Exception as e:
            print(f"Error analyzing fluency: {str(e)}")

    def _calculate_overall_score(self):
        """Calculate overall interview score with relevance based on response quality"""
        transcription = self.results.get('transcription', '').strip()
        
        # Evaluate relevance based on transcription content
        if transcription and len(transcription) > 10:
            words = transcription.lower().split()
            word_count = len(words)
            
            # Relevance: Did they actually answer? (word count indicator)
            if word_count < 20:
                self.results['relevance_score'] = 3
            elif word_count < 50:
                self.results['relevance_score'] = 6
            elif word_count < 150:
                self.results['relevance_score'] = 8
            else:
                self.results['relevance_score'] = 9
                
            # Bonus for specific keywords
            if any(keyword in words for keyword in ['experience', 'project', 'learned', 'developed', 'achieved']):
                self.results['relevance_score'] = min(10, self.results['relevance_score'] + 1)
        else:
            self.results['relevance_score'] = 2
            self.results['feedback'] = "No valid response detected. Please provide a more complete answer."

        # Calculate overall from all components
        scores = [
            self.results['posture_score'] * 0.15,
            self.results['eye_contact_score'] * 0.15,
            self.results['fluency_score'] * 0.25,
            self.results['clarity_score'] * 0.20,
            self.results['confidence_score'] * 0.15,
            self.results['relevance_score'] * 0.10
        ]
        
        self.results['overall_score'] = int(sum(scores))
        self.results['overall_score'] = min(10, max(1, self.results['overall_score']))

    def _generate_ai_feedback(self):
        """Generate professional AI feedback"""
        try:
            from processor import client, MODEL
            if not client:
                self._generate_basic_feedback()
                return

            transcription_preview = self.results['transcription'][:300] if self.results['transcription'] else "[No transcription]"
            
            prompt = f"""
            You are an expert interview coach. Analyze this interview performance:
            
            Metrics (0-10):
            - Posture: {self.results['posture_score']}
            - Eye Contact: {self.results['eye_contact_score']}
            - Fluency: {self.results['fluency_score']}
            - Clarity: {self.results['clarity_score']}
            - Confidence: {self.results['confidence_score']}
            - Answer Relevance: {self.results['relevance_score']}
            
            Speech Analysis:
            - Filler Words: {self.results['filler_count']} ({', '.join(self.results['filler_words'][:5]) if self.results['filler_words'] else 'None'})
            - Words/Minute: {self.results['words_per_minute']}
            - Sample: "{transcription_preview}"
            
            Provide JSON feedback with:
            {{"
                "strengths": "1 sentence on what was done well",
                "improvements": "1 sentence on key improvement area",
                "word_upgrades": ["2-3 specific vocabulary suggestions based on the response"],
                "next_steps": "1 sentence actionable advice"
            }}
            
            Be specific, encouraging, and constructive. Return ONLY valid JSON.
            """

            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert AI interview coach providing specific, actionable feedback."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )

            import json
            feedback_json = json.loads(completion.choices[0].message.content)
            
            if 'strengths' in feedback_json:
                self.results['suggestions'].insert(0, f"✓ Strengths: {feedback_json['strengths']}")
            if 'improvements' in feedback_json:
                self.results['suggestions'].insert(1, f"✓ Focus Area: {feedback_json['improvements']}")
            if 'word_upgrades' in feedback_json:
                self.results['word_suggestions'] = feedback_json['word_upgrades']
            if 'next_steps' in feedback_json:
                self.results['suggestions'].append(f"✓ Next: {feedback_json['next_steps']}")

        except Exception as e:
            print(f"AI feedback generation failed: {str(e)}")
            self._generate_basic_feedback()

    def _generate_basic_feedback(self):
        """Generate feedback without AI (fallback)"""
        if not self.results['suggestions']:
            if self.results['overall_score'] >= 7:
                self.results['suggestions'].append("✓ Great job! You demonstrated strong communication skills.")
            elif self.results['overall_score'] >= 5:
                self.results['suggestions'].append("✓ Good effort. Focus on fluency and reducing filler words for next time.")
            else:
                self.results['suggestions'].append("✓ Keep practicing. Ensure clear audio and better eye contact.")

    @staticmethod
    def get_word_suggestions(transcription):
        """Generate vocabulary upgrade suggestions"""
        suggestions = []
        words = transcription.lower().split()
        
        for word in words:
            for weak, upgrades in WORD_UPGRADES.items():
                if weak in word or word in weak:
                    suggestions.append(f"Instead of '{weak}', try: {upgrades[0]}")
                    if len(suggestions) >= 3:
                        return list(set(suggestions))
        
        return list(set(suggestions))[:3]



