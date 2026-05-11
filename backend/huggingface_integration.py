"""
Hugging Face API Integration for InterviewX
NLP and ML models for advanced interview features
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

class HuggingFaceAPI:
    """Hugging Face API integration for NLP and ML models"""
    
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.base_url = "https://api-inference.huggingface.co/models"
        
        if not self.api_key:
            print("Warning: HUGGINGFACE_API_KEY not found in environment variables")
    
    def analyze_sentiment_and_emotions(
        self,
        text: str,
        models: List[str] = ["distilbert-base-uncased-finetuned-sst-2-english", "j-hartmann/emotion-english-distilroberta-base"]
    ) -> Dict[str, Any]:
        """Analyze sentiment and emotions from text using Hugging Face models"""
        try:
            if not self.api_key:
                return self._generate_fallback_sentiment_analysis(text)
            
            results = {}
            
            for model in models:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "inputs": text
                }
                
                response = requests.post(
                    f"{self.base_url}/{model}",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if "distilbert" in model:
                        # Sentiment analysis
                        sentiment_label = result[0][0]["label"]
                        sentiment_score = result[0][0]["score"]
                        results["sentiment"] = {
                            "label": sentiment_label,
                            "score": sentiment_score,
                            "confidence": sentiment_score
                        }
                    elif "emotion" in model:
                        # Emotion analysis
                        emotions = result[0]
                        top_emotion = max(emotions, key=lambda x: x["score"])
                        results["emotions"] = {
                            "primary_emotion": top_emotion["label"],
                            "confidence": top_emotion["score"],
                            "all_emotions": emotions
                        }
            
            return {
                "success": True,
                "text": text,
                "analysis": results,
                "models_used": models,
                "analyzed_by": "huggingface"
            }
                
        except Exception as e:
            print(f"Error analyzing sentiment with Hugging Face: {e}")
            return self._generate_fallback_sentiment_analysis(text)
    
    def classify_text(
        self,
        text: str,
        model: str = "facebook/bart-large-mnli"
    ) -> Dict[str, Any]:
        """Classify text using Hugging Face models"""
        try:
            if not self.api_key:
                return self._generate_fallback_classification(text)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "inputs": text,
                "parameters": {
                    "candidate_labels": ["technical", "behavioral", "situational", "problem-solving"],
                    "multi_label": False
                }
            }
            
            response = requests.post(
                f"{self.base_url}/{model}",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if len(result) > 0:
                    classification = result[0]
                    top_label = max(classification, key=lambda x: x["score"])
                    
                    return {
                        "success": True,
                        "text": text,
                        "classification": {
                            "predicted_label": top_label["label"],
                            "confidence": top_label["score"],
                            "all_scores": classification
                        },
                        "model": model,
                        "classified_by": "huggingface"
                    }
                else:
                    return self._generate_fallback_classification(text)
            else:
                print(f"Hugging Face API error: {response.status_code} - {response.text}")
                return self._generate_fallback_classification(text)
                
        except Exception as e:
            print(f"Error classifying text with Hugging Face: {e}")
            return self._generate_fallback_classification(text)
    
    def generate_interview_questions(
        self,
        role: str,
        experience_level: str = "intermediate",
        num_questions: int = 5,
        model: str = "microsoft/DialoGPT-medium"
    ) -> Dict[str, Any]:
        """Generate interview questions using Hugging Face conversational models"""
        try:
            if not self.api_key:
                return self._generate_fallback_questions(role, num_questions)
            
            prompt = f"""
            You are an expert interviewer. Generate {num_questions} diverse interview questions for a {role} position at {experience_level} level.
            
            Requirements:
            1. Mix of behavioral, technical, and situational questions
            2. Questions should reveal problem-solving approach and technical depth
            3. Include questions about specific technologies relevant to {role}
            4. Each question should be realistic and challenging but fair
            5. Include follow-up suggestions for each question
            
            For each question, provide:
            - Clear question text
            - Type (behavioral, technical, situational, problem-solving)
            - Difficulty (beginner, intermediate, advanced)
            - Expected skills/concepts being tested
            - Follow-up suggestions (1-2 specific follow-up questions)
            - Estimated response time
            
            Return ONLY a JSON array with this structure:
            [
                {{
                    "id": 1,
                    "question": "Specific question text here",
                    "type": "technical",
                    "difficulty": "intermediate",
                    "skills_tested": ["React", "API design", "Problem-solving"],
                    "follow_up_suggestions": ["Can you elaborate on your approach?", "What alternatives did you consider?"],
                    "estimated_time": "5-8 minutes",
                    "scoring_criteria": "Technical accuracy, problem-solving approach, communication of trade-offs"
                }},
                ...
            ]
            """
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1024,
                    "temperature": 0.7 + (hash(str(datetime.now())) % 20) / 100,  # Add randomness
                    "do_sample": True,
                    "top_p": 0.95,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                f"{self.base_url}/{model}",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result[0]["generated_text"]
                
                # Parse JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                questions = json.loads(content)
                
                # Add metadata
                for i, question in enumerate(questions):
                    question["id"] = i + 1
                    question["generated_by"] = "huggingface"
                
                return {
                    "success": True,
                    "questions": questions,
                    "generated_by": "huggingface",
                    "model": model,
                    "count": len(questions)
                }
            else:
                print(f"Hugging Face API error: {response.status_code} - {response.text}")
                return self._generate_fallback_questions(role, num_questions)
                
        except Exception as e:
            print(f"Error generating questions with Hugging Face: {e}")
            return self._generate_fallback_questions(role, num_questions)
    
    def extract_keywords_and_topics(
        self,
        text: str,
        model: str = "dslim/bert-base-NER"
    ) -> Dict[str, Any]:
        """Extract keywords and topics from text using NER model"""
        try:
            if not self.api_key:
                return self._generate_fallback_keywords(text)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "inputs": text,
                "parameters": {
                    "aggregation_strategy": "simple"
                }
            }
            
            response = requests.post(
                f"{self.base_url}/{model}",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if len(result) > 0:
                    entities = result[0]
                    
                    # Group entities by type
                    grouped_entities = {}
                    for entity in entities:
                        entity_type = entity.get("entity_group", entity.get("entity", "Unknown"))
                        if entity_type not in grouped_entities:
                            grouped_entities[entity_type] = []
                        grouped_entities[entity_type].append({
                            "text": entity.get("word", ""),
                            "score": entity.get("score", 0),
                            "start": entity.get("start", 0),
                            "end": entity.get("end", 0)
                        })
                    
                    return {
                        "success": True,
                        "text": text,
                        "entities": grouped_entities,
                        "model": model,
                        "extracted_by": "huggingface"
                    }
                else:
                    return self._generate_fallback_keywords(text)
            else:
                print(f"Hugging Face API error: {response.status_code} - {response.text}")
                return self._generate_fallback_keywords(text)
                
        except Exception as e:
            print(f"Error extracting keywords with Hugging Face: {e}")
            return self._generate_fallback_keywords(text)
    
    def _generate_fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Generate fallback sentiment analysis when API is unavailable"""
        return {
            "success": False,
            "text": text,
            "sentiment": {
                "label": "neutral",
                "score": 0.5,
                "confidence": 0.5
            },
            "emotions": {
                "primary_emotion": "neutral",
                "confidence": 0.5,
                "all_emotions": []
            },
            "analyzed_by": "fallback"
        }
    
    def _generate_fallback_classification(self, text: str) -> Dict[str, Any]:
        """Generate fallback classification when API is unavailable"""
        return {
            "success": False,
            "text": text,
            "classification": {
                "predicted_label": "general",
                "confidence": 0.5,
                "all_scores": []
            },
            "model": "fallback",
            "classified_by": "fallback"
        }
    
    def _generate_fallback_questions(self, role: str, num_questions: int) -> Dict[str, Any]:
        """Generate fallback questions when API is unavailable"""
        fallback_questions = [
            {
                "id": 1,
                "question": "Describe your experience with relevant technologies for this role.",
                "type": "technical",
                "difficulty": "intermediate",
                "skills_tested": ["Technical knowledge", "Experience"],
                "follow_up_suggestions": ["Can you provide specific examples?", "What challenges did you face?"],
                "estimated_time": "5-8 minutes",
                "scoring_criteria": "Technical depth, practical experience"
            },
            {
                "id": 2,
                "question": "Tell me about a time you had to solve a complex problem.",
                "type": "behavioral",
                "difficulty": "intermediate",
                "skills_tested": ["Problem-solving", "Communication"],
                "follow_up_suggestions": ["What was the outcome?", "What did you learn?"],
                "estimated_time": "3-5 minutes",
                "scoring_criteria": "Problem-solving approach, results communication"
            }
        ]
        
        return {
            "success": False,
            "questions": fallback_questions[:num_questions],
            "generated_by": "fallback"
        }
    
    def _generate_fallback_keywords(self, text: str) -> Dict[str, Any]:
        """Generate fallback keyword extraction when API is unavailable"""
        words = text.split()
        keywords = [word.lower() for word in words if len(word) > 3][:10]
        
        return {
            "success": False,
            "text": text,
            "entities": {
                "keywords": [{"text": word, "score": 0.8} for word in keywords]
            },
            "model": "fallback",
            "extracted_by": "fallback"
        }

# Global instance
huggingface_api = HuggingFaceAPI()
