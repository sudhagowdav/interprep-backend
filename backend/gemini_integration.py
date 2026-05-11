"""
Gemini API Integration for InterviewX
Advanced question generation and performance analysis
"""

import os
import json
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional

class GeminiAPI:
    """Google Gemini API integration for advanced interview features"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.primary_model = "gemini-2.0-flash"
        self.fallback_model = "gemini-1.5-flash"
        
        if not self.api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables")

    def _make_request(self, endpoint: str, data: Dict[str, Any], model: str = None) -> requests.Response:
        """Make API request with automatic model fallback for 429 errors"""
        if not model:
            model = self.primary_model
            
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        url = f"{self.base_url}/models/{model}:{endpoint}"
        response = requests.post(url, headers=headers, json=data)
        
        # If rate limited on primary, try fallback
        if response.status_code == 429 and model == self.primary_model:
            print(f"Rate limited on {self.primary_model}, trying {self.fallback_model}...")
            url = f"{self.base_url}/models/{self.fallback_model}:{endpoint}"
            response = requests.post(url, headers=headers, json=data)
            
        return response
    
    def generate_interview_questions(
        self, 
        job_description: str,
        experience_level: str = "intermediate",
        num_questions: int = 5,
        role: str = "Software Developer"
    ) -> List[Dict[str, Any]]:
        """Generate contextual interview questions using Gemini"""
        try:
            if not self.api_key:
                return self._generate_fallback_questions(role, num_questions)
            
            prompt = f"""
            You are an expert interviewer and technical recruiter. Generate {num_questions} diverse interview questions for a {role} position at {experience_level} level.
            
            Job Description/Context:
            {job_description}
            
            Requirements:
            1. Mix of behavioral, technical, and situational questions
            2. Questions should reveal problem-solving approach and technical depth
            3. Include questions about specific technologies mentioned in job description
            4. Each question should be realistic and challenging but fair
            5. Include follow-up suggestions for each question
            
            For each question, provide:
            - Question text (clear and specific)
            - Type (behavioral, technical, situational, problem-solving)
            - Difficulty (beginner, intermediate, advanced)
            - Expected skills/concepts being tested
            - Follow-up suggestions (1-2 specific follow-up questions)
            - Estimated response time
            - Scoring criteria
            
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
            
            response = self._make_request("generateContent", data)
            
            if response.status_code == 200:
                result = response.json()
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Parse JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                questions = json.loads(content)
                
                # Add metadata
                for i, question in enumerate(questions):
                    question["id"] = i + 1
                    question["generated_by"] = "gemini"
                    question["job_context"] = job_description[:100] + "..." if len(job_description) > 100 else job_description
                
                return questions
                
            else:
                print(f"Gemini API error: {response.status_code} - {response.text}")
                return self._generate_fallback_questions(role, num_questions)
                
        except Exception as e:
            print(f"Error generating questions with Gemini: {e}")
            return self._generate_fallback_questions(role, num_questions)
    
    def analyze_interview_performance(
        self,
        questions: List[Dict[str, Any]],
        responses: List[Dict[str, Any]],
        job_description: str = ""
    ) -> Dict[str, Any]:
        """Analyze overall interview performance using Gemini"""
        try:
            if not self.api_key:
                return self._generate_fallback_analysis(questions, responses)
            
            # Prepare interview transcript for analysis
            transcript = []
            for i, (question, response) in enumerate(zip(questions, responses)):
                transcript.append(f"Q{i+1}: {question.get('question', '')}")
                transcript.append(f"A{i+1}: {response.get('response_text', '')}")
            
            full_transcript = "\n".join(transcript)
            
            prompt = f"""
            You are an expert hiring manager and interview coach. Analyze this complete interview performance and provide comprehensive feedback.
            
            Job Context: {job_description}
            
            Interview Transcript:
            {full_transcript}
            
            Provide detailed analysis covering:
            1. Overall Performance Score (0-100)
            2. Communication Skills Assessment
            3. Technical Competency Evaluation
            4. Problem-Solving Approach Analysis
            5. Cultural Fit Assessment
            6. Strengths Demonstrated
            7. Areas for Improvement
            8. Specific Recommendations
            9. Hiring Recommendation (Hire/Consider/Reject)
            10. Next Steps for Candidate
            
            Consider:
            - Question responses quality and depth
            - Communication clarity and confidence
            - Technical accuracy and problem-solving
            - Professionalism and engagement
            - Alignment with job requirements
            
            Return ONLY a JSON object:
            {{
                "overall_score": <0-100>,
                "communication_score": <0-100>,
                "technical_score": <0-100>,
                "problem_solving_score": <0-100>,
                "cultural_fit_score": <0-100>,
                "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
                "improvement_areas": ["<area 1>", "<area 2>", "<area 3>"],
                "recommendations": ["<recommendation 1>", "<recommendation 2>"],
                "hiring_recommendation": "<Hire/Consider/Reject>",
                "next_steps": ["<step 1>", "<step 2>"],
                "detailed_feedback": "<comprehensive 2-3 paragraph feedback>"
            }}
            """
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "topK": 32,
                    "topP": 0.95,
                    "maxOutputTokens": 4096,
                    "responseMimeType": "application/json"
                }
            }
            
            response = requests.post(
                f"{self.base_url}/models/gemini-2.0-flash:generateContent",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Parse JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                analysis = json.loads(content)
                analysis["analyzed_by"] = "gemini"
                analysis["analysis_timestamp"] = str(datetime.now())
                
                return analysis
                
            else:
                print(f"Gemini API error: {response.status_code} - {response.text}")
                return self._generate_fallback_analysis(questions, responses)
                
        except Exception as e:
            print(f"Error analyzing performance with Gemini: {e}")
            return self._generate_fallback_analysis(questions, responses)

    def analyze_single_response(
        self,
        question: str,
        response_text: str,
        ideal_answer: str = "",
        role: str = "Software Developer"
    ) -> Dict[str, Any]:
        """Analyze a single interview response against an ideal answer using Gemini"""
        try:
            if not self.api_key:
                return {
                    "success": False,
                    "error": "Gemini API key not configured"
                }
            
            prompt = f"""
            You are an expert interviewer. Analyze the candidate's response to the following question.
            
            Role: {role}
            Question: {question}
            Ideal Answer/Context: {ideal_answer}
            
            Candidate's Response:
            {response_text}
            
            Analysis Task:
            1. Relevance Score (0-10): How well does it answer the question? (BE CRITICAL. If answer is empty or unrelated, score must be 0-1)
            2. Completeness Score (0-10): Did they cover all key technical concepts? (0 if answer is too short)
            3. Clarity & Confidence (0-10): Technical communication quality. (0 if no speech)
            4. Keyword Comparison: Identify specific technical keywords matched and missing.
            5. Strengths: What was technically sound? (None if empty)
            6. Areas for Improvement: Specific technical or communication gaps.
            7. Feedback: Candid, professional advice.
            8. Better Version: A high-quality model answer.
            
            SCORING POLICY:
            - If the response is empty, "what", "I don't know", or nonsensical: All scores MUST be 0-1.
            - If the response is good but lacks depth: Score 5-6.
            - Only exceptional, complete answers should get 9-10.
            - DO NOT give the same scores (like 7 or 8) for every answer. Differentiate based on transcript quality.
            
            Return ONLY a JSON object:
            {{
                "relevance_score": <0-10>,
                "completeness_score": <0-10>,
                "clarity_score": <0-10>,
                "keywords_matched": ["<keyword 1>", ...],
                "keywords_missing": ["<keyword 1>", ...],
                "strengths": ["<strength 1>", ...],
                "improvement_areas": ["<area 1>", ...],
                "feedback": "<candid feedback>",
                "better_version": "<high quality version>"
            }}
            """
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 32,
                    "topP": 0.95,
                    "maxOutputTokens": 4096,
                    "responseMimeType": "application/json"
                }
            }
            
            response = requests.post(
                f"{self.base_url}/models/gemini-2.0-flash:generateContent?key={self.api_key}",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Parse JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                analysis = json.loads(content)
                analysis["success"] = True
                analysis["transcription"] = response_text
                return analysis
            else:
                print(f"Gemini API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            print(f"Error in analyze_single_response: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    
    def generate_role_playing_scenarios(
        self,
        role: str,
        scenario_type: str = "behavioral",
        difficulty: str = "intermediate"
    ) -> List[Dict[str, Any]]:
        """Generate realistic role-playing interview scenarios"""
        try:
            if not self.api_key:
                return self._generate_fallback_scenarios(role, scenario_type)
            
            prompt = f"""
            You are an expert interviewer conducting a {scenario_type} interview for a {role} position at {difficulty} level.
            
            Generate 3 realistic interview scenarios that include:
            1. The interviewer's opening and setup
            2. The candidate's potential response areas
            3. Follow-up questions the interviewer should ask
            4. Evaluation criteria for the scenario
            
            Make scenarios realistic and challenging but fair. Include specific details about the situation, company context, and technical challenges.
            
            Return ONLY a JSON array:
            [
                {{
                    "id": 1,
                    "scenario_type": "{scenario_type}",
                    "difficulty": "{difficulty}",
                    "title": "Scenario title",
                    "setup": "Detailed scenario setup and context",
                    "interviewer_dialogue": ["Opening question 1", "Follow-up 1", "Follow-up 2"],
                    "candidate_response_areas": ["Expected response areas"],
                    "evaluation_criteria": ["Criteria 1", "Criteria 2", "Criteria 3"],
                    "ideal_outcome": "What the interviewer is looking for"
                }},
                ...
            ]
            """
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.8,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 3072,
                    "responseMimeType": "application/json"
                }
            }
            
            response = requests.post(
                f"{self.base_url}/models/gemini-2.0-flash:generateContent",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Parse JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                scenarios = json.loads(content)
                
                # Add metadata
                for scenario in scenarios:
                    scenario["generated_by"] = "gemini"
                    scenario["role"] = role
                
                return scenarios
                
            else:
                print(f"Gemini API error: {response.status_code} - {response.text}")
                return self._generate_fallback_scenarios(role, scenario_type)
                
        except Exception as e:
            print(f"Error generating scenarios with Gemini: {e}")
            return self._generate_fallback_scenarios(role, scenario_type)
    
    def _generate_fallback_questions(self, role: str, num_questions: int) -> List[Dict[str, Any]]:
        """Generate fallback questions when API is unavailable"""
        fallback_questions = {
            "Software Developer": [
                {
                    "id": 1,
                    "question": "Describe your experience with cloud platforms and microservices architecture.",
                    "type": "technical",
                    "difficulty": "intermediate",
                    "skills_tested": ["Cloud computing", "Microservices", "System design"],
                    "follow_up_suggestions": ["What scaling challenges did you face?", "How did you handle service discovery?"],
                    "estimated_time": "5-8 minutes",
                    "scoring_criteria": "Technical depth, practical experience, problem-solving approach"
                },
                {
                    "id": 2,
                    "question": "Tell me about a time you had to debug a complex production issue.",
                    "type": "behavioral",
                    "difficulty": "intermediate",
                    "skills_tested": ["Debugging", "Problem-solving", "Pressure management"],
                    "follow_up_suggestions": ["What was the root cause?", "How did you prevent similar issues?"],
                    "estimated_time": "3-5 minutes",
                    "scoring_criteria": "Systematic approach, communication, learning from experience"
                }
            ]
        }
        
        questions = fallback_questions.get(role, [])
        return questions[:num_questions]
    
    def _generate_fallback_analysis(self, questions: List[Dict[str, Any]], responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate fallback performance analysis"""
        return {
            "overall_score": 75,
            "communication_score": 70,
            "technical_score": 75,
            "problem_solving_score": 80,
            "cultural_fit_score": 72,
            "strengths": ["Technical knowledge", "Problem-solving approach"],
            "improvement_areas": ["Communication clarity", "Depth of examples"],
            "recommendations": ["Provide more specific examples", "Practice STAR method"],
            "hiring_recommendation": "Consider",
            "next_steps": ["Technical interview", "Reference check"],
            "detailed_feedback": "Candidate shows good technical foundation with room for improvement in communication depth.",
            "analyzed_by": "fallback"
        }
    
    def _generate_fallback_scenarios(self, role: str, scenario_type: str) -> List[Dict[str, Any]]:
        """Generate fallback role-playing scenarios"""
        return [
            {
                "id": 1,
                "scenario_type": scenario_type,
                "difficulty": "intermediate",
                "title": "System Design Challenge",
                "setup": f"You're interviewing a {role} for a senior position. The candidate needs to design a scalable system.",
                "interviewer_dialogue": [
                    "Design a URL shortening service like bit.ly.",
                    "What are the key components?",
                    "How would you handle 1 million requests per day?"
                ],
                "candidate_response_areas": ["Architecture decisions", "Scalability considerations", "Trade-offs"],
                "evaluation_criteria": ["Technical depth", "Practical thinking", "Communication skills"],
                "ideal_outcome": "Candidate demonstrates systematic approach to system design"
            }
        ]

# Global instance
gemini_api = GeminiAPI()
