"""
Groq API Integration for InterviewX
Fast AI inference for question generation and analysis
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

class GroqAPI:
    """Groq API integration for fast AI inference"""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        
        if not self.api_key:
            print("Warning: GROQ_API_KEY not found in environment variables")
    
    def generate_interview_questions(
        self,
        role: str,
        experience_level: str = "intermediate",
        num_questions: int = 5,
        job_description: str = "",
        model: str = "llama-3.1-8b-instant"
    ) -> Dict[str, Any]:
        """Generate interview questions using Groq"""
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
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.7 + (hash(str(datetime.now())) % 10) / 100  # Add randomness
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                questions = json.loads(content)
                
                # Add metadata
                for i, question in enumerate(questions):
                    question["id"] = i + 1
                    question["generated_by"] = "groq"
                    question["job_context"] = job_description[:100] + "..." if len(job_description) > 100 else job_description
                
                return {
                    "success": True,
                    "questions": questions,
                    "generated_by": "groq",
                    "model": model,
                    "count": len(questions)
                }
                
            else:
                print(f"Groq API error: {response.status_code} - {response.text}")
                return self._generate_fallback_questions(role, num_questions)
                
        except Exception as e:
            print(f"Error generating questions with Groq: {e}")
            return self._generate_fallback_questions(role, num_questions)
    
    def analyze_interview_performance(
        self,
        questions: List[Dict[str, Any]],
        responses: List[Dict[str, Any]],
        job_description: str = "",
        model: str = "llama-3.1-8b-instant"
    ) -> Dict[str, Any]:
        """Analyze interview performance using Groq"""
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
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                analysis = json.loads(content)
                analysis["analyzed_by"] = "groq"
                analysis["model"] = model
                analysis["analysis_timestamp"] = str(datetime.now())
                
                return analysis
                
            else:
                print(f"Groq API error: {response.status_code} - {response.text}")
                return self._generate_fallback_analysis(questions, responses)
                
        except Exception as e:
            print(f"Error analyzing performance with Groq: {e}")
            return self._generate_fallback_analysis(questions, responses)
    
    def generate_ai_interviewer(
        self,
        role: str,
        interview_style: str = "professional",
        model: str = "llama-3.1-8b-instant"
    ) -> Dict[str, Any]:
        """Generate AI interviewer using Groq"""
        try:
            if not self.api_key:
                return self._generate_fallback_interviewer(role)
            
            system_prompt = f"""
            You are an expert interviewer conducting a {interview_style} interview for a {role} position.
            
            Your role:
            - Ask realistic, challenging questions
            - Listen actively to responses and ask relevant follow-ups
            - Maintain professional but conversational tone
            - Adapt difficulty based on candidate responses
            - Provide feedback when appropriate
            - Keep responses concise (2-3 sentences max)
            
            Interview style: {interview_style}
            Target role: {role}
            Model: {model}
            
            Ask questions that reveal:
            - Technical knowledge and depth
            - Problem-solving approach
            - Communication skills
            - Cultural fit and collaboration
            - Leadership potential
            
            Be conversational but professional. Use natural language, not robotic.
            """
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Hello! I'm here to interview for the {role} position. Let's start when you're ready."
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result["choices"][0]["message"]["content"]
                
                return {
                    "success": True,
                    "interviewer_response": ai_response,
                    "model": model,
                    "style": interview_style,
                    "role": role,
                    "generated_by": "groq"
                }
            else:
                print(f"Groq API error: {response.status_code} - {response.text}")
                return self._generate_fallback_interviewer(role)
                
        except Exception as e:
            print(f"Error generating interviewer with Groq: {e}")
            return self._generate_fallback_interviewer(role)
    
    def assess_technical_answer(
        self,
        question: str,
        answer: str,
        context: str = "",
        model: str = "llama-3.1-8b-instant"
    ) -> Dict[str, Any]:
        """Assess technical answer using Groq"""
        try:
            if not self.api_key:
                return self._generate_fallback_assessment()
            
            prompt = f"""
            You are a senior technical interviewer and expert in evaluating interview responses.
            
            Question: {question}
            Candidate's Answer: {answer}
            Context: {context}
            
            Assess the technical correctness and quality of this answer on a scale of 1-10:
            
            Evaluation Criteria:
            1. Technical Accuracy (0-10): Is the information technically correct?
            2. Problem-Solving Approach (0-10): Does the answer show good problem-solving?
            3. Code Quality (0-10): If applicable, is the solution well-structured?
            4. Best Practices (0-10): Does the answer follow industry best practices?
            5. Scalability Considerations (0-10): Does the candidate think about scale?
            6. Security Awareness (0-10): Security implications mentioned?
            7. Communication of Technical Concepts (0-10): How clearly are technical ideas explained?
            8. Innovation/Creativity (0-10): Creative or novel approaches suggested?
            
            For each criterion, provide:
            - Score (0-10)
            - Specific feedback on what was done well or poorly
            - Suggestions for improvement
            
            Also provide:
            - Overall technical assessment score (0-10)
            - Strengths demonstrated
            - Critical gaps or misconceptions
            - Specific recommendations for improvement
            - Comparison to expert-level expectations
            
            Return ONLY a JSON object:
            {{
                "technical_accuracy_score": <0-10>,
                "problem_solving_score": <0-10>,
                "code_quality_score": <0-10>,
                "best_practices_score": <0-10>,
                "scalability_score": <0-10>,
                "security_score": <0-10>,
                "communication_score": <0-10>,
                "innovation_score": <0-10>,
                "overall_technical_score": <0-10>,
                "strengths": ["<strength 1>", "<strength 2>"],
                "weaknesses": ["<weakness 1>", "<weakness 2>"],
                "improvement_suggestions": ["<suggestion 1>", "<suggestion 2>"],
                "missing_concepts": ["<concept 1>", "<concept 2>"],
                "detailed_feedback": "<comprehensive technical assessment>"
            }}
            """
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1500,
                "temperature": 0.2
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                assessment = json.loads(content)
                assessment["assessed_by"] = "groq"
                assessment["model"] = model
                assessment["assessment_timestamp"] = str(datetime.now())
                
                return assessment
                
            else:
                print(f"Groq API error: {response.status_code} - {response.text}")
                return self._generate_fallback_assessment()
                
        except Exception as e:
            print(f"Error assessing with Groq: {e}")
            return self._generate_fallback_assessment()
    
    def _generate_fallback_questions(self, role: str, num_questions: int) -> Dict[str, Any]:
        """Generate fallback questions when API is unavailable"""
        fallback_questions = [
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
        
        return {
            "success": False,
            "questions": fallback_questions[:num_questions],
            "generated_by": "fallback"
        }
    
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
    
    def _generate_fallback_interviewer(self, role: str) -> Dict[str, Any]:
        """Generate fallback interviewer when API is unavailable"""
        return {
            "success": False,
            "interviewer_response": f"Hello! I'm ready to conduct your {role} interview. Let's begin with our first question.",
            "model": "fallback",
            "style": "professional",
            "role": role,
            "generated_by": "fallback"
        }
    
    def _generate_fallback_assessment(self) -> Dict[str, Any]:
        """Generate fallback technical assessment"""
        return {
            "technical_accuracy_score": 7,
            "problem_solving_score": 7,
            "code_quality_score": 7,
            "best_practices_score": 7,
            "scalability_score": 7,
            "security_score": 7,
            "communication_score": 7,
            "innovation_score": 6,
            "overall_technical_score": 7,
            "strengths": ["Basic technical knowledge", "Clear communication"],
            "weaknesses": ["Limited depth", "Could be more specific"],
            "improvement_suggestions": ["Provide more detailed examples", "Show deeper technical understanding"],
            "missing_concepts": [],
            "detailed_feedback": "Candidate shows basic understanding with room for technical depth improvement.",
            "assessed_by": "fallback"
        }

# Global instance
groq_api = GroqAPI()
