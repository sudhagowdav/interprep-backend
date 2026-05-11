"""
Processor module for InterviewX - AI and PDF processing logic
Handles resume parsing, question generation, and feedback analysis using Groq API
"""

import PyPDF2
import json
from typing import List, Dict, Any, Tuple
import os
from groq import Groq

import requests
import time
from groq import Groq

# Initialize Groq client
try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Warning: Groq client initialization error: {e}")
    client = None

# Updated model (llama-3.1-70b-versatile was decommissioned)
MODEL = "llama-3.3-70b-versatile"
APYHUB_TOKEN = os.getenv("APYHUB_TOKEN")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from PDF resume"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""


def extract_skills_from_resume(resume_text: str) -> List[Dict[str, str]]:
    """Extract skills from resume using AI"""
    try:
        if not client:
            return [{"name": "Python", "proficiency": "Intermediate"},
                    {"name": "JavaScript", "proficiency": "Intermediate"}]

        prompt = f"""
        Analyze the following resume text and extract all technical and non-technical skills.
        Return ONLY a valid JSON array with objects containing 'name' and 'proficiency' fields.
        Proficiency levels should be: 'Beginner', 'Intermediate', 'Advanced', or 'Expert'.

        Resume Text:
        {resume_text}

        Return only JSON array, no other text.
        """

        message = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.choices[0].message.content
        # Parse JSON from response
        skills = json.loads(response_text)
        return skills
    except Exception as e:
        print(f"Error extracting skills: {e}")
        return [{"name": "Python", "proficiency": "Intermediate"}]


def extract_experience_summary(resume_text: str) -> str:
    """Extract work experience summary from resume"""
    try:
        if not client:
            return "Unable to generate summary at this moment. Please ensure Groq API key is configured."

        prompt = f"""
        Based on the following resume, provide a brief 2-3 sentence summary of candidate's
        professional experience and background.

        Resume Text:
        {resume_text}
        """

        message = client.chat.completions.create(
            model=MODEL,
            max_tokens=512,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.choices[0].message.content
    except Exception as e:
        print(f"Error extracting experience summary: {e}")
        return ""


def generate_interview_questions(
        resume_text: str,
        num_questions: int = 5) -> List[str]:
    """Generate 5 situational interview questions based on resume"""
    try:
        if not client:
            return [
                "Tell me about a challenging project you've worked on and how you overcame obstacles.",
                "Describe a time when you had to work with a difficult team member. How did you handle it?",
                "Give an example of when you showed leadership skills.",
                "Tell me about a time you failed. What did you learn from it?",
                "How do you stay updated with new technologies and trends in your field?"
            ]

        prompt = f"""
        Based on the following resume, generate {num_questions} unique and diverse situational interview questions
        that would be relevant for candidate's background and experience.
        Each question should be specific to their skills and experience mentioned in the resume.
        Ensure questions are different from typical generic questions and vary each time they are generated.

        Resume Text:
        {resume_text}

        Return ONLY a JSON array of strings containing the questions, no other text.
        Example format: ["Question 1?", "Question 2?", ...]
        """

        message = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.choices[0].message.content
        questions = json.loads(response_text)
        return questions
    except Exception as e:
        print(f"Error generating interview questions: {e}")
        return []


def analyze_response(question: str, user_response: str,
                     resume_context: str = "") -> Dict[str, Any]:
    """Analyze user's response and generate feedback scores using Groq (priority) or Gemini (fallback)"""
    try:
        # 1. Try Groq API first (Highest reliability and speed)
        try:
            from groq_integration import groq_api
            assessment = groq_api.assess_technical_answer(question, user_response, resume_context)
            # If we got a valid-looking assessment from Groq (it might not have a 'success' key)
            if assessment and (assessment.get('success') or assessment.get('overall_technical_score') is not None):
                return {
                    "clarity_score": assessment.get('communication_score', 0),
                    "confidence_score": assessment.get('communication_score', 0),
                    "relevance_score": assessment.get('overall_technical_score', 0),
                    "strengths": ", ".join(assessment.get('strengths', ["N/A"])) if isinstance(assessment.get('strengths'), list) else assessment.get('strengths', "N/A"),
                    "areas_for_improvement": ", ".join(assessment.get('improvement_suggestions', ["N/A"])) if isinstance(assessment.get('improvement_suggestions'), list) else assessment.get('improvement_suggestions', "N/A"),
                    "overall_feedback": assessment.get('detailed_feedback', "No response detected."),
                    "ideal_answer": ", ".join(assessment.get('missing_concepts', ["N/A"])) if isinstance(assessment.get('missing_concepts'), list) else assessment.get('missing_concepts', "N/A"),
                    "keywords_matched": assessment.get('keywords_matched', []),
                    "keywords_missing": assessment.get('keywords_missing', [])
                }
        except Exception as e:
            print(f"Groq analysis failed: {e}. Falling back to Gemini.")

        # 2. Try Gemini API (Fallback 1)
        try:
            from gemini_integration import gemini_api
            if gemini_api and gemini_api.api_key:
                gemini_feedback = gemini_api.analyze_single_response(question, user_response, ideal_answer=resume_context)
                if gemini_feedback and gemini_feedback.get('success'):
                    return {
                        "clarity_score": gemini_feedback.get('clarity_score', 0),
                        "confidence_score": gemini_feedback.get('clarity_score', 0),
                        "relevance_score": gemini_feedback.get('relevance_score', 0),
                        "strengths": ", ".join(gemini_feedback.get('strengths', [])) if isinstance(gemini_feedback.get('strengths'), list) else gemini_feedback.get('strengths', ''),
                        "areas_for_improvement": ", ".join(gemini_feedback.get('improvement_areas', [])) if isinstance(gemini_feedback.get('improvement_areas'), list) else gemini_feedback.get('improvement_areas', ''),
                        "overall_feedback": gemini_feedback.get('feedback', ''),
                        "ideal_answer": gemini_feedback.get('better_version', ''),
                        "keywords_matched": gemini_feedback.get('keywords_matched', []),
                        "keywords_missing": gemini_feedback.get('keywords_missing', [])
                    }
        except Exception as e:
            print(f"Gemini analysis in processor failed: {e}")

        # 3. Final Fallback: Basic Analysis (if AI services are down or rate-limited)
        if not user_response or len(user_response.strip()) < 5:
            return {
                "clarity_score": 3,
                "confidence_score": 3,
                "relevance_score": 3,
                "strengths": "Brief response",
                "areas_for_improvement": "Provide more detailed answer",
                "overall_feedback": "Response is too brief to evaluate properly.",
                "ideal_answer": "A comprehensive answer would include specific examples and details."
            }
        else:
            # Basic scoring based on response length and content
            word_count = len(user_response.split())
            clarity_score = min(10, max(3, word_count // 10))
            confidence_score = min(10, max(3, word_count // 15))
            relevance_score = min(10, max(3, word_count // 12))
            
            return {
                "clarity_score": clarity_score,
                "confidence_score": confidence_score,
                "relevance_score": relevance_score,
                "strengths": "Clear communication" if word_count > 10 else "Basic response",
                "areas_for_improvement": "Add more specific examples" if clarity_score < 6 else "Good response",
                "overall_feedback": f"Response contains {word_count} words and shows {'good' if clarity_score > 6 else 'basic'} understanding.",
                "ideal_answer": "A comprehensive answer would include specific examples, technical details, and clear explanations."
            }
    except Exception as e:
        print(f"Error analyzing response: {e}")
        return {
            "clarity_score": 5,
            "confidence_score": 5,
            "relevance_score": 5,
            "strengths": "N/A",
            "areas_for_improvement": "N/A",
            "overall_feedback": "Unable to analyze at this moment"
        }


def calculate_interview_readiness(
        skills_count: int, avg_feedback_score: float) -> Tuple[float, float]:
    """Calculate interview readiness score and percentage"""
    # Skills factor (0-40 points)
    skills_score = min(skills_count * 4, 40)

    # Feedback factor (0-60 points)
    feedback_score = avg_feedback_score * 6

    # Total score (0-100)
    overall_score = skills_score + feedback_score
    readiness_percentage = min(overall_score, 100)

    return overall_score, readiness_percentage


def generate_session_report(resume_text: str,
                            interview_responses: List[Dict[str,
                                                           Any]],
                            feedback_scores: List[Dict[str,
                                                       float]]) -> Dict[str,
                                                                        Any]:
    """Generate comprehensive session report"""
    try:
        avg_clarity = sum(
            f.get(
                'clarity_score',
                0) for f in feedback_scores) / len(feedback_scores) if feedback_scores else 0
        avg_confidence = sum(
            f.get(
                'confidence_score',
                0) for f in feedback_scores) / len(feedback_scores) if feedback_scores else 0
        avg_relevance = sum(
            f.get(
                'relevance_score',
                0) for f in feedback_scores) / len(feedback_scores) if feedback_scores else 0

        report = {
            "average_clarity": round(avg_clarity, 1),
            "average_confidence": round(avg_confidence, 1),
            "average_relevance": round(avg_relevance, 1),
            "questions_answered": len(interview_responses),
            "key_strengths": extract_strengths_summary(feedback_scores),
            "improvement_areas": extract_improvement_areas(feedback_scores)
        }

        return report
    except Exception as e:
        print(f"Error generating session report: {e}")
        return {}


def extract_strengths_summary(feedback_scores: List[Dict[str, Any]]) -> str:
    """Extract common strengths from feedback"""
    strengths = []
    for feedback in feedback_scores:
        if 'strengths' in feedback:
            strengths.append(feedback['strengths'])
    return "; ".join(strengths[:3]) if strengths else "N/A"


def extract_improvement_areas(feedback_scores: List[Dict[str, Any]]) -> str:
    """Extract common improvement areas from feedback"""
    improvements = []
    for feedback in feedback_scores:
        if 'areas_for_improvement' in feedback:
            improvements.append(feedback['areas_for_improvement'])
    return "; ".join(improvements[:3]) if improvements else "N/A"


def analyze_resume_ats(resume_text: str, job_description: str, pdf_path: str = None) -> Dict[str, Any]:
    """Analyze resume against job description for ATS fit using ApyHub or Groq"""
    # 1. Try ApyHub Sharp API if token and PDF path are available
    if APYHUB_TOKEN and pdf_path and os.path.exists(pdf_path):
        try:
            print("Using ApyHub Sharp API for ATS Analysis...")
            url = "https://api.apyhub.com/sharpapi/api/v1/hr/resume_job_match_score"
            headers = {
                'Accept': 'application/json',
                'apy-token': APYHUB_TOKEN
            }
            
            payload = {
                'content': job_description,
                'language': 'English'
            }
            
            with open(pdf_path, 'rb') as f:
                files = [('file', (os.path.basename(pdf_path), f, 'application/pdf'))]
                response = requests.post(url, headers=headers, data=payload, files=files)
            
            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data.get("job_id")
                
                if job_id:
                    # Poll for results (asynchronous API)
                    status_url = f"https://api.apyhub.com/sharpapi/api/v1/hr/job/status/{job_id}"
                    max_retries = 10
                    for _ in range(max_retries):
                        time.sleep(3) # Wait 3 seconds between polls
                        status_response = requests.get(status_url, headers=headers)
                        if status_response.status_code == 200:
                            result_data = status_response.json()
                            if result_data.get("status") == "success":
                                # Extract data from Sharp API format
                                # Based on typical Sharp API response structure
                                data = result_data.get("result", {})
                                return {
                                    "score": data.get("match_score", 0),
                                    "strengths": [data.get("experience_match", "Good match")],
                                    "improvements": [data.get("gaps", "No major gaps")],
                                    "missing_keywords": data.get("missing_skills", []),
                                    "overall_feedback": data.get("summary", "Analysis completed using Sharp API.")
                                }
                        elif status_response.status_code != 202: # 202 means still processing
                            break
        except Exception as e:
            print(f"ApyHub Sharp API failed: {e}. Falling back to Groq.")

    # 2. Fallback to Groq if ApyHub fails or is not configured
    try:
        if not client:
            return {
                "score": 0,
                "strengths": ["AI Service Unavailable"],
                "improvements": ["Please check API configuration"],
                "missing_keywords": [],
                "overall_feedback": "Could not perform analysis as the AI service is currently offline."
            }

        prompt = f"""
        Act as an expert ATS (Applicant Tracking System) software and Technical Recruiter.
        Analyze the following Resume against the provided Job Description.

        Resume:
        {resume_text}

        Job Description:
        {job_description}

        Provide your analysis in the following strict JSON format:
        {{
            "score": <overall ATS match score 0-100>,
            "strengths": ["<list of 3 key strengths/matches>"],
            "improvements": ["<list of specific, actionable changes to make to resume>"],
            "missing_keywords": ["<list of important keywords from the JD missing in the resume>"],
            "overall_feedback": "<2-3 sentence overall assessment>"
        }}

        Return ONLY JSON object, no other text.
        """

        message = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.choices[0].message.content
        
        # Robustly extract JSON block
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            response_text = json_match.group(1).strip()
        else:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                # If we found double braces like {{...}}, strip one set
                temp_text = response_text[start_idx:end_idx+1].strip()
                if temp_text.startswith('{{') and temp_text.endswith('}}'):
                    temp_text = temp_text[1:-1].strip()
                response_text = temp_text
            
        try:
            analysis = json.loads(response_text)
            return {
                "score": analysis.get("score", 0) if isinstance(analysis.get("score", 0), (int, float)) else 0,
                "strengths": analysis.get("strengths", []) if isinstance(analysis.get("strengths", []), list) else [],
                "improvements": analysis.get("improvements", []) if isinstance(analysis.get("improvements", []), list) else [],
                "missing_keywords": analysis.get("missing_keywords", []) if isinstance(analysis.get("missing_keywords", []), list) else [],
                "overall_feedback": analysis.get("overall_feedback", "Analysis completed.") if isinstance(analysis.get("overall_feedback", ""), str) else "Analysis completed."
            }
        except json.JSONDecodeError:
            return {
                "score": 0,
                "strengths": ["Analysis failed"],
                "improvements": ["Please try again with a clearer JD"],
                "missing_keywords": [],
                "overall_feedback": "Could not parse AI response. Raw output: " + response_text[:100]
            }
    except Exception as e:
        print(f"Error analyzing ATS resume: {str(e)}")
        return {
            "score": 0,
            "strengths": ["Error analyzing"],
            "improvements": [f"Error: {str(e)}"],
            "missing_keywords": [],
            "overall_feedback": "An error occurred during the analysis process. Please try again later."
        }


def generate_english_teacher_response(user_text: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, str]:
    """Generate a response as an English Teacher using Groq."""
    try:
        if not client:
            return {
                "response": "I am currently offline. Please ensure Groq API key is configured.",
                "correction": "N/A"
            }
            
        history_text = ""
        if conversation_history:
            for msg in conversation_history[-4:]: # Only use the last few messages for context
                history_text += msg.get('role', 'user').capitalize() + ": " + msg.get('text', '') + "\n"
        
        prompt = """
        You are FluentEdge, an expert AI Communication Coach. The user is practicing their spoken English and professional articulation with you.
        Your goals are:
        1. Keep the conversation flowing naturally and engagingly.
        2. Keep your response concise (1-3 sentences) so it's easy for text-to-speech.
        3. If the user makes a grammatical error or awkward phrasing, politely provide a correction.
        
        User's current sentence: """ + user_text + """
        
        Recent conversation context:
        """ + history_text + """
        
        You MUST respond in the following JSON format ONLY:
        {{
            "response": "<Your natural conversational reply to the user>",
            "correction": "<Specific correction if they made an error, otherwise null>"
        }}
        """

        message = client.chat.completions.create(
            model=MODEL,
            max_tokens=256,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.choices[0].message.content
        
        # Robustly extract JSON
        import re
        import json
        
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            extracted_text = json_match.group(1).strip()
        else:
            # Aggressively find the first { and last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                # If we found double braces like {{...}}, strip one set
                temp_text = response_text[start_idx:end_idx+1].strip()
                if temp_text.startswith('{{') and temp_text.endswith('}}'):
                    temp_text = temp_text[1:-1].strip()
                extracted_text = temp_text
            else:
                extracted_text = response_text
                
        try:
            result = json.loads(extracted_text)
            return {
                "response": result.get("response", "Keep up the good work!"),
                "correction": result.get("correction")
            }
        except json.JSONDecodeError:
            # Fallback for plain text response if JSON fails
            return {
                "response": response_text.strip(),
                "correction": None
            }
    except Exception as e:
        print(f"Error in English teacher: {e}")
        return {
            "response": "I'm having trouble processing that right now. Let's try again.",
            "correction": None
        }
