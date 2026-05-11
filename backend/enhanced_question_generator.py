"""
Enhanced AI Question Generator for InterviewX
Provides diverse, contextual, and adaptive question generation
"""

import json
import random
from typing import List, Dict, Any, Optional
from processor import client, MODEL

class EnhancedQuestionGenerator:
    """Advanced question generation with multiple strategies"""
    
    def __init__(self):
        self.question_templates = {
            'behavioral': [
                "Tell me about a time when you had to lead a team through a difficult challenge.",
                "Describe a situation where you had to make a quick decision with limited information.",
                "How do you handle competing priorities from different stakeholders?",
                "Give an example of when you had to adapt to a major change mid-project.",
                "Describe a time when you received constructive criticism and how you responded.",
                "How do you motivate team members during challenging periods?",
                "Tell me about a time when you had to present bad news to stakeholders.",
                "Walk me through how you would handle [specific scenario] step by step.",
                "Describe your approach to learning new technologies or skills.",
                "How do you measure success in your role, and what metrics do you track?",
                "Tell me about a time when you had to collaborate with a difficult colleague."
            ],
            'technical': [
                "How would you design [specific system] from scratch?",
                "Explain the trade-offs between [technology A] and [technology B].",
                "Describe how you would optimize [specific process] for performance.",
                "How do you ensure code quality and maintainability in your projects?",
                "Walk me through your debugging process when faced with a complex issue.",
                "How would you implement [specific feature] in [framework/language]?",
                "Describe your experience with [specific technology or tool].",
                "How do you approach testing and quality assurance?",
                "Explain [technical concept] to someone non-technical.",
                "What's your experience with [specific domain like databases, security, etc.]?"
            ],
            'situational': [
                "Describe a time when you had to handle an angry customer or client.",
                "How would you handle it if you discovered a critical bug right before a major release?",
                "Tell me about a time when you had to work with limited resources.",
                "Describe how you would handle [specific workplace conflict].",
                "How do you prioritize tasks when everything seems urgent?",
                "Walk me through how you would handle [specific emergency situation].",
                "Describe a time when you had to convince stakeholders to adopt your approach.",
                "How do you handle pressure when working under tight deadlines?"
            ],
            'problem_solving': [
                "Describe the most complex problem you've solved in your career.",
                "How do you approach problems when you don't have all the information you need?",
                "Tell me about a time when you had to think outside the box to solve a problem.",
                "Describe your process for troubleshooting and root cause analysis.",
                "How do you validate that your solution actually solves the problem?",
                "Walk me through a time when you had to reverse engineer a solution.",
                "How do you balance between perfect solutions and practical constraints?"
            ]
        }
        
        self.difficulty_levels = {
            'beginner': {
                'focus': 'Fundamental concepts and basic implementation',
                'complexity': 'Low to Medium',
                'keywords': ['basic', 'fundamentals', 'introduction', 'getting started']
            },
            'intermediate': {
                'focus': 'Practical application and integration',
                'complexity': 'Medium to High',
                'keywords': ['practical', 'implementation', 'integration', 'real-world']
            },
            'advanced': {
                'focus': 'Optimization, architecture, and edge cases',
                'complexity': 'High to Very High',
                'keywords': ['optimization', 'architecture', 'scalability', 'performance']
            }
        }

    def generate_contextual_questions(
        self, 
        role: str, 
        experience_level: str,
        resume_context: str = "",
        num_questions: int = 5,
        difficulty: str = 'intermediate'
    ) -> List[Dict[str, Any]]:
        """Generate contextual questions based on role and experience"""
        try:
            if not client:
                return self._generate_fallback_questions(role, num_questions)
            
            # Create context-aware prompt
            context_prompt = f"""
            Generate {num_questions} diverse interview questions for a {experience_level} {role} position.
            
            Candidate Context:
            - Resume/CV: {resume_context[:200] if resume_context else "Not provided"}
            - Experience Level: {experience_level}
            - Target Difficulty: {difficulty}
            
            Requirements:
            1. Mix different question types: behavioral, technical, situational, and problem-solving
            2. Make questions specific to the role and industry
            3. Include progressive difficulty - some easier, some challenging
            4. Ensure questions are open-ended to encourage detailed responses
            5. Add role-specific scenarios they might actually encounter
            
            Return ONLY a JSON array of objects with this structure:
            [
                {{
                    "id": <unique_number>,
                    "type": "<behavioral|technical|situational|problem_solving>",
                    "question": "<specific question text>",
                    "follow_up": "<1-2 potential follow-up questions>",
                    "difficulty": "<beginner|intermediate|advanced>",
                    "focus_area": "<specific skill or competency being tested>",
                    "scoring_criteria": "<what makes a good answer>"
                }}
            ]
            """
            
            message = client.chat.completions.create(
                model=MODEL,
                max_tokens=1500,
                messages=[
                    {"role": "system", "content": context_prompt},
                    {"role": "user", "content": f"Generate interview questions for {role} at {experience_level} level"}
                ],
                response_format={"type": "json_object"}
            )
            
            content = message.choices[0].message.content
            questions = self._parse_ai_response(content)
            
            # Add metadata and scoring criteria
            for i, question in enumerate(questions):
                question.update({
                    'id': i + 1,
                    'difficulty': difficulty,
                    'estimated_time': self._estimate_response_time(question['type'], difficulty)
                })
            
            return questions
            
        except Exception as e:
            print(f"Error generating contextual questions: {str(e)}")
            return self._generate_fallback_questions(role, num_questions)

    def generate_adaptive_questions(
        self,
        role: str,
        previous_answers: List[Dict[str, Any]],
        target_weakness: Optional[str] = None,
        num_questions: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate adaptive questions based on previous performance"""
        try:
            if not client:
                return self._generate_fallback_questions(role, num_questions)
            
            # Analyze previous answers to identify patterns
            analysis_prompt = f"""
            Analyze these previous interview answers and identify patterns:
            
            Previous Answers:
            {json.dumps(previous_answers, indent=2)}
            
            Target Weakness (if any): {target_weakness or "None identified"}
            
            Generate 3 adaptive follow-up questions that:
            1. Address identified weaknesses or areas needing improvement
            2. Test deeper understanding in areas where they struggled
            3. Explore new competencies while building on strengths
            
            Role: {role}
            
            Return ONLY a JSON array with the same structure as before.
            """
            
            message = client.chat.completions.create(
                model=MODEL,
                max_tokens=1200,
                messages=[
                    {"role": "system", "content": analysis_prompt},
                    {"role": "user", "content": "Generate adaptive follow-up questions"}
                ],
                response_format={"type": "json_object"}
            )
            
            content = message.choices[0].message.content
            questions = self._parse_ai_response(content)
            
            return questions
            
        except Exception as e:
            print(f"Error generating adaptive questions: {str(e)}")
            return self._generate_fallback_questions(role, num_questions)

    def generate_industry_specific_questions(
        self,
        industry: str,
        role: str,
        num_questions: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate industry-specific questions"""
        try:
            if not client:
                return self._generate_fallback_questions(role, num_questions)
            
            industry_context = f"""
            Industry: {industry}
            Role: {role}
            
            Generate questions that reflect current industry trends, challenges, and best practices.
            Include questions about:
            - Industry-specific tools and technologies
            - Current market challenges and opportunities
            - Regulatory or compliance considerations
            - Emerging trends and future outlook
            """
            
            message = client.chat.completions.create(
                model=MODEL,
                max_tokens=1200,
                messages=[
                    {"role": "system", "content": industry_context},
                    {"role": "user", "content": f"Generate {industry} industry questions for {role}"}
                ],
                response_format={"type": "json_object"}
            )
            
            content = message.choices[0].message.content
            questions = self._parse_ai_response(content)
            
            return questions
            
        except Exception as e:
            print(f"Error generating industry questions: {str(e)}")
            return self._generate_fallback_questions(role, num_questions)

    def generate_coding_challenge_questions(
        self,
        programming_language: str,
        difficulty: str = 'intermediate',
        num_questions: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate live coding challenge questions"""
        try:
            if not client:
                return self._generate_fallback_questions(role, num_questions)
            
            coding_prompt = f"""
            Generate {num_questions} practical coding challenge questions for {programming_language} at {difficulty} level.
            
            Each question should include:
            1. A realistic coding problem they might encounter
            2. Specific constraints or requirements
            3. Expected time complexity discussion
            4. Follow-up questions about optimization or edge cases
            
            Language: {programming_language}
            Difficulty: {difficulty}
            
            Return questions that test both problem-solving ability and language knowledge.
            Structure:
            [
                {{
                    "id": <unique_number>,
                    "type": "coding_challenge",
                    "question": "<complete problem description>",
                    "constraints": "<specific requirements>",
                    "starter_code": "<optional starting point>",
                    "expected_time_complexity": "<time/space discussion>",
                    "follow_up": "<deeper technical questions>",
                    "difficulty": "{difficulty}"
                }}
            ]
            """
            
            message = client.chat.completions.create(
                model=MODEL,
                max_tokens=1500,
                messages=[
                    {"role": "system", "content": coding_prompt},
                    {"role": "user", "content": f"Generate {programming_language} coding challenges"}
                ],
                response_format={"type": "json_object"}
            )
            
            content = message.choices[0].message.content
            questions = self._parse_ai_response(content)
            
            return questions
            
        except Exception as e:
            print(f"Error generating coding questions: {str(e)}")
            return self._generate_fallback_questions(role, num_questions)

    def _parse_ai_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse AI response and ensure proper format"""
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            
            # Handle different response formats
            if isinstance(data, dict):
                for key in ['questions', 'data', 'interview_questions']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                # If no obvious key, check if there's only one key that is a list
                lists = [v for v in data.values() if isinstance(v, list)]
                if len(lists) == 1:
                    return lists[0]
            
            return data if isinstance(data, list) else []
            
        except json.JSONDecodeError as e:
            print(f"Error parsing AI response: {e}")
            return []

    def _generate_fallback_questions(self, role: str, num_questions: int) -> List[Dict[str, Any]]:
        """Generate fallback questions when AI is unavailable"""
        templates = {
            'software_developer': [
                "What's your experience with cloud platforms like AWS, Azure, or GCP?",
                "How do you approach API design and documentation?",
                "Describe your experience with microservices architecture.",
                "What's your approach to testing and code quality?",
                "How do you stay updated with new technologies and frameworks?"
            ],
            'data_scientist': [
                "What machine learning algorithms have you implemented in production?",
                "How do you handle data quality and missing values?",
                "Describe your experience with data visualization tools.",
                "What's your approach to A/B testing and experimentation?"
            ],
            'product_manager': [
                "How do you prioritize features in your product roadmap?",
                "Describe your experience with user research and data analysis.",
                "How do you handle conflicting requirements from stakeholders?",
                "What metrics do you use to measure product success?"
            ]
        }
        
        questions = []
        role_templates = templates.get(role.lower(), templates.get('general_interview', []))
        
        for i, template in enumerate(role_templates[:num_questions]):
            questions.append({
                'id': i + 1,
                'type': 'fallback',
                'question': template,
                'difficulty': 'intermediate',
                'focus_area': 'General competency'
            })
        
        return questions

    def _estimate_response_time(self, question_type: str, difficulty: str) -> str:
        """Estimate appropriate response time based on question type and difficulty"""
        base_times = {
            'behavioral': {'beginner': '2-3 min', 'intermediate': '3-5 min', 'advanced': '5-8 min'},
            'technical': {'beginner': '5-10 min', 'intermediate': '8-15 min', 'advanced': '15-25 min'},
            'situational': {'beginner': '2-4 min', 'intermediate': '4-8 min', 'advanced': '8-15 min'},
            'problem_solving': {'beginner': '10-15 min', 'intermediate': '15-25 min', 'advanced': '25-40 min'},
            'coding_challenge': {'beginner': '15-30 min', 'intermediate': '30-45 min', 'advanced': '45-60 min'}
        }
        
        return base_times.get(question_type, {}).get(difficulty, '5-10 min')

# Global instance
enhanced_generator = EnhancedQuestionGenerator()
