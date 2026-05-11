"""
Main Flask application for InterPrep API
Handles all API endpoints for resume upload, question generation, response submission, and feedback
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import uuid
import json
import traceback

from config import (
    UPLOAD_FOLDER, ALLOWED_EXTENSIONS, SERVER_HOST, SERVER_PORT,
    FLASK_ENV, NUM_INTERVIEW_QUESTIONS, CORS_ORIGINS, MAX_UPLOAD_SIZE
)
from database import (
    init_database,
    create_session,
    get_session,
    get_all_sessions,
    save_interview_question,
    get_interview_questions,
    save_user_response,
    get_user_responses,
    save_feedback,
    get_feedback_for_session,
    save_skills,
    get_skills_for_session,
    save_interview_score,
    get_interview_score,
    update_session_resume,
    update_session_ats_score)
from processor import (
    extract_text_from_pdf,
    extract_skills_from_resume,
    extract_experience_summary,
    generate_interview_questions,
    analyze_response,
    calculate_interview_readiness,
    generate_session_report,
    analyze_resume_ats,
    generate_english_teacher_response)
# Video service for real-time streaming
from video_service import (
    get_video_list, get_random_video, get_video_by_filename
)
from video_manager import video_manager
from amazon_polly_integration import amazon_polly_api
from gemini_integration import gemini_api
from groq_integration import groq_api
from huggingface_integration import huggingface_api
from elevenlabs_integration import elevenlabs_api
from auth import (
    register_user,
    login_user,
    verify_token,
    get_user,
    logout_user,
    create_auth_tables)
from interview_questions import (
    get_all_available_roles, get_self_introduction_question,
    get_beginner_questions, get_intermediate_questions
)
from interview_analyzer import InterviewAnalyzer

current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(current_dir, "..", "frontend", "build")

app = Flask(__name__, static_folder=frontend_path, static_url_path='/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})

# Initialize database
init_database()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {"status": "healthy", "message": "InterPrep API is running"}), 200


@app.route('/favicon.ico')
def favicon():
    return '', 204


# ==================== AUTHENTICATION ENDPOINTS ====================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.json
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()
        full_name = data.get('full_name', '').strip()

        # Validation
        if not email or not password or not full_name:
            return jsonify({
                "success": False,
                "message": "Email, password, and full name are required"
            }), 400

        if len(password) < 6:
            return jsonify({
                "success": False,
                "message": "Password must be at least 6 characters"
            }), 400

        result = register_user(email, password, full_name)

        if result['success']:
            return jsonify(result), 201
        else:
            return jsonify(result), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Registration error: {str(e)}"
        }), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.json
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()

        if not email or not password:
            return jsonify({
                "success": False,
                "message": "Email and password are required"
            }), 400

        result = login_user(email, password)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 401
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Login error: {str(e)}"
        }), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout user"""
    try:
        data = request.json
        token = data.get('token', '')

        if not token:
            return jsonify({
                "success": False,
                "message": "Token is required"
            }), 400

        if logout_user(token):
            return jsonify({
                "success": True,
                "message": "Logged out successfully"
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Logout failed"
            }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Logout error: {str(e)}"
        }), 500


@app.route('/api/auth/me', methods=['GET'])
def get_current_user():
    """Get current authenticated user"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')

        if not token:
            return jsonify({
                "success": False,
                "message": "Token is required"
            }), 401

        user_id = verify_token(token)

        if not user_id:
            return jsonify({
                "success": False,
                "message": "Invalid or expired token"
            }), 401

        user = get_user(user_id)

        if not user:
            return jsonify({
                "success": False,
                "message": "User not found"
            }), 404

        return jsonify({
            "success": True,
            "user": user
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving user: {str(e)}"
        }), 500


# ==================== INTERVIEW SESSION ENDPOINTS ====================

@app.route('/api/session/create', methods=['POST'])
def create_new_session():
    """Create a new interview session"""
    try:
        data = request.json
        user_name = data.get('user_name', 'Anonymous')

        session_id = create_session(user_name)

        return jsonify({
            "success": True,
            "session_id": session_id,
            "message": "Session created successfully"
        }), 201
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error creating session: {str(e)}"
        }), 500


@app.route('/api/session/<int:session_id>', methods=['GET'])
def get_session_info(session_id):
    """Get session information"""
    try:
        session = get_session(session_id)

        if not session:
            return jsonify({
                "success": False,
                "message": "Session not found"
            }), 404

        return jsonify({
            "success": True,
            "session": session
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving session: {str(e)}"
        }), 500


@app.route('/api/sessions', methods=['GET'])
def get_all_user_sessions():
    """Get all interview sessions for the current environment"""
    try:
        sessions = get_all_sessions()
        return jsonify({
            "success": True,
            "sessions": sessions,
            "count": len(sessions)
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving sessions: {str(e)}"
        }), 500


@app.route('/api/session/<int:session_id>', methods=['DELETE'])
def remove_session(session_id):
    """Delete an interview session"""
    try:
        from database import delete_session
        success = delete_session(session_id)
        if success:
            return jsonify({"success": True, "message": "Session deleted"}), 200
        else:
            return jsonify({"success": False, "message": "Failed to delete session"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/resume/upload', methods=['POST'])
def upload_resume():
    """Upload and process resume"""
    try:
        session_id = request.form.get('session_id')

        if not session_id:
            return jsonify({
                "success": False,
                "message": "session_id is required"
            }), 400

        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "message": "No file provided"
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                "success": False,
                "message": "No file selected"
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "message": "Only PDF files are allowed"
            }), 400

        # Save file
        filename = secure_filename(
            f"{session_id}_{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Update session
        update_session_resume(int(session_id), filepath)

        # Extract resume text
        resume_text = extract_text_from_pdf(filepath)

        # Extract skills
        skills = extract_skills_from_resume(resume_text)
        save_skills(int(session_id), skills)

        # Extract experience summary
        experience_summary = extract_experience_summary(resume_text)

        # Calculate preliminary ATS score (using skills count as a simple metric if no JD)
        # or we can use a genericJD
        ats_analysis = analyze_resume_ats(resume_text, "A professional role requiring the skills mentioned in the resume", filepath)
        ats_score = ats_analysis.get('score', 0)
        update_session_ats_score(int(session_id), ats_score)

        return jsonify({
            "success": True,
            "message": "Resume uploaded and processed successfully",
            "file_path": filepath,
            "skills": skills,
            "experience_summary": experience_summary,
            "ats_score": ats_score
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error uploading resume: {str(e)}"
        }), 500


@app.route('/api/resume/analyze-ats', methods=['POST'])
def analyze_resume_ats_endpoint():
    """Analyze resume against job description for ATS fit"""
    try:
        job_description = request.form.get('job_description', '')
        
        if not job_description:
            return jsonify({
                "success": False,
                "message": "job_description is required"
            }), 400

        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "message": "No file provided"
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                "success": False,
                "message": "No file selected"
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "message": "Only PDF/Doc files are allowed"
            }), 400

        # Save file temporarily
        filename = secure_filename(f"ats_analyze_{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract resume text
        resume_text = extract_text_from_pdf(filepath)
        
        # Analyze using Groq or Sharp API
        analysis_result = analyze_resume_ats(resume_text, job_description, filepath)
        
        # Optionally delete temp file
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "success": True,
            "message": "Resume analyzed successfully",
            "analysis": analysis_result
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error analyzing ATS resume: {str(e)}"
        }), 500


@app.route('/api/skills/<int:session_id>', methods=['GET'])
def get_session_skills(session_id):
    """Get extracted skills for session"""
    try:
        skills = get_skills_for_session(session_id)

        return jsonify({
            "success": True,
            "skills": skills,
            "skill_count": len(skills)
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving skills: {str(e)}"
        }), 500


@app.route('/api/questions/generate', methods=['POST'])
def generate_questions():
    """Generate interview questions based on resume"""
    try:
        data = request.json
        session_id = data.get('session_id')

        if not session_id:
            return jsonify({
                "success": False,
                "message": "session_id is required"
            }), 400

        session = get_session(session_id)
        if not session or not session['resume_path']:
            return jsonify({
                "success": False,
                "message": "No resume found for this session"
            }), 404

        # Extract resume text
        resume_text = extract_text_from_pdf(session['resume_path'])

        # Generate questions
        questions = generate_interview_questions(
            resume_text, NUM_INTERVIEW_QUESTIONS)

        # Save questions to database
        saved_questions = []
        for idx, question in enumerate(questions, 1):
            question_id = save_interview_question(session_id, idx, question)
            saved_questions.append({
                "id": question_id,
                "number": idx,
                "text": question
            })

        return jsonify({
            "success": True,
            "message": "Interview questions generated successfully",
            "questions": saved_questions
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating questions: {str(e)}"
        }), 500


@app.route('/api/questions/<int:session_id>', methods=['GET'])
def get_session_questions(session_id):
    """Get interview questions for session"""
    try:
        questions = get_interview_questions(session_id)

        return jsonify({
            "success": True,
            "questions": questions
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving questions: {str(e)}"
        }), 500


@app.route('/api/response/submit', methods=['POST'])
def submit_response():
    """Submit user response to interview question"""
    try:
        data = request.json
        session_id = data.get('session_id')
        question_id = data.get('question_id')
        response_text = data.get('response_text', '')
        video_path = data.get('video_path')
        response_time = data.get('response_time', 0)

        if not session_id or not question_id:
            return jsonify({
                "success": False,
                "message": "session_id and question_id are required"
            }), 400

        # Save response
        response_id = save_user_response(
            session_id, question_id, response_text, video_path, response_time
        )

        # Get question text for analysis
        questions = get_interview_questions(session_id)
        question_text = next((q['question_text']
                             for q in questions if q['id'] == question_id), "")

        # Analyze response
        feedback = {}
        if gemini_api and gemini_api.api_key:
            try:
                gemini_feedback = gemini_api.analyze_single_response(question_text, response_text)
                if gemini_feedback.get('success'):
                    feedback = {
                        "clarity_score": gemini_feedback.get('clarity_score', 7),
                        "confidence_score": gemini_feedback.get('clarity_score', 7), # Using clarity for confidence as proxy if missing
                        "relevance_score": gemini_feedback.get('relevance_score', 7),
                        "overall_feedback": gemini_feedback.get('feedback', ''),
                        "strengths": ", ".join(gemini_feedback.get('strengths', [])),
                        "areas_for_improvement": ", ".join(gemini_feedback.get('improvement_areas', [])),
                        "ideal_answer": gemini_feedback.get('better_version', '')
                    }
            except Exception as e:
                print(f"Gemini analysis failed, falling back to Groq: {e}")
        
        if not feedback:
            feedback = analyze_response(question_text, response_text)

        # Save feedback
        feedback_id = save_feedback(
            session_id, question_id, response_id,
            feedback.get('clarity_score', 5),
            feedback.get('confidence_score', 5),
            feedback.get('relevance_score', 5),
            feedback.get('overall_feedback', '')
        )

        return jsonify({
            "success": True,
            "message": "Response submitted and analyzed successfully",
            "response_id": response_id,
            "feedback_id": feedback_id,
            "feedback": feedback
        }), 201
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error submitting response: {str(e)}"
        }), 500


@app.route('/api/responses/<int:session_id>', methods=['GET'])
def get_session_responses(session_id):
    """Get all responses for a session"""
    try:
        responses = get_user_responses(session_id)

        return jsonify({
            "success": True,
            "responses": responses
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving responses: {str(e)}"
        }), 500


@app.route('/api/feedback/<int:session_id>', methods=['GET'])
def get_session_feedback(session_id):
    """Get all feedback for a session"""
    try:
        feedback = get_feedback_for_session(session_id)

        return jsonify({
            "success": True,
            "feedback": feedback
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving feedback: {str(e)}"
        }), 500


@app.route('/api/score/calculate', methods=['POST'])
def calculate_score():
    """Calculate interview readiness score"""
    try:
        data = request.json
        session_id = data.get('session_id')

        if not session_id:
            return jsonify({
                "success": False,
                "message": "session_id is required"
            }), 400

        # Get skills and feedback
        skills = get_skills_for_session(session_id)
        feedback = get_feedback_for_session(session_id)

        # Calculate average feedback score
        if feedback:
            avg_score = sum(
                (f.get('clarity_score', 0) + f.get('confidence_score', 0) + f.get('relevance_score', 0)) / 3
                for f in feedback
            ) / len(feedback)
        else:
            avg_score = 0

        # Calculate overall score
        overall_score, readiness_percentage = calculate_interview_readiness(
            len(skills), avg_score)

        # Save score
        save_interview_score(session_id, overall_score, readiness_percentage)

        return jsonify({
            "success": True,
            "overall_score": overall_score,
            "readiness_percentage": readiness_percentage
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error calculating score: {str(e)}"
        }), 500


@app.route('/api/score/<int:session_id>', methods=['GET'])
def get_score(session_id):
    """Get interview score for session"""
    try:
        score = get_interview_score(session_id)

        if not score:
            return jsonify({
                "success": False,
                "message": "No score found for this session"
            }), 404

        return jsonify({
            "success": True,
            "score": score
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving score: {str(e)}"
        }), 500


@app.route('/api/report/<int:session_id>', methods=['GET'])
def get_session_report(session_id):
    """Get comprehensive session report"""
    try:
        session = get_session(session_id)
        if not session:
            return jsonify({
                "success": False,
                "message": "Session not found"
            }), 404

        responses = get_user_responses(session_id)
        feedback = get_feedback_for_session(session_id)
        skills = get_skills_for_session(session_id)
        score = get_interview_score(session_id)
        
        # Calculate score if not already present
        if not score:
            try:
                # Calculate average feedback score
                if feedback:
                    avg_score = sum(
                        (f.get('clarity_score', 0) + f.get('confidence_score', 0) + f.get('relevance_score', 0)) / 3
                        for f in feedback
                    ) / len(feedback)
                else:
                    avg_score = 0

                # Calculate overall score
                overall_score, readiness_percentage = calculate_interview_readiness(
                    len(skills), avg_score)

                # Save score
                save_interview_score(session_id, overall_score, readiness_percentage)
                score = get_interview_score(session_id)
            except Exception as e:
                print(f"Error calculating score during report generation: {e}")

        # Get resume text if available
        resume_text = ""
        if session['resume_path']:
            resume_text = extract_text_from_pdf(session['resume_path'])

        # Generate report
        report = generate_session_report(resume_text, responses, feedback)

        return jsonify({
            "success": True,
            "session_info": {
                "user_name": session['user_name'],
                "created_at": session['created_at']
            },
            "report": report,
            "score": score,
            "responses": responses,
            "feedback": feedback,
            "total_questions": len(get_interview_questions(session_id)),
            "responses_count": len(responses),
            "skills_count": len(skills)
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating report: {str(e)}"
        }), 500


# ==================== NEW INTERVIEW ENDPOINTS ====================

@app.route('/api/roles', methods=['GET'])
def get_interview_roles():
    """Get all available interview roles"""
    try:
        roles = get_all_available_roles()
        return jsonify({
            "success": True,
            "roles": roles,
            "count": len(roles)
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error fetching roles: {str(e)}"
        }), 500


@app.route('/api/interview/questions/<role>/<level>', methods=['GET'])
def get_interview_questions_by_role(role, level):
    """Get interview questions for a specific role and level"""
    try:
        questions = []
        # Support dynamic question generation via AI
        is_dynamic = request.args.get('dynamic', 'false').lower() == 'true'
        count = int(request.args.get('count', 5))

        if level == 'self':
            question = get_self_introduction_question(role)
            if question:
                questions = [question]
        elif level == 'beginner':
            questions = get_beginner_questions(role, count=count)
        elif level == 'intermediate':
            questions = get_intermediate_questions(role, count=count)
        elif level == 'advanced' or level == 'advance':
            from interview_questions import get_advanced_questions
            questions = get_advanced_questions(role, count=count)
        else:
            return jsonify({
                "success": False,
                "message": "Invalid level. Use: self, beginner, intermediate, or advanced"
            }), 400

        if not questions:
            return jsonify({
                "success": False,
                "message": f"No questions found for role: {role}, level: {level}"
            }), 404

        return jsonify({
            "success": True,
            "role": role,
            "level": level,
            "is_dynamic": is_dynamic,
            "questions": questions,
            "count": len(questions)
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error fetching questions: {str(e)}"
        }), 500


@app.route('/api/interview/questions/custom', methods=['POST'])
def generate_custom_interview_questions():
    """Generate interview questions for a custom job description using AI"""
    try:
        data = request.json
        job_title = data.get('job_title', 'Custom Role')
        job_description = data.get('job_description', '')
        level = data.get('level', 'intermediate')
        count = int(data.get('count', 5))
        
        if not job_description:
            return jsonify({
                "success": False,
                "message": "job_description is required"
            }), 400

        # Use processor's Groq client
        from processor import client, MODEL
        
        if not client:
            return jsonify({
                "success": False,
                "message": "AI service currently unavailable"
            }), 503
            
        prompt = f"""
        Act as an expert technical recruiter. 
        Generate {count} unique and challenging interview questions for a {level}-level {job_title} role.
        Use this Job Description/Context: {job_description}
        
        Provide the questions in the following strict JSON format:
        [
            {{
                "id": 1,
                "question": "Clear, concise interview question",
                "expected_keywords": ["important_term1", "important_term2"]
            }}
        ]
        
        Return ONLY the JSON array, no other text.
        """
        
        message = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": "You are a world-class hiring manager specializing in technical and professional recruitment."},
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.choices[0].message.content
        
        # Robustly extract JSON block
        import re
        import json
        
        try:
            # Try to find JSON array
            json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', response_text)
            if json_match:
                extracted_text = json_match.group(0)
            else:
                # Handle double square brackets if AI is confused
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    temp_text = response_text[start_idx:end_idx+1].strip()
                    if temp_text.startswith('[[') and temp_text.endswith(']]'):
                        temp_text = temp_text[1:-1].strip()
                    extracted_text = temp_text
                else:
                    extracted_text = response_text
            
            questions = json.loads(extracted_text)
        except json.JSONDecodeError:
            # Emergency fallback: provide static generic questions
            questions = [
                {"id": 1, "question": "Could you tell me more about your specific experience in this field?", "expected_keywords": []},
                {"id": 2, "question": "What attracted you to this specific job description?", "expected_keywords": []},
                {"id": 3, "question": "How do you handle challenging situations at work?", "expected_keywords": []}
            ]
            
        return jsonify({
            "success": True,
            "job_title": job_title,
            "questions": questions,
            "count": len(questions)
        }), 200
    except Exception as e:
        print(f"Error generating custom questions: {e}")
        return jsonify({
            "success": False,
            "message": f"Error generating custom questions: {str(e)}"
        }), 500


@app.route('/api/interview/video/upload', methods=['POST'])
def upload_interview_video():
    """Upload and analyze interview video"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "message": "No video file provided"
            }), 400

        file = request.files['file']
        session_id_raw = request.form.get('session_id')
        question_id = request.form.get('question_id')
        frontend_transcript = request.form.get('transcript', '')

        # Robust session_id handling
        if not session_id_raw or session_id_raw == 'null' or session_id_raw == 'undefined':
            print("WARNING: Received null session_id. Creating fallback session.")
            session_id = create_session('Candidate')
        else:
            try:
                session_id = int(session_id_raw)
            except ValueError:
                print(f"WARNING: Invalid session_id '{session_id_raw}'. Creating fallback session.")
                session_id = create_session('Candidate')

        if file.filename == '':
            return jsonify({
                "success": False,
                "message": "No file selected"
            }), 400

        # Check file extension
        allowed_video_ext = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
        if not (
                '.' in file.filename and file.filename.rsplit(
                    '.',
                    1)[1].lower() in allowed_video_ext):
            return jsonify({
                "success": False,
                "message": "Only video files are allowed (mp4, avi, mov, mkv, webm)"
            }), 400

        # Save video using video manager
        result = video_manager.save_video(
            file,
            session_id,
            int(question_id) if question_id and question_id != 'null' else None
        )

        # Get the saved video path for analysis
        video_path = result.get('file_path') if result.get('success') else None
        
        if video_path:
            # Analyze video using enhanced analyzer
            from enhanced_video_analyzer import EnhancedVideoAnalyzer
            analyzer = EnhancedVideoAnalyzer(video_path)
            analysis_results = analyzer.analyze()
            
            # Fallback to frontend transcript if no speech detected by analyzer
            # But ONLY if it's a real transcript (not a placeholder)
            dummy_phrases = ["Analysing your response...", "Listening to your answer...", "Connecting...", "null", "undefined"]
            is_dummy = any(phrase in frontend_transcript for phrase in dummy_phrases) or len(frontend_transcript.strip()) < 3
            
            if not analysis_results.get('speech_detected') and frontend_transcript and not is_dummy:
                analysis_results['speech_detected'] = True
                analysis_results['transcription'] = frontend_transcript
                analysis_results['word_count'] = len(frontend_transcript.split())
        else:
            analysis_results = {
                'error': 'Failed to save video',
                'speech_detected': True if frontend_transcript else False,
                'transcription': frontend_transcript,
                'fluency_score': 0,
                'confidence_score': 0,
                'clarity_score': 0,
                'eye_contact_score': 0,
                'posture_score': 0
            }

        # Only proceed with analysis if speech was actually detected
        if not analysis_results.get('speech_detected', False):
            # No meaningful speech detected - return zero scores
            analysis_results.update({
                'fluency_score': 0,
                'confidence_score': 0,
                'clarity_score': 0,
                'relevance_score': 0,
                'eye_contact_score': 0,
                'posture_score': 0,
                'feedback': 'No speech detected. Please speak clearly into the microphone for a proper analysis.',
                'ideal_answer': 'A good answer should be clear and concise.',
                'strengths': 'N/A - No response',
                'areas_for_improvement': 'N/A - No response',
                'relevance_score': 0
            })
        else:
            # Speech detected - analyze content
            transcription = analysis_results.get('transcription', '')
            
            # Fetch question text - Prioritize text sent from frontend to avoid mismatch
            q_text = request.form.get('question_text', '')
            
            if not q_text and question_id:
                questions_list = get_interview_questions(int(session_id)) if session_id else []
                q_text = next((q['question_text'] for q in questions_list if q['id'] == int(question_id)), "")
                if not q_text:
                    # Fallback if question_id is just index
                    try:
                        all_qs = get_beginner_questions('software_developer') # generic fallback
                        q_text = all_qs[int(question_id)-1]['question'] if int(question_id) <= len(all_qs) else ""
                    except: q_text = ""
            
            if q_text and transcription:
                text_analysis = analyze_response(q_text, transcription)
                # Merge results with safety check
                if text_analysis:
                    analysis_results['ideal_answer'] = text_analysis.get('ideal_answer', 'Could not generate an ideal answer.')
                    analysis_results['feedback'] = text_analysis.get('overall_feedback', 'No feedback available.')
                    analysis_results['relevance_score'] = text_analysis.get('relevance_score', 0)
                    analysis_results['keywords_matched'] = text_analysis.get('keywords_matched', [])
                    analysis_results['keywords_missing'] = text_analysis.get('keywords_missing', [])
                    analysis_results['strengths'] = text_analysis.get('strengths', 'N/A')
                    analysis_results['areas_for_improvement'] = text_analysis.get('areas_for_improvement', 'N/A')
                else:
                    analysis_results['ideal_answer'] = 'AI Analysis unavailable'
                    analysis_results['feedback'] = 'AI analysis failed. Please try again.'
                    analysis_results['relevance_score'] = 0
                    
            # Use AI-enhanced scores if available
            if 'ai_strengths' in analysis_results:
                analysis_results['ai_strengths'] = analysis_results['ai_strengths']
                analysis_results['ai_improvements'] = analysis_results['ai_improvements']

        # Save response with video path
        if session_id and question_id:
            # Get transcription from analysis results
            transcription = analysis_results.get('transcription', '')
            response_id = save_user_response(
                int(session_id), int(question_id),
                transcription,
                video_path,
                0
            )

            # Save feedback with correctly mapped scores
            clarity_score = analysis_results.get('clarity_score', 0)
            confidence_score = analysis_results.get('confidence_score', 0)
            relevance_score = analysis_results.get('relevance_score', 0)
            
            # Additional scores saved in feedback text
            posture_score = analysis_results.get('posture_score', 0)
            fluency_score = analysis_results.get('fluency_score', 0)
            eye_contact_score = analysis_results.get('eye_contact_score', 0)
            
            # Build comprehensive feedback message
            feedback_message = f"Speech Detected: {analysis_results.get('speech_detected', False)}"
            if analysis_results.get('speech_detected', False):
                feedback_message += f"\nClarity: {clarity_score}/10"
                feedback_message += f"\nConfidence: {confidence_score}/10"
                feedback_message += f"\nPosture: {posture_score}/10"
                feedback_message += f"\nFluency: {fluency_score}/10"
                feedback_message += f"\nEye Contact: {eye_contact_score}/10"
                feedback_message += f"\nContent Relevance: {relevance_score}/10"
                if analysis_results.get('word_count'):
                    feedback_message += f"\nWords Spoken: {analysis_results.get('word_count', 0)}"
                if analysis_results.get('ai_strengths'):
                    feedback_message += f"\nStrengths: {analysis_results.get('ai_strengths', '')}"
                if analysis_results.get('ai_improvements'):
                    feedback_message += f"\nImprovements: {analysis_results.get('ai_improvements', '')}"
            
            feedback_id = save_feedback(
                int(session_id), int(question_id), response_id,
                clarity_score,
                confidence_score,
                relevance_score,
                feedback_message
            )

        return jsonify({
            "success": True,
            "message": "Video analyzed successfully",
            "video_path": video_path,
            "analysis": analysis_results
        }), 200

    except Exception as e:
        print(f"CRITICAL ERROR in upload_interview_video: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Error uploading/analyzing video: {str(e)}"
        }), 500


@app.route('/api/interview/live-feedback', methods=['POST'])
def analyze_interview_response():
    """Analyze interview response - both text and optional video"""
    try:
        data = request.json
        role = data.get('role')
        level = data.get('level')
        response_text = data.get('response_text', '')
        question_text = data.get('question_text', '')

        if not response_text or not question_text:
            return jsonify({
                "success": False,
                "message": "response_text and question_text are required"
            }), 400

        # Analyze response
        feedback = analyze_response(question_text, response_text)

        # Get word suggestions
        word_suggestions = InterviewAnalyzer.get_word_suggestions(
            response_text)

        # Count filler words
        filler_count = 0
        for filler in [
            'uhh',
            'umm',
            'uh',
            'um',
            'like',
            'you know',
            'basically',
                'actually']:
            filler_count += response_text.lower().count(filler)

        return jsonify({
            "success": True,
            "feedback": feedback,
            "word_suggestions": word_suggestions,
            "filler_word_count": filler_count,
            "additional_tips": [
                "Maintain confident tone and pace",
                "Use specific examples when possible",
                "Avoid filler words and unnecessary pauses",
                "Practice good grammar and articulation"
            ]
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error analyzing response: {str(e)}"
        }), 500


@app.route('/api/english-teacher', methods=['POST'])
def english_teacher_interaction():
    """Real-time interaction with English Teacher AI using Hugging Face models via Groq"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        history = data.get('history', [])
        
        if not text:
            return jsonify({
                "success": False,
                "message": "Text is required"
            }), 400
            
        result = generate_english_teacher_response(text, history)
        
        return jsonify({
            "success": True,
            "response": result.get("response", ""),
            "correction": result.get("correction", None)
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error communicating with teacher: {str(e)}"
        }), 500


# ==================== ENHANCED QUESTION ENDPOINTS ====================

@app.route('/api/questions/enhanced/<role>/<level>', methods=['GET'])
def get_enhanced_questions(role, level):
    """Get enhanced AI-generated questions"""
    try:
        experience_level = request.args.get('experience', 'intermediate')
        resume_context = request.args.get('resume_context', '')
        
        questions = enhanced_generator.generate_contextual_questions(
            role, experience_level, resume_context
        )
        
        return jsonify({
            "success": True,
            "questions": questions,
            "role": role,
            "level": level,
            "experience_level": experience_level
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating enhanced questions: {str(e)}"
        }), 500


@app.route('/api/questions/adaptive', methods=['POST'])
def generate_adaptive_questions():
    """Generate adaptive questions based on previous performance"""
    try:
        data = request.json
        role = data.get('role', 'software_developer')
        previous_answers = data.get('previous_answers', [])
        target_weakness = data.get('target_weakness')
        
        questions = enhanced_generator.generate_adaptive_questions(
            role, previous_answers, target_weakness
        )
        
        return jsonify({
            "success": True,
            "questions": questions,
            "adaptive": True
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating adaptive questions: {str(e)}"
        }), 500


@app.route('/api/videos/<int:session_id>', methods=['GET'])
def get_session_videos(session_id):
    """Get all videos for a session"""
    try:
        videos = video_manager.get_session_videos(session_id)
        return jsonify({
            "success": True,
            "videos": videos,
            "count": len(videos)
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving videos: {str(e)}"
        }), 500


@app.route('/api/videos/<int:video_id>', methods=['GET'])
def get_video_details(video_id):
    """Get specific video details"""
    try:
        video = video_manager.get_video(video_id)
        if video:
            return jsonify({
                "success": True,
                "video": video
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Video not found"
            }), 404
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving video: {str(e)}"
        }), 500


@app.route('/api/videos/<int:video_id>', methods=['DELETE'])
def delete_video(video_id):
    """Delete a video"""
    try:
        result = video_manager.delete_video(video_id)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error deleting video: {str(e)}"
        }), 500


@app.route('/api/videos/stats', methods=['GET'])
def get_video_storage_stats():
    """Get video storage statistics"""
    try:
        stats = video_manager.get_storage_stats()
        return jsonify({
            "success": True,
            "stats": stats
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error getting storage stats: {str(e)}"
        }), 500


@app.route('/api/videos/<filename>', methods=['GET'])
def serve_video(filename):
    """Serve video files"""
    try:
        file_path = os.path.join(video_manager.storage_path, filename)
        
        if os.path.exists(file_path):
            return send_from_directory(
                video_manager.storage_path,
                filename,
                as_attachment=True
            )
        else:
            return jsonify({
                "success": False,
                "message": "Video not found"
            }), 404
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error serving video: {str(e)}"
        }), 500


# ==================== GEMINI API ENDPOINTS ====================

@app.route('/api/gemini/questions/generate', methods=['POST'])
def generate_gemini_questions():
    """Generate questions using Gemini AI"""
    try:
        data = request.json
        job_description = data.get('job_description', '')
        experience_level = data.get('experience_level', 'intermediate')
        role = data.get('role', 'Software Developer')
        num_questions = data.get('num_questions', 5)
        
        questions = gemini_api.generate_interview_questions(
            job_description, experience_level, num_questions, role
        )
        
        return jsonify({
            "success": True,
            "questions": questions,
            "generated_by": "gemini",
            "count": len(questions)
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating questions: {str(e)}"
        }), 500


@app.route('/api/gemini/performance/analyze', methods=['POST'])
def analyze_interview_performance_gemini():
    """Analyze interview performance using Gemini AI"""
    try:
        data = request.json
        questions = data.get('questions', [])
        responses = data.get('responses', [])
        job_description = data.get('job_description', '')
        
        analysis = gemini_api.analyze_interview_performance(
            questions, responses, job_description
        )
        
        return jsonify({
            "success": True,
            "analysis": analysis,
            "analyzed_by": "gemini"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error analyzing performance: {str(e)}"
        }), 500


@app.route('/api/gemini/scenarios/generate', methods=['POST'])
def generate_role_playing_scenarios():
    """Generate role-playing interview scenarios using Gemini"""
    try:
        data = request.json
        role = data.get('role', 'Software Developer')
        scenario_type = data.get('scenario_type', 'behavioral')
        difficulty = data.get('difficulty', 'intermediate')
        
        scenarios = gemini_api.generate_role_playing_scenarios(
            role, scenario_type, difficulty
        )
        
        return jsonify({
            "success": True,
            "scenarios": scenarios,
            "generated_by": "gemini",
            "count": len(scenarios)
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating scenarios: {str(e)}"
        }), 500




# ==================== OPENAI WHISPER API ENDPOINTS ====================

@app.route('/api/whisper/transcribe', methods=['POST'])
def transcribe_audio_whisper():
    """Transcribe audio using OpenAI Whisper API"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "message": "No audio file provided"
            }), 400
        
        audio_file = request.files['file']
        model = request.form.get('model', 'whisper-1')
        language = request.form.get('language', 'en')
        
        # Transcribe using Whisper API
        result = whisper_api.transcribe_audio_file(audio_file, model)
        
        return jsonify({
            "success": True,
            "transcription": result.get('transcription', ''),
            "model": model,
            "language": result.get('language', language),
            "duration": result.get('duration', 0),
            "transcribed_by": "whisper"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error transcribing audio: {str(e)}"
        }), 500


@app.route('/api/whisper/transcribe/video', methods=['POST'])
def transcribe_video_whisper():
    """Extract audio from video and transcribe using Whisper"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "message": "No video file provided"
            }), 400
        
        video_file = request.files['file']
        model = request.form.get('model', 'whisper-1')
        language = request.form.get('language', 'en')
        
        # Transcribe video using Whisper API
        result = whisper_api.transcribe_video_audio(video_file, model)
        
        return jsonify({
            "success": True,
            "transcription": result.get('transcription', ''),
            "model": model,
            "language": result.get('language', language),
            "duration": result.get('duration', 0),
            "transcribed_by": "whisper"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error transcribing video: {str(e)}"
        }), 500


@app.route('/api/whisper/transcribe/diarize', methods=['POST'])
def transcribe_with_diarization():
    """Transcribe with speaker diarization"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "message": "No audio file provided"
            }), 400
        
        audio_file = request.files['file']
        model = request.form.get('model', 'whisper-1')
        language = request.form.get('language', 'en')
        num_speakers = int(request.form.get('num_speakers', 2))
        
        # Transcribe with diarization
        result = whisper_api.transcribe_with_diarization(audio_file, model, num_speakers)
        
        return jsonify({
            "success": True,
            "transcription": result.get('transcription', ''),
            "segments": result.get('segments', []),
            "speakers": result.get('speakers', {}),
            "model": model,
            "language": result.get('language', language),
            "transcribed_by": "whisper"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error in diarization: {str(e)}"
        }), 500


# ==================== ELEVENLABS API ENDPOINTS ====================

@app.route('/api/elevenlabs/speech/generate', methods=['POST'])
def generate_speech_elevenlabs():
    """Generate speech from text using ElevenLabs"""
    try:
        data = request.json
        text = data.get('text', '')
        voice_id = data.get('voice_id', 'rachel')
        model_id = data.get('model_id', 'eleven_multilingual_v2')
        stability = float(data.get('stability', 0.75))
        similarity_boost = float(data.get('similarity_boost', 0.75))
        style = data.get('style', 'professional')
        
        speech = elevenlabs_api.generate_speech_from_text(
            text, voice_id, model_id, stability, similarity_boost, style
        )
        
        return jsonify({
            "success": True,
            "speech": speech,
            "generated_by": "elevenlabs"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating speech: {str(e)}"
        }), 500


@app.route('/api/elevenlabs/voices', methods=['GET'])
def get_available_voices_elevenlabs():
    """Get available voices from ElevenLabs"""
    try:
        result = elevenlabs_api.get_available_voices()
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error getting voices: {str(e)}"
        }), 500


@app.route('/api/elevenlabs/interviewer/generate', methods=['POST'])
def generate_interviewer_script_elevenlabs():
    """Generate complete interviewer script using ElevenLabs"""
    try:
        data = request.json
        script_type = data.get('script_type', 'interview')
        role = data.get('role', 'interviewer')
        voice_id = data.get('voice_id', 'rachel')
        model_id = data.get('model_id', 'eleven_multilingual_v2')
        style = data.get('style', 'professional')
        num_questions = data.get('num_questions', 5)
        
        script = elevenlabs_api.generate_interviewer_script(
            script_type, role, voice_id, model_id, style, num_questions
        )
        
        return jsonify({
            "success": True,
            "script": script,
            "generated_by": "elevenlabs"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating script: {str(e)}"
        }), 500


# ==================== AMAZON POLLY API ENDPOINTS ====================

@app.route('/api/polly/speech/generate', methods=['POST'])
def generate_speech_polly():
    """Generate speech from text using Amazon Polly"""
    try:
        data = request.json
        text = data.get('text', '')
        voice_id = data.get('voice_id', 'Joanna')
        output_format = data.get('output_format', 'mp3')
        sample_rate = data.get('sample_rate', 22050)
        language_code = data.get('language_code', 'en-US')
        engine = data.get('engine', 'neural')
        
        speech = amazon_polly_api.generate_speech_from_text(
            text, voice_id, output_format, sample_rate, language_code, engine
        )
        
        return jsonify({
            "success": True,
            "speech": speech,
            "generated_by": "amazon_polly"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating speech: {str(e)}"
        }), 500


@app.route('/api/polly/voices', methods=['GET'])
def get_available_voices_polly():
    """Get available voices from Amazon Polly"""
    try:
        voices = amazon_polly_api.get_available_voices()
        
        return jsonify({
            "success": True,
            "voices": voices,
            "generated_by": "amazon_polly"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error getting voices: {str(e)}"
        }), 500


@app.route('/api/polly/speech/ssml', methods=['POST'])
def generate_speech_ssml_polly():
    """Generate speech with SSML using Amazon Polly"""
    try:
        data = request.json
        ssml_text = data.get('ssml_text', '')
        voice_id = data.get('voice_id', 'Joanna')
        output_format = data.get('output_format', 'mp3')
        sample_rate = data.get('sample_rate', 22050)
        
        speech = amazon_polly_api.generate_speech_with_ssml(
            ssml_text, voice_id, sample_rate
        )
        
        return jsonify({
            "success": True,
            "speech": speech,
            "generated_by": "amazon_polly"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating SSML speech: {str(e)}"
        }), 500


# ==================== GROQ API ENDPOINTS ====================

@app.route('/api/groq/questions/generate', methods=['POST'])
def generate_questions_groq():
    """Generate interview questions using Groq"""
    try:
        data = request.json
        role = data.get('role', 'Software Developer')
        experience_level = data.get('experience_level', 'intermediate')
        num_questions = data.get('num_questions', 5)
        job_description = data.get('job_description', '')
        model = data.get('model', 'llama-3.1-8b-instant')
        
        result = groq_api.generate_interview_questions(
            role, experience_level, num_questions, job_description, model
        )
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating questions: {str(e)}"
        }), 500


@app.route('/api/groq/performance/analyze', methods=['POST'])
def analyze_performance_groq():
    """Analyze interview performance using Groq"""
    try:
        data = request.json
        questions = data.get('questions', [])
        responses = data.get('responses', [])
        job_description = data.get('job_description', '')
        model = data.get('model', 'llama-3.1-8b-instant')
        
        analysis = groq_api.analyze_interview_performance(
            questions, responses, job_description, model
        )
        
        return jsonify({
            "success": True,
            "analysis": analysis,
            "analyzed_by": "groq"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error analyzing performance: {str(e)}"
        }), 500


@app.route('/api/groq/interviewer/generate', methods=['POST'])
def generate_interviewer_groq():
    """Generate AI interviewer using Groq"""
    try:
        data = request.json
        role = data.get('role', 'Software Developer')
        interview_style = data.get('interview_style', 'professional')
        model = data.get('model', 'llama-3.1-8b-instant')
        
        interviewer = groq_api.generate_ai_interviewer(
            role, interview_style, model
        )
        
        return jsonify({
            "success": True,
            "interviewer": interviewer,
            "generated_by": "groq"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating interviewer: {str(e)}"
        }), 500


@app.route('/api/groq/technical/assess', methods=['POST'])
def assess_technical_groq():
    """Assess technical correctness using Groq"""
    try:
        data = request.json
        question = data.get('question', '')
        answer = data.get('answer', '')
        context = data.get('context', '')
        model = data.get('model', 'llama-3.1-8b-instant')
        
        assessment = groq_api.assess_technical_answer(
            question, answer, context, model
        )
        
        return jsonify({
            "success": True,
            "assessment": assessment,
            "assessed_by": "groq"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error assessing technical: {str(e)}"
        }), 500


# ==================== HUGGING FACE API ENDPOINTS ====================

@app.route('/api/hf/sentiment/analyze', methods=['POST'])
def analyze_sentiment_hf():
    """Analyze sentiment and emotions using Hugging Face"""
    try:
        data = request.json
        text = data.get('text', '')
        models = data.get('models', ['distilbert-base-uncased-finetuned-sst-2-english'])
        
        result = huggingface_api.analyze_sentiment_and_emotions(
            text, models
        )
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error analyzing sentiment: {str(e)}"
        }), 500


@app.route('/api/hf/classify', methods=['POST'])
def classify_text_hf():
    """Classify text using Hugging Face"""
    try:
        data = request.json
        text = data.get('text', '')
        model = data.get('model', 'facebook/bart-large-mnli')
        
        classification = huggingface_api.classify_text(
            text, model
        )
        
        return jsonify({
            "success": True,
            "classification": classification,
            "classified_by": "huggingface"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error classifying text: {str(e)}"
        }), 500


@app.route('/api/hf/questions/generate', methods=['POST'])
def generate_questions_hf():
    """Generate questions using Hugging Face"""
    try:
        data = request.json
        role = data.get('role', 'Software Developer')
        experience_level = data.get('experience_level', 'intermediate')
        num_questions = data.get('num_questions', 5)
        model = data.get('model', 'microsoft/DialoGPT-medium')
        
        questions = huggingface_api.generate_interview_questions(
            role, experience_level, num_questions, model
        )
        
        return jsonify({
            "success": True,
            "questions": questions,
            "generated_by": "huggingface"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating questions: {str(e)}"
        }), 500


@app.route('/api/hf/keywords/extract', methods=['POST'])
def extract_keywords_hf():
    """Extract keywords using Hugging Face"""
    try:
        data = request.json
        text = data.get('text', '')
        model = data.get('model', 'dslim/bert-base-NER')
        
        keywords = huggingface_api.extract_keywords_and_topics(
            text, model
        )
        
        return jsonify({
            "success": True,
            "keywords": keywords,
            "extracted_by": "huggingface"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error extracting keywords: {str(e)}"
        }), 500


# ==================== STATIC FILE SERVING ====================

@app.route("/")
def serve():
    return send_from_directory(app.static_folder, 'index.html')

# ==================== VIDEO STREAMING ENDPOINTS ====================

@app.route('/api/videos/list', methods=['GET'])
def list_videos():
    """Get list of all available videos"""
    try:
        videos = get_video_list()
        return jsonify({
            "success": True,
            "videos": videos,
            "count": len(videos)
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error listing videos: {str(e)}"
        }), 500


@app.route('/api/videos/random', methods=['GET'])
def get_random_video_endpoint():
    """Get a random video for interviewer"""
    try:
        video = get_random_video()
        if not video:
            return jsonify({
                "success": False,
                "message": "No videos available"
            }), 404
        
        return jsonify({
            "success": True,
            "video": {
                "filename": video['filename'],
                "url": f"/api/videos/stream/{video['filename']}",
                "size": video['size']
            }
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error getting random video: {str(e)}"
        }), 500


@app.route('/api/videos/stream/<filename>', methods=['GET'])
def stream_video(filename):
    """Stream video file from recordings folder"""
    try:
        from config import RECORDINGS_FOLDER
        video_path = os.path.join(RECORDINGS_FOLDER, filename)
        
        if not os.path.exists(video_path):
            return jsonify({
                "success": False,
                "message": "Video not found"
            }), 404
        
        # Determine mimetype based on extension
        mimetype = 'video/mp4'
        if filename.lower().endswith('.webm'):
            mimetype = 'video/webm'
        elif filename.lower().endswith('.mov'):
            mimetype = 'video/quicktime'

        # Check for Range header for streaming support
        range_header = request.headers.get('Range', None)
        if range_header:
            return send_file(
                video_path,
                mimetype=mimetype,
                as_attachment=False,
                download_name=filename
            ), 206
        else:
            return send_file(
                video_path,
                mimetype=mimetype,
                as_attachment=False,
                download_name=filename
            ), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error streaming video: {str(e)}"
        }), 500


@app.route('/api/videos/download/<filename>', methods=['GET'])
def download_video(filename):
    """Download video file"""
    try:
        from config import RECORDINGS_FOLDER
        video_path = os.path.join(RECORDINGS_FOLDER, filename)
        
        if not os.path.exists(video_path):
            return jsonify({
                "success": False,
                "message": "Video not found"
            }), 404
        
        return send_file(
            video_path,
            as_attachment=True,
            download_name=filename
        ), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error downloading video: {str(e)}"
        }), 500


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({"success": False, "message": "API route not found"}), 404
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    app.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=FLASK_ENV == 'development')
