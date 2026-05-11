"""
Database module for InterPrep - SQLite operations
Handles all database interactions for storing user sessions, scores, and interview data
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

DATABASE_PATH = "interviewx.db"


def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Users/Sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            resume_path TEXT,
            ats_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Interview Questions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interview_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            question_number INTEGER,
            question_text TEXT NOT NULL,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    # User Responses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            response_text TEXT,
            video_path TEXT,
            response_time INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (question_id) REFERENCES interview_questions(id)
        )
    """)

    # Feedback/Scores table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            response_id INTEGER,
            clarity_score REAL,
            confidence_score REAL,
            relevance_score REAL,
            feedback_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (question_id) REFERENCES interview_questions(id),
            FOREIGN KEY (response_id) REFERENCES user_responses(id)
        )
    """)

    # Skills Extracted table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            skill_name TEXT NOT NULL,
            proficiency_level TEXT,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    # Interview Readiness Score table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interview_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            overall_score REAL,
            readiness_percentage REAL,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    # Video Recordings table (Added for VideoManager compatibility)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            question_id INTEGER,
            filename TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully!")


def create_session(user_name: str, resume_path: Optional[str] = None) -> int:
    """Create a new interview session"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO sessions (user_name, resume_path)
        VALUES (?, ?)
    """, (user_name, resume_path))

    conn.commit()
    session_id = cursor.lastrowid
    conn.close()

    return session_id


def get_session(session_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve session details"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def get_all_sessions() -> List[Dict[str, Any]]:
    """Retrieve all sessions with their scores"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT s.*, 
               COALESCE(i.overall_score, 0) as overall_score,
               COALESCE(i.readiness_percentage, 0) as readiness_percentage,
               COALESCE(f_agg.avg_score, 0) as avg_feedback,
               COALESCE(f_agg.q_count, 0) as question_count
        FROM sessions s
        LEFT JOIN (
            SELECT session_id, overall_score, readiness_percentage, 
            MAX(calculated_at) as last_calc
            FROM interview_scores 
            GROUP BY session_id
        ) i ON s.id = i.session_id
        LEFT JOIN (
            SELECT session_id, 
                   AVG(clarity_score + confidence_score + relevance_score) / 3 as avg_score,
                   COUNT(id) as q_count
            FROM feedback
            GROUP BY session_id
        ) f_agg ON s.id = f_agg.session_id
        ORDER BY s.created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def save_interview_question(
        session_id: int,
        question_number: int,
        question_text: str):
    """Save generated interview question"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO interview_questions (session_id, question_number, question_text)
        VALUES (?, ?, ?)
    """, (session_id, question_number, question_text))

    conn.commit()
    question_id = cursor.lastrowid
    conn.close()

    return question_id


def get_interview_questions(session_id: int) -> List[Dict[str, Any]]:
    """Get all interview questions for a session"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM interview_questions
        WHERE session_id = ?
        ORDER BY question_number
    """, (session_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def save_user_response(
        session_id: int,
        question_id: int,
        response_text: str = None,
        video_path: str = None,
        response_time: int = 0):
    """Save user's response to interview question"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO user_responses (session_id, question_id, response_text, video_path, response_time)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, question_id, response_text, video_path, response_time))

    conn.commit()
    response_id = cursor.lastrowid
    conn.close()

    return response_id


def get_user_responses(session_id: int) -> List[Dict[str, Any]]:
    """Get all user responses for a session"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM user_responses
        WHERE session_id = ?
    """, (session_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def save_feedback(session_id: int, question_id: int, response_id: int,
                  clarity_score: float, confidence_score: float,
                  relevance_score: float, feedback_text: str):
    """Save AI-generated feedback for a response"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO feedback (session_id, question_id, response_id, clarity_score,
                             confidence_score, relevance_score, feedback_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (session_id, question_id, response_id, clarity_score,
          confidence_score, relevance_score, feedback_text))

    conn.commit()
    feedback_id = cursor.lastrowid
    conn.close()

    return feedback_id


def get_feedback_for_session(session_id: int) -> List[Dict[str, Any]]:
    """Get all feedback for a session"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM feedback
        WHERE session_id = ?
    """, (session_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def save_skills(session_id: int, skills: List[Dict[str, str]]):
    """Save extracted skills from resume"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    for skill in skills:
        cursor.execute("""
            INSERT INTO skills (session_id, skill_name, proficiency_level)
            VALUES (?, ?, ?)
        """, (session_id, skill.get('name'), skill.get('proficiency', 'Intermediate')))

    conn.commit()
    conn.close()


def get_skills_for_session(session_id: int) -> List[Dict[str, Any]]:
    """Get all extracted skills for a session"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM skills
        WHERE session_id = ?
    """, (session_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def save_interview_score(
        session_id: int,
        overall_score: float,
        readiness_percentage: float):
    """Save interview readiness score"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO interview_scores (session_id, overall_score, readiness_percentage)
        VALUES (?, ?, ?)
    """, (session_id, overall_score, readiness_percentage))

    conn.commit()
    conn.close()


def get_interview_score(session_id: int) -> Optional[Dict[str, Any]]:
    """Get interview score for a session"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM interview_scores
        WHERE session_id = ?
        ORDER BY calculated_at DESC
        LIMIT 1
    """, (session_id,))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def update_session_resume(session_id: int, resume_path: str):
    """Update session with resume path"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE sessions
        SET resume_path = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (resume_path, session_id))

    conn.commit()
    conn.close()


def update_session_ats_score(session_id: int, ats_score: float):
    """Update session with ATS score"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE sessions
        SET ats_score = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (ats_score, session_id))

    conn.commit()
    conn.close()


def delete_session(session_id: int):
    """Delete a session and all associated data"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    try:
        # Delete from all tables referencing session_id
        cursor.execute("DELETE FROM feedback WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM user_responses WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM interview_questions WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM skills WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM interview_scores WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM video_recordings WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting session {session_id}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


# Initialize database on module import
if not os.path.exists(DATABASE_PATH):
    init_database()
