"""
Authentication module for InterviewX
Handles user registration, login, and session management
"""

import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os

DATABASE_PATH = "interviewx.db"


def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_auth_tables():
    """Create authentication tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def register_user(email: str, password: str, full_name: str) -> Dict[str, Any]:
    """Register a new user"""
    try:
        create_auth_tables()

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            conn.close()
            return {"success": False, "message": "Email already registered"}

        # Hash password and create user
        password_hash = hash_password(password)
        cursor.execute("""
            INSERT INTO users (email, password_hash, full_name)
            VALUES (?, ?, ?)
        """, (email, password_hash, full_name))

        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return {
            "success": True,
            "user_id": user_id,
            "email": email,
            "message": "User registered successfully"
        }
    except Exception as e:
        return {"success": False, "message": f"Registration failed: {str(e)}"}


def login_user(email: str, password: str) -> Dict[str, Any]:
    """Login user and create session"""
    try:
        create_auth_tables()

        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Find user
        cursor.execute(
            "SELECT id, password_hash, full_name FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()

        if not user:
            conn.close()
            return {"success": False, "message": "Invalid email or password"}

        # Verify password
        password_hash = hash_password(password)
        if user['password_hash'] != password_hash:
            conn.close()
            return {"success": False, "message": "Invalid email or password"}

        # Create session token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=30)

        cursor.execute("""
            INSERT INTO user_sessions (user_id, token, expires_at)
            VALUES (?, ?, ?)
        """, (user['id'], token, expires_at))

        conn.commit()
        conn.close()

        return {
            "success": True,
            "user": {
                "id": user['id'],
                "email": email,
                "full_name": user['full_name']
            },
            "token": token,
            "message": "Login successful"
        }
    except Exception as e:
        return {"success": False, "message": f"Login failed: {str(e)}"}


def verify_token(token: str) -> Optional[int]:
    """Verify session token and return user_id"""
    try:
        create_auth_tables()

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id FROM user_sessions
            WHERE token = ? AND expires_at > datetime('now')
        """, (token,))

        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None
    except Exception:
        return None


def get_user(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user information"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, email, full_name, created_at FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        conn.close()

        return dict(user) if user else None
    except Exception:
        return None


def logout_user(token: str) -> bool:
    """Logout user by invalidating session"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM user_sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()

        return True
    except Exception:
        return False
