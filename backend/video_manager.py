"""
Video Storage and Management System for InterviewX
Handles video upload, storage, compression, and retrieval
"""

import os
import uuid
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
import sqlite3

class VideoManager:
    """Manages video files for interview recordings"""
    
    def __init__(self, storage_path: str = "recordings"):
        self.storage_path = storage_path
        self.ensure_storage_directory()
        
    def ensure_storage_directory(self):
        """Create storage directory if it doesn't exist"""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)
            print(f"Created video storage directory: {self.storage_path}")
    
    def save_video(self, video_blob, session_id: int, question_id: int, 
                   filename: Optional[str] = None) -> Dict[str, Any]:
        """Save video blob to storage"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"session_{session_id}_q{question_id}_{timestamp}.webm"
            
            file_path = os.path.join(self.storage_path, filename)
            
            # Save video file
            if hasattr(video_blob, 'save'):
                video_blob.save(file_path)
                # Need to seek(0) if we want to read it again, but we just saved it
                size = os.path.getsize(file_path)
            else:
                with open(file_path, 'wb') as f:
                    f.write(video_blob)
                size = len(video_blob)
            
            # Save metadata to database
            self._save_video_metadata(session_id, question_id, filename, size)
            
            return {
                "success": True,
                "filename": filename,
                "file_path": file_path,
                "size": len(video_blob),
                "message": "Video saved successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error saving video: {str(e)}"
            }
    
    def _save_video_metadata(self, session_id: int, question_id: int, 
                       filename: str, size: int):
        """Save video metadata to database"""
        try:
            conn = sqlite3.connect("interviewx.db")
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO video_recordings 
                (session_id, question_id, filename, file_size, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, question_id, filename, size, datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving video metadata: {e}")
    
    def get_session_videos(self, session_id: int) -> List[Dict[str, Any]]:
        """Get all videos for a session"""
        try:
            conn = sqlite3.connect("interviewx.db")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, filename, file_size, created_at 
                FROM video_recordings 
                WHERE session_id = ? 
                ORDER BY created_at DESC
            """, (session_id,))
            
            videos = []
            for row in cursor.fetchall():
                videos.append({
                    "id": row["id"],
                    "filename": row["filename"],
                    "file_size": row["file_size"],
                    "created_at": row["created_at"],
                    "file_url": f"/api/videos/{row['filename']}"
                })
            
            conn.close()
            return videos
            
        except Exception as e:
            print(f"Error retrieving videos: {e}")
            return []
    
    def get_video(self, video_id: int) -> Optional[Dict[str, Any]]:
        """Get specific video details"""
        try:
            conn = sqlite3.connect("interviewx.db")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, filename, file_size, created_at 
                FROM video_recordings 
                WHERE id = ?
            """, (video_id,))
            
            video = cursor.fetchone()
            conn.close()
            
            if video:
                return {
                    "id": video["id"],
                    "filename": video["filename"],
                    "file_size": video["file_size"],
                    "created_at": video["created_at"],
                    "file_url": f"/api/videos/{video['filename']}"
                }
            return None
            
        except Exception as e:
            print(f"Error retrieving video: {e}")
            return None
    
    def delete_video(self, video_id: int) -> Dict[str, Any]:
        """Delete a video and its metadata"""
        try:
            conn = sqlite3.connect("interviewx.db")
            cursor = conn.cursor()
            
            # Get video info first
            cursor.execute("""
                SELECT filename FROM video_recordings WHERE id = ?
            """, (video_id,))
            video = cursor.fetchone()
            
            if video:
                file_path = os.path.join(self.storage_path, video["filename"])
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted video file: {video['filename']}")
            
            # Delete from database
            cursor.execute("""
                DELETE FROM video_recordings WHERE id = ?
            """, (video_id,))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "message": f"Video {video['filename'] if video else 'unknown'} deleted successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error deleting video: {str(e)}"
            }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            total_size = 0
            file_count = 0
            
            if os.path.exists(self.storage_path):
                for filename in os.listdir(self.storage_path):
                    file_path = os.path.join(self.storage_path, filename)
                    if os.path.isfile(file_path):
                        file_count += 1
                        total_size += os.path.getsize(file_path)
            
            return {
                "total_files": file_count,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "storage_path": self.storage_path
            }
            
        except Exception as e:
            return {
                "total_files": 0,
                "total_size_mb": 0,
                "storage_path": self.storage_path,
                "error": str(e)
            }

# Initialize video manager
video_manager = VideoManager()
