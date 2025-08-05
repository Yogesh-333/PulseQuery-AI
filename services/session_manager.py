from typing import Dict, Optional, List
import uuid
import time
import json
from datetime import datetime, timezone, timedelta
import logging

def serialize_metadata_for_db(obj):
    """Convert lists and other non-primitive types to strings for ChromaDB compatibility"""
    if isinstance(obj, dict):
        return {k: serialize_metadata_for_db(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Convert list to comma-separated string
        return ','.join(str(x) for x in obj)
    else:
        return obj

def deserialize_metadata_from_db(obj):
    """Convert comma-separated strings back to lists when loading from DB"""
    if isinstance(obj, dict):
        return {k: deserialize_metadata_from_db(v) for k, v in obj.items()}
    elif isinstance(obj, str):
        # Check if this looks like a permissions list
        if ',' in obj and all(part.strip() in ['read', 'write', 'optimize', 'admin'] for part in obj.split(',')):
            # Convert comma-separated permissions back to list
            return [x.strip() for x in obj.split(',')]
        else:
            return obj
    else:
        return obj

class SessionManager:
    def __init__(self, db_manager=None, session_timeout_hours: int = 24):
        """Initialize session manager with proper ChromaDB serialization"""
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager  # ChromaDB manager for persistence
        self.session_timeout_hours = session_timeout_hours
        
        # In-memory session cache for fast access
        self.active_sessions = {}
        
        # Load existing sessions from database
        if self.db_manager:
            self._load_sessions_from_db()
        
        self.logger.info("SessionManager initialized with ChromaDB serialization support")

    def create_session(self, user_info: Dict) -> str:
        """Create a new session for authenticated user with proper serialization"""
        try:
            session_id = str(uuid.uuid4())
            
            # Create session data with serialized permissions for DB storage
            session_data = {
                "session_id": session_id,
                "user_id": user_info["user_id"],
                "user_name": user_info["name"],
                "user_role": user_info["role"],
                "user_department": user_info["department"],
                "permissions": user_info["permissions"],  # Keep as list for memory
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "expires_at": (datetime.now(timezone.utc) + timedelta(hours=self.session_timeout_hours)).isoformat(),
                "is_active": True
            }
            
            # Store in memory cache (keep original format with list permissions)
            self.active_sessions[session_id] = session_data
            
            # Store in database with serialized format
            if self.db_manager:
                self._save_session_to_db(session_data)
            
            self.logger.info(f"Session created for user {user_info['user_id']}: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to create session for {user_info['user_id']}: {e}")
            raise

    def validate_session(self, session_id: str) -> Optional[Dict]:
        """Validate session and return user info if valid"""
        try:
            if not session_id:
                return None
            
            session = self.active_sessions.get(session_id)
            if not session:
                # Try loading from database
                session = self._load_session_from_db(session_id)
                if session:
                    self.active_sessions[session_id] = session
            
            if not session:
                return None
            
            # Check if session is expired
            expires_at = datetime.fromisoformat(session["expires_at"])
            if datetime.now(timezone.utc) > expires_at:
                self.logger.info(f"Session expired: {session_id}")
                self.terminate_session(session_id)
                return None
            
            # Update last activity
            session["last_activity"] = datetime.now(timezone.utc).isoformat()
            
            # Update in database
            if self.db_manager:
                self._save_session_to_db(session)
            
            return {
                "user_id": session["user_id"],
                "user_name": session["user_name"],
                "user_role": session["user_role"],
                "user_department": session["user_department"],
                "permissions": session["permissions"],  # This will be a list
                "session_id": session_id
            }
            
        except Exception as e:
            self.logger.error(f"Session validation error for {session_id}: {e}")
            return None

    def terminate_session(self, session_id: str) -> bool:
        """Terminate a session"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session["is_active"] = False
                session["terminated_at"] = datetime.now(timezone.utc).isoformat()
                
                # Update in database
                if self.db_manager:
                    self._save_session_to_db(session)
                
                # Remove from active cache
                del self.active_sessions[session_id]
                
                self.logger.info(f"Session terminated: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to terminate session {session_id}: {e}")
            return False

    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get all active sessions for a user"""
        user_sessions = []
        for session in self.active_sessions.values():
            if session["user_id"] == user_id and session["is_active"]:
                user_sessions.append({
                    "session_id": session["session_id"],
                    "created_at": session["created_at"],
                    "last_activity": session["last_activity"],
                    "expires_at": session["expires_at"]
                })
        return user_sessions

    def get_session_statistics(self) -> Dict:
        """Get session statistics"""
        active_count = len(self.active_sessions)
        
        # Count by role
        role_counts = {}
        for session in self.active_sessions.values():
            role = session["user_role"]
            role_counts[role] = role_counts.get(role, 0) + 1
        
        return {
            "total_active_sessions": active_count,
            "sessions_by_role": role_counts,
            "session_timeout_hours": self.session_timeout_hours
        }

    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now(timezone.utc)
        expired_sessions = []
        
        for session_id, session in list(self.active_sessions.items()):
            expires_at = datetime.fromisoformat(session["expires_at"])
            if current_time > expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.terminate_session(session_id)
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def _save_session_to_db(self, session_data: Dict):
        """Save session to database with proper serialization for ChromaDB"""
        if not self.db_manager:
            return
        
        try:
            # Serialize metadata to handle lists and other complex types
            serialized_session_data = serialize_metadata_for_db(session_data.copy())
            
            # Use the sessions collection from ChromaDB
            self.db_manager.sessions_collection.upsert(
                documents=[json.dumps(serialized_session_data)],
                metadatas=[serialized_session_data],
                ids=[serialized_session_data["session_id"]]
            )
            
            self.logger.debug(f"Session saved to DB: {serialized_session_data['session_id']}")
            
        except Exception as e:
            self.logger.error(f"Failed to save session to DB: {e}")

    def _load_session_from_db(self, session_id: str) -> Optional[Dict]:
        """Load session from database with proper deserialization"""
        if not self.db_manager:
            return None
        
        try:
            results = self.db_manager.sessions_collection.get(ids=[session_id])
            if results["ids"]:
                session_data = results["metadatas"][0]
                # Deserialize metadata to restore lists
                deserialized_session_data = deserialize_metadata_from_db(session_data)
                self.logger.debug(f"Session loaded from DB: {session_id}")
                return deserialized_session_data
        except Exception as e:
            self.logger.error(f"Failed to load session from DB: {e}")
        
        return None

    def _load_sessions_from_db(self):
        """Load all active sessions from database on startup with deserialization"""
        if not self.db_manager:
            return
        
        try:
            # Get all sessions from database
            results = self.db_manager.sessions_collection.get()
            
            if results["ids"]:
                current_time = datetime.now(timezone.utc)
                loaded_count = 0
                
                for session_data in results["metadatas"]:
                    # Deserialize the session data
                    deserialized_session_data = deserialize_metadata_from_db(session_data)
                    
                    if deserialized_session_data.get("is_active"):
                        # Check if session is still valid
                        try:
                            expires_at = datetime.fromisoformat(deserialized_session_data["expires_at"])
                            if current_time < expires_at:
                                session_id = deserialized_session_data["session_id"]
                                self.active_sessions[session_id] = deserialized_session_data
                                loaded_count += 1
                        except (ValueError, KeyError) as e:
                            self.logger.warning(f"Invalid session data in DB, skipping: {e}")
                            continue
                        
            self.logger.info(f"Loaded {loaded_count} active sessions from database")
            
        except Exception as e:
            self.logger.error(f"Failed to load sessions from database: {e}")

    def refresh_session(self, session_id: str) -> bool:
        """Refresh session expiration time"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session["last_activity"] = datetime.now(timezone.utc).isoformat()
                session["expires_at"] = (datetime.now(timezone.utc) + timedelta(hours=self.session_timeout_hours)).isoformat()
                
                # Update in database
                if self.db_manager:
                    self._save_session_to_db(session)
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to refresh session {session_id}: {e}")
            return False

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get detailed session information"""
        session = self.active_sessions.get(session_id)
        if session:
            return {
                "session_id": session["session_id"],
                "user_id": session["user_id"],
                "user_name": session["user_name"],
                "user_role": session["user_role"],
                "user_department": session["user_department"],
                "permissions": session["permissions"],
                "created_at": session["created_at"],
                "last_activity": session["last_activity"],
                "expires_at": session["expires_at"],
                "is_active": session["is_active"]
            }
        return None

    def update_session_activity(self, session_id: str):
        """Update session last activity timestamp"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_activity"] = datetime.now(timezone.utc).isoformat()
            # Save to database
            if self.db_manager:
                self._save_session_to_db(self.active_sessions[session_id])
