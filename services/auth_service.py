from typing import Dict, Optional
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
import logging

class AuthService:
    def __init__(self):
        """Initialize authentication service with hardcoded users"""
        self.logger = logging.getLogger(__name__)
        
        # Hardcoded users for demonstration (in production, use proper database)
        self.users = {
            "doctor1": {
                "name": "Dr. Alice Johnson",
                "role": "Cardiologist",
                "department": "Cardiology",
                "password_hash": self._hash_password("password123"),
                "permissions": ["read", "write", "optimize"]
            },
            "nurse1": {
                "name": "Nurse Bob Smith", 
                "role": "Registered Nurse",
                "department": "Emergency",
                "password_hash": self._hash_password("nurse123"),
                "permissions": ["read", "optimize"]
            },
            "admin1": {
                "name": "Admin Eve Wilson",
                "role": "System Administrator", 
                "department": "IT",
                "password_hash": self._hash_password("admin123"),
                "permissions": ["read", "write", "optimize", "admin"]
            },
            "resident1": {
                "name": "Dr. Charlie Brown",
                "role": "Medical Resident",
                "department": "Internal Medicine", 
                "password_hash": self._hash_password("resident123"),
                "permissions": ["read"]
            }
        }
        
        self.logger.info(f"AuthService initialized with {len(self.users)} users")

    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 (in production, use bcrypt)"""
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate_user(self, user_id: str, password: str) -> Optional[Dict]:
        """Authenticate user credentials"""
        try:
            user = self.users.get(user_id)
            if not user:
                self.logger.warning(f"Authentication failed: user {user_id} not found")
                return None
            
            password_hash = self._hash_password(password)
            if password_hash != user["password_hash"]:
                self.logger.warning(f"Authentication failed: invalid password for {user_id}")
                return None
            
            # Return user info without password hash
            user_info = {
                "user_id": user_id,
                "name": user["name"],
                "role": user["role"],
                "department": user["department"],
                "permissions": user["permissions"]
            }
            
            self.logger.info(f"Authentication successful for {user_id}")
            return user_info
            
        except Exception as e:
            self.logger.error(f"Authentication error for {user_id}: {e}")
            return None

    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """Get user information by ID"""
        user = self.users.get(user_id)
        if not user:
            return None
        
        return {
            "user_id": user_id,
            "name": user["name"],
            "role": user["role"], 
            "department": user["department"],
            "permissions": user["permissions"]
        }

    def validate_permission(self, user_id: str, required_permission: str) -> bool:
        """Check if user has required permission"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        return required_permission in user["permissions"]

    def get_all_users(self) -> Dict:
        """Get list of all users (for admin purposes)"""
        users_list = {}
        for user_id, user_data in self.users.items():
            users_list[user_id] = {
                "name": user_data["name"],
                "role": user_data["role"],
                "department": user_data["department"],
                "permissions": user_data["permissions"]
            }
        return users_list
