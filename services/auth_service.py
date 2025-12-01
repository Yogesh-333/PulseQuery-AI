
from typing import Dict, Optional
import logging
from models import User, db

class AuthService:
    def __init__(self):
        """Initialize authentication service"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("AuthService initialized with Database backend")

    def authenticate_user(self, user_id: str, password: str) -> Optional[Dict]:
        """Authenticate user credentials against database"""
        try:
            user = User.query.filter_by(username=user_id).first()
            
            if not user:
                self.logger.warning(f"Authentication failed: user {user_id} not found")
                return None
            
            if not user.check_password(password):
                self.logger.warning(f"Authentication failed: invalid password for {user_id}")
                return None
            
            # Return user info
            user_info = {
                "user_id": user.username,
                "name": user.name,
                "role": user.role,
                "department": user.department,
                "permissions": self._get_permissions_for_role(user.role)
            }
            
            self.logger.info(f"Authentication successful for {user_id}")
            return user_info
            
        except Exception as e:
            self.logger.error(f"Authentication error for {user_id}: {e}")
            return None

    def _get_permissions_for_role(self, role: str) -> list:
        """Get permissions based on role"""
        permissions = {
            "admin": ["read", "write", "optimize", "admin"],
            "doctor": ["read", "write", "optimize"],
            "lab": ["read", "write"],
            "patient": ["read"]
        }
        return permissions.get(role, [])

    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """Get user information by ID"""
        user = User.query.filter_by(username=user_id).first()
        if not user:
            return None
        
        return {
            "user_id": user.username,
            "name": user.name,
            "role": user.role, 
            "department": user.department,
            "permissions": self._get_permissions_for_role(user.role)
        }

    def create_user(self, username, password, name, role, department=None, email=None, phone=None, address=None):
        """Create a new user"""
        try:
            if User.query.filter_by(username=username).first():
                return None, "Username already exists"
                
            user = User(
                username=username,
                role=role,
                name=name,
                department=department,
                email=email,
                phone=phone,
                address=address
            )
            user.set_password(password)
            
            db.session.add(user)
            db.session.commit()
            
            return user.to_dict(), None
        except Exception as e:
            db.session.rollback()
            return None, str(e)

    def get_all_users(self):
        """Get all users"""
        try:
            users = User.query.all()
            return [user.to_dict() for user in users]
        except Exception as e:
            self.logger.error(f"Error fetching users: {e}")
            return []

    def delete_user(self, user_id):
        """Delete a user by ID (username)"""
        try:
            user = User.query.filter_by(username=user_id).first()
            if not user:
                return False, "User not found"
            
            db.session.delete(user)
            db.session.commit()
            self.logger.info(f"User deleted: {user_id}")
            return True, None
        except Exception as e:
            self.logger.error(f"Error deleting user {user_id}: {e}")
            db.session.rollback()
            return False, str(e)

    def update_user(self, user_id, data):
        """Update a user"""
        try:
            user = User.query.filter_by(username=user_id).first()
            if not user:
                return None, "User not found"
            
            if 'name' in data:
                user.name = data['name']
            if 'role' in data:
                user.role = data['role']
            if 'department' in data:
                user.department = data['department']
            if 'email' in data:
                user.email = data['email']
            if 'phone' in data:
                user.phone = data['phone']
            if 'address' in data:
                user.address = data['address']
            if 'password' in data and data['password']:
                user.set_password(data['password'])
                
            db.session.commit()
            return user.to_dict(), None
        except Exception as e:
            db.session.rollback()
            return None, str(e)
