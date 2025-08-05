from functools import wraps
from flask import request, jsonify, session
import logging

logger = logging.getLogger(__name__)

def require_auth(session_manager):
    """Decorator to require authentication for endpoints"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get session ID from request headers or session
            session_id = request.headers.get('X-Session-ID') or session.get('session_id')
            
            if not session_id:
                return jsonify({
                    "error": "Authentication required",
                    "code": "NO_SESSION"
                }), 401
            
            # Validate session
            user_info = session_manager.validate_session(session_id)
            if not user_info:
                return jsonify({
                    "error": "Invalid or expired session",
                    "code": "INVALID_SESSION"
                }), 401
            
            # Add user info to request context
            request.current_user = user_info
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def require_permission(session_manager, permission):
    """Decorator to require specific permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # First check authentication
            session_id = request.headers.get('X-Session-ID') or session.get('session_id')
            
            if not session_id:
                return jsonify({
                    "error": "Authentication required",
                    "code": "NO_SESSION"
                }), 401
            
            user_info = session_manager.validate_session(session_id)
            if not user_info:
                return jsonify({
                    "error": "Invalid or expired session",
                    "code": "INVALID_SESSION"  
                }), 401
            
            # Check permission
            if permission not in user_info.get("permissions", []):
                return jsonify({
                    "error": f"Permission '{permission}' required",
                    "code": "INSUFFICIENT_PERMISSIONS"
                }), 403
            
            request.current_user = user_info
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def optional_auth(session_manager):
    """Decorator to optionally include user info if authenticated"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            session_id = request.headers.get('X-Session-ID') or session.get('session_id')
            
            request.current_user = None
            if session_id:
                user_info = session_manager.validate_session(session_id)
                if user_info:
                    request.current_user = user_info
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator