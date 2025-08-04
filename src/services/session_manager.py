# src/services/session_manager.py

import datetime
from typing import Dict, Any, List

class SessionManager:
    """
    Manages user session history and prompt optimization results.
    Starts with in-memory storage, with a plan for ChromaDB migration.
    """
    _instance = None # Singleton pattern
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._instance._sessions = {} # {session_id: session_data}
            cls._instance._next_session_id = 1
        return cls._instance

    def add_session_data(self, user_id: str, session_data: Dict[str, Any]) -> int:
        """
        Adds a new session entry for a user.
        Args:
            user_id (str): Identifier for the user (e.g., doctor's ID, patient ID).
            session_data (Dict[str, Any]): Dictionary containing session details
                                            (original_query, candidates, scores, etc.).
        Returns:
            int: The unique session ID generated for this entry.
        """
        session_id = self._next_session_id
        self._next_session_id += 1

        if user_id not in self._sessions:
            self._sessions[user_id] = {}
        
        # Add timestamp for auditability
        session_data['timestamp'] = datetime.datetime.now().isoformat()
        
        self._sessions[user_id][session_id] = session_data
        print(f"Session {session_id} added for user '{user_id}'.")
        return session_id

    def get_session_history(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all session history for a specific user.
        Args:
            user_id (str): The ID of the user.
        Returns:
            List[Dict[str, Any]]: A list of session data dictionaries, sorted by timestamp.
        """
        user_sessions = self._sessions.get(user_id, {})
        # Sort sessions by timestamp for chronological order
        sorted_sessions = sorted(user_sessions.values(), key=lambda x: x.get('timestamp', ''))
        return sorted_sessions

    def get_session_by_id(self, user_id: str, session_id: int) -> Dict[str, Any] | None:
        """
        Retrieves a specific session by its ID for a given user.
        """
        return self._sessions.get(user_id, {}).get(session_id)

    def clear_all_sessions(self):
        """Clears all sessions from memory. (For testing/development only)."""
        self._sessions = {}
        self._next_session_id = 1
        print("All sessions cleared from memory.")

# --- Testing the SessionManager (add to your if __name__ == "__main__": block in prompt_optimizer.py for now) ---
# We will temporarily add this test to prompt_optimizer.py's __main__ block
# to demonstrate its usage, but in the final structure, a higher-level service
# will orchestrate it.