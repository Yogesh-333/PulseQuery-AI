
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json

# Add project root to path
sys.path.append(os.getcwd())

# Mock dependencies before importing app
sys.modules['core.medgemma_inference'] = MagicMock()
sys.modules['core.rag_system'] = MagicMock()
sys.modules['llama_cpp'] = MagicMock()

from app import create_app, RAG_AVAILABLE

class TestRAGChat(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Mock SessionManager and AuthService to bypass auth
        self.app.session_manager = MagicMock()
        self.app.session_manager.get_session.return_value = {'user_id': 'doctor1', 'role': 'doctor'}
        self.app.session_manager.validate_session.return_value = True
        
        # Mock MedGemma
        self.app.medgemma = MagicMock()
        self.app.medgemma.generate_text.return_value = {'generated_text': 'Medical Assistant Response: Based on the history...'}
        
        # Mock RAG System
        self.app.rag_system = MagicMock()
        self.app.rag_system.search_relevant_context.return_value = ["Document 1: Patient had flu in 2020."]
        
        # Mock Database models
        self.patient_mock = MagicMock()
        self.patient_mock.name = "John Doe"
        self.patient_mock.age = 30
        self.patient_mock.gender = "Male"
        self.patient_mock.medical_history = "None"
        
        self.vital_mock = MagicMock()
        self.vital_mock.bp = "120/80"
        self.vital_mock.heart_rate = 70
        self.vital_mock.temperature = 37.0
        
    @patch('models.Patient')
    @patch('models.Vital')
    def test_rag_query_construction(self, mock_vital, mock_patient):
        # Setup DB mocks
        mock_patient.query.get.return_value = self.patient_mock
        mock_vital.query.filter_by.return_value.order_by.return_value.first.return_value = self.vital_mock
        
        # Make request
        with self.client.session_transaction() as sess:
            sess['user_id'] = 'doctor1'
            
        # We need to bypass the @require_auth decorator which might check session
        # Since we mocked session_manager, it should be fine if the decorator uses it.
        # However, the decorator in app.py uses app.session_manager.
        
        response = self.client.post('/api/consultation/chat', 
                                  json={'query': 'what is the past history', 'patient_id': 1},
                                  headers={'Authorization': 'Bearer test_token'})
        
        # Verify RAG search was called with correct query
        expected_rag_query = "what is the past history for patient John Doe"
        self.app.rag_system.search_relevant_context.assert_called_with(expected_rag_query, max_docs=3)
        
        # Verify MedGemma was called with prompt containing instruction
        call_args = self.app.medgemma.generate_text.call_args
        prompt_sent = call_args[0][0]
        
        self.assertIn("INSTRUCTION: If there is no past history", prompt_sent)
        self.assertIn("reply EXACTLY \"no past history found\"", prompt_sent)
        self.assertIn("Relevant Medical Records:", prompt_sent)
        self.assertIn("Document 1: Patient had flu in 2020.", prompt_sent)
        
        print("\n✅ Test Passed: RAG query constructed correctly and prompt contains instructions.")

if __name__ == '__main__':
    unittest.main()
