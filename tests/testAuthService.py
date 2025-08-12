import unittest
from services.auth_service import AuthService


class TestAuthService(unittest.TestCase):

    def setUp(self):
        """Set up AuthService instance before each test"""
        self.auth_service = AuthService()

    def test_authenticate_valid_user(self):
        """Test authentication with correct credentials"""
        user_info = self.auth_service.authenticate_user("doctor1", "password123")
        self.assertIsNotNone(user_info)
        self.assertEqual(user_info["user_id"], "doctor1")
        self.assertIn("read", user_info["permissions"])

    def test_authenticate_invalid_user(self):
        """Test authentication with a non-existing user"""
        user_info = self.auth_service.authenticate_user("ghost", "password123")
        self.assertIsNone(user_info)

    def test_authenticate_wrong_password(self):
        """Test authentication with incorrect password"""
        user_info = self.auth_service.authenticate_user("doctor1", "wrongpassword")
        self.assertIsNone(user_info)

    def test_get_user_info_valid(self):
        """Test fetching user info for a valid user"""
        user_info = self.auth_service.get_user_info("nurse1")
        self.assertIsNotNone(user_info)
        self.assertEqual(user_info["name"], "Nurse Bob Smith")

    def test_get_user_info_invalid(self):
        """Test fetching user info for an invalid user"""
        user_info = self.auth_service.get_user_info("unknown_user")
        self.assertIsNone(user_info)

    def test_validate_permission_true(self):
        """Test permission check when permission exists"""
        self.assertTrue(self.auth_service.validate_permission("admin1", "admin"))

    def test_validate_permission_false(self):
        """Test permission check when permission does not exist"""
        self.assertFalse(self.auth_service.validate_permission("resident1", "write"))

    def test_get_all_users(self):
        """Test getting all users"""
        users = self.auth_service.get_all_users()
        self.assertIsInstance(users, dict)
        self.assertIn("doctor1", users)
        self.assertIn("name", users["doctor1"])


if __name__ == "__main__":
    unittest.main()
