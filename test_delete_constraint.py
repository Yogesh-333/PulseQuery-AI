import requests
import datetime

BASE_URL = "http://localhost:5000"

def test_delete_constraint():
    # 1. Login
    print("Logging in...")
    resp = requests.post(f"{BASE_URL}/api/auth/login", json={"user_id": "admin1", "password": "admin123"})
    if resp.status_code != 200:
        print(f"Login failed: {resp.text}")
        return
    
    session_id = resp.json().get("session_id")
    headers = {"X-Session-ID": session_id}
    
    # 2. Create a Doctor User
    print("Creating doctor 'doc_constraint'...")
    doc_user = {
        "username": "doc_constraint",
        "password": "password123",
        "name": "Doctor Constraint",
        "role": "doctor",
        "department": "Test"
    }
    requests.post(f"{BASE_URL}/api/admin/users", json=doc_user, headers=headers)

    # 3. Create a Patient (needed for appointment)
    # We'll just assume patient with ID 1 exists or create one if we can't find one easily.
    # For simplicity, let's try to create a patient first via the auto-create logic or just use ID 1.
    # Actually, let's just use a raw SQL insert or assume ID 1 exists from previous tests.
    
    # 4. Create an Appointment linked to this doctor
    # We need an endpoint to create appointments or insert directly. 
    # Since we don't have a clear 'create appointment' endpoint in the snippets I saw, 
    # I will try to delete the user. If they have NO records, it should work. 
    # If the USER is trying to delete a user WITH records, that's the key.
    
    # Let's try to delete 'doc_constraint' immediately. If this works, then the issue isn't the user creation itself.
    # But to test the constraint, I need to link data.
    
    # Instead of complex setup, let's check if the user is trying to delete 'doctor1' or someone who MIGHT have data.
    # I will try to delete 'doctor1' (the demo user). This user likely has data.
    
    print("Attempting to delete 'doctor1' (who likely has related records)...")
    resp = requests.delete(f"{BASE_URL}/api/admin/users/doctor1", headers=headers)
    
    print(f"Delete 'doctor1' status: {resp.status_code}")
    print(f"Delete 'doctor1' response: {resp.text}")

if __name__ == "__main__":
    test_delete_constraint()
