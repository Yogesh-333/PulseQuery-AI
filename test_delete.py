import requests
import json

BASE_URL = "http://localhost:5000"

def test_delete():
    # 1. Login
    print("Logging in with password123...")
    resp = requests.post(f"{BASE_URL}/api/auth/login", json={"user_id": "admin1", "password": "password123"})
    
    if resp.status_code != 200:
        print(f"Login failed with password123: {resp.text}")
        print("Trying admin123...")
        resp = requests.post(f"{BASE_URL}/api/auth/login", json={"user_id": "admin1", "password": "admin123"})
        
    if resp.status_code != 200:
        print(f"Login failed with admin123: {resp.text}")
        return
    
    data = resp.json()
    session_id = data.get("session_id")
    print(f"Logged in. Session ID: {session_id}")
    
    headers = {"X-Session-ID": session_id}
    
    # 2. Create User
    print("Creating user 'deletetest'...")
    new_user = {
        "username": "deletetest",
        "password": "password123",
        "name": "Delete Test",
        "role": "doctor",
        "department": "Test"
    }
    resp = requests.post(f"{BASE_URL}/api/admin/users", json=new_user, headers=headers)
    if resp.status_code == 200:
        print("User created.")
    elif "Username already exists" in resp.text:
        print("User already exists, proceeding to delete.")
    else:
        print(f"Create failed: {resp.text}")
        return

    # 3. Delete User
    print("Deleting user 'deletetest'...")
    resp = requests.delete(f"{BASE_URL}/api/admin/users/deletetest", headers=headers)
    print(f"Delete status: {resp.status_code}")
    print(f"Delete response: {resp.text}")

if __name__ == "__main__":
    test_delete()
