import requests
import json

BASE_URL = "http://localhost:5000"
SESSION_ID = None

def login():
    global SESSION_ID
    resp = requests.post(f"{BASE_URL}/api/auth/login", json={"user_id": "admin1", "password": "admin123"})
    if resp.status_code == 200:
        SESSION_ID = resp.json()['session_id']
        print("✅ Login successful")
        return True
    else:
        print(f"❌ Login failed: {resp.text}")
        return False

def create_patient_user():
    headers = {'X-Session-ID': SESSION_ID}
    data = {
        "username": "proptest5",
        "password": "password123",
        "name": "Prop Test Patient 5",
        "role": "patient",
        "department": "General"
    }
    resp = requests.post(f"{BASE_URL}/api/admin/users", json=data, headers=headers)
    if resp.status_code == 200:
        print("✅ User 'Prop Test Patient' created")
        return True
    else:
        print(f"❌ Create user failed: {resp.text}")
        return False

def check_patient_exists(name, should_exist=True):
    headers = {'X-Session-ID': SESSION_ID}
    resp = requests.get(f"{BASE_URL}/api/patients", headers=headers)
    if resp.status_code == 200:
        patients = resp.json()
        found = any(p['name'] == name for p in patients)
        if found == should_exist:
            print(f"✅ Patient '{name}' {'found' if found else 'not found'} as expected")
            return True
        else:
            print(f"❌ Patient '{name}' {'found' if found else 'not found'}, expected {'found' if should_exist else 'not found'}")
            return False
    else:
        print(f"❌ Get patients failed: {resp.text}")
        return False

def delete_user(username):
    headers = {'X-Session-ID': SESSION_ID}
    resp = requests.delete(f"{BASE_URL}/api/admin/users/{username}", headers=headers, json={})
    if resp.status_code == 200:
        print(f"✅ User '{username}' deleted")
        return True
    else:
        print(f"❌ Delete user failed: {resp.text}")
        return False

if __name__ == "__main__":
    if login():
        # 1. Create User
        if create_patient_user():
            # 2. Check Patient Exists
            if check_patient_exists("Prop Test Patient 5", True):
                # 3. Delete User
                if delete_user("proptest5"):
                    # 4. Check Patient Gone
                    check_patient_exists("Prop Test Patient 5", False)
