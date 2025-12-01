import requests
import os

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

def test_upload():
    headers = {'X-Session-ID': SESSION_ID}
    
    # Create dummy file
    with open("dummy.txt", "w") as f:
        f.write("This is a test medical document.")
        
    files = {'file': ('dummy.txt', open('dummy.txt', 'rb'))}
    data = {'doc_type': 'medical', 'language': 'en'}
    
    print("Attempting upload...")
    resp = requests.post(f"{BASE_URL}/api/rag/upload", files=files, data=data, headers=headers)
    
    if resp.status_code == 200:
        print(f"✅ Upload successful: {resp.json()}")
    else:
        print(f"❌ Upload failed ({resp.status_code}): {resp.text}")

    # Cleanup
    os.remove("dummy.txt")

if __name__ == "__main__":
    if login():
        test_upload()
