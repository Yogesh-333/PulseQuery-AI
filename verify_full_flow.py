
import requests
import json
import sys

BASE_URL = "http://localhost:5000"

def print_step(step):
    print(f"\n{'='*50}\n{step}\n{'='*50}")

def verify_flow():
    session = requests.Session()
    
    # 1. Login as Doctor
    print_step("1. Testing Doctor Login")
    try:
        login_payload = {"user_id": "doctor1", "password": "password123"}
        response = session.post(f"{BASE_URL}/api/auth/login", json=login_payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code != 200 or not response.json().get('success'):
            print("❌ Login failed!")
            return
        
        print("✅ Login successful!")
        user_data = response.json()['user']
        print(f"User: {user_data['name']} ({user_data['role']})")
        
    except Exception as e:
        print(f"❌ Login error: {e}")
        return

    # 2. Fetch Patients (Doctor Dashboard)
    print_step("2. Fetching Patients (Doctor Dashboard)")
    try:
        response = session.get(f"{BASE_URL}/api/patients")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            patients = response.json()
            print(f"✅ Fetched {len(patients)} patients")
            for p in patients:
                print(f" - ID: {p['id']}, Name: {p['name']}, Age: {p['age']}")
            
            if not patients:
                print("⚠️ No patients found to test consultation with.")
                return
            
            patient_id = patients[0]['id']
        else:
            print(f"❌ Failed to fetch patients: {response.text}")
            return

    except Exception as e:
        print(f"❌ Fetch patients error: {e}")
        return

    # 3. Fetch Patient Details (Consultation View)
    print_step(f"3. Fetching Details for Patient ID {patient_id}")
    try:
        response = session.get(f"{BASE_URL}/api/patients/{patient_id}")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            patient = response.json()
            print(f"✅ Fetched details for {patient['name']}")
            print(f" - Medical History: {patient['medical_history']}")
            print(f" - Vitals: {patient.get('latest_vitals')}")
        else:
            print(f"❌ Failed to fetch patient details: {response.text}")
            return

    except Exception as e:
        print(f"❌ Fetch details error: {e}")
        return

    # 4. Test AI Consultation
    print_step("4. Testing AI Consultation Chat")
    try:
        chat_payload = {
            "query": "What are the risks for this patient given their history?",
            "patient_id": patient_id
        }
        print(f"Sending query: {chat_payload['query']}")
        
        # Note: This might take a while if running on CPU
        response = session.post(f"{BASE_URL}/api/consultation/chat", json=chat_payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ AI Response received:")
            print(f"Response: {result.get('response')}")
        else:
            print(f"❌ AI Chat failed: {response.text}")

    except Exception as e:
        print(f"❌ AI Chat error: {e}")

if __name__ == "__main__":
    verify_flow()
