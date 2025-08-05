import requests
import json

def test_milestone1():
    """Test all Milestone 1 endpoints"""
    base_url = "http://localhost:5000"
    
    endpoints = [
        "/",
        "/health", 
        "/api/status"
    ]
    
    print("🧪 Testing Milestone 1 Endpoints")
    print("=" * 40)
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}")
            if response.status_code == 200:
                print(f"✅ {endpoint}: SUCCESS")
                data = response.json()
                if 'milestone' in data:
                    print(f"   📍 Milestone: {data['milestone']}")
                if 'status' in data:
                    print(f"   📊 Status: {data['status']}")
            else:
                print(f"❌ {endpoint}: FAILED ({response.status_code})")
        except Exception as e:
            print(f"❌ {endpoint}: ERROR - {e}")
        print()
    
    print("🔄 Milestone 1 testing complete!")
    print("Ready to proceed to Milestone 2: MedGemma Integration")

if __name__ == "__main__":
    test_milestone1()
