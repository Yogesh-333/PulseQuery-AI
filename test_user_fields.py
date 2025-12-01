import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000/api"

def test_user_fields():
    print("🚀 Starting User Fields Test...")
    
    session = requests.Session()
    
    # 0. Authenticate as admin
    print("0. Authenticating as admin...")
    try:
        auth_response = session.post(f"{BASE_URL}/auth/login", json={
            "user_id": "admin_manual",
            "password": "admin123"
        })
        if auth_response.status_code == 200:
            print("✅ Authenticated successfully.")
        else:
            print(f"❌ Authentication failed: {auth_response.status_code} - {auth_response.text}")
            return
    except Exception as e:
        print(f"❌ Exception during authentication: {e}")
        return
    
    # 1. Create a new user with all fields
    username = f"fieldtest_{int(time.time())}"
    password = "password123"
    name = "Field Test User"
    role = "doctor"
    department = "Cardiology"
    email = "test@example.com"
    phone = "123-456-7890"
    address = "123 Test St, Test City"
    
    payload = {
        "username": username,
        "password": password,
        "name": name,
        "role": role,
        "department": department,
        "email": email,
        "phone": phone,
        "address": address
    }
    
    print(f"\n1. Creating user '{username}' with all fields...")
    try:
        response = session.post(f"{BASE_URL}/admin/users", json=payload)
        if response.status_code == 200:
            print("✅ User created successfully.")
            user_data = response.json().get('user', {})
            # Verify fields in response
            if (user_data.get('email') == email and 
                user_data.get('phone') == phone and 
                user_data.get('address') == address):
                print("✅ Response contains correct new fields.")
            else:
                print(f"❌ Response missing or incorrect fields: {user_data}")
        else:
            print(f"❌ Failed to create user: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Exception during creation: {e}")
        return

    # 2. Verify persistence by fetching user list
    print(f"\n2. Verifying persistence via GET /admin/users...")
    try:
        response = session.get(f"{BASE_URL}/admin/users")
        if response.status_code == 200:
            users = response.json()
            target_user = next((u for u in users if u['username'] == username), None)
            if target_user:
                if (target_user.get('email') == email and 
                    target_user.get('phone') == phone and 
                    target_user.get('address') == address):
                    print("✅ User found with correct persisted fields.")
                else:
                    print(f"❌ User found but fields mismatch: {target_user}")
            else:
                print("❌ User not found in list.")
        else:
            print(f"❌ Failed to fetch users: {response.status_code}")
    except Exception as e:
        print(f"❌ Exception during fetch: {e}")

    # 3. Update user fields
    print(f"\n3. Updating user fields...")
    new_email = "updated@example.com"
    new_phone = "987-654-3210"
    update_payload = {
        "email": new_email,
        "phone": new_phone
    }
    
    try:
        response = session.put(f"{BASE_URL}/admin/users/{username}", json=update_payload)
        if response.status_code == 200:
            print("✅ User updated successfully.")
            updated_user = response.json().get('user', {})
            if (updated_user.get('email') == new_email and 
                updated_user.get('phone') == new_phone and
                updated_user.get('address') == address): # Address should remain unchanged
                print("✅ Update response correct.")
            else:
                print(f"❌ Update response incorrect: {updated_user}")
        else:
            print(f"❌ Failed to update user: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Exception during update: {e}")

    # 4. Clean up
    print(f"\n4. Cleaning up (Deleting user)...")
    try:
        # Send empty JSON body to avoid 400 Bad Request if Content-Type is set but body is empty
        response = session.delete(f"{BASE_URL}/admin/users/{username}", json={})
        if response.status_code == 200:
            print("✅ User deleted successfully.")
        else:
            print(f"❌ Failed to delete user: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Exception during cleanup: {e}")

if __name__ == "__main__":
    test_user_fields()
