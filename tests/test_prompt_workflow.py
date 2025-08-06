import requests
import json

def test_prompt_optimization_workflow():
    """Test the complete prompt optimization workflow"""
    base_url = "http://localhost:5000"
    
    # Test data
    test_query = "Patient Name: Rogers, Pamela\nAge: 45\nChief Complaint: Chest pain and shortness of breath"
    
    print("🧪 Testing Prompt Optimization Workflow...")
    
    # Step 1: Login
    login_response = requests.post(f"{base_url}/api/auth/login", 
                                 json={"user_id": "doctor1", "password": "password123"})
    
    if login_response.status_code == 200:
        session_id = login_response.json()["session_id"]
        print("✅ Login successful")
        
        # Step 2: Optimize prompt
        headers = {"X-Session-ID": session_id}
        optimize_response = requests.post(f"{base_url}/api/prompt/optimize",
                                        json={"query": test_query, "use_context": True},
                                        headers=headers)
        
        if optimize_response.status_code == 200:
            optimization_data = optimize_response.json()
            print("✅ Prompt optimization successful")
            print(f"   Query Type: {optimization_data['query_type']}")
            print(f"   Medical Specialty: {optimization_data['medical_specialty']}")
            print(f"   Optimized Prompt Length: {optimization_data['metrics']['length']} chars")
            
            # Step 3: Generate from optimized prompt
            optimized_prompt = optimization_data['optimized_prompt']
            generate_response = requests.post(f"{base_url}/api/prompt/generate-final",
                                            json={"final_prompt": optimized_prompt},
                                            headers=headers)
            
            if generate_response.status_code == 200:
                generation_data = generate_response.json()
                print("✅ AI generation successful")
                print(f"   Response Length: {generation_data['generation_info']['response_length']} chars")
                print("✅ Complete workflow test passed!")
                return True
            else:
                print(f"❌ Generation failed: {generate_response.text}")
        else:
            print(f"❌ Optimization failed: {optimize_response.text}")
    else:
        print(f"❌ Login failed: {login_response.text}")
    
    return False

if __name__ == "__main__":
    test_prompt_optimization_workflow()
