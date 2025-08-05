# test_model.py
from llama_cpp import Llama
import time

print("🔄 Loading model...")
model = Llama(
    model_path="models/medgemma-4b-it-Q8_0.gguf",
    n_ctx=2048,      # Context window
    n_threads=4,     # CPU threads (adjust based on your CPU)
    verbose=False    # Reduce logging
)

print("✅ Model loaded successfully!")

# Test simple generation
test_prompts = [
    "Hello, how are you?",
    "Explain fever and nausea symptoms.",
    "Patient Name: Rogers, Pamela. Create a medical report."
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n🧪 Test {i}: {prompt}")
    
    start_time = time.time()
    result = model(
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,    # Higher temperature
        top_p=0.9,         # Nucleus sampling
        top_k=40,          # Top-k sampling
        repeat_penalty=1.1, # Prevent loops
        stop=[]            # Don't stop early
    )
    end_time = time.time()
    
    generated_text = result['choices'][0]['text']
    print(f"📝 Generated ({len(generated_text)} chars, {end_time-start_time:.2f}s):")
    print(f"   {generated_text[:200]}{'...' if len(generated_text) > 200 else ''}")
    
    if not generated_text:
        print("❌ Empty generation detected!")
    else:
        print("✅ Generation successful!")

print("\n🏁 Model testing complete!")
