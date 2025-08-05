"""
MedGemma Inference Module with GPU Support and CPU Fallback
Supports automatic GPU detection and graceful fallback to CPU
"""

import os
import time
import threading
from typing import Dict, Any, Optional
from llama_cpp import Llama
import traceback

class MedGemmaInference:
    """
    MedGemma model inference class with GPU support and CPU fallback
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model: Optional[Llama] = None
        self.device = "unknown"
        self.loading_status = {
            "status": "not_started",
            "progress": 0,
            "error": None,
            "device": "unknown"
        }
        self._loading_thread = None
        self._ready = False
        
        print(f"🔍 Initializing MedGemma with model: {model_path}")
        
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Check file size
        file_size = os.path.getsize(model_path) / (1024**3)  # GB
        print(f"📊 Model file size: {file_size:.2f} GB")
    
    def start_background_loading(self):
        """Start loading the model in background thread"""
        if self._loading_thread and self._loading_thread.is_alive():
            print("⚠️ Model already loading in background")
            return
        
        print("🔄 Starting background model loading...")
        self._loading_thread = threading.Thread(target=self._load_model, daemon=True)
        self._loading_thread.start()
    
    def _check_gpu_support(self) -> bool:
        """Check if GPU/CUDA support is available"""
        try:
            # Try to import CUDA-related functions
            from llama_cpp import llama_cpp
            return hasattr(llama_cpp, 'llama_supports_gpu_offload')
        except Exception as e:
            print(f"🔍 GPU check failed: {e}")
            return False
    
    def _load_model(self):
        """Load the model with GPU support and CPU fallback"""
        try:
            self.loading_status = {
                "status": "loading",
                "progress": 10,
                "error": None,
                "device": "detecting"
            }
            
            # Check GPU availability
            gpu_available = self._check_gpu_support()
            print(f"🔍 GPU support available: {gpu_available}")
            
            if gpu_available:
                # Try GPU first
                try:
                    print("🚀 Attempting to load model with GPU acceleration...")
                    self.loading_status["progress"] = 30
                    
                    self.model = Llama(
                        model_path=self.model_path,
                        n_ctx=2048,                    # Context window
                        n_threads=4,                   # CPU threads for non-GPU operations
                        n_gpu_layers=25,               # Offload layers to GPU (adjust for 1050Ti)
                        n_batch=512,                   # Batch size
                        verbose=True,                  # Show detailed loading info
                        use_mmap=True,                 # Memory mapping
                        use_mlock=False                # Don't lock memory
                    )
                    
                    self.device = "cuda"
                    print("✅ Model loaded successfully with GPU acceleration!")
                    
                except Exception as gpu_error:
                    print(f"⚠️ GPU loading failed: {gpu_error}")
                    print("🔄 Falling back to CPU loading...")
                    self._load_cpu_model()
            else:
                print("🔄 GPU not available, loading on CPU...")
                self._load_cpu_model()
            
            # Final status update
            self.loading_status = {
                "status": "loaded",
                "progress": 100,
                "error": None,
                "device": self.device
            }
            self._ready = True
            print(f"✅ Model ready on {self.device.upper()}")
            
        except Exception as e:
            error_msg = f"Model loading failed: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            
            self.loading_status = {
                "status": "failed",
                "progress": 0,
                "error": error_msg,
                "device": "none"
            }
            self._ready = False
    
    def _load_cpu_model(self):
        """Load model on CPU only"""
        try:
            print("🔄 Loading model on CPU...")
            self.loading_status["progress"] = 60
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=2048,                    # Context window
                n_threads=6,                   # More CPU threads when not using GPU
                n_batch=256,                   # Smaller batch for CPU
                verbose=True,                  # Show loading info
                use_mmap=True,                 # Memory mapping
                use_mlock=False                # Don't lock memory
            )
            
            self.device = "cpu"
            self.loading_status["progress"] = 90
            print("✅ Model loaded successfully on CPU")
            
        except Exception as e:
            print(f"❌ CPU loading also failed: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference"""
        return self._ready and self.model is not None
    
    def get_loading_status(self) -> Dict[str, Any]:
        """Get current loading status"""
        return {
            **self.loading_status,
            "model_path": self.model_path,
            "ready": self.is_ready()
        }
    
    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate text using the loaded model
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.1-1.0)
        
        Returns:
            Dictionary with generated text and metadata
        """
        if not self.is_ready():
            raise RuntimeError("Model not ready. Check loading status.")
        
        print(f"🔍 DEBUG: Generating text")
        print(f"   Prompt length: {len(prompt)} characters")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Temperature: {temperature}")
        print(f"   Model device: {self.device}")
        print(f"   First 100 chars of prompt: {prompt[:100]}...")
        
        start_time = time.time()
        
        try:
            # Ensure minimum temperature to avoid empty responses
            safe_temp = max(temperature, 0.3)
            safe_max_tokens = min(max_tokens, 800)  # Cap to prevent issues
            
            print(f"🔄 Calling model.generate() with safe_max_tokens={safe_max_tokens}, safe_temp={safe_temp}")
            
            # Generate text with optimized parameters
            result = self.model(
                prompt=prompt,
                max_tokens=safe_max_tokens,
                temperature=safe_temp,
                top_p=0.9,                     # Nucleus sampling
                top_k=40,                      # Top-k sampling
                repeat_penalty=1.1,           # Prevent repetition
                stop=[],                      # No early stopping
                echo=False,                   # Don't echo prompt
                stream=False                  # Complete generation
            )
            
            print(f"📤 Raw model response: {result}")
            
            # Extract generated text
            generated_text = ""
            output_tokens = 0
            
            if 'choices' in result and len(result['choices']) > 0:
                generated_text = result['choices'][0]['text']
                if 'usage' in result:
                    output_tokens = result['usage'].get('completion_tokens', 0)
            
            print(f"📝 Extracted text length: {len(generated_text)}")
            
            if generated_text:
                print(f"✅ Generated text preview: {generated_text[:50]}...")
            else:
                print("❌ WARNING: Generated text is EMPTY!")
                print(f"   Raw choice[0]: {result['choices'][0] if 'choices' in result and result['choices'] else 'No choices'}")
            
            end_time = time.time()
            generation_time = end_time - start_time
            tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
            
            print(f"🎯 Final result: {len(generated_text)} chars generated")
            
            return {
                'generated_text': generated_text,
                'output_tokens': output_tokens,
                'generation_time': generation_time,
                'tokens_per_second': tokens_per_second,
                'model': f'MedGemma-{"GPU" if self.device == "cuda" else "CPU"}',
                'device': self.device,
                'temperature': safe_temp,
                'max_tokens': safe_max_tokens,
                'backend': 'llama_cpp'
            }
            
        except Exception as e:
            error_msg = f"Text generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            
            return {
                'generated_text': '',
                'output_tokens': 0,
                'generation_time': 0,
                'tokens_per_second': 0,
                'model': 'MedGemma-ERROR',
                'device': self.device,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'backend': 'llama_cpp',
                'error': error_msg
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'ready': self.is_ready(),
            'loading_status': self.loading_status,
            'model_size_gb': os.path.getsize(self.model_path) / (1024**3) if os.path.exists(self.model_path) else 0
        }
    
    def unload(self):
        """Unload the model to free memory"""
        if self.model:
            del self.model
            self.model = None
            self._ready = False
            print("🗑️ Model unloaded from memory")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'model') and self.model:
            self.unload()

# Utility function for testing
def test_model(model_path: str, prompt: str = "Explain fever and nausea symptoms."):
    """
    Test function to verify model loading and generation
    
    Args:
        model_path: Path to the GGUF model file
        prompt: Test prompt to generate
    """
    print("🧪 Testing MedGemma model...")
    
    try:
        # Initialize model
        inference = MedGemmaInference(model_path)
        
        # Start loading
        inference.start_background_loading()
        
        # Wait for loading (with timeout)
        max_wait = 300  # 5 minutes
        wait_time = 0
        
        while not inference.is_ready() and wait_time < max_wait:
            status = inference.get_loading_status()
            print(f"⏳ Loading... {status['status']} ({status['progress']}%)")
            time.sleep(10)
            wait_time += 10
        
        if not inference.is_ready():
            print("❌ Model loading timed out")
            return
        
        print("✅ Model loaded successfully!")
        
        # Test generation
        print(f"🧪 Testing generation with prompt: {prompt[:50]}...")
        result = inference.generate_text(prompt, max_tokens=200, temperature=0.7)
        
        print(f"📝 Generated text ({len(result['generated_text'])} chars):")
        print(f"   {result['generated_text'][:300]}...")
        print(f"⚡ Performance: {result['tokens_per_second']:.2f} tokens/sec on {result['device']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        test_model(model_path)
    else:
        print("Usage: python medgemma_inference.py <model_path>")
        print("Example: python medgemma_inference.py models/medgemma-4b-it-Q8_0.gguf")
