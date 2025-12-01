import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    print("Attempting to import RAGSystem...")
    from core.rag_system import RAGSystem
    print("✅ Import successful!")
    
    print("Attempting to initialize RAGSystem...")
    rag = RAGSystem(data_dir='data/chromadb')
    print("✅ Initialization successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
