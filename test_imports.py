import sys
import os

print("Testing RAG system imports...")

try:
    from core.rag_system import RAGSystem
    print("✅ RAGSystem import successful")
except ImportError as e:
    print(f"❌ RAGSystem import failed: {e}")

try:
    from database.chromadb_manager import ChromaDBManager  
    print("✅ ChromaDBManager import successful")
except ImportError as e:
    print(f"❌ ChromaDBManager import failed: {e}")

try:
    import chromadb
    print("✅ ChromaDB package available")
except ImportError as e:
    print(f"❌ ChromaDB package missing: {e}")

try:
    import sentence_transformers
    print("✅ SentenceTransformers available")
except ImportError as e:
    print(f"❌ SentenceTransformers missing: {e}")

print(f"\nPython path: {sys.path}")
print(f"Current directory: {os.getcwd()}")
