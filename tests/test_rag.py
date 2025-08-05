# test_rag_direct.py
import sys
import os
sys.path.append('.')

try:
    print("🧪 Testing RAG system directly...")
    
    # Import RAG system
    from core.rag_system import RAGSystem
    print("✅ RAG system imported")
    
    # Create data directory
    data_dir = 'data/chromadb'
    os.makedirs(data_dir, exist_ok=True)
    print(f"✅ Directory created: {data_dir}")
    
    # Initialize RAG system with medical embeddings
    print("🔄 Initializing RAG with medical embeddings...")
    rag = RAGSystem(
        data_dir=data_dir,
        embedding_model="medical"  # This should use MedEmbed
    )
    print("✅ RAG system initialized successfully")
    
    # Test basic functionality
    stats = rag.get_system_stats()
    print(f"📊 RAG stats: {stats}")
    
    print("✅ RAG system test completed successfully")
    
except ImportError as import_error:
    print(f"❌ Import error: {import_error}")
except Exception as e:
    print(f"❌ RAG system test failed: {e}")
    import traceback
    traceback.print_exc()
