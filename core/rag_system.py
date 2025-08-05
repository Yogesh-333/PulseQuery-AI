"""
Enhanced RAG System with MedEmbed Embeddings for Medical Documents
Specialized embeddings for improved medical document retrieval and understanding
"""

import os
import uuid
import logging
import traceback
import torch
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import tempfile
import shutil

# ✅ UPDATED: Fixed deprecated imports
try:
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Database for persistence
import sqlite3

class DocumentDatabase:
    """SQLite database for document metadata persistence"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    file_name TEXT,
                    doc_type TEXT,
                    language TEXT,
                    file_size INTEGER,
                    chunks_created INTEGER,
                    processed_at TEXT,
                    embedding_model TEXT,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TEXT,
                    last_accessed TEXT,
                    metadata TEXT
                )
            ''')
    
    def store_document_metadata(self, metadata: Dict[str, Any]):
        """Store document metadata"""
        try:
            doc_id = str(uuid.uuid4())
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO documents 
                    (id, file_name, doc_type, language, file_size, processed_at, embedding_model, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc_id,
                    metadata.get('file_name', ''),
                    metadata.get('doc_type', 'medical'),
                    metadata.get('language', 'en'),
                    metadata.get('file_size', 0),
                    metadata.get('processed_at', ''),
                    metadata.get('embedding_model', ''),
                    str(metadata)
                ))
        except Exception as e:
            logging.error(f"Failed to store document metadata: {e}")
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all document metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT * FROM documents ORDER BY processed_at DESC')
                columns = [desc[0] for desc in cursor.description]
                
                docs = []
                for row in cursor.fetchall():
                    doc = dict(zip(columns, row))
                    docs.append(doc)
                
                return docs
        except Exception as e:
            logging.error(f"Failed to get documents: {e}")
            return []

class RAGSystem:
    """
    Enhanced RAG system with MedEmbed-base embeddings for medical documents
    """
    
    def __init__(self, data_dir: str = "data/chromadb", embedding_model: str = "medical"):
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # ✅ FIXED: Complete embedding configurations with all expected keys
        self.embedding_configs = {
            "medical": "abhinand/MedEmbed-base-v0.1",           # Primary medical model
            "medembed": "abhinand/MedEmbed-base-v0.1",          # Alias for compatibility
            "medembed_small": "abhinand/MedEmbed-small-v0.1",   # Smaller variant
            "medembed_large": "abhinand/MedEmbed-large-v0.1",   # Larger variant
            "clinical": "emilyalsentzer/Bio_ClinicalBERT",      # Alternative clinical model
            "general": "sentence-transformers/all-mpnet-base-v2"  # Fallback general model
        }
        
        # ✅ FIXED: Validate embedding model key exists
        if embedding_model not in self.embedding_configs:
            available_models = list(self.embedding_configs.keys())
            raise ValueError(f"Embedding model '{embedding_model}' not found. "
                           f"Available models: {available_models}")
        
        # Select embedding model
        self.embedding_model_name = self.embedding_configs[embedding_model]
        self.embedding_model_key = embedding_model
        
        print(f"🧠 Selected embedding model: {self.embedding_model_name}")
        
        # Storage
        self.vector_store = None
        self.embeddings = None
        self.db_manager = None
        self.documents_metadata = []
        self.fallback_storage = {}
        
        # Text processing with medical optimization
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,   # Smaller chunks for precise medical information
            chunk_overlap=150,  # More overlap to preserve medical context
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
        )
        
        # Initialize components
        self._init_directories()
        self._init_embeddings()
        self._init_vector_store()
        self._init_database()
        
        print("✅ RAG System initialized with medical embeddings")
    
    def _init_directories(self):
        """Create necessary directories"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "documents"), exist_ok=True)
        print(f"📁 Data directory: {self.data_dir}")
    
    def _init_embeddings(self):
        """Initialize medical-specific embedding model"""
        try:
            print(f"🔄 Loading medical embeddings: {self.embedding_model_name}")
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🖥️ Using device: {device}")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={
                    'device': device,
                    'trust_remote_code': True  # For specialized models
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32  # Optimize batch size
                }
            )
            
            print(f"✅ Medical embeddings initialized: {self.embedding_model_name}")
            print(f"🧠 Device: {device} | Model: Medical-specialized")
            
        except Exception as e:
            print(f"❌ Medical embeddings failed: {e}")
            print("🔄 Falling back to general model...")
            
            # Fallback to general model
            try:
                self.embedding_model_name = self.embedding_configs["general"]
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                print(f"✅ Fallback embeddings initialized: {self.embedding_model_name}")
            except Exception as fallback_error:
                print(f"❌ Fallback embeddings also failed: {fallback_error}")
                self.embeddings = None
    
    def _init_vector_store(self):
        """Initialize ChromaDB vector store with medical embeddings"""
        try:
            if self.embeddings:
                self.vector_store = Chroma(
                    persist_directory=self.data_dir,
                    embedding_function=self.embeddings,
                    collection_name="medical_documents_medembed"  # Medical-specific collection
                )
                print("✅ ChromaDB vector store initialized with medical embeddings")
                
                # Check existing documents
                try:
                    collection_count = self.vector_store._collection.count()
                    print(f"📊 Existing documents in collection: {collection_count}")
                except:
                    print("📊 New collection created")
                    
            else:
                # Fallback to in-memory storage
                self.vector_store = None
                self.fallback_storage = {}
                print("⚠️ Using fallback in-memory storage")
                
        except Exception as e:
            print(f"❌ Vector store initialization failed: {e}")
            self.vector_store = None
            self.fallback_storage = {}
    
    def _init_database(self):
        """Initialize SQLite database for metadata"""
        try:
            db_path = os.path.join(self.data_dir, "documents.db")
            self.db_manager = DocumentDatabase(db_path)
            print("✅ Document database initialized")
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            self.db_manager = None
    
    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        ✅ CRITICAL FIX: Sanitize metadata to ensure all values are strings
        Fixes 'tuple' object has no attribute 'lower' error
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, tuple):
                sanitized[key] = str(value[0]) if value else ""
                print(f"🔧 Fixed tuple metadata: {key} = {value} -> {sanitized[key]}")
            elif isinstance(value, list):
                sanitized[key] = ', '.join(str(v) for v in value) if value else ""
                print(f"🔧 Fixed list metadata: {key}")
            elif value is None:
                sanitized[key] = ""
            else:
                sanitized[key] = str(value)
        
        return sanitized
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document with enhanced error handling and metadata sanitization"""
        print(f"🔍 Loading document: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        print(f"📄 File type: {file_extension}")
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            print(f"📑 Loaded {len(documents)} document pages/sections")
            
            # ✅ CRITICAL FIX: Sanitize all metadata
            for i, doc in enumerate(documents):
                # Sanitize metadata to prevent tuple/list errors
                doc.metadata = self.sanitize_metadata(doc.metadata)
                
                # Add processing metadata
                doc.metadata.update({
                    'file_name': os.path.basename(file_path),
                    'file_size': str(os.path.getsize(file_path)),
                    'processed_at': datetime.now(timezone.utc).isoformat(),
                    'document_index': str(i),
                    'embedding_model': self.embedding_model_name
                })
                
                print(f"✅ Sanitized metadata for doc {i}")
            
            return documents
            
        except Exception as e:
            print(f"❌ Document loading failed: {e}")
            traceback.print_exc()
            raise
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks optimized for medical content"""
        print(f"🔪 Chunking {len(documents)} documents with medical-optimized splitting...")
        
        chunks = []
        for doc_idx, document in enumerate(documents):
            # Split document into chunks
            doc_chunks = self.text_splitter.split_documents([document])
            
            # Add chunk metadata
            for chunk_idx, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    'chunk_index': str(chunk_idx),
                    'total_chunks': str(len(doc_chunks)),
                    'chunk_size': str(len(chunk.page_content)),
                    'parent_doc_index': str(doc_idx),
                    'chunk_type': 'medical_optimized'
                })
                chunks.append(chunk)
        
        print(f"✅ Created {len(chunks)} medical-optimized chunks from {len(documents)} documents")
        return chunks
    
    def embed_and_store(self, chunks: List[Document], doc_type: str = 'medical', language: str = 'en') -> Dict[str, Any]:
        """Embed chunks using medical embeddings and store in vector database"""
        print(f"🔄 Embedding {len(chunks)} chunks with medical embeddings...")
        print(f"🧠 Using model: {self.embedding_model_name}")
        
        try:
            # Add additional metadata
            for chunk in chunks:
                chunk.metadata.update({
                    'doc_type': doc_type,
                    'language': language,
                    'embedding_model': self.embedding_model_name,
                    'embedding_type': 'medical_specialized',
                    'ingestion_time': datetime.now(timezone.utc).isoformat()
                })
            
            if self.vector_store:
                # Store in ChromaDB with medical embeddings
                ids = [str(uuid.uuid4()) for _ in chunks]
                
                print("🔄 Computing medical embeddings...")
                self.vector_store.add_documents(documents=chunks, ids=ids)
                print(f"✅ Stored {len(chunks)} chunks with medical embeddings in ChromaDB")
                
                # Persist the database
                if hasattr(self.vector_store, 'persist'):
                    self.vector_store.persist()
                    print("💾 Vector database persisted")
                
            else:
                # Fallback storage
                for i, chunk in enumerate(chunks):
                    chunk_id = str(uuid.uuid4())
                    self.fallback_storage[chunk_id] = {
                        'content': chunk.page_content,
                        'metadata': chunk.metadata,
                        'id': chunk_id
                    }
                print(f"✅ Stored {len(chunks)} chunks in fallback storage")
            
            # Store metadata in database
            if self.db_manager:
                for chunk in chunks:
                    self.db_manager.store_document_metadata(chunk.metadata)
            
            return {
                'success': True,
                'chunks_created': len(chunks),
                'doc_type': doc_type,
                'language': language,
                'embedding_model': self.embedding_model_name,
                'embedding_type': 'medical_specialized'
            }
            
        except Exception as e:
            print(f"❌ Medical embedding and storage failed: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'chunks_created': 0
            }
    
    def ingest_document_from_file(self, file_path: str, doc_type: str = 'medical', language: str = 'en') -> Dict[str, Any]:
        """Complete document ingestion pipeline with medical embeddings"""
        print(f"🔍 Processing with medical embeddings: {os.path.basename(file_path)}")
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            print(f"📄 File size: {file_size} bytes")
            
            # Load documents
            documents = self.load_document(file_path)
            
            # Chunk documents with medical optimization
            chunks = self.chunk_documents(documents)
            
            # Embed and store with medical embeddings
            result = self.embed_and_store(chunks, doc_type, language)
            
            # Add file info to result
            result.update({
                'file_name': os.path.basename(file_path),
                'file_size': file_size,
                'documents_loaded': len(documents),
                'extraction_method': 'Medical-optimized',
                'embedding_model_used': self.embedding_model_name
            })
            
            if result['success']:
                print(f"✅ Successfully processed {file_path} with medical embeddings")
                print(f"📊 Created {result['chunks_created']} chunks with medical embeddings")
            else:
                print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to ingest document {file_path}: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'file_name': os.path.basename(file_path) if file_path else 'unknown',
                'chunks_created': 0
            }
    
    def search_relevant_context(self, query: str, max_docs: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using medical-specialized semantic similarity
        """
        print(f"🔍 Medical search for: {query[:100]}...")
        print(f"🧠 Using medical embeddings: {self.embedding_model_name}")
        
        try:
            if self.vector_store:
                # ChromaDB search with medical embeddings
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=max_docs
                )
                
                formatted_results = []
                for doc, score in results:
                    # Convert distance to similarity score (ChromaDB returns distance)
                    similarity_score = 1.0 - score if score <= 1.0 else 1.0 / (1.0 + score)
                    
                    formatted_results.append({
                        'id': str(uuid.uuid4()),
                        'text': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity': float(similarity_score),
                        'relevance_score': float(similarity_score),
                        'embedding_model': self.embedding_model_name,
                        'search_type': 'medical_semantic'
                    })
                
                print(f"✅ Found {len(formatted_results)} relevant medical documents")
                
                # Log similarity scores for debugging
                for i, result in enumerate(formatted_results):
                    print(f"   Doc {i+1}: {result['similarity']:.3f} similarity (medical)")
                
                return formatted_results
                
            else:
                # Enhanced fallback: Medical term-aware text matching
                results = []
                query_lower = query.lower()
                
                # Medical-specific keywords for enhanced matching
                medical_terms = ['patient', 'diagnosis', 'treatment', 'symptoms', 'medication', 
                               'clinical', 'medical', 'therapy', 'disease', 'condition']
                
                for doc_id, doc_data in self.fallback_storage.items():
                    content_lower = doc_data['content'].lower()
                    
                    # Enhanced relevance scoring
                    term_matches = sum(1 for word in query_lower.split() if word in content_lower)
                    medical_term_boost = sum(1 for term in medical_terms 
                                           if term in content_lower and term in query_lower)
                    
                    relevance = term_matches + (medical_term_boost * 2)  # Boost medical terms
                    
                    if relevance > 0:
                        similarity = min(relevance / max(len(query.split()), 1), 1.0)
                        results.append({
                            'id': doc_id,
                            'text': doc_data['content'],
                            'metadata': doc_data['metadata'],
                            'similarity': similarity,
                            'relevance_score': relevance,
                            'search_type': 'medical_fallback'
                        })
                
                # Sort by relevance
                results.sort(key=lambda x: x['relevance_score'], reverse=True)
                results = results[:max_docs]
                
                print(f"✅ Found {len(results)} relevant documents (medical fallback search)")
                return results
                
        except Exception as e:
            print(f"❌ Medical search failed: {e}")
            traceback.print_exc()
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics with medical embedding info"""
        try:
            total_docs = 0
            if self.vector_store:
                try:
                    collection = self.vector_store._collection
                    total_docs = collection.count()
                except:
                    total_docs = "Unknown"
            else:
                total_docs = len(self.fallback_storage)
            
            return {
                'total_documents': total_docs,
                'vector_store_type': 'ChromaDB' if self.vector_store else 'Fallback',
                'embedding_model': self.embedding_model_name,
                'embedding_type': 'Medical-specialized',
                'data_directory': self.data_dir,
                'database_available': self.db_manager is not None,
                'gpu_available': torch.cuda.is_available(),
                'device_used': 'cuda' if torch.cuda.is_available() else 'cpu',
                'status': 'Ready (Medical Embeddings)',
                'chunking_strategy': 'Medical-optimized'
            }
            
        except Exception as e:
            return {
                'total_documents': 0,
                'status': 'Error',
                'error': str(e)
            }
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get metadata for all documents with medical embedding info"""
        try:
            if self.db_manager:
                return self.db_manager.get_all_documents()
            else:
                docs = []
                for doc_id, doc_data in self.fallback_storage.items():
                    docs.append({
                        'id': doc_id,
                        'metadata': doc_data['metadata']
                    })
                return docs
                
        except Exception as e:
            print(f"❌ Failed to get documents: {e}")
            return []

# Test function
def test_medical_rag_system():
    """Test the RAG system with medical embeddings"""
    print("🧪 Testing Medical RAG System with MedEmbed...")
    
    try:
        # Initialize with medical embeddings
        rag = RAGSystem(embedding_model="medical")
        
        # Test system stats
        stats = rag.get_system_stats()
        print(f"📊 Medical system stats: {stats}")
        
        # Test medical search
        medical_queries = [
            "chest pain and shortness of breath",
            "diabetes type 2 management",
            "post operative complications"
        ]
        
        for query in medical_queries:
            results = rag.search_relevant_context(query)
            print(f"🔍 '{query}': {len(results)} medical documents found")
        
        print("✅ Medical RAG System test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Medical RAG System test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_medical_rag_system()
