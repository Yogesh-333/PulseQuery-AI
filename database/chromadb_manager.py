import chromadb
import os
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import logging

class ChromaDBManager:
    def __init__(self, persist_directory: str = "data/chromadb"):
        """Initialize ChromaDB with persistent storage"""
        self.persist_directory = persist_directory
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Initialize collections
        self.documents_collection = self.client.get_or_create_collection(
            name="medical_documents"
        )
        self.sessions_collection = self.client.get_or_create_collection(
            name="user_sessions"
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ChromaDB initialized with {len(self.get_all_documents())} existing documents")

    def add_document(self, text: str, metadata: Dict, doc_id: Optional[str] = None) -> str:
        """Add a document chunk to the collection"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # Add timestamp to metadata
        metadata["ingestion_time"] = datetime.utcnow().isoformat()
        
        self.documents_collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return doc_id

    def search_documents(self, query: str, n_results: int = 5, 
                        doc_type: Optional[str] = None) -> List[Dict]:
        """Search documents using semantic similarity"""
        where_filter = {}
        if doc_type:
            where_filter["doc_type"] = doc_type
        
        results = self.documents_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        # Format results
        formatted_results = []
        if results['ids'][0]:  # Check if we have results
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity": 1 - results['distances'][0][i] if 'distances' in results else 0.0
                })
        
        return formatted_results

    def get_all_documents(self) -> List[Dict]:
        """Get all documents in the collection"""
        try:
            results = self.documents_collection.get()
            formatted_docs = []
            
            if results['ids']:
                for i in range(len(results['ids'])):
                    formatted_docs.append({
                        "id": results['ids'][i],
                        "text": results['documents'][i][:200] + "..." if len(results['documents'][i]) > 200 else results['documents'][i],
                        "metadata": results['metadatas'][i]
                    })
            
            return formatted_docs
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return []

    def get_collection_stats(self) -> Dict:
        """Get statistics about the document collection"""
        try:
            doc_count = self.documents_collection.count()
            return {
                "total_documents": doc_count,
                "collection_name": "medical_documents",
                "storage_path": self.persist_directory
            }
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {"total_documents": 0, "error": str(e)}

    def delete_document(self, doc_id: str) -> bool:
        """Delete a specific document"""
        try:
            self.documents_collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def add_session(self, session_data: Dict) -> str:
        """Add session data"""
        session_id = str(uuid.uuid4())
        session_data["timestamp"] = datetime.utcnow().isoformat()
        
        self.sessions_collection.add(
            documents=[str(session_data)],
            metadatas=[session_data],
            ids=[session_id]
        )
        
        return session_id
