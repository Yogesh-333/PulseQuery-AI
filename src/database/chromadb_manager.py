# src/database/chromadb_manager.py

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import uuid # To generate unique IDs for documents

class ChromaDBManager:
    """
    Manages interactions with ChromaDB for document storage and retrieval.
    """
    def __init__(self,
                 persist_directory: str = "chroma_db",
                 collection_name: str = "medical_knowledge_base",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the ChromaDB client and collection.
        Args:
            persist_directory (str): Path to store the ChromaDB data (for persistent client).
            collection_name (str): Name of the collection to interact with.
            embedding_model_name (str): The model name for generating embeddings (must match
                                        the one used for ingestion and query embedding).
        """
        # Using a persistent client so data is saved across runs
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Define the embedding function for the collection
        # This ensures embeddings are consistent during ingestion and retrieval
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
            # You can also specify other settings here if needed, e.g., metadata
        )
        print(f"ChromaDB initialized with collection: '{collection_name}' at '{persist_directory}'")
        print(f"Current document count in collection: {self.collection.count()}")

    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Adds documents to the ChromaDB collection.
        Documents should be a list of dictionaries with 'text' and optional 'metadata'.
        Example: [{'text': '...', 'metadata': {'source': 'guidelineX'}}]
        
        Args:
            documents (List[Dict[str, str]]): List of documents. Each dict must have a 'text' key.
                                              A unique ID will be generated for each if not provided.
        """
        texts = [doc['text'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        # Generate unique IDs for documents
        ids = [str(uuid.uuid4()) for _ in documents] 

        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(documents)} documents to ChromaDB.")
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")

    def query_documents(self, query_texts: List[str], n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a semantic search on the ChromaDB collection.
        Args:
            query_texts (List[str]): List of texts to query with.
            n_results (int): Number of similar documents to retrieve.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing retrieved 'document',
                                  'metadata', and 'distance' for each query.
        """
        results = self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] # Specify what to retrieve
        )
        
        # Flatten results for easier consumption
        retrieved_data = []
        for i, query_text in enumerate(query_texts):
            query_results = {
                "query": query_text,
                "results": []
            }
            if results['documents'] and results['documents'][i]:
                for j in range(len(results['documents'][i])):
                    query_results['results'].append({
                        "document": results['documents'][i][j],
                        "metadata": results['metadatas'][i][j] if results['metadatas'] and results['metadatas'][i] else {},
                        "distance": results['distances'][i][j] if results['distances'] and results['distances'][i] else None
                    })
            retrieved_data.append(query_results)
            
        return retrieved_data

    def get_document_count(self) -> int:
        """Returns the number of documents in the collection."""
        return self.collection.count()

    def delete_collection(self):
        """Deletes the entire collection. Use with caution!"""
        self.client.delete_collection(name=self.collection.name)
        print(f"Collection '{self.collection.name}' deleted.")


# --- Testing the ChromaDBManager (add this to prompt_optimizer.py's __main__ block) ---
# We will temporarily add this test to prompt_optimizer.py's __main__ block
# to demonstrate its usage.