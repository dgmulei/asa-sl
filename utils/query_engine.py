from typing import List, Dict, Any
from dataclasses import dataclass
import chromadb
import numpy as np

@dataclass
class QueryResult:
    text: str
    metadata: Dict[str, Any]
    relevance_score: float

class QueryEngine:
    def __init__(self, collection: Any):
        """Initialize the query engine with a Chroma collection."""
        self.collection = collection

    def query(self, query_text: str, n_results: int = 3) -> List[QueryResult]:
        """
        Query the vector database and return relevant documents.
        
        Args:
            query_text: The query string
            n_results: Number of results to return
            
        Returns:
            List of QueryResult objects containing matched text, metadata, and relevance scores
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            query_results = []
            if results['documents'] and results['documents'][0]:  # Check if we have results
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    # Convert distance to similarity score (1 - normalized_distance)
                    similarity = 1.0 - (float(distance) / 2.0)  # Normalize distance to [0,1]
                    
                    query_result = QueryResult(
                        text=doc,
                        metadata=metadata,
                        relevance_score=similarity
                    )
                    query_results.append(query_result)
            
            return query_results
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return []
