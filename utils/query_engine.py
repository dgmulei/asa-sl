from typing import List, Dict, Any
from dataclasses import dataclass
import chromadb

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
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        query_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            query_result = QueryResult(
                text=doc,
                metadata=metadata,
                relevance_score=1 - distance  # Convert distance to similarity score
            )
            query_results.append(query_result)
        
        return query_results