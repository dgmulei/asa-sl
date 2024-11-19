from typing import List, Dict, Any, Union, Optional, Sequence, cast
from dataclasses import dataclass
import sys
from typing_extensions import TypedDict, NotRequired

# Import pysqlite3 through pip-installed package
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from chromadb.api.types import (
    CollectionMetadata,
    Documents,
    Embeddings,
    Metadata as ChromaMetadata
)
import numpy as np

class DocumentMetadata(TypedDict):
    source: NotRequired[str]
    page: NotRequired[int]
    
@dataclass
class QueryResult:
    text: str
    metadata: ChromaMetadata
    distance: float

class QueryEngine:
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add_documents(self, texts: List[str], metadatas: List[ChromaMetadata], ids: List[str]) -> None:
        if not texts or not metadatas or not ids:
            return
            
        # Convert lists to ChromaDB's expected types
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_text: str, n_results: int = 3) -> List[QueryResult]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        query_results: List[QueryResult] = []
        
        # Safely extract results with defaults
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[{}] * len(documents)])[0]
        distances = (results.get("distances") or [[0.0] * len(documents)])[0]
        
        # Create QueryResult objects only if we have documents
        if documents:
            for doc, meta, dist in zip(documents, metadatas, distances):
                query_results.append(
                    QueryResult(
                        text=str(doc),
                        metadata=meta or {},
                        distance=float(dist)
                    )
                )
        
        return query_results
