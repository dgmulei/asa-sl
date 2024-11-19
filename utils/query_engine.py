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
from chromadb.config import Settings
from chromadb.api.types import (
    CollectionMetadata,
    Documents,
    Embeddings,
    Metadata as ChromaMetadata,
    QueryResult as ChromaQueryResult
)
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DocumentMetadata(TypedDict):
    source: NotRequired[str]
    page: NotRequired[int]
    
@dataclass
class QueryResult:
    text: str
    metadata: ChromaMetadata
    distance: float

class QueryEngine:
    def __init__(self, collection_name: str = "real_estate_docs"):
        # Initialize ChromaDB with persistent settings
        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )
        
        # Use persistent client with settings
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=settings
        )
        
        # Get the collection - don't create if it doesn't exist
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Successfully connected to collection: {collection_name}")
        except ValueError as e:
            logger.warning(f"Collection {collection_name} does not exist yet")
            self.collection = None
    
    def add_documents(self, texts: List[str], metadatas: List[ChromaMetadata], ids: List[str]) -> None:
        if not texts or not metadatas or not ids or not self.collection:
            return
            
        # Add documents to collection
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_text: str, n_results: int = 3) -> List[QueryResult]:
        if not self.collection:
            logger.warning("No collection available for querying")
            return []
            
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            query_results: List[QueryResult] = []
            
            # Check if we have valid results
            if not results or not isinstance(results, dict):
                logger.warning("Query returned no results")
                return []
            
            # Get the results with safe defaults
            result_dict = cast(Dict[str, List[List[Any]]], results)
            documents_list = result_dict.get("documents", [[]])
            metadatas_list = result_dict.get("metadatas", [[]])
            distances_list = result_dict.get("distances", [[]])
            
            # Ensure we have at least one result
            if not documents_list or not documents_list[0]:
                return []
                
            # Get the first (and only) set of results
            documents = documents_list[0]
            metadatas = metadatas_list[0] if metadatas_list else [{}] * len(documents)
            distances = distances_list[0] if distances_list else [0.0] * len(documents)
            
            # Create QueryResult objects
            for doc, meta, dist in zip(documents, metadatas, distances):
                query_results.append(
                    QueryResult(
                        text=str(doc),
                        metadata=meta or {},
                        distance=float(dist)
                    )
                )
            
            return query_results
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return []
